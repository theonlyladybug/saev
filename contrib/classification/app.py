import os.path

import beartype
import einops.layers.torch
import gradio as gr
import torch
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.visuals

from . import training

####################
# Global Constants #
####################

DEBUG = True
"""Whether we are debugging."""

n_sae_latents = 3
"""Number of SAE latents to show."""

n_sae_examples = 4
"""Number of SAE examples per latent to show."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Hardware accelerator, if any."""

model_ckpt = "ViT-B-16/openai"
"""CLIP checkpoint."""

sae_ckpt = "rscsjxgd"
"""Which SAE checkpoint to use."""

max_frequency = 1e-2
"""Maximum frequency. Any feature that fires more than this is ignored."""

##########
# Models #
##########


@jaxtyped(typechecker=beartype.beartype)
class SplitClip(torch.nn.Module):
    def __init__(self, *, n_end_layers: int):
        super().__init__()
        import open_clip

        if model_ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                model_ckpt, cache_dir=saev.helpers.get_cache_dir()
            )
        else:
            arch, ckpt = model_ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=saev.helpers.get_cache_dir()
            )
        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.vit = model.eval()
        assert not isinstance(self.vit, open_clip.timm_model.TimmModel)

        self.n_end_layers = n_end_layers

    @staticmethod
    def _expand_token(token, batch_size: int):
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def forward_start(self, x: Float[Tensor, "batch channels width height"]):
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self._expand_token(self.vit.class_embedding, x.shape[0]).to(x.dtype), x],
            dim=1,
        )
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)

        x = self.vit.patch_dropout(x)
        x = self.vit.ln_pre(x)
        for r in self.vit.transformer.resblocks[: -self.n_end_layers]:
            x = r(x)
        return x

    def forward_end(self, x: Float[Tensor, "batch n_patches dim"]):
        for r in self.vit.transformer.resblocks[-self.n_end_layers :]:
            x = r(x)

        x = self.vit.ln_post(x)
        pooled, _ = self.vit._global_pool(x)
        if self.vit.proj is not None:
            pooled = pooled @ self.vit.proj

        return pooled


split_vit = SplitClip(n_end_layers=1)
split_vit = split_vit.to(device)

clf_ckpt_fpath = "/home/stevens.994/projects/saev/checkpoints/contrib/classification/lr_0_001__wd_0_1/model.pt"
clf = training.load_model(clf_ckpt_fpath)
clf = clf.to(device).eval()

sae_ckpt_fpath = f"/home/stevens.994/projects/saev/checkpoints/{sae_ckpt}/sae.pt"
sae = saev.nn.load(sae_ckpt_fpath)
sae.to(device).eval()


############
# Datasets #
############

acts_dataset = saev.activations.Dataset(
    saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/30a3ef8c5467730c8edf1f9f459e8d1da9eaa6ba13f1868c9b4d5dd2f9bc3dbe",
        scale_mean=not DEBUG,
        scale_norm=not DEBUG,
        layer=-2,
    )
)

to_array = v2.Compose([
    v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
    v2.CenterCrop((448, 448)),
    v2.ToImage(),
    einops.layers.torch.Rearrange("channels width height -> width height channels"),
])

human_dataset = saev.activations.ImageFolder(
    "/research/nfs_su_809/workspace/stevens.994/datasets/cub2011/test",
    transform=to_array,
)

vit_dataset = saev.activations.ImageFolder(
    "/research/nfs_su_809/workspace/stevens.994/datasets/cub2011/test",
    transform=saev.activations.make_img_transform("clip", model_ckpt),
)

examples_dataset = saev.activations.ImageFolder(
    "/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini",
    transform=v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.CenterCrop(size=(448, 448)),
    ]),
)

#############
# Variables #
#############


@beartype.beartype
def load_tensor(path: str) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


ckpt_data_root = f"/research/nfs_su_809/workspace/stevens.994/saev/features/{sae_ckpt}-high-freq/sort_by_patch"

top_img_i = load_tensor(os.path.join(ckpt_data_root, "top_img_i.pt"))
top_values = load_tensor(os.path.join(ckpt_data_root, "top_values.pt"))
sparsity = load_tensor(os.path.join(ckpt_data_root, "sparsity.pt"))

mask = torch.ones((sae.cfg.d_sae), dtype=bool)
mask = mask & (sparsity < max_frequency)


@beartype.beartype
def make_img(
    elem: saev.visuals.GridElement, *, upper: float | None = None
) -> Image.Image:
    # Resize to 256x256 and crop to 224x224
    resize_size_px = (512, 512)
    resize_w_px, resize_h_px = resize_size_px
    crop_size_px = (448, 448)
    crop_w_px, crop_h_px = crop_size_px
    crop_coords_px = (
        (resize_w_px - crop_w_px) // 2,
        (resize_h_px - crop_h_px) // 2,
        (resize_w_px + crop_w_px) // 2,
        (resize_h_px + crop_h_px) // 2,
    )

    img = elem.img.resize(resize_size_px).crop(crop_coords_px)
    img = saev.imaging.add_highlights(
        img, elem.patches.numpy(), upper=upper, opacity=0.5
    )
    return img


#############
# Inference #
#############


def get_image(image_i: int) -> Image.Image:
    image = human_dataset[image_i]["image"]
    return Image.fromarray(image.numpy())


@torch.inference_mode
def get_sae_examples(
    image_i: int, patches: list[int]
) -> list[None | Image.Image | int]:
    """
    Given a particular cell, returns some highlighted images showing what feature fires most on this cell.
    """
    if not patches:
        return [None] * 12 + [-1] * 3

    vit_acts_MD = torch.stack([
        acts_dataset[image_i * acts_dataset.metadata.n_patches_per_img + i]["act"]
        for i in patches
    ]).to(device)

    _, f_x_MS, _ = sae(vit_acts_MD)
    f_x_S = f_x_MS.sum(axis=0)

    latents = torch.argsort(f_x_S, descending=True).cpu()
    latents = latents[mask[latents]][:n_sae_latents].tolist()

    images = []
    for latent in latents:
        elems, seen_i_im = [], set()
        for i_im, values_p in zip(top_img_i[latent].tolist(), top_values[latent]):
            if i_im in seen_i_im:
                continue

            example = examples_dataset[i_im]
            elems.append(
                saev.visuals.GridElement(example["image"], example["label"], values_p)
            )
            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[latent].numel() > 0:
            upper = top_values[latent].max().item()

        latent_images = [make_img(elem, upper=upper) for elem in elems[:n_sae_examples]]

        while len(latent_images) < n_sae_examples:
            latent_images += [None]

        images.extend(latent_images)

    return images + latents


@torch.inference_mode
def get_pred_dist(i: int) -> dict[str, float]:
    sample = vit_dataset[i]
    x = sample["image"][None, ...].to(device)
    x_BPD = split_vit.forward_start(x)
    x_BD = split_vit.forward_end(x_BPD)

    logits_BC = clf(x_BD)

    probs = torch.nn.functional.softmax(logits_BC[0], dim=0).cpu().tolist()
    return {name: prob for name, prob in zip(vit_dataset.classes, probs)}


#############
# Interface #
#############


with gr.Blocks() as demo:
    image_number = gr.Number(label="Test Example")
    input_image = gr.Image(label="Input Image", format="png")
    get_input_image_btn = gr.Button(value="Get Input Image")
    get_input_image_btn.click(
        get_image,
        inputs=[image_number],
        outputs=input_image,
        api_name="get-image",
    )

    patch_numbers = gr.CheckboxGroup(label="Image Patch", choices=list(range(196)))
    top_latent_numbers = gr.CheckboxGroup(label="Top Latents")
    top_latent_numbers = [
        gr.Number(label=f"Top Latents #{j + 1}") for j in range(n_sae_latents)
    ]
    sae_example_images = [
        gr.Image(label=f"Latent #{j}, Example #{i + 1}", format="png")
        for i in range(n_sae_examples)
        for j in range(n_sae_latents)
    ]
    get_sae_examples_btn = gr.Button(value="Get SAE Examples")
    get_sae_examples_btn.click(
        get_sae_examples,
        inputs=[image_number, patch_numbers],
        outputs=sae_example_images + top_latent_numbers,
        api_name="get-sae-examples",
    )

    pred_dist = gr.Label(label="Pred. Dist.")
    get_pred_dist_btn = gr.Button(value="Get Pred. Distribution")
    get_pred_dist_btn.click(
        get_pred_dist,
        inputs=[image_number],
        outputs=[pred_dist],
        api_name="get-preds",
    )

if __name__ == "__main__":
    demo.launch()
