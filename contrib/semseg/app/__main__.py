"""
https://www.gradio.app/guides/environment-variables
https://www.gradio.app/guides/querying-gradio-apps-with-curl
"""

import os.path
import random

import beartype
import einops
import einops.layers.torch
import gradio as gr
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.config
import saev.nn
import saev.visuals

from . import training

####################
# Global Constants #
####################


DEBUG = False
"""Whether we are debugging."""

max_frequency = 1e-2
"""Maximum frequency. Any feature that fires more than this is ignored."""

ckpt = "oebd6e6i"
"""Which SAE checkpoint to use."""

n_sae_latents = 3
"""Number of SAE latents to show."""

n_sae_examples = 4
"""Number of SAE examples per latent to show."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Hardware accelerator, if any."""


####################
# Helper Functions #
####################


@beartype.beartype
def load_tensor(path: str) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


##########
# Models #
##########


sae_ckpt_fpath = f"/home/stevens.994/projects/saev/checkpoints/{ckpt}/sae.pt"
sae = saev.nn.load(sae_ckpt_fpath)
sae.to(device).eval()


head_ckpt_fpath = "/home/stevens.994/projects/saev/checkpoints/contrib/semseg/lr_0_001__wd_0_001/model_step8000.pt"
head = training.load(head_ckpt_fpath)
head = head.to(device).eval()


class RestOfDinoV2(torch.nn.Module):
    def __init__(self, *, n_end_layers: int):
        super().__init__()
        self.vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.n_end_layers = n_end_layers

    def forward_start(self, x: Float[Tensor, "batch channels width height"]):
        x_BPD = self.vit.prepare_tokens_with_masks(x)
        for blk in self.vit.blocks[: -self.n_end_layers]:
            x_BPD = blk(x_BPD)

        return x_BPD

    def forward_end(self, x_BPD: Float[Tensor, "batch n_patches dim"]):
        for blk in self.vit.blocks[-self.n_end_layers :]:
            x_BPD = blk(x_BPD)

        x_BPD = self.vit.norm(x_BPD)
        return x_BPD[:, self.vit.num_register_tokens + 1 :]


rest_of_vit = RestOfDinoV2(n_end_layers=1)
rest_of_vit = rest_of_vit.to(device)


####################
# Global Variables #
####################


ckpt_data_root = (
    f"/research/nfs_su_809/workspace/stevens.994/saev/features/{ckpt}/sort_by_patch"
)

top_img_i = load_tensor(os.path.join(ckpt_data_root, "top_img_i.pt"))
top_values = load_tensor(os.path.join(ckpt_data_root, "top_values.pt"))
sparsity = load_tensor(os.path.join(ckpt_data_root, "sparsity.pt"))


mask = torch.ones((sae.cfg.d_sae), dtype=bool)
mask = mask & (sparsity < max_frequency)


############
# Datasets #
############


in1k_dataset = saev.activations.get_dataset(
    saev.config.ImagenetDataset(),
    img_transform=v2.Compose([
        v2.Resize(size=(512, 512)),
        v2.CenterCrop(size=(448, 448)),
    ]),
)


acts_dataset = saev.activations.Dataset(
    saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/a1f842330bb568b2fb05c15d4fa4252fb7f5204837335000d9fd420f120cd03e",
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


human_dataset = saev.activations.Ade20k(
    saev.config.Ade20kDataset(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/"
    ),
    img_transform=to_array,
    seg_transform=to_array,
)


vit_dataset = saev.activations.Ade20k(
    saev.config.Ade20kDataset(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/"
    ),
    img_transform=v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.CenterCrop(size=(224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ]),
)


#######################
# Inference Functions #
#######################


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

            example = in1k_dataset[i_im]
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
def get_true_labels(image_i: int) -> Image.Image:
    seg = human_dataset[image_i]["segmentation"]
    image = seg_to_img(seg)
    return image


@torch.inference_mode
def get_pred_labels(i: int) -> list[Image.Image | list[int]]:
    sample = vit_dataset[i]
    x = sample["image"][None, ...].to(device)
    x_BPD = rest_of_vit.forward_start(x)
    x_BPD = rest_of_vit.forward_end(x_BPD)

    x_WHD = einops.rearrange(x_BPD, "() (w h) dim -> w h dim", w=16, h=16)

    logits_WHC = head(x_WHD)

    pred_WH = logits_WHC.argmax(axis=-1)
    preds = einops.rearrange(pred_WH, "w h -> (w h)").tolist()
    return [seg_to_img(upsample(pred_WH)), preds]


@beartype.beartype
def unscaled(x: float, max_obs: float) -> float:
    """Scale from [-10, 10] to [10 * -max_obs, 10 * max_obs]."""
    return map_range(x, (-10.0, 10.0), (-10.0 * max_obs, 10.0 * max_obs))


@beartype.beartype
def map_range(
    x: float,
    domain: tuple[float | int, float | int],
    range: tuple[float | int, float | int],
):
    a, b = domain
    c, d = range
    if not (a <= x <= b):
        raise ValueError(f"x={x:.3f} must be in {[a, b]}.")
    return c + (x - a) * (d - c) / (b - a)


@torch.inference_mode
def get_modified_labels(
    i: int,
    latent1: int,
    latent2: int,
    latent3: int,
    value1: float,
    value2: float,
    value3: float,
) -> list[Image.Image | list[int]]:
    sample = vit_dataset[i]
    x = sample["image"][None, ...].to(device)
    x_BPD = rest_of_vit.forward_start(x)

    x_hat_BPD, f_x_BPS, _ = sae(x_BPD)

    err_BPD = x_BPD - x_hat_BPD

    values = torch.tensor(
        [
            unscaled(float(value), top_values[latent].max().item())
            for value, latent in [
                (value1, latent1),
                (value2, latent2),
                (value3, latent3),
            ]
        ],
        device=device,
    )
    f_x_BPS[..., torch.tensor([latent1, latent2, latent3], device=device)] = values

    # Reproduce the SAE forward pass after f_x
    modified_x_hat_BPD = (
        einops.einsum(
            f_x_BPS,
            sae.W_dec,
            "batch patches d_sae, d_sae d_vit -> batch patches d_vit",
        )
        + sae.b_dec
    )
    modified_BPD = err_BPD + modified_x_hat_BPD

    modified_BPD = rest_of_vit.forward_end(modified_BPD)

    logits_BPC = head(modified_BPD)
    pred_P = logits_BPC[0].argmax(axis=-1)
    pred_WH = einops.rearrange(pred_P, "(w h) -> w h", w=16, h=16)
    return seg_to_img(upsample(pred_WH)), pred_P.tolist()


@jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode
def upsample(
    x_WH: Int[Tensor, "width_ps height_ps"],
) -> UInt8[Tensor, "width_px height_px"]:
    return (
        torch.nn.functional.interpolate(
            x_WH.view((1, 1, 16, 16)).float(),
            scale_factor=28,
        )
        .view((448, 448))
        .type(torch.uint8)
    )


@jaxtyped(typechecker=beartype.beartype)
def make_colors() -> UInt8[np.ndarray, "n 3"]:
    values = (0, 51, 102, 153, 204, 255)
    colors = []
    for r in values:
        for g in values:
            for b in values:
                colors.append((r, g, b))
    # Fixed seed
    random.Random(42).shuffle(colors)
    colors = np.array(colors, dtype=np.uint8)

    # Fixed colors for example 3122
    colors[2] = np.array([201, 249, 255], dtype=np.uint8)
    colors[4] = np.array([151, 204, 4], dtype=np.uint8)
    colors[13] = np.array([104, 139, 88], dtype=np.uint8)
    colors[16] = np.array([54, 48, 32], dtype=np.uint8)
    colors[26] = np.array([45, 125, 210], dtype=np.uint8)
    colors[46] = np.array([238, 185, 2], dtype=np.uint8)
    colors[52] = np.array([88, 91, 86], dtype=np.uint8)
    colors[72] = np.array([76, 46, 5], dtype=np.uint8)
    colors[94] = np.array([12, 15, 10], dtype=np.uint8)

    return colors


@jaxtyped(typechecker=beartype.beartype)
def seg_to_img(map: UInt8[Tensor, "width height *channel"]) -> Image.Image:
    map = map.cpu().numpy()
    if map.ndim == 3:
        map = einops.rearrange(map, "w h () -> w h")
    colored = np.zeros((448, 448, 3), dtype=np.uint8)
    for i, color in enumerate(make_colors()):
        colored[map == i + 1, :] = color

    return Image.fromarray(colored)


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


with gr.Blocks() as demo:
    image_number = gr.Number(label="Validation Example")
    input_image = gr.Image(label="Input Image", format="png")
    get_input_image_btn = gr.Button(value="Get Input Image")
    get_input_image_btn.click(
        get_image,
        inputs=[image_number],
        outputs=input_image,
        api_name="get-image",
    )

    patch_numbers = gr.CheckboxGroup(label="Image Patch", choices=list(range(256)))
    top_latent_numbers = gr.CheckboxGroup(label="Top Latents")
    top_latent_numbers = [
        gr.Number(label="Top Latents #{j+1}") for j in range(n_sae_latents)
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
    semseg_image = gr.Image(label="Semantic Segmentaions", format="png")
    semseg_colors = gr.CheckboxGroup(
        label="Sem Seg Colors", choices=list(range(1, 151))
    )

    get_pred_labels_btn = gr.Button(value="Get Pred. Labels")
    get_pred_labels_btn.click(
        get_pred_labels,
        inputs=[image_number],
        outputs=[semseg_image, semseg_colors],
        api_name="get-pred-labels",
    )

    get_true_labels_btn = gr.Button(value="Get True Label")
    get_true_labels_btn.click(
        get_true_labels,
        inputs=[image_number],
        outputs=semseg_image,
        api_name="get-true-labels",
    )

    latent_numbers = [gr.Number(label=f"Latent {i + 1}") for i in range(3)]
    value_sliders = [
        gr.Slider(label=f"Value {i + 1}", minimum=-10, maximum=10) for i in range(3)
    ]

    get_modified_labels_btn = gr.Button(value="Get Modified Label")
    get_modified_labels_btn.click(
        get_modified_labels,
        inputs=[image_number] + latent_numbers + value_sliders,
        outputs=[semseg_image, semseg_colors],
        api_name="get-modified-labels",
    )

if __name__ == "__main__":
    demo.launch()
