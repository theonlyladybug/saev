import base64
import functools
import io
import logging
import math
import os
import pathlib
import random
import typing

import beartype
import einops.layers.torch
import gradio as gr
import numpy as np
import open_clip
import torch
import torchvision
from jaxtyping import Float, jaxtyped
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("app.py")


####################
# Global Constants #
####################

DEBUG = True
"""Whether we are debugging."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Hardware accelerator, if any."""

CWD = pathlib.Path(".")

logger.info("Set global constants.")


###########
# Helpers #
###########


@beartype.beartype
def get_cache_dir() -> str:
    """
    Get cache directory from environment variables, defaulting to the current working directory (.)

    Returns:
        A path to a cache directory (might not exist yet).
    """
    cache_dir = ""
    for var in ("HF_HOME", "HF_HUB_CACHE"):
        cache_dir = cache_dir or os.environ.get(var, "")
    return cache_dir or "."


dataset = torchvision.datasets.ImageFolder(
    root="/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini/"
)


@beartype.beartype
def get_dataset_img(i: int) -> tuple[Image.Image, str]:
    img, tgt = dataset[i]
    label = dataset.classes[tgt]
    # iNat21 specific trick
    label = " ".join(label.split("_")[1:])
    return img, label


@beartype.beartype
def make_img(
    img: Image.Image,
    patches: Float[Tensor, " n_patches"],
    *,
    upper: int | None = None,
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

    img = img.resize(resize_size_px).crop(crop_coords_px)
    img = add_highlights(img, patches.numpy(), upper=upper, opacity=0.5)
    return img


##########
# Models #
##########


@jaxtyped(typechecker=beartype.beartype)
class SplitClip(torch.nn.Module):
    def __init__(self, vit_ckpt: str, *, n_end_layers: int):
        super().__init__()

        if vit_ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=get_cache_dir()
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


vit_lookup: dict[str, tuple[str, str]] = {
    "bioclip/inat21": ("clip", "hf-hub:imageomics/bioclip"),
    "clip/inat21": ("clip", "ViT-B-16/openai"),
}


@beartype.beartype
@functools.cache
def load_split_vit(model: str) -> tuple[SplitClip, object]:
    # Translate model key to ckpt. Use the model as the default.
    model_family, model_ckpt = vit_lookup[model]
    split_vit = SplitClip(model_ckpt, n_end_layers=1).to(device).eval()
    vit_transform = saev.activations.make_img_transform(model_family, model_ckpt)
    logger.info("Loaded Split ViT: %s.", model)
    return split_vit, vit_transform


sae_lookup = {
    "bioclip/inat21": "gpnn7x3p",
    "clip/inat21": "rscsjxgd",
}


@beartype.beartype
@functools.cache
def load_sae(model: str) -> saev.nn.SparseAutoencoder:
    sae_ckpt = sae_lookup[model]
    sae_ckpt_fpath = CWD / "checkpoints" / sae_ckpt / "sae.pt"
    sae = saev.nn.load(sae_ckpt_fpath.as_posix())
    sae.to(device).eval()
    logger.info("Loaded SAE: %s -> %s.", model, sae_ckpt)
    return sae


############
# Datasets #
############

human_transform = v2.Compose([
    v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
    v2.CenterCrop((448, 448)),
    v2.ToImage(),
    einops.layers.torch.Rearrange("channels width height -> width height channels"),
])


logger.info("Loaded all datasets.")

#############
# Variables #
#############


@beartype.beartype
def load_tensor(path: str | pathlib.Path) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


top_img_i = load_tensor(
    "/research/nfs_su_809/workspace/stevens.994/saev/features/gpnn7x3p-high-freq/sort_by_patch/top_img_i.pt"
)
top_values = load_tensor(
    "/research/nfs_su_809/workspace/stevens.994/saev/features/gpnn7x3p-high-freq/sort_by_patch/top_values.pt"
)
sparsity = load_tensor(
    "/research/nfs_su_809/workspace/stevens.994/saev/features/gpnn7x3p-high-freq/sort_by_patch/sparsity.pt"
)


#############
# Inference #
#############


@beartype.beartype
def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf8")


@beartype.beartype
def get_image(image_i: int) -> list[str]:
    image, label = get_dataset_img(image_i)
    image = human_transform(image)

    return [pil_to_base64(Image.fromarray(image.numpy())), label]


@beartype.beartype
def get_random_class_image(cls: int) -> Image.Image:
    raise NotImplementedError()
    # TODO
    indices = [i for i, tgt in enumerate(image_labels) if tgt == cls]
    i = random.choice(indices)

    image = get_dataset_img(i)
    image = human_transform(image)
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

    logger.info("Getting SAE examples for patches %s.", patches)

    img = get_dataset_img(image_i)
    x = vit_transform(img)[None, ...].to(device)
    x_BPD = split_vit.forward_start(x)
    # Need to add 1 to account for [CLS] token.
    vit_acts_MD = x_BPD[0, [p + 1 for p in patches]].to(device)

    _, f_x_MS, _ = sae(vit_acts_MD)
    f_x_S = f_x_MS.sum(axis=0)

    latents = torch.argsort(f_x_S, descending=True).cpu()
    latents = latents[mask[latents]][:n_sae_latents].tolist()

    images = []
    for latent in latents:
        img_patch_pairs, seen_i_im = [], set()
        for i_im, values_p in zip(top_img_i[latent].tolist(), top_values[latent]):
            if i_im in seen_i_im:
                continue

            example_img = get_dataset_img(i_im)
            img_patch_pairs.append((example_img, values_p))
            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[latent].numel() > 0:
            upper = top_values[latent].max().item()

        latent_images = [
            make_img(img, patches.to(float), upper=upper)
            for img, patches in img_patch_pairs[:n_sae_examples]
        ]

        while len(latent_images) < n_sae_examples:
            latent_images += [None]

        images.extend(latent_images)

    return images + latents


@beartype.beartype
def unscaled(x: float | int, max_obs: float | int) -> float:
    """Scale from [-20, 20] to [20 * -max_obs, 20 * max_obs]."""
    return map_range(x, (-20.0, 20.0), (-20.0 * max_obs, 20.0 * max_obs))


@beartype.beartype
def map_range(
    x: float | int,
    domain: tuple[float | int, float | int],
    range: tuple[float | int, float | int],
):
    a, b = domain
    c, d = range
    if not (a <= x <= b):
        raise ValueError(f"x={x:.3f} must be in {[a, b]}.")
    return c + (x - a) * (d - c) / (b - a)


@jaxtyped(typechecker=beartype.beartype)
def add_highlights(
    img: Image.Image,
    patches: Float[np.ndarray, " n_patches"],
    *,
    upper: int | None = None,
    opacity: float = 0.9,
) -> Image.Image:
    if not len(patches):
        return img

    iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
    iw_px, ih_px = img.size
    pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
    assert iw_np * ih_np == len(patches)

    # Create a transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Using semi-transparent red (255, 0, 0, alpha)
    for p, val in enumerate(patches):
        assert upper is not None
        val /= upper + 1e-9
        x_np, y_np = p % iw_np, p // ih_np
        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(int(val * 256), 0, 0, int(opacity * val * 256)),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)


@beartype.beartype
class Example(typing.TypedDict):
    # Document this class and variables. AI!
    url: str
    label: str


@beartype.beartype
class SaeActivation(typing.TypedDict):
    """Represents the activation pattern of a single SAE latent across patches.

    This captures how strongly a particular SAE latent fires on different patches of an input image.
    """

    latent: int
    """The index of the SAE latent being measured."""

    activations: list[float]
    """The activation values of this latent across different patches. Each value represents how strongly this latent fired on a particular patch."""

    examples: list[Example]


@beartype.beartype
@torch.inference_mode
def get_sae_activations(
    image: Image.Image, latents: dict[str, list[int]]
) -> dict[str, list[SaeActivation]]:
    """
    Args:
        image: Image to get SAE activations for.
        latents: A lookup from model name (string) to a list of latents to report latents for (integers).

    Returns:
        A lookup from model name (string) to a list of SaeActivations, one for each latent in the `latents` argument.
    """
    response = {}
    for model_name, requested_latents in latents.items():
        sae_activations = []
        split_vit, vit_transform = load_split_vit(model_name)
        sae = load_sae(model_name)
        x = vit_transform(image)[None, ...].to(device)
        vit_acts_PD = split_vit.forward_start(x)[0]

        _, f_x_PS, _ = sae(vit_acts_PD)
        # Ignore [CLS] token and get just the requested latents.
        acts_SP = einops.rearrange(f_x_PS[1:], "patches n_latents -> n_latents patches")
        for latent in requested_latents:
            acts = acts_SP[latent].cpu().tolist()
            sae_activations.append(SaeActivation(latent=latent, activations=acts))
        response[model_name] = sae_activations
    return response


#############
# Interface #
#############


with gr.Blocks() as demo:
    image_number = gr.Number(label="Test Example", precision=0)
    class_number = gr.Number(label="Test Class", precision=0)
    input_image = gr.Image(
        label="Input Image",
        sources=["upload", "clipboard"],
        type="pil",
        interactive=True,
    )

    input_image_base64 = gr.Text(label="Image in Base64")
    input_image_label = gr.Text(label="Image Label")
    get_input_image_btn = gr.Button(value="Get Input Image")
    get_input_image_btn.click(
        get_image,
        inputs=[image_number],
        outputs=[input_image_base64, input_image_label],
        api_name="get-image",
        postprocess=False,
    )

    latents_json = gr.JSON(label="Latents", value={})
    activations_json = gr.JSON(label="Activations", value={})

    get_sae_activations_btn = gr.Button(value="Get SAE Activations")
    get_sae_activations_btn.click(
        get_sae_activations,
        inputs=[input_image, latents_json],
        outputs=[activations_json],
        api_name="get-sae-activations",
    )


if __name__ == "__main__":
    demo.launch()
