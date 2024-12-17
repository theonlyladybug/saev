import os.path
import random

import beartype
import einops
import einops.layers.torch
import gradio as gr
import numpy as np
import torch
from jaxtyping import Int, UInt8, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

import contrib.semseg.training
import saev.activations
import saev.config
import saev.nn
import saev.visuals

####################
# Global Constants #
####################


max_frequency = 1e-2
"""Maximum frequency. Any feature that fires more than this is ignored."""

ckpt = "oebd6e6i"
"""Which SAE checkpoint to use."""

n_sae_latents = 3
"""Number of SAE latents to show."""

n_sae_examples = 4
"""Number of SAE examples per latent to show."""


####################
# Helper Functions #
####################


@beartype.beartype
def load_tensor(path: str) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


####################
# Global Variables #
####################


sae_ckpt_fpath = f"/home/stevens.994/projects/saev/checkpoints/{ckpt}/sae.pt"
sae = saev.nn.load(sae_ckpt_fpath)
sae.eval()


head_ckpt_fpath = "/home/stevens.994/projects/saev/checkpoints/contrib/semseg/lr_0_001__wd_0_001/model_step8000.pt"
head = contrib.semseg.training.load(head_ckpt_fpath)
head.eval()


ckpt_data_root = (
    f"/research/nfs_su_809/workspace/stevens.994/saev/features/{ckpt}/sort_by_patch"
)

top_img_i = load_tensor(os.path.join(ckpt_data_root, "top_img_i.pt"))
top_values = load_tensor(os.path.join(ckpt_data_root, "top_values.pt"))
sparsity = load_tensor(os.path.join(ckpt_data_root, "sparsity.pt"))


mask = torch.ones((sae.cfg.d_sae), dtype=bool)
mask = mask & (sparsity < max_frequency)


# Datasets
############


in1k_dataset = saev.activations.get_dataset(
    saev.config.ImagenetDataset(),
    img_transform=v2.Compose([
        v2.Resize(size=512),
        v2.CenterCrop(size=(448, 448)),
    ]),
)


acts_dataset = saev.activations.Dataset(
    saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/1864947033ca8b8a171a482644a948a6e6489e3249469373c78dfeeb0a75bcd4",
        scale_mean=True,
        scale_norm=True,
    )
)


to_array = v2.Compose([
    v2.Resize(512, interpolation=v2.InterpolationMode.NEAREST),
    v2.CenterCrop((448, 448)),
    v2.ToImage(),
    einops.layers.torch.Rearrange("channels width height -> width height channels"),
])


ade20k_dataset = saev.activations.Ade20k(
    saev.config.Ade20kDataset(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/"
    ),
    img_transform=to_array,
    seg_transform=to_array,
)


#######################
# Inference Functions #
#######################


def get_image(image_i: int) -> Image.Image:
    image = ade20k_dataset[image_i]["image"]
    return Image.fromarray(image.numpy())


@torch.inference_mode
def get_sae_examples(
    image_i: int, patches: list[int]
) -> list[None | Image.Image | int]:
    """
    Given a particular cell, returns some highlighted images showing what feature fires most on this cell.
    """
    if not patches:
        return [None, None, None, None, -1]

    vit_acts_MD = torch.stack([
        acts_dataset[image_i * acts_dataset.metadata.n_patches_per_img + i]["act"]
        for i in patches
    ])

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

        latent_images = [
            saev.visuals.make_img(elem, upper=upper) for elem in elems[:n_sae_examples]
        ]

        while len(latent_images) < n_sae_examples:
            latent_images += [None]

        images.extend(latent_images)

    return images + latents


@torch.inference_mode
def get_true_labels(image_i: int) -> Image.Image:
    seg = ade20k_dataset[image_i]["segmentation"]
    image = seg_to_img(seg)
    return image


@torch.inference_mode
def get_pred_labels(image_i: int) -> list[Image.Image | list[int]]:
    acts_PD = torch.stack([
        acts_dataset[image_i * acts_dataset.metadata.n_patches_per_img + i]["act"]
        for i in range(acts_dataset.metadata.n_patches_per_img)
    ])
    acts_WHD = einops.rearrange(acts_PD, "(w h) d -> w h d", w=16, h=16)

    logits_WHC = head(acts_WHD)

    pred_WH = logits_WHC.argmax(axis=-1)
    preds = einops.rearrange(pred_WH, "w h -> (w h)").tolist()
    return seg_to_img(upsample(pred_WH)), preds


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
def get_modified_labels(image_i: int, latent: int, value: float) -> Image.Image:
    acts_PD = torch.stack([
        acts_dataset[image_i * acts_dataset.metadata.n_patches_per_img + i]["act"]
        for i in range(acts_dataset.metadata.n_patches_per_img)
    ])
    acts_WHD = einops.rearrange(acts_PD, "(w h) d -> w h d", w=16, h=16)

    x_hat_WHD, f_x_WHS, _ = sae(acts_WHD)

    err_WHD = acts_WHD - x_hat_WHD

    value = unscaled(float(value), top_values[latent].max().item())
    f_x_WHS[..., latent] = value

    # Reproduce the SAE forward pass after f_x
    modified_x_hat_WHD = (
        einops.einsum(
            f_x_WHS, sae.W_dec, "width height d_sae, d_sae d_vit -> width height d_vit"
        )
        + sae.b_dec
    )
    modified_WHD = err_WHD + modified_x_hat_WHD

    logits_WHC = head(modified_WHD)
    pred_WH = logits_WHC.argmax(axis=-1)
    preds = einops.rearrange(pred_WH, "w h -> (w h)").tolist()
    return seg_to_img(upsample(pred_WH)), preds


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

    return colors


@jaxtyped(typechecker=beartype.beartype)
def seg_to_img(map: UInt8[Tensor, "width height *channel"]) -> Image.Image:
    map = map.numpy()
    if map.ndim == 3:
        map = einops.rearrange(map, "w h () -> w h")
    colored = np.zeros((448, 448, 3), dtype=np.uint8)
    for i, color in enumerate(make_colors()):
        colored[map == i + 1, :] = color

    return Image.fromarray(colored)


with gr.Blocks() as demo:
    image_number = gr.Number(label="Validation Example")
    input_image = gr.Image(label="Input Image")
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
        gr.Image(label=f"Latent #{j}, Example #{i + 1}")
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
    semseg_image = gr.Image()
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

    latent_number = gr.Number(label="Latent")
    value_slider = gr.Slider(label="Value", minimum=-10, maximum=10)

    get_modified_labels_btn = gr.Button(value="Get Modified Label")
    get_modified_labels_btn.click(
        get_modified_labels,
        inputs=[image_number, latent_number, value_slider],
        outputs=semseg_image,
        api_name="get-modified-labels",
    )

demo.launch()
