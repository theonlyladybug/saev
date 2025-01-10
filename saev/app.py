import base64
import dataclasses
import functools
import logging
import math
import os
import pathlib
import time
import typing

import beartype
import einops.layers.torch
import gradio as gr
import line_profiler
import numpy as np
import open_clip
import PIL.Image
import pyvips
import torch
import torchvision
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("app.py")


###########
# Globals #
###########


RESIZE_SIZE = (512, 512)

CROP_SIZE = (448, 448)

CROP_COORDS = (
    (RESIZE_SIZE[0] - CROP_SIZE[0]) // 2,
    (RESIZE_SIZE[1] - CROP_SIZE[1]) // 2,
    (RESIZE_SIZE[0] + CROP_SIZE[0]) // 2,
    (RESIZE_SIZE[1] + CROP_SIZE[1]) // 2,
)

DEBUG = True
"""Whether we are debugging."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Hardware accelerator, if any."""

CWD = pathlib.Path(".")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for a Vision Transformer (ViT) and Sparse Autoencoder (SAE) model pair.

    Stores paths and configuration needed to load and run a specific ViT+SAE combination.
    """

    vit_family: str
    """The family of ViT model, e.g. 'clip' for CLIP models."""

    vit_ckpt: str
    """Checkpoint identifier for the ViT model, either as HuggingFace path or model/checkpoint pair."""

    sae_ckpt: str
    """Identifier for the SAE checkpoint to load."""

    tensor_dpath: pathlib.Path
    """Directory containing precomputed tensors for this model combination."""


MODEL_LOOKUP: dict[str, ModelConfig] = {
    "bioclip/inat21": ModelConfig(
        "clip",
        "hf-hub:imageomics/bioclip",
        "gpnn7x3p",
        pathlib.Path(
            "/research/nfs_su_809/workspace/stevens.994/saev/features/gpnn7x3p-high-freq/sort_by_patch/"
        ),
    ),
    "clip/inat21": ModelConfig(
        "clip",
        "ViT-B-16/openai",
        "rscsjxgd",
        pathlib.Path(
            "/research/nfs_su_809/workspace/stevens.994/saev/features/rscsjxgd-high-freq/sort_by_patch/"
        ),
    ),
}


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
    return cache_dir or CWD


class VipsImageFolder(torchvision.datasets.ImageFolder):
    """
    Clone of ImageFolder that returns pyvips.Image instead of PIL.Image.Image.
    """

    def __init__(
        self,
        root: str,
        transform: typing.Callable | None = None,
        target_transform: typing.Callable | None = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=self._vips_loader,
        )

    @staticmethod
    def _vips_loader(path: str) -> torch.Tensor:
        """Load and convert image to tensor using pyvips."""
        image = pyvips.Image.new_from_file(path, access="sequential")
        return image


dataset = VipsImageFolder(
    root="/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini/"
)


@beartype.beartype
def get_dataset_image(i: int) -> tuple[pyvips.Image, str]:
    """
    Get raw image and processed label from dataset.

    Returns:
        Tuple of pyvips.Image and classname.
    """
    img, tgt = dataset[i]
    species_label = dataset.classes[tgt]
    # iNat21 specific: Remove taxonomy prefix
    species_name = " ".join(species_label.split("_")[1:])
    return img, species_name


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


@beartype.beartype
@functools.cache
def load_split_vit(model: str) -> tuple[SplitClip, object]:
    # Translate model key to ckpt. Use the model as the default.
    model_cfg = MODEL_LOOKUP[model]
    split_vit = SplitClip(model_cfg.vit_ckpt, n_end_layers=1).to(DEVICE).eval()
    vit_transform = saev.activations.make_img_transform(
        model_cfg.vit_family, model_cfg.vit_ckpt
    )
    logger.info("Loaded Split ViT: %s.", model)
    return split_vit, vit_transform


@beartype.beartype
@functools.cache
def load_sae(model: str) -> saev.nn.SparseAutoencoder:
    model_cfg = MODEL_LOOKUP[model]
    sae_ckpt_fpath = CWD / "checkpoints" / model_cfg.sae_ckpt / "sae.pt"
    sae = saev.nn.load(sae_ckpt_fpath.as_posix())
    sae.to(DEVICE).eval()
    logger.info("Loaded SAE: %s -> %s.", model, model_cfg.sae_ckpt)
    return sae


@beartype.beartype
def load_tensor(path: str | pathlib.Path) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


@beartype.beartype
@functools.cache
def load_tensors(
    model: str,
) -> tuple[Int[Tensor, "d_sae top_k"], Float[Tensor, "d_sae top_k n_patches"]]:
    model_cfg = MODEL_LOOKUP[model]
    top_img_i = load_tensor(model_cfg.tensor_dpath / "top_img_i.pt")
    top_values = load_tensor(model_cfg.tensor_dpath / "top_values.pt")
    return top_img_i, top_values


############
# Datasets #
############


def to_sized(img_v_raw: pyvips.Image) -> pyvips.Image:
    """Convert raw vips image to standard model input size (resize + crop)."""
    # Calculate scaling factors to reach RESIZE_SIZE
    hscale = RESIZE_SIZE[0] / img_v_raw.width
    vscale = RESIZE_SIZE[1] / img_v_raw.height
    
    # Resize then crop to CROP_COORDS
    resized = img_v_raw.resize(hscale, vscale=vscale)
    return resized.crop(*CROP_COORDS)


human_transform = v2.Compose([
    v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
    v2.CenterCrop((448, 448)),
])


logger.info("Loaded all datasets.")


@beartype.beartype
@line_profiler.profile
def vips_to_base64(image: pyvips.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(
        image.write_to_buffer(".png")
    ).decode("utf8")


@beartype.beartype
def get_image(image_i: int) -> list[str]:
    image, label = get_dataset_image(image_i)
    image = human_transform(image)

    return [vips_to_base64(image), label]


@jaxtyped(typechecker=beartype.beartype)
def add_highlights(
    img_v_sized: pyvips.Image,
    patches: np.ndarray,
    *,
    upper: float | None = None,
    opacity: float = 0.9,
) -> pyvips.Image:
    """Add colored highlights to an image based on patch activation values.

    Overlays a colored highlight on each patch of the image, with intensity proportional
    to the activation value for that patch. Used to visualize which parts of an image
    most strongly activated a particular SAE latent.

    Args:
        img: The base image to highlight
        patches: Array of activation values, one per patch
        upper: Optional maximum value to normalize activations against
        opacity: Opacity of the highlight overlay (0-1)

    Returns:
        A new image with colored highlights overlaid on the original
    """
    if not len(patches):
        return img_v

    # Calculate patch grid dimensions
    grid_w = grid_h = int(math.sqrt(len(patches)))
    assert grid_w * grid_h == len(patches)

    patch_w = img.width // grid_w
    patch_h = img.height // grid_h
    breakpoint()
    assert patch_w == patch_h

    # Convert image to RGBA if needed
    if img.bands < 4:
        img = img.bandjoin(255)  # Add alpha channel

    # Create overlay by processing each patch
    overlay = None
    for idx, val in enumerate(patches):
        assert upper is not None
        val = val / (upper + 1e-9)

        x = (idx % grid_w) * patch_w
        y = (idx // grid_w) * patch_h

        # Create patch overlay
        patch_overlay = pyvips.Image.new_from_array([
            [val * opacity]  # Alpha value for this patch
        ]).resize(patch_w)

        # Create RGBA for this patch (red + alpha)
        patch_rgba = patch_overlay.bandjoin([
            patch_overlay * 0,
            patch_overlay * 0,
            patch_overlay,
        ])

        if overlay is None:
            overlay = patch_rgba.embed(x, y, img.width, img.height)
        else:
            patch_positioned = patch_rgba.embed(x, y, img.width, img.height)
            overlay = overlay.composite(patch_positioned, "over")

    return img.composite(overlay, "over")


@beartype.beartype
class Example(typing.TypedDict):
    """Represents an example image and its associated label.

    Used to store examples of SAE latent activations for visualization.
    """

    orig_url: str
    """The URL or path to access the original example image."""
    highlighted_url: str
    """The URL or path to access the SAE-highlighted image."""
    label: str
    """The class label or description associated with this example."""


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
    """Top examples for this latent."""


@beartype.beartype
def pil_to_vips(pil_img: PIL.Image.Image) -> pyvips.Image:
    # Convert to numpy array
    np_array = np.asarray(pil_img)
    # Handle different formats
    if np_array.ndim == 2:  # Grayscale
        return pyvips.Image.new_from_memory(
            np_array.tobytes(),
            np_array.shape[1],  # width
            np_array.shape[0],  # height
            1,  # bands
            "uchar",
        )
    else:  # RGB/RGBA
        return pyvips.Image.new_from_memory(
            np_array.tobytes(),
            np_array.shape[1],  # width
            np_array.shape[0],  # height
            np_array.shape[2],  # bands
            "uchar",
        )


@beartype.beartype
def vips_to_pil(vips_img: PIL.Image.Image) -> PIL.Image.Image:
    # Convert to numpy array
    np_array = vips_img.numpy()
    # Convert numpy array to PIL Image
    return PIL.Image.fromarray(np_array)


@beartype.beartype
@torch.inference_mode
@line_profiler.profile
def get_sae_activations(
    img_p: PIL.Image.Image, latents: dict[str, list[int]]
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
        x = vit_transform(img_p)[None, ...].to(DEVICE)
        vit_acts_PD = split_vit.forward_start(x)[0]

        _, f_x_PS, _ = sae(vit_acts_PD)
        # Ignore [CLS] token and get just the requested latents.
        acts_SP = einops.rearrange(f_x_PS[1:], "patches n_latents -> n_latents patches")
        logger.info("Got SAE activations for '%s'.", model_name)
        top_img_i, top_values = load_tensors(model_name)
        logger.info("Loaded top SAE activations for '%s'.", model_name)

        for latent in progress(requested_latents, every=1):
            acts = acts_SP[latent].cpu().tolist()
            raw_examples, seen_i_im = [], set()
            for i_im, values_p in zip(top_img_i[latent], top_values[latent]):
                if i_im in seen_i_im:
                    continue
                ex_img, ex_label = get_dataset_image(i_im.item())
                ex_img = human_transform(ex_img)
                raw_examples.append((ex_img, values_p.numpy(), ex_label))
                # Only need 4 example images per latent.
                if len(seen_i_im) >= 4:
                    break

            upper = None
            if top_values[latent].numel() > 0:
                upper = top_values[latent].max().item()

            examples = []
            for ex_img, values_p, ex_label in raw_examples:
                breakpoint()
                highlighted_img = add_highlights(ex_img, values_p, upper=upper)
                example = Example(
                    orig_url=vips_to_base64(ex_img),
                    highlighted_url=vips_to_base64(highlighted_img),
                    label=ex_label,
                )
                examples.append(example)

            sae_activations.append(
                SaeActivation(latent=latent, activations=acts, examples=examples)
            )
        response[model_name] = sae_activations
    return response


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


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
