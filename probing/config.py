import dataclasses
import os
import typing

import beartype

from saev import config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Probe:
    ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to the sae.pt file."""
    data: config.DataLoad = dataclasses.field(default_factory=config.DataLoad)
    """ViT activations for probing tasks."""
    n_workers: int = 8
    """Number of dataloader workers."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""
    device: str = "cuda"
    """Which accelerator to use."""
    images: config.ImageFolderDataset = dataclasses.field(
        default_factory=config.ImageFolderDataset
    )
    """Where the raw images are."""
    dump_to: str = os.path.join(".", "logs", "probes")
    """Where to save images."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Topk:
    """.. todo:: document."""

    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: config.DataLoad = dataclasses.field(default_factory=config.DataLoad)
    """Data configuration."""
    images: config.ImagenetDataset | config.ImageFolderDataset = dataclasses.field(
        default_factory=config.ImagenetDataset
    )
    """Which images to use."""
    top_k: int = 16
    """How many images per SAE feature to store."""
    n_workers: int = 16
    """Number of dataloader workers."""
    topk_batch_size: int = 1024 * 16
    """Number of examples to apply top-k op to."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""
    epsilon: float = 1e-9
    """Value to add to avoid log(0)."""
    sort_by: typing.Literal["cls", "img", "patch"] = "cls"
    """How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."""
    device: str = "cuda"
    """Which accelerator to use."""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""
    log_freq_range: tuple[float, float] = (-6.0, -2.0)
    """Log10 frequency range for which to save images."""
    log_value_range: tuple[float, float] = (-1.0, 1.0)
    """Log10 frequency range for which to save images."""
    include_latents: list[int] = dataclasses.field(default_factory=list)
    """Latents to always include, no matter what."""

    @property
    def root(self) -> str:
        return os.path.join(self.dump_to, f"sort_by_{self.sort_by}")

    @property
    def top_values_fpath(self) -> str:
        return os.path.join(self.root, "top_values.pt")

    @property
    def top_img_i_fpath(self) -> str:
        return os.path.join(self.root, "top_img_i.pt")

    @property
    def top_patch_i_fpath(self) -> str:
        return os.path.join(self.root, "top_patch_i.pt")

    @property
    def mean_values_fpath(self) -> str:
        return os.path.join(self.root, "mean_values.pt")

    @property
    def sparsity_fpath(self) -> str:
        return os.path.join(self.root, "sparsity.pt")
