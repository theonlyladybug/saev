import dataclasses
import os.path

import saev.config


@dataclasses.dataclass(frozen=True)
class Score:
    sae_ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to SAE checkpoint"""

    batch_size: int = 2048
    """Batch size for SAE inference."""
    n_workers: int = 32
    """Number of dataloader workers."""

    threshold: float = 0.0
    """Threshold for feature activation."""
    top_k: int = 5
    """Number of top features to manually analyze."""

    imgs: saev.config.ImageFolderDataset = dataclasses.field(
        default_factory=saev.config.ImageFolderDataset
    )
    """Where curated examples are stored"""
    acts: saev.config.DataLoad = dataclasses.field(default_factory=saev.config.DataLoad)
    """SAE activations for the curated examples."""

    # Where to save analysis results
    dump_to: str = os.path.join(".", "logs", "contrib", "semprobe")
    """Where to save results/visualizations."""

    device: str = "cuda"
    """Hardware device."""
