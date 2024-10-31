"""
Configs for saev experiments.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.
"""

import dataclasses
import os

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Imagenet:
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name. Don't need to change this."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(
            self.name, split="train", trust_remote_code=True
        )
        return len(dataset)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TreeOfLife:
    """
    Configuration for the TreeOfLife-10M webdataset.

    Webdatasets are designed for random sampling of the entire dataset so that over multiple epochs, every sample is seen, on average, the same number of times. However, for training sparse autoencoders, we need to calculate ViT activations exactly once for each example in the dataset. Webdatasets support this through the [`wids`](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets) library.

    Here is a short discussion of the steps required to use saev with webdatasets.

    First, you will need to use `widsindex` (installed with the webdataset library) to create an metadata file used by wids. You can see an example file [here](https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-train.json). To generate my own metadata file, I ran this command:

    ```sh
    uv run widsindex create --name treeoflife-10m --output treeoflife-10m.json '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar'
    ```

    It took a long time (more than an hour, less than 3 hours) and generated a `treeoflife-10m.json` file.
    """

    metadata: str = "treeoflife-10m.json"
    """Path to dataset shards."""
    label_key: str = ".taxonomic_name.txt"
    """Which key to use as the label."""

    @property
    def n_imgs(self) -> int:
        """Return number of images in the dataset by reading the metadata file and summing the `nsamples` fields."""
        import json

        import numpy as np

        with open(self.metadata) as fd:
            metadata = json.load(fd)

        return (
            np.array([shard["nsamples"] for shard in metadata["shardlist"]])
            .sum()
            .item()
        )


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Laion:
    name: str = "laion/relaion2B-multi-research-safe"
    """Name of dataset on HuggingFace."""
    cache_dir: str = "/fs/scratch/PAS2136/samuelstevens/cache/laion"
    """Where to save the webdataset files that are downloaded."""
    n_imgs: int = 10_000_000
    """Number of images in this dataset (fixed at 10M)."""

    @property
    def url_list_filepath(self) -> str:
        """Path to file with list of URLs."""
        return os.path.join(self.cache_dir, "urls.jsonl")

    @property
    def tar_dir(self) -> str:
        return os.path.join(self.cache_dir, "shards")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Activations:
    # Data Generating Function (Model + Training Distibuion)
    data: Imagenet | TreeOfLife | Laion = dataclasses.field(default_factory=Imagenet)
    """Which dataset to use."""
    width: int = 224
    """Image width."""
    height: int = 224
    """Image height."""
    model: str = "ViT-L-14/openai"
    """Model string, for use with open_clip."""
    vit_batch_size: int = 1024
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_vit: int = 1024
    """Dimension of the ViT activations (depends on model)."""
    n_layers: int = 6
    """How many of the last ViT layers to save."""
    n_patches_per_img: int = 256
    """Number of ViT patches per image (depends on model)."""
    n_patches_per_shard: int = 2_400_000
    """Number of activations per shard; 2.4M is approximately 10GB for 1024-dimensional 4-byte activations."""

    seed: int = 42
    """Random seed."""
    ssl: bool = True
    """Whether to use SSL."""

    # Hardware
    device: str = "cuda"
    """Which device to use."""
    slurm: bool = False
    """Whether to use `submitit` to run jobs on a Slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account string."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    n_workers: int = 8
    """Number of dataloader workers."""

    # Training
    n_reinit_batches: int = 15
    """Number of batches to use for SAE re-init."""
    n_patches: int = 1_000_000_000
    """Number of SAE training examples."""
    exp_factor: int = 64
    """Expansion factor for SAE."""
    l1_coeff: float = 0.00008

    lr: float = 0.0004
    """Learning rate."""
    n_lr_warmup: int = 500
    """Number of learning rate warmup steps."""
    sae_batch_size: int = 1024
    """Batch size for SAE training."""

    use_ghost_grads: bool = True

    feature_sampling_window: int = 64

    resample_batches: int = 32

    dead_feature_window: int = 64

    dead_feature_threshold: float = 1e-6

    # Logging
    track: bool = True
    """Whether to track with WandB."""
    wandb_project: str = "saev"

    log_every: int = 10
    """How often to log to WandB."""
    ckpt_path: str = os.path.join(".", "checkpoints")
    """Where to save checkpoints."""

    # Misc.
    device: str = "cuda"

    seed: int = 42
    """Random seed."""

    slurm: bool = False
    """Whether to use `submitit` to run jobs on a Slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account string."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""

    @property
    def reinit_size(self) -> int:
        return self.n_reinit_batches * self.sae_batch_size
