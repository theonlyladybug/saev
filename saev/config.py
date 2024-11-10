"""
All configs for all saev jobs.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.
"""

import dataclasses
import os
import typing

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImagenetDataset:
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name on HuggingFace. Don't need to change this.."""
    split: str = "train"
    """Dataset split. For the default ImageNet-1K dataset, can either be 'train', 'validation' or 'test'."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        dataset = datasets.load_dataset(
            self.name, split=self.split, trust_remote_code=True
        )
        return len(dataset)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TreeOfLifeDataset:
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
    """"""
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
class LaionDataset:
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
class Inat21Dataset:
    """Configuration for iNat2021 dataset."""

    root: str = "./inat21"
    """Where the images are stored."""
    split: str = "train"
    """Dataset split. Can either be 'train', 'val' or 'train_mini'."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        n = 0
        for _, _, files in os.walk(os.path.join(self.root, self.split)):
            n += len(files)
        return n


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BrodenDataset:
    """Configuration for Broden dataset."""

    root: str = os.path.join(".", "broden")
    """Where the Broden dataset is stored."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires reading a file. If you need to reference this number very often, cache it in a local variable."""
        with open(os.path.join(self.root, "index.csv")) as fd:
            return sum(1 for _ in fd)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Activations:
    """
    Configuration for calculating and saving ViT activations.
    """

    data: (
        ImagenetDataset
        | TreeOfLifeDataset
        | LaionDataset
        | Inat21Dataset
        | BrodenDataset
    ) = dataclasses.field(default_factory=ImagenetDataset)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    model_org: typing.Literal["open-clip", "timm", "dinov2"] = "open-clip"
    """Where to load models from."""
    model_ckpt: str = "ViT-L-14/openai"
    """Specific model checkpoint."""
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
class DataLoad:
    """
    Configuration for loading activation data from disk.
    """

    shard_root: str = os.path.join(".", "shards")
    """Directory with .bin shards and a metadata.json file."""
    patches: typing.Literal["cls", "patches", "meanpool"] = "cls"
    """Which kinds of patches to use. 'cls' indicates just the [CLS] token (if any). 'patches' indicates it will return all patches. 'meanpool' returns the mean of all image patches."""
    layer: int | typing.Literal["all", "meanpool"] = -1
    """.. todo: document this field."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration"""
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

    log_every: int = 25
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


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Webapp:
    """.. todo:: document."""

    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration."""
    images: ImagenetDataset | TreeOfLifeDataset | LaionDataset | Inat21Dataset = (
        dataclasses.field(default_factory=ImagenetDataset)
    )
    """Which images to use."""
    top_k: int = 16
    """How many images per SAE feature to store."""
    n_workers: int = 16
    """Number of dataloader workers."""
    topk_batch_size: int = 1024 * 16
    """Number of examples to apply top-k op to."""
    sae_batch_size: int = 1024
    """Batch size for SAE inference."""
    epsilon: float = 1e-9
    """Value to add to avoid log(0)."""
    sort_by: typing.Literal["cls", "img", "patch"] = "cls"
    """How to find the top k images. 'cls' picks images where the SAE latents of the ViT's [CLS] token are maximized without any patch highligting. 'img' picks images that maximize the sum of an SAE latent over all patches in the image, highlighting the patches. 'patch' pickes images that maximize an SAE latent over all patches (not summed), highlighting the patches and only showing unique images."""
    device: str = "cuda"
    """Which accelerator to use."""
    dump_to: str = os.path.join(".", "data")
    """Where to save data."""

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


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImagenetEvaluate:
    # Model
    ckpt: str = os.path.join(".", "checkpoints", "abcdefg")
    """Path to the sae.pt file."""
    # Data
    train_shard_root: str = os.path.join(".", "imagenet-1k-shards", "train")
    """Train shards root directory."""
    val_shard_root: str = os.path.join(".", "imagenet-1k-shards", "val")
    """Validation shards root directory."""
    n_workers: int = 16
    """Number of dataloader workers."""
    # Optimization
    sgd_batch_size: int = 1024 * 16
    """Batch size for linear classifier."""
    n_steps: int = 12500
    """Number of SGD steps."""
    # Hardware
    device: str = "cuda"
    """Which accelerator to use."""
    # Misc
    log_every: int = 5
    """How often to log progress."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class HistogramsEvaluate:
    ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to the sae.pt file."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration."""
    n_workers: int = 8
    """Number of dataloader workers."""
    sae_batch_size: int = 1024 * 8
    """SAE inference batch size."""
    device: str = "cuda"
    """Which accelerator to use."""
    log_every: int = 10
    """How often to log progress."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class BrodenEvaluate:
    ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to the sae.pt file."""
    patch_size: tuple[int, int] = (16, 16)
    """Original patch size in pixels."""
    root: str = "./broden"
    """Root of the Broden dataset."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """ViT activations for Broden."""
    n_workers: int = 8
    """Number of dataloader workers."""
    batch_size: int = 1024
    """ViT and SAE inference batch size."""
    sample_range: tuple[int, int] = (200, 1_000)
    """Range of samples per label. Will skip samples with fewer samples and will truncate classes with more samples."""
    dump_to: str = os.path.join(".", "logs", "broden")
    """Where to save charts."""
    device: str = "cuda"
    """Which accelerator to use."""
    log_every: int = 10
    """How often to log progress."""
    seed: int = 42
    """Random seed."""
    debug: bool = False
    """Debugging?"""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Evaluate:
    histograms: HistogramsEvaluate = dataclasses.field(
        default_factory=HistogramsEvaluate
    )
    """Histogram config."""
    broden: BrodenEvaluate = dataclasses.field(default_factory=BrodenEvaluate)
    """Broden feature probing config."""

    imagenet: ImagenetEvaluate = dataclasses.field(default_factory=ImagenetEvaluate)
    """ImageNet-1K config."""

    slurm: bool = False
    """Whether to use `submitit` to run jobs on a Slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account string."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""
