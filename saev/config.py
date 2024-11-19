"""
All configs for all saev jobs.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.

Also contains code for expanding configs with lists into lists of configs (grid search).
Might be expanded in the future to support pseudo-random sampling from distributions to support random hyperparameter search, as in [this file](https://github.com/samuelstevens/sax/blob/main/sax/sweep.py).
"""

import collections.abc
import dataclasses
import itertools
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
class ProbeDataset:
    """Configuration for manual probing dataset."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires loading the dataset. If you need to reference this number very often, cache it in a local variable."""
        import datasets

        n = 0

        name = "samuelstevens/sae-probing"
        cfgs = datasets.get_dataset_config_names(name)
        for cfg in sorted(cfgs):
            if cfg == "default":
                continue

            dataset = datasets.load_dataset(name, cfg, split="train")
            n += len(dataset)

        return n


DatasetConfig = (
    ImagenetDataset
    | TreeOfLifeDataset
    | LaionDataset
    | Inat21Dataset
    | BrodenDataset
    | ProbeDataset
)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Activations:
    """
    Configuration for calculating and saving ViT activations.
    """

    data: DatasetConfig = dataclasses.field(default_factory=ImagenetDataset)
    """Which dataset to use."""
    dump_to: str = os.path.join(".", "shards")
    """Where to write shards."""
    model_org: typing.Literal["clip", "siglip", "timm", "dinov2"] = "clip"
    """Where to load models from."""
    model_ckpt: str = "ViT-L-14/openai"
    """Specific model checkpoint."""
    vit_batch_size: int = 1024
    """Batch size for ViT inference."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_vit: int = 1024
    """Dimension of the ViT activations (depends on model)."""
    layers: list[int] = dataclasses.field(default_factory=lambda: [-2])
    """How many of the last ViT layers to save."""
    n_patches_per_img: int = 256
    """Number of ViT patches per image (depends on model)."""
    cls_token: bool = True
    """Whether the model has a [CLS] token."""
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
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    n_random_samples: int = 2**19
    """Number of random samples used to calculate dataset means at startup."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class SparseAutoencoder:
    d_vit: int = 1024
    exp_factor: int = 64
    """Expansion factor for SAE."""
    sparsity_coeff: float = 0.00008
    """How much to weight sparsity loss term."""
    n_reinit_samples: int = 1024 * 16 * 32
    """Number of samples to use for SAE re-init. Anthropic proposes initializing b_dec to the geometric median of the dataset here: https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias. We use the regular mean."""
    ghost_grads: bool = False
    """Whether to use ghost grads."""
    remove_parallel_grads: bool = True
    """Whether to remove gradients parallel to W_dec columns (which will be ignored because we force the columns to have unit norm). See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization for the original discussion from Anthropic."""
    normalize_w_dec: bool = True
    """Whether to make sure W_dec has unit norm columns. See https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder for original citation."""
    seed: int = 0
    """Random seed."""

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration"""
    n_workers: int = 32
    """Number of dataloader workers."""
    n_patches: int = 100_000_000
    """Number of SAE training examples."""
    sae: SparseAutoencoder = dataclasses.field(default_factory=SparseAutoencoder)
    """SAE configuration."""
    n_sparsity_warmup: int = 0
    """Number of sparsity coefficient warmup steps."""
    lr: float = 0.0004
    """Learning rate."""
    n_lr_warmup: int = 500
    """Number of learning rate warmup steps."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE training."""

    feature_sampling_window: int = 64

    dead_feature_window: int = 64

    dead_feature_threshold: float = 1e-6

    # Logging
    track: bool = True
    """Whether to track with WandB."""
    wandb_project: str = "saev"
    """WandB project name."""
    tag: str = ""
    """Tag to add to WandB run."""
    log_every: int = 25
    """How often to log to WandB."""
    ckpt_path: str = os.path.join(".", "checkpoints")
    """Where to save checkpoints."""

    device: typing.Literal["cuda", "cpu"] = "cuda"
    """Hardware device."""
    seed: int = 42
    """Random seed."""
    slurm: bool = False
    """Whether to use `submitit` to run jobs on a Slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account string."""
    log_to: str = os.path.join(".", "logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Webapp:
    """.. todo:: document."""

    ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """Data configuration."""
    images: ImagenetDataset | Inat21Dataset | ProbeDataset = dataclasses.field(
        default_factory=ImagenetDataset
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
    log_to: str = os.path.join(".", "logs")
    """Where to write charts to."""


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
    k: int = 16
    """How man images at most to save."""


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


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Probe:
    ckpt: str = os.path.join(".", "checkpoints", "abcdefg", "sae.pt")
    """Path to the sae.pt file."""
    data: DataLoad = dataclasses.field(default_factory=DataLoad)
    """ViT activations for probing tasks."""
    n_workers: int = 8
    """Number of dataloader workers."""
    sae_batch_size: int = 1024 * 16
    """Batch size for SAE inference."""
    device: str = "cuda"
    """Which accelerator to use."""
    images: ProbeDataset = dataclasses.field(default_factory=ProbeDataset)
    """Where the raw images are."""
    dump_to: str = os.path.join(".", "logs", "probes")
    """Where to save images."""


##########
# SWEEPS #
##########


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(expand(sweep_dct)):
        # .sae is a nested field that cannot be naively expanded.
        sae_dct = dct.pop("sae")
        if sae_dct:
            sae_dct["seed"] = sae_dct.pop("seed", cfg.sae.seed) + cfg.seed + d
            dct["sae"] = dataclasses.replace(cfg.sae, **sae_dct)

        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs


@beartype.beartype
def expand(config: dict[str, object]) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands dicts with (nested) lists into a list of (nested) dicts.
    """
    yield from _expand_discrete(config)


@beartype.beartype
def _expand_discrete(
    config: dict[str, object],
) -> collections.abc.Iterator[dict[str, object]]:
    """
    Expands any (possibly nested) list values in `config`
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    elif isinstance(value, dict):
        # Expand
        for c, v in itertools.product(
            _expand_discrete(config), _expand_discrete(value)
        ):
            yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}
