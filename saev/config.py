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
class ImageFolderDataset:
    """Configuration for a generic image folder dataset."""

    root: str = os.path.join(".", "data", "split")
    """Where the class folders with images are stored."""

    @property
    def n_imgs(self) -> int:
        """Number of images in the dataset. Calculated on the fly, but is non-trivial to calculate because it requires walking the directory structure. If you need to reference this number very often, cache it in a local variable."""
        n = 0
        for _, _, files in os.walk(self.root):
            n += len(files)
        return n


DatasetConfig = ImagenetDataset | TreeOfLifeDataset | LaionDataset | ImageFolderDataset


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
    """Which layers to save. By default, the second-to-last layer."""
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
    layer: int | typing.Literal["all", "meanpool"] = -2
    """.. todo: document this field."""
    clamp: float = 1e5
    """Maximum value for activations; activations will be clamped to within [-clamp, clamp]`."""
    n_random_samples: int = 2**19
    """Number of random samples used to calculate approximate dataset means at startup."""
    scale_mean: bool = True
    """Whether to subtract approximate dataset means from examples."""
    scale_norm: bool = True
    """Whether to scale average dataset norm to sqrt(d_vit)."""


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
class Visuals:
    pass


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

        # .data is a nested field that cannot be naively expanded.
        data_dct = dct.pop("data")
        if data_dct:
            dct["data"] = dataclasses.replace(cfg.data, **data_dct)

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
