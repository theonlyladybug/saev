"""
Configs for saev experiments.

## Import Times

This module should be very fast to import so that `python main.py --help` is fast.
This means that the top-level imports should not include big packages like numpy, torch, etc.
For example, `TreeOfLife.n_imgs` imports numpy when it's needed, rather than importing it at the top level.
"""

import dataclasses

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Imagenet:
    """Configuration for HuggingFace Imagenet."""

    name: str = "ILSVRC/imagenet-1k"
    """Dataset name. Probably don't want to change this."""

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
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    # Data Generating Function (Model + Training Distibuion)
    image_width: int = 224

    image_height: int = 224

    model: str = "ViT-L-14/openai"
    """Model string, for use with open_clip."""
    module_name: str = "resid"

    block_layer: int = -2

    data: Imagenet | TreeOfLife = dataclasses.field(default_factory=Imagenet)
    """Which dataset to use."""
    n_workers: int = 8
    """Number of dataloader workers."""
    d_vit: int = 1024
    """Dimension of the ViT activations (depends on model, module_name, and block_layer)."""

    # Training
    n_reinit_batches: int = 15
    """Number of batches to use for SAE re-init."""
    n_epochs: int = 3
    """Number of SAE training epochs."""
    vit_batch_size: int = 1024
    """Batch size for ViT inference."""
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
    ckpt_path: str = "checkpoints"

    # Misc.
    device: str = "cuda"

    seed: int = 42
    """Random seed."""
    dtype: str = "float32"

    slurm: bool = False
    """Whether to use `submitit` to run jobs on a Slurm cluster."""
    slurm_acct: str = "PAS2136"
    """Slurm account string."""
    log_to: str = "./logs"
    """Where to log Slurm job stdout/stderr."""

    @property
    def reinit_size(self) -> int:
        return self.n_reinit_batches * self.sae_batch_size

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor
