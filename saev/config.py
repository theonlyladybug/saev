import dataclasses

import beartype


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Huggingface:
    """Configuration for datasets from HuggingFace."""

    name: str = "ILSVRC/imagenet-1k"

    @property
    def n_imgs(self) -> int:
        import datasets

        dataset = datasets.load_dataset(
            self.name, split="train", trust_remote_code=True
        )
        return len(dataset)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Webdataset:
    """
    Configuration for webdataset (like TreeOfLife-10M).

    Webdatasets are designed for random sampling of the entire dataset so that over multiple epochs, every sample is seen, on average, the same number of times. However, for training sparse autoencoders, we need to calculate ViT activations exactly once for each example in the dataset. Webdatasets support this through the [`wids`](https://github.com/webdataset/webdataset?tab=readme-ov-file#the-wids-library-for-indexed-webdatasets) library.

    Here is a short discussion of the steps required to use saev with webdatasets.

    First, you will need to use `widsindex` (installed with the webdataset library) to create an metadata file used by wids. You can see an example file [here](https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-train.json). To generate my own metadata file, I ran this command:

    ```
    uv run widsindex create --name treeoflife-10m --output meta.json '/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar'
    ```

    It took a long time (more than an hour) and generated a `meta.json` file.
    """

    url: str = "/fs/ess/PAS2136/open_clip/data/evobio10m-v3.3/224x224/train/shard-{000000..000159}.tar"
    """Path to dataset shards."""
    n_imgs: int = 9562377
    """Number of images in dataset."""


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
    module_name: str = "resid"
    block_layer: int = -2
    data: Huggingface | Webdataset = dataclasses.field(default_factory=Huggingface)
    n_workers: int = 8
    """Number of dataloader workers."""

    # SAE Parameters
    d_in: int = 1024

    # Activation Store Parameters
    n_epochs: int = 3
    n_batches_in_store: int = 15
    vit_batch_size: int = 1024

    # SAE Parameters
    expansion_factor: int = 64

    # Training Parameters
    l1_coefficient: float = 0.00008
    lr: float = 0.0004
    lr_warm_up_steps: int = 500
    batch_size: int = 1024

    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = 64
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 64
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-6

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "saev"
    wandb_log_freq: int = 10

    # Misc
    device: str = "cuda"
    seed: int = 42
    dtype: str = "float32"
    checkpoint_path: str = "checkpoints"

    @property
    def store_size(self) -> int:
        return self.n_batches_in_store * self.batch_size

    @property
    def d_sae(self) -> int:
        return self.d_in * self.expansion_factor

    @property
    def run_name(self) -> str:
        return (
            f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-epochs-{self.n_epochs}"
        )
