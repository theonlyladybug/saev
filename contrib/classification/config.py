import dataclasses
import os

import beartype

import saev.config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    learning_rate: float = 1e-4
    """Linear layer learning rate."""
    weight_decay: float = 1e-3
    """Weight decay  for AdamW."""
    n_steps: int = 400
    """Number of training steps for linear layer."""
    batch_size: int = 1024
    """Training batch size for linear layer."""
    n_workers: int = 32
    """Number of dataloader workers."""
    train_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=lambda: saev.config.DataLoad(patches="cls")
    )
    """Configuration for the saved Flowers102 training ViT activations."""
    val_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=lambda: saev.config.DataLoad(patches="cls")
    )
    """Configuration for the saved Flowers102 validation ViT activations."""
    train_imgs: saev.config.ImageFolderDataset = dataclasses.field(
        default_factory=saev.config.ImageFolderDataset
    )
    """Configuration for the Flowers102 training images."""
    val_imgs: saev.config.ImageFolderDataset = dataclasses.field(
        default_factory=saev.config.ImageFolderDataset
    )
    """Configuration for the Flowers102 validation images."""
    eval_every: int = 100
    """How many epochs between evaluations."""
    device: str = "cuda"
    "Hardware to train on." ""
    ckpt_path: str = os.path.join(".", "checkpoints", "contrib", "classification")
    seed: int = 42
    """Random seed."""
    log_to: str = os.path.join(".", "logs", "contrib", "classification")


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(saev.config.expand(sweep_dct)):
        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs
