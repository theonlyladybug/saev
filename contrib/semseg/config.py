import dataclasses
import os.path

import beartype

import saev.config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    learning_rate: float = 1e-4
    """Linear layer learning rate."""
    weight_decay: float = 1e-3
    """Weight decay  for AdamW."""
    n_epochs: int = 200
    """Number of training epochs for linear layer."""
    batch_size: int = 1024
    """Training batch size for linear layer."""
    n_workers: int = 32
    """Number of dataloader workers."""
    train_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=saev.config.DataLoad
    )
    """Configuration for the saved ADE20K training ViT activations."""
    val_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=saev.config.DataLoad
    )
    """Configuration for the saved ADE20K validation ViT activations."""
    imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=saev.config.Ade20kDataset
    )
    """Configuration for the ADE20K dataset."""
    eval_every: int = 50
    """How many epochs between evaluations."""
    device: str = "cuda"
    "Hardware to train on." ""
    ckpt_path: str = os.path.join(".", "checkpoints", "semseg")
    seed: int = 42
    """Random seed."""
    log_to: str = os.path.join(".", "logs")


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(saev.config.expand(sweep_dct)):
        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs
