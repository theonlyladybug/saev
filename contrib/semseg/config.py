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
@dataclasses.dataclass(frozen=True)
class Visuals:
    sae_ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    acts: saev.config.DataLoad = dataclasses.field(default_factory=saev.config.DataLoad)
    """Configuration for the saved ADE20K training ViT activations."""
    imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=lambda: saev.config.Ade20kDataset(split="training")
    )
    """Configuration for the ADE20K training dataset."""
    batch_size: int = 128
    """Batch size for calculating F1 scores."""
    n_workers: int = 32
    """Number of dataloader workers."""
    label_threshold: float = 0.9
    device: str = "cuda"
    "Hardware for SAE inference." ""
    ade20k_cls: int = 29
    """ADE20K class to probe for."""
    k: int = 32
    """Top K features to save."""


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(saev.config.expand(sweep_dct)):
        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs
