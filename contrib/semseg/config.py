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
    n_epochs: int = 400
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
    patch_size_px: tuple[int, int] = (14, 14)
    """Patch size in pixels."""
    eval_every: int = 100
    """How many epochs between evaluations."""
    device: str = "cuda"
    "Hardware to train on."
    ckpt_path: str = os.path.join(".", "checkpoints", "contrib", "semseg")
    seed: int = 42
    """Random seed."""
    log_to: str = os.path.join(".", "logs", "contrib", "semseg")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Visuals:
    sae_ckpt: str = os.path.join(".", "checkpoints", "sae.pt")
    """Path to the sae.pt file."""
    ade20k_cls: int = 29
    """ADE20K class to probe for."""
    k: int = 32
    """Top K features to save."""
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
    "Hardware for SAE inference."


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Validation:
    ckpt_root: str = os.path.join(".", "checkpoints", "contrib", "semseg")
    """Root to all checkpoints to evaluate."""
    dump_to: str = os.path.join(".", "logs", "contrib", "semseg")
    """Directory to dump results to."""
    acts: saev.config.DataLoad = dataclasses.field(default_factory=saev.config.DataLoad)
    """Configuration for the saved ADE20K validation ViT activations."""
    imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=lambda: saev.config.Ade20kDataset(split="validation")
    )
    """Configuration for the ADE20K validation dataset."""
    patch_size_px: tuple[int, int] = (14, 14)
    """Patch size in pixels."""
    batch_size: int = 128
    """Batch size for calculating F1 scores."""
    n_workers: int = 32
    """Number of dataloader workers."""
    device: str = "cuda"
    "Hardware for linear probe inference."


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Manipulation:
    probe_ckpt: str = os.path.join(
        ".", "checkpoints", "semseg", "lr_0_001__wd_0_1", "model.pt"
    )
    """Linear probe checkpoint."""
    sae_ckpt: str = os.path.join(".", "checkpoints", "abcdef", "sae.pt")
    """SAE checkpoint."""
    ade20k_classes: list[int] = dataclasses.field(default_factory=lambda: [29])
    """One or more ADE20K classes to track."""
    sae_latents: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2])
    """one or more SAE latents to manipulate."""
    acts: saev.config.DataLoad = dataclasses.field(default_factory=saev.config.DataLoad)
    """Configuration for the saved ADE20K validation ViT activations."""
    imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=lambda: saev.config.Ade20kDataset(split="validation")
    )
    """Configuration for the ADE20K validation dataset."""
    batch_size: int = 128
    """Batch size for both linear probe and SAE."""
    n_workers: int = 32
    """Number of dataloader workers."""
    device: str = "cuda"
    "Hardware for linear probe and SAE inference."


@beartype.beartype
def grid(cfg: Train, sweep_dct: dict[str, object]) -> tuple[list[Train], list[str]]:
    cfgs, errs = [], []
    for d, dct in enumerate(saev.config.expand(sweep_dct)):
        try:
            cfgs.append(dataclasses.replace(cfg, **dct, seed=cfg.seed + d))
        except Exception as err:
            errs.append(str(err))

    return cfgs, errs
