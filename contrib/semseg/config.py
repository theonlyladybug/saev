import dataclasses
import os.path

import beartype

import saev.config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Train:
    ckpt_path: str = os.path.join(".", "checkpoints", "faithfulness")
    learning_rate: float = 1e-4
    """Linear layer learning rate."""
    weight_decay: float = 1e-3
    """Weight decay  for AdamW."""
    n_epochs: int = 10
    """Number of training epochs for linear layer."""
    batch_size: int = 32
    """Training batch size for linear layer."""
    n_workers: int = 8
    """Number of dataloader workers."""
    train_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=saev.config.DataLoad
    )
    """Configuration for the saved ADE20K training ViT activations."""
    val_acts: saev.config.DataLoad = dataclasses.field(
        default_factory=saev.config.DataLoad
    )
    """Configuration for the saved ADE20K validation ViT activations."""
    train_imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=lambda: saev.config.Ade20kDataset(split="training")
    )
    """Configuration for the training ADE20K dataset."""
    val_imgs: saev.config.Ade20kDataset = dataclasses.field(
        default_factory=lambda: saev.config.Ade20kDataset(split="validation")
    )
    """Configuration for the validation ADE20K dataset."""
    log_every: int = 10
    """How often to log during training."""
