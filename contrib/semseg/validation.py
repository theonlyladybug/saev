"""
Make some predictions for a bunch of checkpoints.
See which checkpoints have the best validation loss, mean IoU, class-specific IoU, validation accuracy, and qualitative results.

Writes results to CSV files and hparam graphs (in-progress).
"""

import beartype
import torch

from . import config, training


@beartype.beartype
def main(cfg: config.Validation):
    breakpoint()
    dataset = training.Dataset(cfg.acts, cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        persistent_workers=(cfg.n_workers > 0),
    )


@beartype.beartype
def load_ckpts(root: str) -> list[tuple[config.Train, torch.nn.Module]]:
    """
    Loads the latest checkpoints for each directory within root.

    Arguments:
        root: direcotry containing other directories with cfg.json and model_step{step}.pt files.

    Returns:
        List of cfg, model pairs.
    """
