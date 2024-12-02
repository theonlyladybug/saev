"""
Make some predictions for a bunch of checkpoints.
See which checkpoints have the best validation loss, mean IoU, class-specific IoU, validation accuracy, and qualitative results.

Writes results to CSV files and hparam graphs (in-progress).
"""

import json
import os

import beartype
import torch

from . import training

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
        root: directory containing other directories with cfg.json and model_step{step}.pt files.

    Returns:
        List of cfg, model pairs.
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    results = []
    
    # Find all subdirectories that contain cfg.json
    for dname in os.listdir(root):
        dpath = os.path.join(root, dname)
        if not os.path.isdir(dpath):
            continue
            
        cfg_path = os.path.join(dpath, "cfg.json")
        if not os.path.exists(cfg_path):
            continue

        # Load config
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        cfg = config.Train(**cfg_dict)

        # Load latest model checkpoint
        model = training.load_latest(dpath)
        
        results.append((cfg, model))

    if not results:
        raise FileNotFoundError(f"No valid checkpoint directories found in: {root}")

    return results
