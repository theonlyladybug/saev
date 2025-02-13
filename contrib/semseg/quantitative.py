import dataclasses
import os.path

import beartype
import torch

import saev.nn

from . import config, training


@beartype.beartype
def main(cfg: config.Quantitative):
    """Main entry point for quantitative evaluation."""
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than float16 and almost as accurate as float32. This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Load models
    sae = saev.nn.load(cfg.sae_ckpt)
    clf = training.load_latest(cfg.ckpt_root, device=cfg.device)

    # Get validation data
    dataset = training.Dataset(cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        persistent_workers=(cfg.n_workers > 0),
    )

    # For each method (random vector, random feature, etc)
    reports = []
    for fn in (eval_rand_vec, eval_rand_feat, eval_auto_feat):
        report = fn(cfg, sae, clf, dataloader)
        reports.append(report)

    # Save results
    save(reports, os.path.join(cfg.dump_to, "results.csv"))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Report:
    method: str


@beartype.beartype
def save(results: list[Report], dpath: str) -> None:
    raise NotImplementedError()


@beartype.beartype
def eval_rand_vec(cfg: config.Quantitative) -> Report:
    raise NotImplementedError()


@beartype.beartype
def eval_rand_feat(cfg: config.Quantitative) -> Report:
    raise NotImplementedError()


@beartype.beartype
def eval_auto_feat(cfg: config.Quantitative) -> Report:
    raise NotImplementedError()
