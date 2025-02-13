import dataclasses
import os.path

import beartype
import torch
from jaxtyping import Int, jaxtyped

import numpy as np
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
    clf = training.load_latest(cfg.seg_ckpt, device=cfg.device)

    # Get validation data
    dataset = training.Dataset(cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=False
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
class ClassResults:
    """Results for a single class."""

    # Change the comments in ClassResults to use """-""" underneath the field name. AI!

    class_id: int
    class_name: str

    # Original patches that were this class
    n_original_patches: int
    # After intervention, how many changed
    n_changed_patches: int

    # Total patches that weren't this class
    n_other_patches: int
    # After intervention, how many of the other patches changed
    n_other_changed: int

    change_distribution: dict[int, int]
    """What classes did patches change to? Tracks how many times <value> a patch changed from self.class_id to <key>."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Report:
    """Complete results from an intervention experiment."""

    # Which intervention method was used
    method: str

    # Per-class detailed results
    class_results: list[ClassResults]

    # Original patch-wise predictions
    original_preds: Int[np.ndarray, "n_imgs height width"]

    # Modified patch-wise predictions
    modified_preds: Int[np.ndarray, "n_imgs height width"]

    # Magnitude of intervention
    intervention_scale: float

    @property
    def mean_target_change(self) -> float:
        """Percentage of target patches that changed class."""
        total_target = sum(r.n_original_patches for r in self.class_results)
        total_changed = sum(r.n_changed_patches for r in self.class_results)
        return total_changed / total_target if total_target > 0 else 0.0

    @property
    def mean_other_change(self) -> float:
        """Percentage of non-target patches that changed class."""
        total_other = sum(r.n_other_patches for r in self.class_results)
        total_changed = sum(r.n_other_changed for r in self.class_results)
        return total_changed / total_other if total_other > 0 else 0.0

    @property
    def per_class_target_changes(self) -> np.ndarray:
        """Array of per-class change percentages for target patches."""
        return np.array([
            r.n_changed_patches / r.n_original_patches
            if r.n_original_patches > 0
            else 0.0
            for r in self.class_results
        ])

    @property
    def target_change_std(self) -> float:
        """Standard deviation of change percentage across classes."""
        return float(np.std(self.per_class_target_changes))

    def to_csv_row(self) -> dict[str, float]:
        """Convert to a row for the summary CSV."""
        return {
            "method": self.method,
            "target_change": self.mean_target_change,
            "other_change": self.mean_other_change,
            "target_std": self.target_change_std,
        }

    def save_detailed(self, path: str) -> None:
        """Save detailed results including per-class statistics and predictions."""
        # Save tensors and detailed stats to npz file
        np.savez(
            path,
            original_preds=self.original_preds.cpu().numpy(),
            modified_preds=self.modified_preds.cpu().numpy(),
            intervention_vectors=self.intervention_vectors.cpu().numpy(),
            intervention_scale=self.intervention_scale,
            class_results=[dataclasses.asdict(r) for r in self.class_results],
        )


@beartype.beartype
def save(results: list[Report], dpath: str) -> None:
    raise NotImplementedError()


@beartype.beartype
def eval_rand_vec(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    """
    Evaluates the effects
    """
    raise NotImplementedError()


@beartype.beartype
def eval_rand_feat(cfg: config.Quantitative) -> Report:
    raise NotImplementedError()


@beartype.beartype
def eval_auto_feat(cfg: config.Quantitative) -> Report:
    raise NotImplementedError()
