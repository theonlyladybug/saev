import dataclasses
import os.path
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

import saev.nn

from . import config, training


@beartype.beartype
@torch.inference_mode
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
        breakpoint()

    # Save results
    save(reports, os.path.join(cfg.dump_to, "results.csv"))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ClassResults:
    """Results for a single class."""

    class_id: int
    """Numeric identifier for the class."""

    class_name: str
    """Human-readable name of the class."""

    n_orig_patches: int
    """Original patches that were this class."""

    n_changed_patches: int
    """After intervention, how many patches changed."""

    n_other_patches: int
    """Total patches that weren't this class."""

    n_other_changed: int
    """After intervention, how many of the other patches changed."""

    change_distribution: dict[int, int]
    """What classes did patches change to? Tracks how many times <value> a patch changed from self.class_id to <key>."""


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class Report:
    """Complete results from an intervention experiment."""

    method: str
    """Which intervention method was used."""

    class_results: list[ClassResults]
    """Per-class detailed results."""

    intervention_scale: float
    """Magnitude of intervention."""

    @property
    def mean_target_change(self) -> float:
        """Percentage of target patches that changed class."""
        total_target = sum(r.n_orig_patches for r in self.class_results)
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
            r.n_changed_patches / r.n_orig_patches if r.n_orig_patches > 0 else 0.0
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


@beartype.beartype
def save(results: list[Report], dpath: str) -> None:
    raise NotImplementedError()


def argmax_logits(
    logits_BPC: Float[Tensor, "batch patches channels_with_null"],
) -> Float[Tensor, "batch patches"]:
    return logits_BPC[:, :, 1:].argmax(axis=-1) + 1


@beartype.beartype
def eval_rand_vec(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    """
    Evaluates the effects of adding a random unit vector to the patches.

    Args:
        cfg: Configuration for quantitative evaluation
        sae: Trained sparse autoencoder model
        clf: Trained classifier model
        dataloader: DataLoader providing batches of images

    Returns:
        Report containing intervention results, including per-class changes
    """

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        acts: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        # Make this into a different random vector for each batch/patch combination, so it should be (batch, patches, dim) in shape, and each dim-dimensional vector should have unit norm. AI!
        rand_vec = torch.randn(sae.cfg.d_vit, device=cfg.device)
        rand_vec = rand_vec / torch.norm(rand_vec)

        intervention_scale = 20.0  # This could be a config parameter
        rand_vec = rand_vec * intervention_scale

        acts[:, 1:, :] += rand_vec
        return acts

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)

    hooked_vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    register_hook(hooked_vit, hook, cfg.vit_layer, cfg.n_patches_per_img)

    orig_preds, mod_preds = [], []
    for batch in dataloader:
        x_BCWH = batch["image"].to(cfg.device)

        orig_acts = vit(x_BCWH)
        orig_logits = clf(orig_acts[:, 1:, :])
        orig_preds.append(argmax_logits(orig_logits).cpu())

        mod_acts = hooked_vit(x_BCWH)
        mod_logits = clf(mod_acts[:, 1:, :])
        mod_preds.append(argmax_logits(mod_logits).cpu())

    # Concatenate all predictions
    orig_preds = torch.cat(orig_preds, dim=0)
    mod_preds = torch.cat(mod_preds, dim=0)

    # Compute per-class results
    class_results = []
    for class_id in range(1, 151):
        # Count original patches of this class
        orig_mask = orig_preds == class_id
        n_orig = orig_mask.sum().item()

        # Count how many changed
        changed_mask = (mod_preds != orig_preds) & orig_mask
        n_changed = changed_mask.sum().item()

        # Count changes in other patches
        other_mask = ~orig_mask
        n_other = other_mask.sum().item()
        n_other_changed = ((mod_preds != orig_preds) & other_mask).sum().item()

        # Track what classes patches changed to
        changes = {}
        for new_class in range(1, 151):
            if new_class != class_id:
                count = ((mod_preds == new_class) & changed_mask).sum().item()
                if count > 0:
                    changes[new_class] = count

        class_results.append(
            ClassResults(
                class_id=class_id,
                class_name="TODO",  # class_names[class_id],
                n_orig_patches=n_orig,
                n_changed_patches=n_changed,
                n_other_patches=n_other,
                n_other_changed=n_other_changed,
                change_distribution=changes,
            )
        )

    return Report(
        method="random-vector",
        class_results=class_results,
        intervention_scale=intervention_scale,
    )


@beartype.beartype
def eval_rand_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    raise NotImplementedError()


@beartype.beartype
def eval_auto_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    raise NotImplementedError()


@jaxtyped(typechecker=beartype.beartype)
def register_hook(
    vit: torch.nn.Module,
    hook: typing.Callable[[Float[Tensor, "..."]], Float[Tensor, "..."]],
    layer: int,
    n_patches_per_img: int,
):
    patches = vit.get_patches(n_patches_per_img)

    @jaxtyped(typechecker=beartype.beartype)
    def _hook(
        block: torch.nn.Module,
        inputs: tuple,
        outputs: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        x = outputs[:, patches, :]
        x = hook(x)
        outputs[:, patches, :] = x
        return outputs

    return vit.get_residuals()[layer].register_forward_hook(_hook)
