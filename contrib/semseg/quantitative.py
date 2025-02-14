import csv
import dataclasses
import logging
import math
import os
import random
import typing

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.helpers
import saev.nn

from . import config, training

logger = logging.getLogger("contrib.semseg.quantitative")


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
    sae = saev.nn.load(cfg.sae_ckpt).to(cfg.device)
    clf = training.load_latest(cfg.seg_ckpt, device=cfg.device)

    # Get validation data
    dataset = training.Dataset(cfg.imgs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=False
    )

    # For each method (random vector, random feature, etc)
    reports = []
    for fn in (eval_auto_feat,):  # (eval_rand_vec, eval_rand_feat, eval_auto_feat):
        report = fn(cfg, sae, clf, dataloader)
        reports.append(report)

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
    def target_change_std(self) -> float:
        """Standard deviation of change percentage across classes."""
        per_class_target_changes = np.array([
            r.n_changed_patches / r.n_orig_patches if r.n_orig_patches > 0 else 0.0
            for r in self.class_results
        ])
        return float(np.std(per_class_target_changes))

    @property
    def other_change_std(self) -> float:
        """Standard deviation of non-target patch changes across classes."""
        per_class_other_changes = np.array([
            r.n_other_changed / r.n_other_patches if r.n_other_patches > 0 else 0.0
            for r in self.class_results
        ])
        return float(np.std(per_class_other_changes))

    def to_csv_row(self) -> dict[str, float]:
        """Convert to a row for the summary CSV."""
        return {
            "method": self.method,
            "target_change": self.mean_target_change,
            "other_change": self.mean_other_change,
            "target_std": self.target_change_std,
            "other_std": self.other_change_std,
        }


@beartype.beartype
def save(results: list[Report], fpath: str) -> None:
    """
    Save evaluation results to a CSV file.

    Args:
        results: List of Report objects containing evaluation results
        dpath: Path to save the CSV file
    """

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    columns = ["method", "target_change", "target_std", "other_change", "other_std"]

    with open(fpath, "w") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


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
    torch.manual_seed(cfg.seed)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        """
        Adds random unit vectors to patch activations.

        Args:
            x_BPD: Activation tensor with shape (batch_size, n_patches, d_vit) where d_vit is the ViT feature dimension

        Returns:
            Modified activation tensor with random unit vectors added
        """
        batch_size, n_patches, dim = x_BPD.shape
        rand_vecs = torch.randn((batch_size, n_patches, dim), device=cfg.device)
        # Normalize each vector to unit norm along the last dimension
        rand_vecs = rand_vecs / torch.norm(rand_vecs, dim=-1, keepdim=True)

        rand_vecs = rand_vecs * cfg.scale

        x_BPD += rand_vecs
        return x_BPD

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

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="random-vector",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@beartype.beartype
def eval_rand_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    """
    Evaluates the effects of suppressing a random SAE feature.

    Args:
        cfg: Configuration for quantitative evaluation
        sae: Trained sparse autoencoder model
        clf: Trained classifier model
        dataloader: DataLoader providing batches of images

    Returns:
        Report containing intervention results, including per-class changes
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    top_values = torch.load(cfg.top_values, map_location="cpu", weights_only=True)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        latent = random.randrange(0, sae.cfg.d_sae)

        x_hat_BPD, f_x_BPS, _ = sae(x_BPD)

        err_BPD = x_BPD - x_hat_BPD

        value = unscaled(cfg.scale, top_values[latent].max().item())
        f_x_BPS[..., latent] = value

        # Reproduce the SAE forward pass after f_x
        mod_x_hat_BPD = sae.decode(f_x_BPS)
        mod_BPD = err_BPD + mod_x_hat_BPD
        return mod_BPD

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

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="random-feature",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@beartype.beartype
def eval_auto_feat(
    cfg: config.Quantitative,
    sae: saev.nn.SparseAutoencoder,
    clf: torch.nn.Module,
    dataloader,
) -> Report:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    top_values = torch.load(cfg.top_values, map_location="cpu", weights_only=True)

    # First, for each class, we need to pick an appropriate latent.
    # Then we can use that to lookup which latent to suppress for each patch.
    latent_lookup = get_latent_lookup(cfg, sae, dataloader)

    @jaxtyped(typechecker=beartype.beartype)
    def hook(
        x_BPD: Float[Tensor, "batch patches dim"],
    ) -> Float[Tensor, "batch patches dim"]:
        batch_size, n_patches, dim = x_BPD.shape
        x_hat_BPD, f_x_BPS, _ = sae(x_BPD)

        err_BPD = x_BPD - x_hat_BPD

        print(batch["patch_labels"].shape)
        breakpoint()
        latents = latent_lookup[batch["patch_labels"].view(batch_size, -1).int()]
        values = unscaled(cfg.scale, top_values[latents].max().item())
        f_x_BPS[:, latents + 1] = value

        # Reproduce the SAE forward pass after f_x
        mod_x_hat_BPD = sae.decode(f_x_BPS)
        mod_BPD = err_BPD + mod_x_hat_BPD
        return mod_BPD

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

    class_results = compute_class_results(orig_preds, mod_preds)

    return Report(
        method="automatic-feature",
        class_results=class_results,
        intervention_scale=cfg.scale,
    )


@jaxtyped(typechecker=beartype.beartype)
def get_latent_lookup(
    cfg: config.Quantitative, sae: saev.nn.SparseAutoencoder, dataloader
) -> Int[Tensor, "151"]:
    latent_lookup = torch.zeros((151,), dtype=int)
    thresholds = [0, 0.1, 0.3, 1.0]

    vit = saev.activations.make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    recorded_vit = saev.activations.RecordedVisionTransformer(
        vit, cfg.n_patches_per_img, cfg.cls_token, [cfg.vit_layer]
    )

    thresholds_T = torch.tensor(thresholds, device=cfg.device)
    pred_labels_TSN = torch.zeros(
        (
            len(thresholds),
            sae.cfg.d_sae,
            len(dataloader.dataset) * cfg.n_patches_per_img,
        ),
        dtype=torch.uint8,
    )
    true_labels_N = torch.zeros(
        (len(dataloader.dataset) * cfg.n_patches_per_img), dtype=torch.uint8
    )

    for batch in saev.helpers.progress(dataloader):
        x_BCWH = batch["image"].to(cfg.device)
        _, vit_acts_BLPD = recorded_vit(x_BCWH)
        _, sae_acts_BPS, _ = sae(vit_acts_BLPD[:, 0, 1:, :].to(cfg.device))
        sae_acts_SN = einops.rearrange(
            sae_acts_BPS, "batch patches d_sae -> d_sae (batch patches)"
        )

        # TODO: i is basically arange(min, min + batch size * patches per img). Simplify to that.
        i = batch["index"][:, None, None].expand(batch["patch_labels"].shape)
        i = get_patch_i(i, cfg.n_patches_per_img).view(-1)
        true_labels_N[i] = batch["patch_labels"].view(-1)

        # Predictions for each latent is 1 for sae_acts_SI[latent] > threshold, 0 otherwise.
        pred_labels_TSN[:, :, i] = (sae_acts_SN[None] > thresholds_T[:, None, None]).to(
            "cpu", torch.uint8
        )

    logger.info("Made %d predictions.", len(true_labels_N))
    lookup = {}
    with open(os.path.join(cfg.imgs.root, "objectInfo150.txt")) as fd:
        for row in csv.DictReader(fd, delimiter="\t"):
            lookup[int(row["Idx"])] = row["Name"]

    for class_id, class_name in saev.helpers.progress(lookup.items()):
        is_class = true_labels_N == class_id
        is_not_class = ~is_class

        is_right = pred_labels_TSN == true_labels_N
        is_wrong = ~is_right
        logger.info("Got masks for '%s' (%d).", class_name, class_id)

        true_pos_TS = einops.reduce(
            is_right & is_class, "thresholds d_sae patches -> thresholds d_sae", "sum"
        )
        false_pos_TS = einops.reduce(
            is_wrong & is_class, "thresholds d_sae n_image -> thresholds d_sae", "sum"
        )
        false_neg_TS = einops.reduce(
            is_wrong & is_not_class,
            "thresholds d_sae n_image -> thresholds d_sae",
            "sum",
        )
        # Compute F1 scores for all thresholds and features at once
        numerator = 2 * true_pos_TS.float()
        denominator = 2 * true_pos_TS + false_pos_TS + false_neg_TS
        # Add small epsilon to avoid division by zero
        f1_TS = numerator / (denominator + 1e-8)
        
        # Find best threshold and score for each feature
        f1_S, best_thresh_i_S = f1_TS.max(dim=0)
        best_thresholds_S = thresholds_T[best_thresh_i_S]

        # Get top performing features
        topk_scores, topk_latents = torch.topk(f1_S, k=cfg.top_k)

        print(f"Top {cfg.top_k} features for {class_name}:")
        for score, latent in zip(topk_scores, topk_latents):
            print(f"{latent:>6} >{best_thresholds_S[latent]}: {score:.3f}")

        latent_lookup[class_id] = topk_latents[0].item()

    return latent_lookup


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


@jaxtyped(typechecker=beartype.beartype)
def compute_class_results(
    orig_preds: Int[Tensor, "n_imgs patches"], mod_preds: Int[Tensor, "n_imgs patches"]
) -> list[ClassResults]:
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

    return class_results


@jaxtyped(typechecker=beartype.beartype)
def argmax_logits(
    logits_BPC: Float[Tensor, "batch patches channels_with_null"],
) -> Int[Tensor, "batch patches"]:
    return logits_BPC[:, :, 1:].argmax(axis=-1) + 1


@jaxtyped(typechecker=beartype.beartype)
def unscaled(
    x: Float[Tensor, "*batch"], max_obs: float | int
) -> Float[Tensor, "*batch"]:
    """Scale from [-10, 10] to [10 * -max_obs, 10 * max_obs]."""
    return map_range(x, (-10.0, 10.0), (-10.0 * max_obs, 10.0 * max_obs))


@jaxtyped(typechecker=beartype.beartype)
def map_range(
    x: Float[Tensor, "*batch"],
    domain: tuple[float | int, float | int],
    range: tuple[float | int, float | int],
) -> Float[Tensor, "*batch"]:
    a, b = domain
    c, d = range
    if not (a <= x <= b):
        raise ValueError(f"x={x:.3f} must be in {[a, b]}.")
    return c + (x - a) * (d - c) / (b - a)


@jaxtyped(typechecker=beartype.beartype)
def get_patch_i(
    i: Int[Tensor, "batch width height"], n_patches_per_img: int
) -> Int[Tensor, "batch width height"]:
    w = h = int(math.sqrt(n_patches_per_img))

    i = i * n_patches_per_img
    i = i + torch.arange(n_patches_per_img).view(w, h)
    return i
