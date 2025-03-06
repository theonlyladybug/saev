"""
There is some important notation used only in this file to dramatically shorten variable names.

Variables suffixed with `_im` refer to entire images, and variables suffixed with `_p` refer to patches.
"""

import collections.abc
import dataclasses
import json
import logging
import math
import os
import random
import typing

import beartype
import einops
import torch
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from . import activations, config, helpers, imaging, nn

logger = logging.getLogger("visuals")


@beartype.beartype
def safe_load(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


@jaxtyped(typechecker=beartype.beartype)
def gather_batched(
    value: Float[Tensor, "batch n dim"], i: Int[Tensor, "batch k"]
) -> Float[Tensor, "batch k dim"]:
    batch_size, n, dim = value.shape  # noqa: F841
    _, k = i.shape

    batch_i = torch.arange(batch_size, device=value.device)[:, None].expand(-1, k)
    return value[batch_i, i]


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass
class GridElement:
    img: Image.Image
    label: str
    patches: Float[Tensor, " n_patches"]


@beartype.beartype
def make_img(elem: GridElement, *, upper: float | None = None) -> Image.Image:
    # Resize to 256x256 and crop to 224x224
    resize_size_px = (512, 512)
    resize_w_px, resize_h_px = resize_size_px
    crop_size_px = (448, 448)
    crop_w_px, crop_h_px = crop_size_px
    crop_coords_px = (
        (resize_w_px - crop_w_px) // 2,
        (resize_h_px - crop_h_px) // 2,
        (resize_w_px + crop_w_px) // 2,
        (resize_h_px + crop_h_px) // 2,
    )

    img = elem.img.resize(resize_size_px).crop(crop_coords_px)
    img = imaging.add_highlights(img, elem.patches.numpy(), upper=upper)
    return img


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    val1: Float[Tensor, "d_sae k"],
    i1: Int[Tensor, "d_sae k"],
    val2: Float[Tensor, "d_sae k"],
    i2: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    """
    Picks out the new top k values among val1 and val2. Also keeps track of i1 and i2, then indices of the values in the original dataset.

    Args:
        val1: top k original SAE values.
        i1: the patch indices of those original top k values.
        val2: top k incoming SAE values.
        i2: the patch indices of those incoming top k values.
        k: k.

    Returns:
        The new top k values and their patch indices.
    """
    all_val = torch.cat([val1, val2], dim=1)
    new_values, top_i = torch.topk(all_val, k=k, dim=1)

    all_i = torch.cat([i1, i2], dim=1)
    new_indices = torch.gather(all_i, 1, top_i)
    return new_values, new_indices


@beartype.beartype
def batched_idx(
    total_size: int, batch_size: int
) -> collections.abc.Iterator[tuple[int, int]]:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield start, stop


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_vit"], sae: nn.SparseAutoencoder, cfg: config.Visuals
) -> Float[Tensor, "n d_sae"]:
    """
    Get SAE hidden layer activations for a batch of ViT activations.

    Args:
        vit_acts: Batch of ViT activations
        sae: Sparse autoencder.
        cfg: Experimental config.
    """
    sae_acts = []
    for start, end in batched_idx(len(vit_acts), cfg.sae_batch_size):
        _, f_x, *_ = sae(vit_acts[start:end].to(cfg.device))
        sae_acts.append(f_x)

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(cfg.device)
    return sae_acts


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TopKImg:
    ".. todo:: Document this class."

    top_values: Float[Tensor, "d_sae k"]
    top_i: Int[Tensor, "d_sae k"]
    mean_values: Float[Tensor, " d_sae"]
    sparsity: Float[Tensor, " d_sae"]
    distributions: Float[Tensor, "m n"]
    percentiles: Float[Tensor, " d_sae"]


@beartype.beartype
@torch.inference_mode()
def get_topk_img(cfg: config.Visuals) -> TopKImg:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all images: f_x(cls)[i]

    Thus, we will never have duplicate images for a given latent.
    But we also will not have patch-level activations (a nice heatmap).

    Args:
        cfg: Config.

    Returns:
        A tuple of TopKImg and the first m features' activation distributions.
    """
    assert cfg.sort_by == "img"
    assert cfg.data.patches == "cls"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    dataset = activations.Dataset(cfg.data)

    top_values_im_SK = torch.full((sae.cfg.d_sae, cfg.top_k), -1.0, device=cfg.device)
    top_i_im_SK = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )
    sparsity_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    distributions_MN = torch.zeros((cfg.n_distributions, len(dataset)), device="cpu")
    estimator = PercentileEstimator(
        cfg.percentile, len(dataset), shape=(sae.cfg.d_sae,)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.topk_batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        drop_last=False,
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader, desc="picking top-k"):
        vit_acts_BD = batch["act"]
        sae_acts_BS = get_sae_acts(vit_acts_BD, sae, cfg)

        for sae_act_S in sae_acts_BS:
            estimator.update(sae_act_S)

        sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
        distributions_MN[:, batch["image_i"]] = sae_acts_SB[: cfg.n_distributions].to(
            "cpu"
        )

        mean_values_S += einops.reduce(sae_acts_SB, "d_sae batch -> d_sae", "sum")
        sparsity_S += einops.reduce((sae_acts_SB > 0), "d_sae batch -> d_sae", "sum")

        sae_acts_SK, k = torch.topk(sae_acts_SB, k=cfg.top_k, dim=1)
        i_im_SK = batch["image_i"].to(cfg.device)[k]

        all_values_im_2SK = torch.cat((top_values_im_SK, sae_acts_SK), axis=1)

        top_values_im_SK, k = torch.topk(all_values_im_2SK, k=cfg.top_k, axis=1)
        top_i_im_SK = torch.gather(torch.cat((top_i_im_SK, i_im_SK), axis=1), 1, k)

    mean_values_S /= sparsity_S
    sparsity_S /= len(dataset)

    return TopKImg(
        top_values_im_SK,
        top_i_im_SK,
        mean_values_S,
        sparsity_S,
        distributions_MN,
        estimator.estimate.cpu(),
    )


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class TopKPatch:
    ".. todo:: Document this class."

    top_values: Float[Tensor, "d_sae k n_patches_per_img"]
    top_i: Int[Tensor, "d_sae k"]
    mean_values: Float[Tensor, " d_sae"]
    sparsity: Float[Tensor, " d_sae"]
    distributions: Float[Tensor, "m n"]
    percentiles: Float[Tensor, " d_sae"]


@beartype.beartype
@torch.inference_mode()
def get_topk_patch(cfg: config.Visuals) -> TopKPatch:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all patches: f_x(patch)[i]

    Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent.

    Args:
        cfg: Config.

    Returns:
        A tuple of TopKPatch and m randomly sampled activation distributions.
    """
    assert cfg.sort_by == "patch"
    assert cfg.data.patches == "patches"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    dataset = activations.Dataset(cfg.data)

    top_values_p = torch.full(
        (sae.cfg.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img),
        -1.0,
        device=cfg.device,
    )
    top_i_im = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sparsity_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values_S = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

    distributions_MN = torch.zeros((cfg.n_distributions, len(dataset)), device="cpu")
    estimator = PercentileEstimator(
        cfg.percentile, len(dataset), shape=(sae.cfg.d_sae,)
    )

    batch_size = (
        cfg.topk_batch_size
        // dataset.metadata.n_patches_per_img
        * dataset.metadata.n_patches_per_img
    )
    n_imgs_per_batch = batch_size // dataset.metadata.n_patches_per_img

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        # See if you can change this to false and still pass the beartype check.
        drop_last=True,
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader, desc="picking top-k"):
        vit_acts_BD = batch["act"]
        sae_acts_BS = get_sae_acts(vit_acts_BD, sae, cfg)

        for sae_act_S in sae_acts_BS:
            estimator.update(sae_act_S)

        sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
        distributions_MN[:, batch["image_i"]] = sae_acts_SB[: cfg.n_distributions].to(
            "cpu"
        )

        mean_values_S += einops.reduce(sae_acts_SB, "d_sae batch -> d_sae", "sum")
        sparsity_S += einops.reduce((sae_acts_SB > 0), "d_sae batch -> d_sae", "sum")

        i_im = torch.sort(torch.unique(batch["image_i"])).values
        values_p = sae_acts_SB.view(
            sae.cfg.d_sae, len(i_im), dataset.metadata.n_patches_per_img
        )

        # Checks that I did my reshaping correctly.
        assert values_p.shape[1] == i_im.shape[0]
        assert len(i_im) == n_imgs_per_batch

        _, k = torch.topk(sae_acts_SB, k=cfg.top_k, dim=1)
        k_im = k // dataset.metadata.n_patches_per_img

        values_p = gather_batched(values_p, k_im)
        i_im = i_im.to(cfg.device)[k_im]

        all_values_p = torch.cat((top_values_p, values_p), axis=1)
        _, k = torch.topk(all_values_p.max(axis=-1).values, k=cfg.top_k, axis=1)

        top_values_p = gather_batched(all_values_p, k)
        top_i_im = torch.gather(torch.cat((top_i_im, i_im), axis=1), 1, k)

    mean_values_S /= sparsity_S
    sparsity_S /= len(dataset)

    return TopKPatch(
        top_values_p,
        top_i_im,
        mean_values_S,
        sparsity_S,
        distributions_MN,
        estimator.estimate.cpu(),
    )


@beartype.beartype
@torch.inference_mode()
def dump_activations(cfg: config.Visuals):
    """
    For each SAE latent, we want to know which images have the most total "activation".
    That is, we keep track of each patch
    """
    if cfg.sort_by == "img":
        topk = get_topk_img(cfg)
    elif cfg.sort_by == "patch":
        topk = get_topk_patch(cfg)
    else:
        typing.assert_never(cfg.sort_by)

    os.makedirs(cfg.root, exist_ok=True)

    torch.save(topk.top_values, cfg.top_values_fpath)
    torch.save(topk.top_i, cfg.top_img_i_fpath)
    torch.save(topk.mean_values, cfg.mean_values_fpath)
    torch.save(topk.sparsity, cfg.sparsity_fpath)
    torch.save(topk.distributions, cfg.distributions_fpath)
    torch.save(topk.percentiles, cfg.percentiles_fpath)


@jaxtyped(typechecker=beartype.beartype)
def plot_activation_distributions(
    cfg: config.Visuals, distributions: Float[Tensor, "m n"]
):
    import matplotlib.pyplot as plt
    import numpy as np

    m, _ = distributions.shape

    n_rows = int(math.sqrt(m))
    n_cols = n_rows
    fig, axes = plt.subplots(
        figsize=(4 * n_cols, 4 * n_rows),
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        sharey=True,
    )

    _, bins = np.histogram(np.log10(distributions[distributions > 0].numpy()), bins=100)

    percentiles = [90, 95, 99, 100]
    colors = ("red", "darkorange", "gold", "lime")

    for dist, ax in zip(distributions, axes.reshape(-1)):
        vals = np.log10(dist[dist > 0].numpy())

        ax.hist(vals, bins=bins)

        if vals.size == 0:
            continue

        for i, (percentile, color) in enumerate(
            zip(np.percentile(vals, percentiles), colors)
        ):
            ax.axvline(percentile, color=color, label=f"{percentiles[i]}th %-ile")

        for i, (percentile, color) in enumerate(zip(percentiles, colors)):
            estimator = PercentileEstimator(percentile, len(vals))
            for v in vals:
                estimator.update(v)
            ax.axvline(
                estimator.estimate,
                color=color,
                linestyle="--",
                label=f"Est. {percentiles[i]}th %-ile",
            )

    ax.legend()

    fig.tight_layout()
    return fig


@beartype.beartype
@torch.inference_mode()
def main(cfg: config.Visuals):
    """
    .. todo:: document this function.

    Dump top-k images to a directory.

    Args:
        cfg: Configuration object.
    """
    print("cfg", cfg)
    try:
        top_values = safe_load(cfg.top_values_fpath)
        sparsity = safe_load(cfg.sparsity_fpath)
        mean_values = safe_load(cfg.mean_values_fpath)
        top_i = safe_load(cfg.top_img_i_fpath)
        distributions = safe_load(cfg.distributions_fpath)
        _ = safe_load(cfg.percentiles_fpath)
    except FileNotFoundError as err:
        logger.warning("Need to dump files: %s", err)
        dump_activations(cfg)
        return main(cfg)

    d_sae, cached_topk, *rest = top_values.shape
    # Check that the data is at least shaped correctly.
    assert cfg.top_k == cached_topk
    if cfg.sort_by == "img":
        assert len(rest) == 0
    elif cfg.sort_by == "patch":
        assert len(rest) == 1
        n_patches = rest[0]
        assert n_patches > 0
    else:
        typing.assert_never(cfg.sort_by)

    logger.info("Loaded sorted data.")

    os.makedirs(cfg.root, exist_ok=True)
    fig_fpath = os.path.join(
        cfg.root, f"{cfg.n_distributions}_activation_distributions.png"
    )
    plot_activation_distributions(cfg, distributions).savefig(fig_fpath, dpi=300)
    logger.info(
        "Saved %d activation distributions to '%s'.", cfg.n_distributions, fig_fpath
    )

    dataset = activations.get_dataset(cfg.images, img_transform=None)

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range

    mask = (
        (min_log_freq < torch.log10(sparsity))
        & (torch.log10(sparsity) < max_log_freq)
        & (min_log_value < torch.log10(mean_values))
        & (torch.log10(mean_values) < max_log_value)
    )

    neurons = cfg.include_latents
    random_neurons = torch.arange(d_sae)[mask.cpu()].tolist()
    random.seed(cfg.seed)
    random.shuffle(random_neurons)
    neurons += random_neurons[: cfg.n_latents]

    for i in helpers.progress(neurons, desc="saving visuals"):
        neuron_dir = os.path.join(cfg.root, "neurons", str(i))
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        elems = []
        seen_i_im = set()
        for i_im, values_p in zip(top_i[i].tolist(), top_values[i]):
            if i_im in seen_i_im:
                continue

            example = dataset[i_im]
            if cfg.sort_by == "img":
                elem = GridElement(example["image"], example["label"], torch.tensor([]))
            elif cfg.sort_by == "patch":
                elem = GridElement(example["image"], example["label"], values_p)
            else:
                typing.assert_never(cfg.sort_by)
            elems.append(elem)

            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values[i].numel() > 0:
            upper = top_values[i].max().item()

        for j, elem in enumerate(elems):
            img = make_img(elem, upper=upper)
            img.save(os.path.join(neuron_dir, f"{j}.png"))
            with open(os.path.join(neuron_dir, f"{j}.txt"), "w") as fd:
                fd.write(elem.label + "\n")

        # Metadata
        metadata = {
            "neuron": i,
            "log10_freq": torch.log10(sparsity[i]).item(),
            "log10_value": torch.log10(mean_values[i]).item(),
        }
        with open(os.path.join(neuron_dir, "metadata.json"), "w") as fd:
            json.dump(metadata, fd)


@beartype.beartype
class PercentileEstimator:
    def __init__(
        self,
        percentile: float | int,
        total: int,
        lr: float = 1e-3,
        shape: tuple[int, ...] = (),
    ):
        self.percentile = percentile
        self.total = total
        self.lr = lr

        self._estimate = torch.zeros(shape)
        self._step = 0

    def update(self, x):
        """
        Update the estimator with a new value.

        This method maintains the marker positions using the P2 algorithm rules.
        When a new value arrives, it's placed in the appropriate position relative to existing markers, and marker positions are adjusted to maintain their desired percentile positions.

        Arguments:
            x: The new value to incorporate into the estimation
        """
        self._step += 1

        step_size = self.lr * (self.total - self._step) / self.total

        # Is a no-op if it's already on the same device.
        self._estimate = self._estimate.to(x.device)

        self._estimate += step_size * (
            torch.sign(x - self._estimate) + 2 * self.percentile / 100 - 1.0
        )

    @property
    def estimate(self):
        return self._estimate


@beartype.beartype
def test_online_quantile_estimation(true: float, percentile: float):
    import matplotlib.pyplot as plt
    import numpy as np
    import tqdm

    rng = np.random.default_rng(seed=0)
    n = 3_000_000
    estimator = PercentileEstimator(percentile, n)

    dist, preds = np.zeros(n), np.zeros(n)
    for i in tqdm.tqdm(range(n), desc="Getting estimates."):
        sampled = rng.normal(true)
        estimator.update(sampled)
        dist[i] = sampled
        preds[i] = estimator.estimate

    fig, ax = plt.subplots()
    ax.plot(preds, label=f"Pred. {percentile * 100}th %-ile")
    ax.axhline(
        np.percentile(dist, percentile * 100),
        label=f"True {percentile * 100}th %-ile",
        color="tab:red",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("online_median_normal.png")


if __name__ == "__main__":
    import tyro
    tyro.cli(test_online_quantile_estimation)
