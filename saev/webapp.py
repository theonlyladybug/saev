"""

There is some important notation used only in this file to dramatically shorten variable names.

Variables suffixed with `_im` refer to entire images, and variables suffixed with `_p` refer to patches.
"""

import collections.abc
import logging
import math
import os
import pickle
import typing

import beartype
import datasets
import torch
import tqdm
from jaxtyping import Float, Int, jaxtyped
from PIL import Image, ImageDraw
from torch import Tensor

from . import activations, config, helpers, histograms, nn

logger = logging.getLogger("webapp")


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
def add_highlights(
    img: Image.Image,
    patches: Float[Tensor, " n_patches"],
    *,
    upper: float | None = None,
) -> Image.Image:
    iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
    iw_px, ih_px = img.size
    pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
    assert iw_np * ih_np == len(patches)

    # Create a transparent red overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Using semi-transparent red (255, 0, 0, alpha)
    for p, val in enumerate(patches):
        assert upper is not None
        alpha = int(val / upper * 128)
        x_np, y_np = p % iw_np, p // ih_np
        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(255, 0, 0, alpha),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img, overlay)


@beartype.beartype
def make_img_grid(
    imgs: list[tuple[Image.Image, Float[Tensor, " n_patches"]]],
    *,
    upper: float | None = None,
):
    """
    .. todo:: document this function.
    """
    imgs = imgs[:16]
    if len(imgs) < 16:
        logger.warning("Missing images; only have %d.", len(imgs))

    # Resize to 256x256 and crop to 224x224
    resize_size_px = (256, 256)
    resize_w_px, resize_h_px = resize_size_px
    crop_size_px = (224, 224)
    crop_w_px, crop_h_px = crop_size_px
    crop_coords_px = (
        (resize_w_px - crop_w_px) // 2,
        (resize_h_px - crop_h_px) // 2,
        (resize_w_px + crop_w_px) // 2,
        (resize_h_px + crop_h_px) // 2,
    )

    # Create an image grid
    grid_size = 4
    border_size = 2  # White border thickness

    # Create a new image with white background
    grid_w_px = grid_size * crop_w_px + (grid_size - 1) * border_size
    grid_h_px = grid_size * crop_h_px + (grid_size - 1) * border_size
    img_grid = Image.new("RGB", (grid_w_px, grid_h_px), "white")

    # Paste images in the grid
    x_offset, y_offset = 0, 0
    for i, (img, patches) in enumerate(imgs):
        img = img.resize(resize_size_px).crop(crop_coords_px).convert("RGBA")
        img = add_highlights(img, patches, upper=upper)
        img_grid.paste(img, (x_offset, y_offset))

        x_offset += crop_w_px + border_size
        if (i + 1) % grid_size == 0:
            x_offset = 0
            y_offset += crop_h_px + border_size

    return img_grid


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    val1: Float[Tensor, "d_sae k"],
    i1: Int[Tensor, "d_sae k"],
    val2: Float[Tensor, "d_sae k"],
    i2: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    """
    .. todo:: document this function.
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
    vit_acts: Float[Tensor, "n d_vit"], sae: nn.SparseAutoencoder, cfg: config.Webapp
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


@beartype.beartype
@torch.inference_mode()
def get_topk_cls(
    cfg: config.Webapp,
) -> tuple[
    Float[Tensor, "d_sae k 0"],
    Int[Tensor, "d_sae k"],
    Float[Tensor, " d_sae"],
    Float[Tensor, " d_sae"],
]:
    assert cfg.sort_by == "cls"
    breakpoint()


@beartype.beartype
@torch.inference_mode()
def get_topk_img(
    cfg: config.Webapp,
) -> tuple[
    Float[Tensor, "d_sae k n_patches_per_img"],
    Int[Tensor, "d_sae k"],
    Float[Tensor, " d_sae"],
    Float[Tensor, " d_sae"],
]:
    """
    .. todo:: Document this.
    """
    assert cfg.sort_by == "img"
    assert cfg.data.patches == "patches"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    dataset = activations.Dataset(cfg.data)

    top_values_p = torch.full(
        (sae.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img),
        -1.0,
        device=cfg.device,
    )
    top_i_im = torch.zeros((sae.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device)

    sparsity = torch.zeros((sae.d_sae,), device=cfg.device)
    mean_values = torch.zeros((sae.d_sae,), device=cfg.device)

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

    for vit_acts, i_im, _ in helpers.progress(dataloader):
        sae_acts = get_sae_acts(vit_acts, sae, cfg).transpose(0, 1)
        mean_values += sae_acts.sum(dim=1)
        sparsity += (sae_acts > 0).sum(dim=1)

        values_p = sae_acts.view(sae.d_sae, -1, dataset.metadata.n_patches_per_img)
        values_im = values_p.sum(axis=-1)
        i_im = torch.sort(torch.unique(i_im)).values

        # Checks that I did my reshaping correctly.
        assert values_p.shape[1] == i_im.shape[0]
        assert len(i_im) == n_imgs_per_batch

        # Pick out the top 16 images for each latent in this batch.
        values_im, i = torch.topk(values_im, k=cfg.top_k, dim=1)
        # Update patch-level values
        shape_in = (sae.d_sae * n_imgs_per_batch, dataset.metadata.n_patches_per_img)
        shape_out = (sae.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img)
        values_p = values_p.reshape(shape_in)[i.view(-1)].reshape(shape_out)
        # Update image indices
        i_im = i_im.to(cfg.device)[i.view(-1)].view(i.shape)

        # Pick out the top 16 images for each latent overall.
        top_values_im = top_values_p.sum(axis=-1)
        all_values_p = torch.cat((top_values_p, values_p), dim=1)
        all_values_im = torch.cat((top_values_im, values_im), dim=1)
        _, j = torch.topk(all_values_im, k=cfg.top_k, dim=1)

        shape_in = (sae.d_sae * cfg.top_k * 2, dataset.metadata.n_patches_per_img)
        top_values_p = all_values_p.reshape(shape_in)[j.view(-1)].reshape(
            top_values_p.shape
        )

        all_top_i = torch.cat((top_i_im, i_im), dim=1)
        top_i_im = torch.gather(all_top_i, 1, j)

    mean_values /= sparsity
    sparsity /= len(dataset)

    return top_values_p, top_i_im, mean_values, sparsity


@beartype.beartype
@torch.inference_mode()
def get_topk_patch(
    cfg: config.Webapp,
) -> tuple[
    Float[Tensor, "d_sae k n_patches_per_img"],
    Int[Tensor, "d_sae k"],
    Float[Tensor, " d_sae"],
    Float[Tensor, " d_sae"],
]:
    """
    Gets the top k images for each latent in the SAE.
    The top k images are for latent i are sorted by

        max over all patches: f_x(patch)[i]

    Thus, we could end up with duplicate images in the top k, if an image has more than one patch that maximally activates an SAE latent.

    Args:
        cfg: Config.

    Returns:

    """
    assert cfg.sort_by == "patch"
    assert cfg.data.patches == "patches"

    sae = nn.load(cfg.ckpt).to(cfg.device)
    dataset = activations.Dataset(cfg.data)

    top_values_p = torch.full(
        (sae.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img),
        -1.0,
        device=cfg.device,
    )
    top_i_im = torch.zeros((sae.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device)

    sparsity = torch.zeros((sae.d_sae,), device=cfg.device)
    mean_values = torch.zeros((sae.d_sae,), device=cfg.device)

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

    for vit_acts, i_im, i_patch in helpers.progress(dataloader):
        sae_acts = get_sae_acts(vit_acts, sae, cfg).transpose(0, 1)
        mean_values += sae_acts.sum(dim=1)
        sparsity += (sae_acts > 0).sum(dim=1)

        i_im = torch.sort(torch.unique(i_im)).values
        values_p = sae_acts.view(
            sae.d_sae, len(i_im), dataset.metadata.n_patches_per_img
        )

        # Checks that I did my reshaping correctly.
        assert values_p.shape[1] == i_im.shape[0]
        assert len(i_im) == n_imgs_per_batch

        _, k = torch.topk(sae_acts, k=cfg.top_k, dim=1)
        k_im = k // dataset.metadata.n_patches_per_img

        values_p = gather_batched(values_p, k_im)
        i_im = i_im.to(cfg.device)[k_im]

        all_values_p = torch.cat((top_values_p, values_p), axis=1)
        _, k = torch.topk(all_values_p.max(axis=-1).values, k=cfg.top_k, axis=1)

        top_values_p = gather_batched(all_values_p, k)
        top_i_im = torch.gather(torch.cat((top_i_im, i_im), axis=1), 1, k)

    mean_values /= sparsity
    sparsity /= len(dataset)

    return top_values_p, top_i_im, mean_values, sparsity


@beartype.beartype
@torch.inference_mode()
def dump_topk(cfg: config.Webapp):
    """
    For each SAE latent, we want to know which images have the most total "activation".
    That is, we keep track of each patch
    """
    if cfg.sort_by == "img":
        top_values_p, top_img_i, mean_values, sparsity = get_topk_img(cfg)
    elif cfg.sort_by == "cls":
        top_values_p, top_img_i, mean_values, sparsity = get_topk_patch(cfg)
    elif cfg.sort_by == "patch":
        top_values_p, top_img_i, mean_values, sparsity = get_topk_patch(cfg)
    else:
        typing.assert_never(cfg.sort_by)

    os.makedirs(cfg.root, exist_ok=True)

    torch.save(top_values_p, cfg.top_values_fpath)
    torch.save(top_img_i, cfg.top_img_i_fpath)
    torch.save(mean_values, cfg.mean_values_fpath)
    torch.save(sparsity, cfg.sparsity_fpath)


@beartype.beartype
@torch.inference_mode()
def main(cfg: config.Webapp):
    """
    .. todo:: document this function.

    Dump top-k images to a directory.

    Args:
        cfg: Configuration object.
    """

    try:
        top_values_p = safe_load(cfg.top_values_fpath)
        sparsity = safe_load(cfg.sparsity_fpath)
        mean_values = safe_load(cfg.mean_values_fpath)
        top_i = safe_load(cfg.top_img_i_fpath)
    except FileNotFoundError as err:
        logger.warning("Need to dump files: %s", err)
        dump_topk(cfg)
        return main(cfg)

    d_sae, cached_topk, n_patches = top_values_p.shape
    # Check that the data is at least shaped correctly.
    assert cfg.top_k == cached_topk
    if cfg.sort_by == "cls":
        assert n_patches == 0
    elif cfg.sort_by == "img":
        assert n_patches > 0
    elif cfg.sort_by == "patch":
        assert n_patches > 0
    else:
        typing.assert_never(cfg.sort_by)

    logger.info("Loaded sorted data.")

    fig = histograms.plot_log10_hist(sparsity + cfg.epsilon)
    fig_fpath = os.path.join(cfg.dump_to, "feature-freq.png")
    fig.savefig(fig_fpath)
    logger.info("Saved feature frequency histogram to %s.", fig_fpath)

    fig = histograms.plot_log10_hist(mean_values + cfg.epsilon)
    fig_fpath = os.path.join(cfg.dump_to, "feature-val.png")
    fig.savefig(fig_fpath)
    logger.info("Saved feature mean value histogram to %s.", fig_fpath)

    if isinstance(cfg.images, config.TreeOfLifeDataset):
        import wids

        dataset = wids.ShardListDataset(cfg.images.metadata).add_transform(
            lambda sample: {"image": sample[".jpg"]}
        )
    elif isinstance(cfg.images, config.ImagenetDataset):
        dataset = datasets.load_dataset(cfg.images.name, split="train")
    elif isinstance(cfg.images, config.LaionDataset):
        raise NotImplementedError(cfg.images)
    else:
        typing.assert_never(cfg.images)

    # Mask all neurons in the dense cluster
    # TODO: actually need to plot the 2d cluster instead of two 1d histograms
    mask = (
        (-2.5 < torch.log10(sparsity))
        & (torch.log10(sparsity) < 2.0)
        & (torch.log10(mean_values) > -0.5)
    )

    neuron_i = torch.arange(d_sae)[mask.cpu()].tolist()

    for i in tqdm.tqdm(neuron_i, desc="saving visuals"):
        neuron_dir = os.path.join(cfg.root, "neurons", str(i))
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        imgs = []
        seen_i_im = set()
        for i_im, values_p in zip(top_i[i].tolist(), top_values_p[i]):
            if i_im in seen_i_im:
                continue
            imgs.append((dataset[i_im]["image"], values_p))
            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values_p[i].numel() > 0:
            upper = top_values_p[i].max().item()

        img_grid = make_img_grid(imgs, upper=upper)
        img_grid.save(f"{neuron_dir}/top_images.png")

        # Metadata
        metadata = {
            "neuron": i,
            "log10 sparsity": torch.log10(sparsity)[i].item(),
            "mean activation": mean_values[i].item(),
        }
        with open(f"{neuron_dir}/metadata.pkl", "wb") as pickle_file:
            pickle.dump(metadata, pickle_file)
