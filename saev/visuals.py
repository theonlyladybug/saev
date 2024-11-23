"""
There is some important notation used only in this file to dramatically shorten variable names.

Variables suffixed with `_im` refer to entire images, and variables suffixed with `_p` refer to patches.
"""

import collections.abc
import dataclasses
import logging
import os
import pickle
import typing

import beartype
import torch
import tqdm
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


@beartype.beartype
@torch.inference_mode()
def get_topk_cls(
    cfg: config.Visuals,
) -> tuple[
    Float[Tensor, "d_sae k 0"],
    Int[Tensor, "d_sae k"],
    Float[Tensor, " d_sae"],
    Float[Tensor, " d_sae"],
]:
    assert cfg.sort_by == "cls"
    raise NotImplementedError()


@beartype.beartype
@torch.inference_mode()
def get_topk_img(
    cfg: config.Visuals,
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
        (sae.cfg.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img),
        -1.0,
        device=cfg.device,
    )
    top_i_im = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sparsity = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

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

    for vit_acts, i_im, _ in helpers.progress(dataloader, desc="picking top-k"):
        sae_acts = get_sae_acts(vit_acts, sae, cfg).transpose(0, 1)
        mean_values += sae_acts.sum(dim=1)
        sparsity += (sae_acts > 0).sum(dim=1)

        values_p = sae_acts.view(sae.cfg.d_sae, -1, dataset.metadata.n_patches_per_img)
        values_im = values_p.sum(axis=-1)
        i_im = torch.sort(torch.unique(i_im)).values

        # Checks that I did my reshaping correctly.
        assert values_p.shape[1] == i_im.shape[0]
        assert len(i_im) == n_imgs_per_batch

        # Pick out the top 16 images for each latent in this batch.
        values_im, i = torch.topk(values_im, k=cfg.top_k, dim=1)
        # Update patch-level values
        shape_in = (
            sae.cfg.d_sae * n_imgs_per_batch,
            dataset.metadata.n_patches_per_img,
        )
        shape_out = (sae.cfg.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img)
        values_p = values_p.reshape(shape_in)[i.view(-1)].reshape(shape_out)
        # Update image indices
        i_im = i_im.to(cfg.device)[i.view(-1)].view(i.shape)

        # Pick out the top 16 images for each latent overall.
        top_values_im = top_values_p.sum(axis=-1)
        all_values_p = torch.cat((top_values_p, values_p), dim=1)
        all_values_im = torch.cat((top_values_im, values_im), dim=1)
        _, j = torch.topk(all_values_im, k=cfg.top_k, dim=1)

        shape_in = (sae.cfg.d_sae * cfg.top_k * 2, dataset.metadata.n_patches_per_img)
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
    cfg: config.Visuals,
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
        (sae.cfg.d_sae, cfg.top_k, dataset.metadata.n_patches_per_img),
        -1.0,
        device=cfg.device,
    )
    top_i_im = torch.zeros(
        (sae.cfg.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sparsity = torch.zeros((sae.cfg.d_sae,), device=cfg.device)
    mean_values = torch.zeros((sae.cfg.d_sae,), device=cfg.device)

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

    for vit_acts, i_im, _ in helpers.progress(dataloader, desc="picking top-k"):
        sae_acts = get_sae_acts(vit_acts, sae, cfg).transpose(0, 1)
        mean_values += sae_acts.sum(dim=1)
        sparsity += (sae_acts > 0).sum(dim=1)

        i_im = torch.sort(torch.unique(i_im)).values
        values_p = sae_acts.view(
            sae.cfg.d_sae, len(i_im), dataset.metadata.n_patches_per_img
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
def dump_activations(cfg: config.Visuals):
    """
    For each SAE latent, we want to know which images have the most total "activation".
    That is, we keep track of each patch
    """
    if cfg.sort_by == "img":
        top_values_p, top_img_i, mean_values, sparsity = get_topk_img(cfg)
    elif cfg.sort_by == "cls":
        top_values_p, top_img_i, mean_values, sparsity = get_topk_cls(cfg)
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
def main(cfg: config.Visuals):
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
        dump_activations(cfg)
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

    dataset = activations.get_dataset(cfg.images, transform=None)

    min_log_freq, max_log_freq = cfg.log_freq_range
    min_log_value, max_log_value = cfg.log_value_range
    # breakpoint()
    mask = (
        (min_log_freq < torch.log10(sparsity))
        & (torch.log10(sparsity) < max_log_freq)
        & (min_log_value < torch.log10(mean_values))
        & (torch.log10(mean_values) < max_log_value)
    )

    neuron_i = cfg.include_latents + torch.arange(d_sae)[mask.cpu()].tolist()

    for i in tqdm.tqdm(neuron_i, desc="saving visuals"):
        neuron_dir = os.path.join(cfg.root, "neurons", str(i))
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        elems = []
        seen_i_im = set()
        for i_im, values_p in zip(top_i[i].tolist(), top_values_p[i]):
            if i_im in seen_i_im:
                continue
            example = dataset[i_im]
            elem = GridElement(example["image"], example["label"], values_p)
            elems.append(elem)

            seen_i_im.add(i_im)

        # How to scale values.
        upper = None
        if top_values_p[i].numel() > 0:
            upper = top_values_p[i].max().item()

        for i, elem in enumerate(elems):
            img = make_img(elem, upper=upper)
            img.save(os.path.join(neuron_dir, f"{i}.png"))
            with open(os.path.join(neuron_dir, f"{i}.txt"), "w") as fd:
                fd.write(elem.label + "\n")

        # Metadata
        metadata = {
            "neuron": i,
            "log10 sparsity": torch.log10(sparsity)[i].item(),
            "mean activation": mean_values[i].item(),
        }
        with open(f"{neuron_dir}/metadata.pkl", "wb") as pickle_file:
            pickle.dump(metadata, pickle_file)
