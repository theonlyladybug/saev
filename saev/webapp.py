import collections.abc
import logging
import os
import pickle
import typing

import beartype
import datasets
import torch
import tqdm
from jaxtyping import Float, Int, jaxtyped
from PIL import Image
from torch import Tensor

from . import activations, config, helpers, histograms, nn

logger = logging.getLogger("webapp")


@beartype.beartype
def safe_load(path: str) -> object:
    return torch.load(path, map_location="cpu", weights_only=True)


@beartype.beartype
def make_img_grid(imgs: list):
    """
    .. todo:: document this function.
    """
    # Resize to 224x224
    img_width, img_height = 224, 224
    imgs = [img.resize((img_width, img_height)).convert("RGB") for img in imgs]

    # Create an image grid
    grid_size = 4
    border_size = 2  # White border thickness

    # Create a new image with white background
    grid_width = grid_size * img_width + (grid_size - 1) * border_size
    grid_height = grid_size * img_height + (grid_size - 1) * border_size
    img_grid = Image.new("RGB", (grid_width, grid_height), "white")

    # Paste images in the grid
    x_offset, y_offset = 0, 0
    for i, img in enumerate(imgs):
        img_grid.paste(img, (x_offset, y_offset))
        x_offset += img_width + border_size
        if (i + 1) % grid_size == 0:
            x_offset = 0
            y_offset += img_height + border_size
    return img_grid


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    first_values: Float[Tensor, "d_sae k"],
    first_indices: Int[Tensor, "d_sae k"],
    second_values: Float[Tensor, "d_sae k"],
    second_indices: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    """
    .. todo:: document this function.
    """
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
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
def main(cfg: config.Webapp):
    """
    .. todo:: document this function.
    """

    sae = nn.load(cfg.ckpt)
    dataset = activations.Dataset(cfg.data)
    sae = sae.to(cfg.device)

    top_values = torch.zeros((sae.d_sae, cfg.top_k), device=cfg.device)
    top_indices = torch.zeros(
        (sae.d_sae, cfg.top_k), dtype=torch.int, device=cfg.device
    )

    sae_sparsity = torch.zeros((sae.d_sae,), device=cfg.device)
    sae_mean_acts = torch.zeros((sae.d_sae,), device=cfg.device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.topk_batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        # See if you can change this to false and still pass the beartype check.
        drop_last=True,
    )

    logger.info("Loaded SAE and data.")

    for batch in helpers.progress(dataloader):
        torch.cuda.empty_cache()
        vit_acts, indices = batch

        # tensor of size [feature_idx, batch]
        sae_acts = get_sae_acts(vit_acts, sae, cfg).transpose(0, 1)
        del vit_acts
        sae_mean_acts += sae_acts.sum(dim=1)
        sae_sparsity += (sae_acts > 0).sum(dim=1)

        values, i = torch.topk(sae_acts, k=cfg.top_k, dim=1)
        # Convert i, a matrix of indices into this current batch, into indices, a matrix of indices into the global dataset.
        indices = indices.to(cfg.device)[i.view(-1)].view(i.shape)

        top_values, top_indices = get_new_topk(
            top_values, top_indices, values, indices, cfg.top_k
        )

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= len(dataset)

    # Check if the directory exists
    if not os.path.exists(cfg.dump_to):
        # Create the directory if it does not exist
        os.makedirs(cfg.dump_to)

    torch.save(top_indices, f"{cfg.dump_to}/top_indices.pt")
    torch.save(top_values, f"{cfg.dump_to}/top_values.pt")
    torch.save(sae_sparsity, f"{cfg.dump_to}/sae_sparsity.pt")
    torch.save(sae_mean_acts, f"{cfg.dump_to}/sae_mean_acts.pt")

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

    n_neurons, _ = top_values.shape

    fig = histograms.plot_log10_hist(sae_sparsity.cpu())
    fig_fpath = os.path.join(cfg.dump_to, "feature-freq.png")
    fig.savefig(fig_fpath)
    logger.info("Saved feature frequency histogram to %s.", fig_fpath)

    fig = histograms.plot_log10_hist(sae_mean_acts.cpu())
    fig_fpath = os.path.join(cfg.dump_to, "feature-val.png")
    fig.savefig(fig_fpath)
    logger.info("Saved feature mean value histogram to %s.", fig_fpath)

    # Mask all neurons in the dense cluster
    # TODO: actually need to plot the 2d cluster instead of two 1d histograms
    mask = (torch.log10(sae_sparsity) > -2.0) & (torch.log10(sae_mean_acts) > -1.0)

    img_indices = torch.arange(n_neurons)[mask.cpu()].tolist()

    os.makedirs(f"{cfg.dump_to}/neurons", exist_ok=True)
    for i in tqdm.tqdm(img_indices, desc="saving highest activating grids"):
        neuron_dir = f"{cfg.dump_to}/neurons/{i}"
        os.makedirs(neuron_dir, exist_ok=True)

        # Image grid
        imgs = [dataset[img_i]["image"] for img_i in top_indices[i][:16].tolist()]
        img_grid = make_img_grid(imgs)
        img_grid.save(f"{neuron_dir}/top_images.png")

        # Metadata
        metadata = {
            "neuron": i,
            "log10 sparsity": torch.log10(sae_sparsity)[i].item(),
            "mean activation": sae_mean_acts[i].item(),
        }
        with open(f"{neuron_dir}/metadata.pkl", "wb") as pickle_file:
            pickle.dump(metadata, pickle_file)
