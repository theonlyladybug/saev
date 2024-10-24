import collections.abc
import logging
import os
import sys

import beartype
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev
from saev import helpers

# Fix pickle renaming errors.
sys.modules["sae_training"] = saev

logger = logging.getLogger("analysis")


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
def get_vit_acts(
    acts_store: saev.CachedActivationsStore, n: int
) -> tuple[Float[Tensor, "n d_model"], Int[Tensor, " n"]]:
    """
    Args:

    Returns:
        activation tensor for all images.
    """
    n_seen = 0
    batches, indices = [], []
    while n_seen < n:
        batch, i = acts_store.next_indexed_batch()
        batches.append(batch)
        indices.append(i)
        n_seen += len(batch)

    batches = torch.cat(batches, dim=0)
    indices = torch.cat(indices, dim=0)
    return batches, indices


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_model"], sae: saev.SparseAutoencoder
) -> Float[Tensor, "n d_sae"]:
    sae_acts = []
    for start, end in batched_idx(len(vit_acts), sae.cfg.vit_batch_size):
        _, f_x, *_ = sae(vit_acts[start:end])
        sae_acts.append(f_x)

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(sae.cfg.device)
    return sae_acts


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    first_values: Float[Tensor, "d_sae k"],
    first_indices: Int[Tensor, "d_sae k"],
    second_values: Float[Tensor, "d_sae k"],
    second_indices: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices


@beartype.beartype
@torch.inference_mode()
def get_feature_data(
    sae: saev.SparseAutoencoder,
    acts_store: saev.CachedActivationsStore,
    *,
    n_images: int = 32_768,
    k_top_images: int = 10,
    images_per_it: int = 16_384,
    directory: str = "data",
):
    """
    Args:
        sae: The sparse autoencoder to use
        vit: The vision transformer to apply the SAE to.
        n_images: How many images to look at.
        k_top_images: How many images per neuron (SAE feature) to keep.
        directory: Where to write this data.
    """
    torch.cuda.empty_cache()
    sae.eval()

    if n_images > len(acts_store):
        logger.warning(
            "The dataset '%s' only has %d images, but you requested %d images.",
            sae.cfg.dataset_path,
            len(acts_store),
            n_images,
        )
        n_images = len(acts_store)

    top_values = torch.zeros((sae.cfg.d_sae, k_top_images)).to(sae.cfg.device)
    top_indices = torch.zeros((sae.cfg.d_sae, k_top_images), dtype=torch.int)
    top_indices = top_indices.to(sae.cfg.device)

    sae_sparsity = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)
    sae_mean_acts = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)

    dataloader = torch.utils.data.DataLoader(
        acts_store,
        batch_size=images_per_it,
        shuffle=True,
        num_workers=sae.cfg.n_workers,
        drop_last=True,
    )

    for batch in helpers.progress(dataloader):
        torch.cuda.empty_cache()
        vit_acts, indices = batch

        # tensor of size [feature_idx, batch]
        sae_acts = get_sae_acts(vit_acts.to(sae.cfg.device), sae).transpose(0, 1)
        del vit_acts
        sae_mean_acts += sae_acts.sum(dim=1)
        sae_sparsity += (sae_acts > 0).sum(dim=1)

        values, i = torch.topk(sae_acts, k=k_top_images, dim=1)
        # Convert i, a matrix of indices into this current batch, into indices, a matrix of indices into the global dataset.
        indices = indices.to(sae.cfg.device)[i.view(-1)].view(i.shape)

        top_values, top_indices = get_new_topk(
            top_values, top_indices, values, indices, k_top_images
        )

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= len(acts_store)

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    # Compute the label tensor
    top_image_label_indices = acts_store.labels[top_indices.view(-1).cpu()].view(
        top_indices.shape
    )
    torch.save(top_indices, f"{directory}/max_activating_image_indices.pt")
    torch.save(top_values, f"{directory}/max_activating_image_values.pt")
    torch.save(
        top_image_label_indices,
        f"{directory}/max_activating_image_label_indices.pt",
    )
    torch.save(sae_sparsity, f"{directory}/sae_sparsity.pt")
    torch.save(sae_mean_acts, f"{directory}/sae_mean_acts.pt")
    # Should also save label information tensor here!!!


def main(
    ckpt_path: str,
    n_images: int = 524_288,
    k_top_images: int = 20,
    directory: str = "data",
):
    """
    Runs the primary function in this file: `get_feature_data()`.

    Args:
        ckpt_path: The SAE checkpoint to use.
        n_images: number of images to use. Use a smaller number for debugging.
        k_top_images: the number of top images to store per neuron.
    """
    _, sae, acts_store = saev.Session.from_disk(ckpt_path)
    get_feature_data(
        sae,
        acts_store,
        n_images=n_images,
        k_top_images=k_top_images,
        directory=directory,
    )


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    tyro.cli(main)
