import collections.abc
import logging
import os

import beartype
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import activations, config, helpers, nn

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
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_vit"],
    sae: nn.SparseAutoencoder,
    cfg: config.Config,
) -> Float[Tensor, "n d_sae"]:
    """
    Get SAE hidden layer activations for a batch of ViT activations.

    Args:
        vit_acts: Batch of ViT activations
        sae: Sparse autoencder.
        cfg: Experimental config.
    """
    sae_acts = []
    for start, end in batched_idx(len(vit_acts), cfg.vit_batch_size):
        _, f_x, *_ = sae(vit_acts[start:end].to(cfg.device))
        sae_acts.append(f_x)

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(cfg.device)
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
def main(cfg: config.Config, run_id: str, *, top_k: int, root: str):
    """
    Runs the primary function in this file: `get_feature_data()`.

    Args:
        cfg: Experimental config.
        run_id: The SAE run you want to analyze.
        top_k: the number of top images to store per neuron.
        root: Root directory save information to.
    """

    sae = nn.load(cfg, run_id)
    acts_store = activations.CachedActivationsStore(cfg, None, on_missing="error")

    torch.cuda.empty_cache()

    sae.eval()
    sae = sae.to(cfg.device)

    top_values = torch.zeros((cfg.d_sae, top_k)).to(cfg.device)
    top_indices = torch.zeros((cfg.d_sae, top_k), dtype=torch.int)
    top_indices = top_indices.to(cfg.device)

    sae_sparsity = torch.zeros((cfg.d_sae,)).to(cfg.device)
    sae_mean_acts = torch.zeros((cfg.d_sae,)).to(cfg.device)

    dataloader = torch.utils.data.DataLoader(
        acts_store,
        batch_size=1024 * 16,
        shuffle=True,
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

        values, i = torch.topk(sae_acts, k=top_k, dim=1)
        # Convert i, a matrix of indices into this current batch, into indices, a matrix of indices into the global dataset.
        indices = indices.to(cfg.device)[i.view(-1)].view(i.shape)

        top_values, top_indices = get_new_topk(
            top_values, top_indices, values, indices, top_k
        )

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= len(acts_store)

    dump_to = os.path.join(root, run_id)
    # Check if the directory exists
    if not os.path.exists(dump_to):
        # Create the directory if it does not exist
        os.makedirs(dump_to)

    torch.save(top_indices, f"{dump_to}/max_activating_image_indices.pt")
    torch.save(top_values, f"{dump_to}/max_activating_image_values.pt")
    torch.save(sae_sparsity, f"{dump_to}/sae_sparsity.pt")
    torch.save(sae_mean_acts, f"{dump_to}/sae_mean_acts.pt")
    # Will always fail because ToL-10M doesn't have integer labels.
    # Compute the label tensor
    top_image_label_indices = acts_store.labels[top_indices.view(-1).cpu()].view(
        top_indices.shape
    )
    torch.save(
        top_image_label_indices,
        f"{dump_to}/max_activating_image_label_indices.pt",
    )
    # Should also save label information tensor here!!!
