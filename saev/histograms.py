"""
Evaluates SAE quality by counting the number of dead features and the number of dense features.
Optionally makes histogram plots to help human qualitative comparison.


.. todo:: Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.
"""

import logging
import os

import beartype
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import activations, config, helpers, nn

logger = logging.getLogger("histograms")


@jaxtyped(typechecker=beartype.beartype)
def plot_log10_hist(
    frequencies: Float[Tensor, " d_sae"], *, eps: float = 1e-9
) -> tuple[object, object]:
    """
    Plot the histogram of feature frequency.
    """
    fig, ax = plt.subplots()
    frequencies = torch.log10(frequencies)
    ax.hist(frequencies, bins=50)
    fig.tight_layout()
    return fig, ax


@beartype.beartype
@torch.inference_mode()
def evaluate(cfg: config.HistogramsEvaluate):
    """
    Count the number of feature activations on the training set.
    """
    sae = nn.load(cfg.ckpt).to(cfg.device)
    sae.eval()

    dataset = activations.Dataset(cfg.data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        pin_memory=True,
    )

    n_fired = torch.zeros(sae.d_sae).to(cfg.device)

    for vit_acts, _, _ in helpers.progress(dataloader):
        vit_acts = vit_acts.to(cfg.device, non_blocking=True)
        _, f_x, *_ = sae(vit_acts)
        n_fired += (f_x > 0).sum(axis=0)

    freqs = n_fired / len(dataset)

    n_dense = (freqs > 0.01).sum().item()
    n_dead = (freqs == 0).sum().item()
    n_almost_dead = (freqs < 1e-7).sum().item()
    logger.info(
        "Checkpoint %s has %d dense features (%.1f)",
        cfg.ckpt,
        n_dense,
        n_dense / sae.d_sae * 100,
    )
    logger.info(
        "Checkpoint %s has %d dead features (%.1f)",
        cfg.ckpt,
        n_dead,
        n_dead / sae.d_sae * 100,
    )
    logger.info(
        "Checkpoint %s has %d *almost* dead (<1e-7) features (%.1f)",
        cfg.ckpt,
        n_almost_dead,
        n_almost_dead / sae.d_sae * 100,
    )

    fig, ax = plot_log10_hist(freqs.cpu())
    ax.set_title("Feature Frequencies")

    ckpt_name = os.path.basename(os.path.dirname(cfg.ckpt))
    fig_fpath = f"{ckpt_name}-feature-freqs.png"
    fig.savefig(fig_fpath)
    logger.info("Saved chart to '%s'.", fig_fpath)
