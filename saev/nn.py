import io
import json
import logging
import os
import typing

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import activations, config


class Loss(typing.NamedTuple):
    reconstruction: Float[Tensor, ""]
    """Reconstruction loss, typically L2."""
    sparsity: Float[Tensor, ""]
    """Sparsity loss, typically lambda * L1."""
    ghost_grad: Float[Tensor, ""]
    """Ghost gradient loss, if any."""
    l0: Float[Tensor, ""]
    """L0 magnitude of hidden activations."""
    l1: Float[Tensor, ""]
    """L1 magnitude of hidden activations."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        return self.reconstruction + self.sparsity + self.ghost_grad


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    d_vit: int
    exp_factor: int
    sparsity_coeff: float
    ghost_grads: bool

    def __init__(
        self,
        *,
        d_vit: int,
        exp_factor: int,
        sparsity_coeff: float,
        ghost_grads: bool,
    ):
        super().__init__()

        self.d_vit = d_vit
        self.exp_factor = exp_factor

        self.sparsity_coeff = sparsity_coeff
        self.ghost_grads = ghost_grads

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_vit, self.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.d_sae, d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(d_vit))

        self.logger = logging.getLogger("sae")

    @property
    def d_sae(self) -> int:
        return self.d_vit * self.exp_factor

    @jaxtyped(typechecker=beartype.beartype)
    def forward(
        self, x: Float[Tensor, "batch d_model"], dead_neuron_mask=None
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae"], Loss]:
        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = torch.nn.functional.relu(h_pre)

        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )

        mse_loss = (
            torch.pow((x_hat - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        ghost_loss = torch.tensor(0.0, dtype=mse_loss.dtype, device=mse_loss.device)
        # gate on config and training so evals is not slowed down.
        if (
            self.ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            assert dead_neuron_mask is not None

            # ghost protocol

            # 1.
            residual = x - x_hat
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(h_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out *= norm_scaling_factor[:, None].detach()

            # 3.
            ghost_loss = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (ghost_loss + 1e-6)).detach()
            ghost_loss *= mse_rescaling_factor

        ghost_loss = ghost_loss.mean()
        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(axis=1).mean(axis=0)
        l1 = f_x.sum(axis=1).mean(axis=0)
        sparsity_loss = self.sparsity_coeff * l1

        return x_hat, f_x, Loss(mse_loss, sparsity_loss, ghost_loss, l0, l1)

    @torch.no_grad()
    def init_b_dec(self, cfg: config.Train, dataset: activations.Dataset):
        if cfg.n_reinit_batches <= 0:
            self.logger.info("Skipping init_b_dec.")
            return
        previous_b_dec = self.b_dec.clone().cpu()

        all_activations = get_sae_batches(cfg, dataset).detach().cpu()
        mean = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - mean, dim=-1)

        self.logger.info(
            "Prev dist: %.3f; new dist: %.3f",
            previous_distances.median(0).values.mean().item(),
            distances.median(0).values.mean().item(),
        )

        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def normalize_w_dec(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_vit, d_sae d_vit -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_vit -> d_sae d_vit",
        )


@jaxtyped(typechecker=beartype.beartype)
def get_sae_batches(
    cfg: config.Train, dataset: activations.Dataset
) -> Float[Tensor, "reinit_size d_model"]:
    """
    Get a batch of vit activations to re-initialize the SAE.

    Args:
        cfg: Train.
        dataset: Dataset.
    """
    perm = np.random.default_rng(seed=cfg.seed).permutation(len(dataset))
    perm = perm[: cfg.reinit_size]

    examples, _, _ = zip(*[dataset[p.item()] for p in perm])

    return torch.stack(examples)


@beartype.beartype
def dump(fpath: str, sae: SparseAutoencoder):
    sae_kwargs = dict(
        d_vit=sae.d_vit,
        exp_factor=sae.exp_factor,
        sparsity_coeff=sae.sparsity_coeff,
        ghost_grads=sae.ghost_grads,
    )

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        kwargs_str = json.dumps(sae_kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: str, *, device: str = "cpu") -> SparseAutoencoder:
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = SparseAutoencoder(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model
