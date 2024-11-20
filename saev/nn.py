import io
import json
import logging
import os
import typing

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import config


class Loss(typing.NamedTuple):
    mse: Float[Tensor, ""]
    """Reconstruction loss (mean squared error)."""
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
        return self.mse + self.sparsity + self.ghost_grad


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    cfg: config.SparseAutoencoder

    def __init__(self, cfg: config.SparseAutoencoder):
        super().__init__()

        self.cfg = cfg

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_vit, cfg.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_vit))

        self.logger = logging.getLogger(f"sae(seed={cfg.seed})")

    def forward(
        self, x: Float[Tensor, "batch d_model"], dead_neuron_mask: None = None
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
        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        mse_loss = safe_mse(x_hat, x)

        ghost_loss = torch.tensor(0.0, dtype=mse_loss.dtype, device=mse_loss.device)
        # gate on config and training so evals is not slowed down.
        if (
            self.cfg.ghost_grads
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
        sparsity_loss = self.cfg.sparsity_coeff * l1

        return x_hat, f_x, Loss(mse_loss, sparsity_loss, ghost_loss, l0, l1)

    @torch.no_grad()
    def normalize_w_dec(self):
        # Make sure the W_dec is still unit-norm
        if self.cfg.normalize_w_dec:
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """
        if not self.cfg.remove_parallel_grads:
            return

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
def ref_mse(
    x_hat: Float[Tensor, "*d"], x: Float[Tensor, "*d"], norm: bool = True
) -> Float[Tensor, "*d"]:
    mse_loss = torch.pow((x_hat - x.float()), 2)

    if norm:
        mse_loss /= (x**2).sum(dim=-1, keepdim=True).sqrt()
    return mse_loss


@jaxtyped(typechecker=beartype.beartype)
def safe_mse(
    x_hat: Float[Tensor, "*batch d"], x: Float[Tensor, "*batch d"], norm: bool = False
) -> Float[Tensor, "*batch d"]:
    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, axis=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper


@beartype.beartype
def dump(fpath: str, sae: SparseAutoencoder):
    kwargs = vars(sae.cfg)

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        kwargs_str = json.dumps(kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: str, *, device: str = "cpu") -> SparseAutoencoder:
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    cfg = config.SparseAutoencoder(**kwargs)
    model = SparseAutoencoder(cfg)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model
