import io
import json
import os

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import activations, config


@beartype.beartype
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    l1_coeff: float
    use_ghost_grads: bool

    def __init__(self, d_vit: int, d_sae: int, l1_coeff: float, use_ghost_grads: bool):
        super().__init__()

        self.l1_coeff = l1_coeff
        self.use_ghost_grads = use_ghost_grads

        # Initialize the weights.
        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_vit, d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_vit))
        )

        with torch.no_grad():
            # Anthropic normalizes this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = torch.nn.Parameter(torch.zeros(d_vit))

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, x: Float[Tensor, "batch d_model"], dead_neuron_mask=None):
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

        # add config for whether l2 is normalized:
        mse_loss = (
            torch.pow((x_hat - x.float()), 2) / (x**2).sum(dim=-1, keepdim=True).sqrt()
        )

        ghost_loss = torch.tensor(0.0, dtype=mse_loss.dtype, device=mse_loss.device)
        # gate on config and training so evals is not slowed down.
        if self.use_ghost_grads and self.training and dead_neuron_mask.sum() > 0:
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
        sparsity = torch.abs(f_x).sum(dim=1).mean(dim=(0,))
        l1_loss = self.l1_coeff * sparsity
        loss = mse_loss + l1_loss + ghost_loss

        return x_hat, f_x, loss, mse_loss, l1_loss, ghost_loss

    @torch.no_grad()
    def init_b_dec(
        self, cfg: config.Config, acts_store: activations.CachedActivationsStore
    ):
        previous_b_dec = self.b_dec.clone().cpu()

        all_activations = get_sae_batches(cfg, acts_store).detach().cpu()
        mean = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - mean, dim=-1)

        print(f"Prev dist: {previous_distances.median(0).values.mean().item()}")
        print(f"New dist: {distances.median(0).values.mean().item()}")

        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
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


#####################
# SparseAutoencoder #
#####################
# Depends on Config and CachedActivationsStore so has to come after them.


@jaxtyped(typechecker=beartype.beartype)
def get_sae_batches(
    cfg: config.Config, acts_store: activations.CachedActivationsStore
) -> Float[Tensor, "reinit_size d_model"]:
    """
    Get a batch of vit activations to re-initialize the SAE.

    Args:
        cfg: Config.
        acts_store: Activation store.
    """
    examples = []
    perm = np.random.default_rng(seed=cfg.seed).permutation(len(acts_store))
    perm = perm[: cfg.reinit_size]

    examples, _ = acts_store[perm]

    return examples


@beartype.beartype
def dump(cfg: config.Config, sae: SparseAutoencoder, run_id: str):
    filepath = f"{cfg.ckpt_path}/{run_id}/sae.pt"

    sae_kwargs = dict(
        d_vit=cfg.d_vit,
        d_sae=cfg.d_sae,
        l1_coeff=cfg.l1_coeff,
        use_ghost_grads=cfg.use_ghost_grads,
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as fd:
        kwargs_str = json.dumps(sae_kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(cfg: config.Config, run_id: str) -> SparseAutoencoder:
    filepath = f"{cfg.ckpt_path}/{run_id}/sae.pt"

    with open(filepath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = SparseAutoencoder(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=cfg.device)
    model.load_state_dict(state_dict)
    return model
