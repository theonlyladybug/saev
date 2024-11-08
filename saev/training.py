import logging

import beartype
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam

import wandb

from . import activations, config, helpers, nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@beartype.beartype
def train(cfg: config.Train) -> str:
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    dataset = activations.Dataset(cfg.data)
    sae = nn.SparseAutoencoder(
        dataset.d_vit, cfg.exp_factor, cfg.l1_coeff, cfg.use_ghost_grads
    )
    sae.init_b_dec(cfg, dataset)

    # Gotta go fast. This won't make much of a difference because the model only has a couple kernels, so there's not a lot of python overhead.
    # BUG: This doesn't work. TODO: figure out why.
    # sae = torch.compile(sae)

    mode = "online" if cfg.track else "disabled"
    run = wandb.init(project=cfg.wandb_project, config=cfg, mode=mode)

    # track active features
    act_freq_scores = torch.zeros(sae.d_sae, device=cfg.device)
    n_fwd_passes_since_fired = torch.zeros(sae.d_sae, device=cfg.device)
    n_frac_active_tokens = 0

    optimizer = Adam(sae.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda steps: min(1.0, (steps + 1) / cfg.n_lr_warmup),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.sae_batch_size, num_workers=cfg.n_workers, shuffle=True
    )

    dataloader = BatchLimiter(dataloader, cfg.n_patches)

    sae.train()
    sae = sae.to(cfg.device)

    global_step, n_patches_seen = 0, 0

    for vit_acts, _, _ in helpers.progress(dataloader, every=cfg.log_every):
        vit_acts = vit_acts.to(cfg.device, non_blocking=True)
        # Make sure the W_dec is still zero-norm
        sae.set_decoder_norm_to_unit_norm()

        # after resampling, reset the sparsity:
        if (global_step + 1) % cfg.feature_sampling_window == 0:
            # feature_sampling_window divides dead_sampling_window
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
            metrics = {
                "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item()
            }
            run.log(metrics, step=global_step)

            act_freq_scores = torch.zeros(sae.d_sae, device=cfg.device)
            n_frac_active_tokens = 0

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_fwd_passes_since_fired > cfg.dead_feature_window
        ).bool()

        # Forward and Backward Passes
        sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sae(
            vit_acts, ghost_grad_neuron_mask
        )
        did_fire = (feature_acts > 0).float().sum(-2) > 0
        n_fwd_passes_since_fired += 1
        n_fwd_passes_since_fired[did_fire] = 0

        n_patches_seen += len(vit_acts)

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += cfg.sae_batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if (global_step + 1) % cfg.log_every == 0:
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]

                per_token_l2_loss = (sae_out - vit_acts).pow(2).sum(dim=-1).squeeze()
                total_variance = vit_acts.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss / total_variance

                metrics = {
                    # losses
                    "losses/mse_loss": mse_loss.item(),
                    # normalize by l1 coefficient
                    "losses/l1_loss": l1_loss.item() / sae.l1_coeff,
                    "losses/ghost_grad_loss": ghost_grad_loss.item(),
                    "losses/overall_loss": loss.item(),
                    # variance explained
                    "metrics/explained_variance": explained_variance.mean().item(),
                    "metrics/explained_variance_std": explained_variance.std().item(),
                    "metrics/l0": l0.item(),
                    # sparsity
                    "sparsity/mean_passes_since_fired": n_fwd_passes_since_fired.mean().item(),
                    "sparsity/n_passes_since_fired_over_threshold": ghost_grad_neuron_mask.sum().item(),
                    "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/dead_features": (
                        feature_sparsity < cfg.dead_feature_threshold
                    )
                    .float()
                    .mean()
                    .item(),
                    "details/n_patches_seen": n_patches_seen,
                    "details/current_learning_rate": current_learning_rate,
                }
                run.log(metrics, step=global_step)

                logger.info(
                    "step: %d (%.1f%%), loss: %.5f, mse loss: %.5f, l1 loss: %.5f",
                    global_step,
                    n_patches_seen / cfg.n_patches * 100,
                    loss.item(),
                    mse_loss.item(),
                    l1_loss.item() / cfg.l1_coeff,
                )

        loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        global_step += 1

    nn.dump(cfg, sae, run.id)
    run.finish()
    return run.id


class BatchLimiter:
    """
    Limits the number of batches to only return `n_samples` total samples.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, n_samples: int):
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.batch_size = dataloader.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self):
        self.n_seen = 0
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += self.batch_size
                if self.n_seen > self.n_samples:
                    return

            # We try to mitigate the above issue by ignoring the last batch if we don't have drop_last.
            if not self.dataloader.drop_last:
                self.n_seen -= self.batch_size
