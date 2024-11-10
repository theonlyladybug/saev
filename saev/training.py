import dataclasses
import json
import logging
import os.path

import beartype
import torch
import torch.optim.lr_scheduler as lr_scheduler
from jaxtyping import Float, jaxtyped
from torch import Tensor
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
        d_vit=dataset.d_vit,
        exp_factor=cfg.exp_factor,
        sparsity_coeff=cfg.sparsity_coeff,
        ghost_grads=cfg.ghost_grads,
    )
    sae.init_b_dec(cfg, dataset)

    # Gotta go fast. This won't make much of a difference because the model only has a couple kernels, so there's not a lot of python overhead.
    # BUG: This doesn't work. TODO: figure out why.
    # sae = torch.compile(sae)

    mode = "online" if cfg.track else "disabled"
    tags = [cfg.tag] if cfg.tag else []
    run = wandb.init(project=cfg.wandb_project, config=cfg, mode=mode, tags=tags)

    # track active features
    act_freq_scores = torch.zeros(sae.d_sae, device=cfg.device)
    n_fwd_passes_since_fired = torch.zeros(sae.d_sae, device=cfg.device)
    n_frac_active_tokens = 0

    optimizer = Adam(sae.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda steps: min(1.0, (steps + 1) / (cfg.n_lr_warmup + 1e-9)),
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
        if cfg.normalize_w_dec:
            # Make sure the W_dec is still unit-norm
            sae.normalize_w_dec()

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
        x_hat, f_x, loss = sae(vit_acts, ghost_grad_neuron_mask)
        did_fire = (f_x > 0).float().sum(-2) > 0
        n_fwd_passes_since_fired += 1
        n_fwd_passes_since_fired[did_fire] = 0

        n_patches_seen += len(vit_acts)

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (f_x.abs() > 0).float().sum(0)
            n_frac_active_tokens += cfg.sae_batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if (global_step + 1) % cfg.log_every == 0:
                # metrics for currents acts
                l0 = (f_x > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]

                per_token_l2_loss = (x_hat - vit_acts).pow(2).sum(dim=-1).squeeze()
                total_variance = vit_acts.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss / total_variance

                metrics = {
                    # losses
                    "losses/reconstruction": loss.reconstruction.item(),
                    "losses/l1": loss.l1.item(),
                    "losses/ghost_grad": loss.ghost_grad.item(),
                    "losses/loss": loss.loss.item(),
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
                    "sparsity/below_1e-7": (feature_sparsity < 1e-7)
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
                    "loss: %.5f, reconstruction loss: %.5f, sparsity loss: %.5f, l0: %.5f, l1: %.5f",
                    loss.loss.item(),
                    loss.reconstruction.item(),
                    loss.sparsity.item(),
                    loss.l0.item(),
                    loss.l1.item(),
                )

        loss.loss.backward()
        if cfg.remove_parallel_grads:
            sae.remove_parallel_grads()
        optimizer.step()

        global_step += 1

    ckpt_fpath = os.path.join(cfg.ckpt_path, run.id, "sae.pt")
    nn.dump(ckpt_fpath, sae)
    logger.info("Dumped checkpoint to '%s'.", ckpt_fpath)
    cfg_fpath = os.path.join(cfg.ckpt_path, run.id, "cfg.json")
    with open(cfg_fpath, "w") as fd:
        json.dump(dataclasses.asdict(cfg), fd, indent=4)

    # Cheap(-ish) evaluation
    n_dead, n_almost_dead, n_dense = evaluate(cfg, ckpt_fpath)
    metrics = {
        "eval/n_dead": n_dead,
        "eval/n_almost_dead": n_almost_dead,
        "eval/n_dense": n_dense,
    }
    run.log(metrics, step=global_step)

    run.finish()
    return run.id


@jaxtyped(typechecker=beartype.beartype)
def plot_log10_hist(
    frequencies: Float[Tensor, " d_sae"], *, eps: float = 1e-9
) -> tuple[object, object]:
    """
    Plot the histogram of feature frequency.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    frequencies = torch.log10(frequencies + eps)
    ax.hist(frequencies, bins=50)
    return fig, ax


@beartype.beartype
@torch.inference_mode()
def evaluate(cfg: config.Train | config.HistogramsEvaluate, ckpt_fpath: str):
    """
    Evaluates SAE quality by counting the number of dead features and the number of dense features.
    Also makes histogram plots to help human qualitative comparison.

    .. todo:: Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.
    """
    ckpt_name = os.path.basename(os.path.dirname(ckpt_fpath))

    sae = nn.load(ckpt_fpath).to(cfg.device)
    sae.eval()

    dataset = activations.Dataset(cfg.data)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.sae_batch_size,
        num_workers=cfg.n_workers,
        shuffle=False,
        pin_memory=True,
    )

    n_fired = torch.zeros(sae.d_sae).to(cfg.device)

    for vit_acts, _, _ in helpers.progress(
        dataloader, desc="eval", every=cfg.log_every
    ):
        vit_acts = vit_acts.to(cfg.device, non_blocking=True)
        _, f_x, *_ = sae(vit_acts)
        n_fired += (f_x > 0).sum(axis=0)

    freqs = n_fired / len(dataset)

    n_dense = (freqs > 0.01).sum().item()
    n_dead = (freqs == 0).sum().item()
    n_almost_dead = (freqs < 1e-7).sum().item()
    logger.info(
        "Checkpoint %s has %d dense features (%.1f)",
        ckpt_name,
        n_dense,
        n_dense / sae.d_sae * 100,
    )
    logger.info(
        "Checkpoint %s has %d dead features (%.1f%%)",
        ckpt_name,
        n_dead,
        n_dead / sae.d_sae * 100,
    )
    logger.info(
        "Checkpoint %s has %d *almost* dead (<1e-7) features (%.1f)",
        ckpt_name,
        n_almost_dead,
        n_almost_dead / sae.d_sae * 100,
    )

    fig, ax = plot_log10_hist(freqs.cpu())
    ax.set_title(f"{ckpt_name} Feature Frequencies")
    ax.set_xlabel("% of inputs a feature fires on (log10)")
    ax.set_ylabel("number of features")
    fig.tight_layout()

    fig_fpath = os.path.join(cfg.log_to, f"{ckpt_name}-feature-freqs.png")
    fig.savefig(fig_fpath)
    logger.info("Saved chart to '%s'.", fig_fpath)

    return n_dead, n_almost_dead, n_dense


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
