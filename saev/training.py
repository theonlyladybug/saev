import json
import os

import beartype
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam

import wandb

from . import config, modeling


@beartype.beartype
def main(cfg: config.Config) -> tuple[modeling.SparseAutoencoder, modeling.RecordedVit]:
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    vit, sae, acts_store = modeling.Session.from_cfg(cfg)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # train SAE
    batch_size = sae.cfg.batch_size

    n_training_tokens = 0

    # track active features
    act_freq_scores = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
    n_fwd_passes_since_fired = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
    n_frac_active_tokens = 0

    optimizer = Adam(sae.parameters(), lr=sae.cfg.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda steps: min(1.0, (steps + 1) / sae.cfg.lr_warm_up_steps),
    )

    sae.initialize_b_dec(acts_store)

    dataloader = torch.utils.data.DataLoader(
        acts_store,
        batch_size=sae.cfg.batch_size,
        num_workers=sae.cfg.n_workers,
        shuffle=True,
    )

    sae.train()
    sae = sae.to(sae.cfg.device)

    n_steps = 0

    for epoch in range(sae.cfg.n_epochs):
        for batch in dataloader:
            vit_acts, _ = batch
            vit_acts = vit_acts.to(sae.cfg.device)
            # Make sure the W_dec is still zero-norm
            sae.set_decoder_norm_to_unit_norm()

            # after resampling, reset the sparsity:
            if (n_steps + 1) % sae.cfg.feature_sampling_window == 0:
                # feature_sampling_window divides dead_sampling_window
                feature_sparsity = act_freq_scores / n_frac_active_tokens
                log_feature_sparsity = (
                    torch.log10(feature_sparsity + 1e-10).detach().cpu()
                )

                if sae.cfg.log_to_wandb:
                    wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                    wandb.log(
                        {
                            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                            "plots/feature_density_line_chart": wandb_histogram,
                        },
                        step=n_steps,
                    )

                act_freq_scores = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
                n_frac_active_tokens = 0

            scheduler.step()
            optimizer.zero_grad()

            ghost_grad_neuron_mask = (
                n_fwd_passes_since_fired > sae.cfg.dead_feature_window
            ).bool()

            # Forward and Backward Passes
            sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sae(
                vit_acts, ghost_grad_neuron_mask
            )
            did_fire = (feature_acts > 0).float().sum(-2) > 0
            n_fwd_passes_since_fired += 1
            n_fwd_passes_since_fired[did_fire] = 0

            n_training_tokens += batch_size

            with torch.no_grad():
                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
                n_frac_active_tokens += batch_size
                feature_sparsity = act_freq_scores / n_frac_active_tokens

                if sae.cfg.log_to_wandb and (
                    (n_steps + 1) % sae.cfg.wandb_log_freq == 0
                ):
                    # metrics for currents acts
                    l0 = (feature_acts > 0).float().sum(-1).mean()
                    current_learning_rate = optimizer.param_groups[0]["lr"]

                    per_token_l2_loss = (
                        (sae_out - vit_acts).pow(2).sum(dim=-1).squeeze()
                    )
                    total_variance = vit_acts.pow(2).sum(-1)
                    explained_variance = 1 - per_token_l2_loss / total_variance

                    metrics = {
                        # losses
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item()
                        / sae.l1_coefficient,  # normalize by l1 coefficient
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
                            feature_sparsity < sae.cfg.dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    }
                    wandb.log(metrics, step=n_steps)

                    directory = f"metrics/{wandb.run.id}"
                    os.makedirs(directory, exist_ok=True)
                    with open(f"{directory}/step{n_steps}.json", "w") as fd:
                        json.dump(metrics, fd)

                    try:
                        with open(f"metrics/reference/step{n_steps}.json") as fd:
                            reference = json.load(fd)
                            print("Check that `reference` and `metrics` are similar.")
                            print("reference:", reference)
                            print("metrics:", metrics)
                    except FileNotFoundError:
                        pass

            loss.backward()
            sae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()

            n_steps += 1

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/{wandb.run.id}/final_{sae.get_name()}.pt"
    sae.save_model(path)

    if cfg.log_to_wandb:
        wandb.finish()

    return sae, vit
