import logging
import typing

import beartype
import einops
import torch
import tyro

import saev.activations
import saev.nn

from . import config

logger = logging.getLogger("contrib.semprobe")


@beartype.beartype
@torch.inference_mode
def score(cfg: typing.Annotated[config.Score, tyro.conf.arg(name="")]):
    sae = saev.nn.load(cfg.sae_ckpt)
    sae = sae.to(cfg.device)
    logger.info("Loaded SAE.")

    acts_dataset = saev.activations.Dataset(cfg.acts)
    imgs_dataset = saev.activations.ImageFolder(cfg.imgs.root)

    assert len(acts_dataset) // acts_dataset.metadata.n_patches_per_img == len(
        imgs_dataset
    )

    batch_size = (
        cfg.batch_size
        // acts_dataset.metadata.n_patches_per_img
        * acts_dataset.metadata.n_patches_per_img
    )
    n_imgs_per_batch = batch_size // acts_dataset.metadata.n_patches_per_img

    dataloader = torch.utils.data.DataLoader(
        acts_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
        drop_last=False,
    )
    logger.info("Loaded data.")

    pred_labels_SN = torch.zeros((sae.cfg.d_sae, len(imgs_dataset)))
    true_labels_N = torch.zeros((len(imgs_dataset)))
    task_N = torch.zeros((len(imgs_dataset)))

    for batch in dataloader:
        vit_acts_BD = batch["act"].to(cfg.device)
        i_im = torch.sort(torch.unique(batch["image_i"])).values
        assert len(i_im) == n_imgs_per_batch
        _, sae_acts_BS, _ = sae(vit_acts_BD)
        sae_acts_SB = einops.rearrange(sae_acts_BS, "batch d_sae -> d_sae batch")
        sae_acts_SIP = sae_acts_SB.view(
            sae.cfg.d_sae, len(i_im), acts_dataset.metadata.n_patches_per_img
        )
        sae_acts_SI = einops.reduce(
            sae_acts_SIP, "d_sae n_img n_patch -> d_sae n_img", "sum"
        )

        # Predictions for each latent is 1 for sae_acts_SI[latent] > threshold, 0 otherwise.
        for j, i in enumerate(i_im.tolist()):
            true_label = imgs_dataset[i]["label"]
            # TODO: actually write to task_N.
            if "teeth" not in true_label:
                continue

            true_labels_N[i] = 1.0 if "positive" in true_label else 0.0

            pred_labels_SN[:, i] = (sae_acts_SI[:, j] > cfg.threshold).cpu().float()

    logger.info("Made %d predictions.", len(imgs_dataset))
    true_pos_S = einops.reduce(
        (pred_labels_SN == true_labels_N) & (true_labels_N == 1),
        "d_sae n_image -> d_sae",
        "sum",
    )
    false_pos_S = einops.reduce(
        (pred_labels_SN != true_labels_N) & (true_labels_N == 0),
        "d_sae n_image -> d_sae",
        "sum",
    )
    false_neg_S = einops.reduce(
        (pred_labels_SN != true_labels_N) & (true_labels_N == 1),
        "d_sae n_image -> d_sae",
        "sum",
    )
    # TODO: Calculate task-specific f1_S using task_N.
    f1_S = (2 * true_pos_S) / (2 * true_pos_S + false_pos_S + false_neg_S)
    breakpoint()
    print(f1_S, task_N)
    # TODO:
    # 1. Pick out the top K features.
    # 2. Save visuals for these example images.
    # 3. Print command to save the topk images from the original training set using `saev visuals`.


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "score": score,
        "noop": lambda: None,
    })
