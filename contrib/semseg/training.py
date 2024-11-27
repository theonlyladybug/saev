import dataclasses
import json
import logging
import os.path

import beartype
import einops
import torch
from jaxtyping import Int, jaxtyped
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.config

from . import config

logger = logging.getLogger(__name__)

n_classes = 151


@beartype.beartype
def main(cfg: config.Train):
    train_dataloader = get_dataloader(cfg, is_train=True)
    val_dataloader = get_dataloader(cfg, is_train=False)

    model = torch.nn.Linear(train_dataloader.dataset.d_vit, n_classes)
    model = model.to(cfg.device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    global_step = 0

    for epoch in range(cfg.n_epochs):
        model.train()
        for batch in train_dataloader:
            acts = batch["acts"].to(cfg.device)
            patch_labels = batch["patch_labels"].to(cfg.device)
            logits = model(acts)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, n_classes), patch_labels.view(-1)
            )
            loss.backward()
            optim.step()
            optim.zero_grad()

            global_step += 1

        # Show last batch's loss and acc.
        acc = (logits.argmax(axis=-1) == patch_labels).float().mean().item() * 100
        logger.info(
            "epoch: %d, step: %d, loss: %.5f, acc: %.3f",
            epoch,
            global_step,
            loss.item(),
            acc,
        )

        if epoch % cfg.eval_every or epoch + 1 == cfg.n_epochs:
            with torch.inference_mode():
                pred_label_list, true_label_list = [], []
                for batch in val_dataloader:
                    acts_BWHD = batch["acts"].to(cfg.device, non_blocking=True)
                    pixel_labels_BWH = batch["pixel_labels"]
                    true_label_list.append(pixel_labels_BWH)

                    logits_BWHC = model(acts_BWHD)
                    logits_BCWH = einops.rearrange(
                        logits_BWHC,
                        "batch width height classes -> batch classes width height",
                    )
                    upsampled_BCWH = torch.nn.functional.interpolate(
                        logits_BCWH.contiguous(), size=(224, 224), mode="bilinear"
                    )
                    pred_BWH = upsampled_BCWH.argmax(axis=1).cpu()
                    del upsampled_BCWH
                    pred_label_list.append(pred_BWH)

                pred_labels_NWH = torch.cat(pred_label_list).int()
                true_labels_NWH = torch.cat(true_label_list).int()

                logger.info("Evaluated all validation batchs.")
                miou = get_mean_iou(pred_labels_NWH, true_labels_NWH, n_classes)
                acc = (pred_labels_NWH == true_labels_NWH).float().mean() * 100
                logger.info(
                    "epoch: %d, step: %d, miou: %.5f, acc: %.3f",
                    epoch,
                    global_step,
                    miou,
                    acc,
                )

                ckpt_fpath = os.path.join(
                    cfg.ckpt_path, f"model_ep{epoch}_step{global_step}.pt"
                )
                torch.save(model.state_dict(), ckpt_fpath)
                logger.info("Saved checkpoint to '%s'.", ckpt_fpath)
                cfg_fpath = os.path.join(cfg.ckpt_path, "config.json")
                with open(cfg_fpath, "w") as fd:
                    json.dump(dataclasses.asdict(cfg), fd, indent=4)
                logger.info("Saved config to '%s'.", cfg_fpath)


def get_dataloader(cfg: config.Train, *, is_train: bool):
    if is_train:
        shuffle = True
        dataset = Dataset(
            cfg.train_acts, dataclasses.replace(cfg.imgs, split="training")
        )
    else:
        shuffle = False
        dataset = Dataset(
            cfg.val_acts, dataclasses.replace(cfg.imgs, split="validation")
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        shuffle=shuffle,
        persistent_workers=(cfg.n_workers > 0),
    )


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, acts_cfg: saev.config.DataLoad, imgs_cfg: saev.config.Ade20kDataset
    ):
        self.acts = saev.activations.Dataset(acts_cfg)
        to_array = v2.Compose([
            v2.Resize(256, interpolation=v2.InterpolationMode.NEAREST),
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
        ])
        self.imgs = saev.activations.Ade20k(
            imgs_cfg, img_transform=to_array, seg_transform=to_array
        )

        assert len(self.imgs) * 196 == len(self.acts)

    @property
    def d_vit(self) -> int:
        return self.acts.metadata.d_vit

    def __getitem__(self, i: int) -> dict[str, object]:
        # Get activations
        acts = []
        for j, k in enumerate(range(i * 196, (i + 1) * 196)):
            act = self.acts[k]
            assert act.patch_i.item() == j
            assert act.img_i.item() == i
            acts.append(act.vit_acts)
        acts = torch.stack(acts).reshape((14, 14, self.d_vit))

        # Get patch and pixel level semantic labels.
        img = self.imgs[i]
        pixel_labels = img["segmentation"].squeeze()

        patch_labels = (
            einops.rearrange(pixel_labels, "(w pw) (h ph) -> w h (pw ph)", pw=16, ph=16)
            .mode(axis=-1)
            .values
        )

        return {
            "index": i,
            "acts": acts,
            "pixel_labels": pixel_labels,
            "patch_labels": patch_labels,
        }

    def __len__(self) -> int:
        return len(self.imgs)


@torch.no_grad()
@jaxtyped(typechecker=beartype.beartype)
def get_mean_iou(
    y_pred: Int[Tensor, "batch width height"],
    y_true: Int[Tensor, "batch width height"],
    n_classes: int,
    ignore_class: int | None = 0,
) -> float:
    """
    Calculate mean IoU for predicted masks.

    Arguments:
        y_pred:
        y_true:
        n_classes: Number of classes.

    Returns:
        Mean IoU as a float.
    """

    # Convert to one-hot encoded format
    pred_one_hot = torch.nn.functional.one_hot(y_pred.long(), n_classes)
    true_one_hot = torch.nn.functional.one_hot(y_true.long(), n_classes)

    if ignore_class is not None:
        pred_one_hot = torch.cat(
            (pred_one_hot[..., :ignore_class], pred_one_hot[..., ignore_class + 1 :]),
            axis=-1,
        )
        true_one_hot = torch.cat(
            (true_one_hot[..., :ignore_class], true_one_hot[..., ignore_class + 1 :]),
            axis=-1,
        )
    logger.info("Got one-hot encodings for inputs (ignore='%s').", ignore_class)

    # Calculate intersection and union for all classes at once
    # Sum over height and width dimensions
    intersection = torch.logical_and(pred_one_hot, true_one_hot).sum(axis=(0, 1))
    union = torch.logical_or(pred_one_hot, true_one_hot).sum(axis=(0, 1))
    logger.info("Got intersection and union.")

    breakpoint()

    # Handle division by zero
    return (intersection / union).mean().item()
