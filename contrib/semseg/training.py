import collections
import dataclasses
import io
import json
import logging
import os.path

import beartype
import einops
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
from torchvision.transforms import v2

import saev.activations
import saev.config
import saev.training

from . import config

logger = logging.getLogger(__name__)

n_classes = 151


@beartype.beartype
def main(cfgs: list[config.Train]):
    check_cfgs(cfgs)

    cfg = cfgs[0]
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    os.makedirs(cfg.ckpt_path, exist_ok=True)

    train_dataloader = get_dataloader(cfg, is_train=True)
    val_dataloader = get_dataloader(cfg, is_train=False)

    models, params = make_models(cfgs, train_dataloader.dataset.d_vit)
    models = models.to(cfg.device)

    optim = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    global_step = 0

    for epoch in range(cfg.n_epochs):
        models.train()
        for batch in train_dataloader:
            acts = batch["acts"].to(cfg.device)
            patch_labels = batch["patch_labels"].to(cfg.device)
            logits = torch.stack([model(acts) for model in models])
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, n_classes),
                patch_labels.expand(len(cfgs), -1, -1, -1).reshape(-1),
            )
            loss.backward()
            optim.step()
            optim.zero_grad()

            global_step += 1

        # Show last batch's loss and acc.
        accs = (
            logits.argmax(axis=-1) == patch_labels.expand(len(cfgs), -1, -1, -1)
        ).float().mean().item() * 100
        logger.info(
            "epoch: %d, step: %d, (mean) loss: %.5f, (mean) acc: %.3f",
            epoch,
            global_step,
            loss.item(),
            accs,
        )

        if epoch % cfg.eval_every == 0 or epoch + 1 == cfg.n_epochs:
            with torch.inference_mode():
                pred_label_list, true_label_list = [], []
                for batch in val_dataloader:
                    acts_BWHD = batch["acts"].to(cfg.device, non_blocking=True)
                    pixel_labels_BWH = batch["pixel_labels"]
                    true_label_list.append(pixel_labels_BWH)

                    logits_MBWHC = torch.stack([model(acts_BWHD) for model in models])
                    logits_MB_CWH = einops.rearrange(
                        logits_MBWHC,
                        "models batch width height classes -> (models batch) classes width height",
                    )

                    pred_MB_WH = batched_upsample_and_pred(
                        logits_MB_CWH, size=(224, 224), mode="bilinear"
                    )
                    del logits_MB_CWH

                    pred_MBWH = einops.rearrange(
                        pred_MB_WH,
                        "(models batch) width height -> models batch width height",
                        models=len(models),
                    )
                    pred_label_list.append(pred_MBWH)

                pred_labels_MNWH = torch.cat(pred_label_list, dim=1).int()
                true_labels_MNWH = (
                    torch.cat(true_label_list).int().expand(len(models), -1, -1, -1)
                )

                logger.info("Evaluated all validation batchs.")
                mious = get_mean_ious(
                    pred_labels_MNWH,
                    true_labels_MNWH.expand(len(models), -1, -1, -1),
                    n_classes,
                )
                acc_M = (pred_labels_MNWH == true_labels_MNWH).float().mean(
                    dim=(1, 2, 3)
                ) * 100
            logger.info(
                "epoch: %d, step: %d, best miou: %.5f, best acc: %.3f",
                epoch,
                global_step,
                mious.max().item(),
                acc_M.max().item(),
            )

            for cfg, model in zip(cfgs, models):
                dump(cfg, model, suffix=f"ep{epoch}_step{global_step}")


@beartype.beartype
def dump(
    cfg: config.Train,
    model: torch.nn.Module,
    *,
    step: int | None = None,
):
    """
    Save a model checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).
    """
    dpath = os.path.join(
        cfg.ckpt_path,
        f"lr_{cfg.learning_rate}__wd_{cfg.weight_decay}".replace(".", "_"),
    )
    os.makedirs(dpath, exist_ok=True)

    kwargs = dict(in_features=model.in_features, out_features=model.out_features)

    fname = "model"
    if step is not None:
        fname += f"_step{step}"

    fpath = os.path.join(dpath, f"{fname}.pt")
    with open(fpath, "wb") as fd:
        kwargs_str = json.dumps(kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(model.state_dict(), fd)

    fpath = os.path.join(dpath, "cfg.json")
    with open(fpath, "w") as fd:
        json.dump(dataclasses.asdict(cfg), fd)


@beartype.beartype
def load_latest(dpath: str, *, device: str = "cpu") -> torch.nn.Module:
    """
    Loads the latest checkpoint by picking out the checkpoint file in dpath with the largest _step{step} suffix.

    Arguments:
        dpath: Directory to search.
        device: optional torch device to pass to load.
    """


@beartype.beartype
def load(fpath: str, *, device: str = "cpu") -> torch.nn.Module:
    """
    Loads a sparse autoencoder from disk.
    """
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = torch.nn.Linear(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model


CANNOT_PARALLELIZE = set([
    "n_epochs",
    "batch_size",
    "n_workers",
    "train_acts",
    "val_acts",
    "imgs",
    "eval_every",
    "device",
])


@beartype.beartype
def check_cfgs(cfgs: list[config.Train]):
    # Check that any differences in configs are supported by our parallel training run.
    seen = collections.defaultdict(list)
    for cfg in cfgs:
        for key, value in vars(cfg).items():
            seen[key].append(value)

    bad_keys = {}
    for key, values in seen.items():
        if key in CANNOT_PARALLELIZE and len(set(values)) != 1:
            bad_keys[key] = values

    if bad_keys:
        msg = ", ".join(f"'{key}': {values}" for key, values in bad_keys.items())
        raise ValueError(f"Cannot parallelize training over: {msg}")


@beartype.beartype
def make_models(
    cfgs: list[config.Train], d_vit: int
) -> tuple[torch.nn.ModuleList, list[dict[str, object]]]:
    param_groups = []
    models = []
    for cfg in cfgs:
        model = torch.nn.Linear(d_vit, n_classes)
        models.append(model)
        # Use an empty LR because our first step is warmup.
        param_groups.append({
            "params": model.parameters(),
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        })

    return torch.nn.ModuleList(models), param_groups


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


@jaxtyped(typechecker=beartype.beartype)
@torch.no_grad()
def get_mean_ious(
    y_pred: Int[Tensor, "models batch width height"],
    y_true: Int[Tensor, "models batch width height"],
    n_classes: int,
    ignore_class: int | None = 0,
) -> Float[Tensor, " models"]:
    """
    Calculate mean IoU for predicted masks.

    Arguments:
        y_pred:
        y_true:
        n_classes: Number of classes.

    Returns:
        Mean IoU as a float tensor.
    """

    mious = []
    for y_pred_, y_true_ in zip(y_pred, y_true):
        # Convert to one-hot encoded format
        pred_one_hot = torch.nn.functional.one_hot(y_pred_.long(), n_classes)
        true_one_hot = torch.nn.functional.one_hot(y_true_.long(), n_classes)

        if ignore_class is not None:
            if ignore_class == 0:
                pred_one_hot = pred_one_hot[..., 1:]
                true_one_hot = true_one_hot[..., 1:]
            else:
                pred_one_hot = torch.cat(
                    (
                        pred_one_hot[..., :ignore_class],
                        pred_one_hot[..., ignore_class + 1 :],
                    ),
                    axis=-1,
                )
                true_one_hot = torch.cat(
                    (
                        true_one_hot[..., :ignore_class],
                        true_one_hot[..., ignore_class + 1 :],
                    ),
                    axis=-1,
                )
        logger.info(
            "Got one-hot encodings for inputs (ignore_class='%s').", ignore_class
        )

        # Calculate intersection and union for all classes at once
        intersection = einops.reduce(
            torch.logical_and(pred_one_hot, true_one_hot), "n w h c -> c", "sum"
        )
        union = einops.reduce(
            torch.logical_or(pred_one_hot, true_one_hot), "n w h c -> c", "sum"
        )
        logger.info("Got intersection and union.")

        if (union == 0).any():
            logger.warning(
                "At least one class is not present in data: '%s'.",
                torch.nonzero(union == 0),
            )

        # Handle division by zero
        miou = (intersection / union).mean().item()
        mious.append(miou)

    return torch.tensor(mious)


@jaxtyped(typechecker=beartype.beartype)
def batched_upsample_and_pred(
    tensor: Float[Tensor, "n channels width height"],
    *,
    size: tuple[int, int],
    mode: str,
    batch_size: int = 128,
) -> Int[Tensor, "n {size[0]} {size[1]}"]:
    preds = []

    for start, end in batched_idx(len(tensor), batch_size):
        upsampled_BCWH = torch.nn.functional.interpolate(
            tensor[start:end].contiguous(), size=size, mode=mode
        )
        pred_BWH = upsampled_BCWH.argmax(axis=1).cpu()
        del upsampled_BCWH
        preds.append(pred_BWH)

    return torch.cat(preds)


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
