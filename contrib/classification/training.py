"""
Train a linear probe on mean patch activations from a ViT.
"""

import collections
import csv
import dataclasses
import io
import json
import logging
import os

import altair as alt
import beartype
import polars as pl
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev.activations

from . import config

logger = logging.getLogger(__name__)


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

    x_train_ND = load_acts(cfg.train_acts).to(cfg.device)
    y_train_N = load_targets(cfg.train_imgs).to(cfg.device)
    x_val_ND = load_acts(cfg.val_acts).to(cfg.device)
    y_val_N = load_targets(cfg.val_imgs).to(cfg.device)

    _, d_vit = x_train_ND.shape
    n_classes = len(set(y_train_N.tolist()))

    models, params = make_models(cfgs, d_vit, n_classes)
    models = models.to(cfg.device)
    optim = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    for step in range(cfg.n_steps):
        models.train()
        logits_MNC = torch.stack([model(x_train_ND) for model in models])
        loss = torch.nn.functional.cross_entropy(
            logits_MNC.view(-1, n_classes),
            y_train_N.expand(len(cfgs), -1).reshape(-1),
        )
        loss.backward()
        optim.step()
        optim.zero_grad()

        train_accs_M = (
            (logits_MNC.argmax(axis=-1) == y_train_N.expand(len(cfgs), -1))
            .float()
            .mean(axis=1)
        )

        models.eval()
        logits_MNC = torch.stack([model(x_val_ND) for model in models])
        val_accs_M = (
            (logits_MNC.argmax(axis=-1) == y_val_N.expand(len(cfgs), -1))
            .float()
            .mean(axis=1)
        )

        logger.info(
            "step: %d, mean train loss: %.5f, max train acc: %.3f%%, max val acc: %.3f%%",
            step,
            loss.item(),
            train_accs_M.max().item() * 100,
            val_accs_M.max().item() * 100,
        )

    for cfg, model in zip(cfgs, models):
        dump_model(cfg, model)

    os.makedirs(cfg.log_to, exist_ok=True)

    # Save CSV file
    class_headers = load_class_headers(cfg.train_imgs)
    csv_fpath = os.path.join(cfg.log_to, "results.csv")
    with open(csv_fpath, "w") as fd:
        writer = csv.writer(fd)
        writer.writerow(
            ["learning_rate", "weight_decay", "train_acc", "val_acc"] + class_headers
        )
        for c, train_acc, val_acc in zip(
            cfgs, train_accs_M.tolist(), val_accs_M.tolist()
        ):
            writer.writerow([c.learning_rate, c.weight_decay, train_acc, val_acc])

    # Save hyperparameter sweep charts
    df = pl.read_csv(csv_fpath).with_columns(
        pl.col("weight_decay").add(1e-9).alias("weight_decay")
    )
    alt.Chart(df).mark_point().encode(
        alt.X(alt.repeat("column"), type="quantitative").scale(type="log"),
        alt.Y(alt.repeat("row"), type="quantitative").scale(zero=False),
    ).repeat(
        row=["train_acc", "val_acc"], column=["learning_rate", "weight_decay"]
    ).save(os.path.join(cfg.log_to, "hparam-sweeps.png"))


@beartype.beartype
def dump_model(cfg: config.Train, model: torch.nn.Module):
    """
    Save a model checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).
    """
    dpath = os.path.join(
        cfg.ckpt_path,
        f"lr_{cfg.learning_rate}__wd_{cfg.weight_decay}".replace(".", "_"),
    )
    os.makedirs(dpath, exist_ok=True)

    kwargs = dict(in_features=model.in_features, out_features=model.out_features)

    fpath = os.path.join(dpath, "model.pt")
    with open(fpath, "wb") as fd:
        kwargs_str = json.dumps(kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(model.state_dict(), fd)

    fpath = os.path.join(dpath, "cfg.json")
    with open(fpath, "w") as fd:
        json.dump(dataclasses.asdict(cfg), fd)


@beartype.beartype
def load_model(fpath: str, *, device: str = "cpu") -> torch.nn.Module:
    """
    Loads a linear layer from disk.
    """
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    model = torch.nn.Linear(**kwargs)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


@jaxtyped(typechecker=beartype.beartype)
def load_acts(cfg: saev.config.DataLoad) -> Float[Tensor, "n d_vit"]:
    dataset = saev.activations.Dataset(cfg)
    acts = torch.stack([dataset[i]["act"] for i in range(len(dataset))])
    return acts


@jaxtyped(typechecker=beartype.beartype)
def load_targets(cfg: saev.config.ImageFolderDataset) -> Int[Tensor, " n"]:
    dataset = saev.activations.ImageFolder(cfg.root)
    targets = torch.tensor([tgt for sample, tgt in dataset.samples])
    return targets


@jaxtyped(typechecker=beartype.beartype)
def load_class_headers(cfg: saev.config.ImageFolderDataset) -> list[str]:
    dataset = saev.activations.ImageFolder(cfg.root)
    unique_targets = sorted(set([tgt for sample, tgt in dataset.samples]))
    labels = [dataset.classes[tgt] for tgt in unique_targets]
    return labels


@beartype.beartype
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, acts_cfg: saev.config.DataLoad, imgs_cfg: saev.config.ImageFolderDataset
    ):
        self.acts = saev.activations.Dataset(acts_cfg)

        img_dataset = saev.activations.ImageFolder(imgs_cfg.root)
        self.targets = [tgt for sample, tgt in img_dataset.samples]
        self.labels = [img_dataset.classes[tgt] for tgt in self.targets]

    @property
    def d_vit(self) -> int:
        return self.acts.metadata.d_vit

    @property
    def n_classes(self) -> int:
        return len(set(self.targets))

    def __getitem__(self, i: int) -> dict[str, object]:
        act_D = self.acts[i]["act"]
        label = self.labels[i]
        target = self.targets[i]

        return {"index": i, "acts": act_D, "labels": label, "targets": target}

    def __len__(self) -> int:
        assert len(self.acts) == len(self.targets)
        return len(self.targets)


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
    cfgs: list[config.Train], d_in: int, d_out: int
) -> tuple[torch.nn.ModuleList, list[dict[str, object]]]:
    param_groups = []
    models = []
    for cfg in cfgs:
        model = torch.nn.Linear(d_in, d_out)
        models.append(model)
        # Use an empty LR because our first step is warmup.
        param_groups.append({
            "params": model.parameters(),
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        })

    return torch.nn.ModuleList(models), param_groups
