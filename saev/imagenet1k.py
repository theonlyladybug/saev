"""
Runs linear probing on ImageNet-1K using the original model activations as well as the SAE reconstructions and calculates the change as a percentage.
"""

import enum
import logging
import os
import typing

import beartype
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from . import activations, config, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

#########################
# Hyperparameter Sweeps #
#########################


lrs = [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2]


class Layer(enum.StrEnum):
    LAST = "last"
    ALL = "all"


class Aggregation(enum.StrEnum):
    CLASS = "cls"
    MEANPOOL = "meanpool"
    BOTH = "both"


############
# Modeling #
############


@jaxtyped(typechecker=beartype.beartype)
class Linear(torch.nn.Module):
    """Linear classifier that handles the different kinds of data that might be used as input."""

    def __init__(self, agg: Aggregation, layer: Layer, d_vit: int, n_classes: int):
        super().__init__()
        self.agg = agg
        self.layer = layer

        d_in = d_vit
        if self.agg == Aggregation.BOTH:
            d_in = d_vit * 2
        self.linear = torch.nn.Linear(d_in, n_classes)

        # TODO: how to we cleanly init weights in torch.nn.Module
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(
        self, x: Float[Tensor, "batch n_layers all_patches d_vit"]
    ) -> Float[Tensor, "batch n_classes"]:
        if self.agg == Aggregation.CLASS:
            x = x[:, :, 0, :]
        elif self.agg == Aggregation.MEANPOOL:
            x = x[:, :, 1:, :].mean(dim=2)
        elif self.agg == Aggregation.BOTH:
            x_cls = x[:, :, 0, :]
            x_meanpool = x[:, :, 1:, :].mean(axis=2)
            x = torch.cat((x_cls, x_meanpool), axis=-1)
        else:
            typing.assert_never(self.agg)

        if self.layer == Layer.LAST:
            x = x[:, -1, :]
        elif self.layer == Layer.ALL:
            x = x.mean(axis=1)
        else:
            typing.assert_never(self.agg)

        return self.linear(x)


@jaxtyped(typechecker=beartype.beartype)
class LinearDict(torch.nn.Module):
    def __init__(self, dct: torch.nn.ModuleDict):
        super().__init__()
        self.dct = torch.nn.ModuleDict()
        self.dct.update(dct)

    def forward(
        self, x: Float[Tensor, "batch n_layers all_patches d_vit"]
    ) -> dict[str, Float[Tensor, "batch n_classes"]]:
        return {k: v.forward(x) for k, v in self.dct.items()}

    def __len__(self) -> int:
        return len(self.dct)


@beartype.beartype
def make_classifiers(
    d_vit: int, n_classes: int
) -> tuple[LinearDict, list[dict[str, object]]]:
    param_groups = []
    dct = torch.nn.ModuleDict()
    for agg in Aggregation:
        for layer in Layer:
            for lr in lrs:
                linear = Linear(agg, layer, d_vit, n_classes)
                param_groups.append({"params": linear.parameters(), "lr": lr})
                # We can use .0e because there's only one sigfig per learning rate.
                key = f"agg_{agg}-layer_{layer}-lr_{lr:.0e}"
                assert key not in dct
                dct[key] = linear

    return LinearDict(dct), param_groups


############
# Datasets #
############


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """ """

    def __init__(self, root: str, labels: Int[Tensor, " n_imgs"]):
        if not os.path.isdir(root):
            raise RuntimeError(f"Activations are not saved at '{root}'.")

        metadata_fpath = os.path.join(root, "metadata.json")
        self.metadata = activations.Metadata.load(metadata_fpath)
        self.root = root
        self.labels = labels

    @jaxtyped(typechecker=beartype.beartype)
    def __getitem__(
        self, i: int
    ) -> tuple[Float[Tensor, "n_layers all_patches d_vit"], Int[Tensor, "*batch"]]:
        n_imgs_per_shard = (
            self.metadata.n_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.n_patches_per_img + 1)
        )

        shape = (
            n_imgs_per_shard,
            len(self.metadata.layers),
            self.metadata.n_patches_per_img + 1,
            self.metadata.d_vit,
        )

        if isinstance(i, int):
            shard = i // n_imgs_per_shard
            pos = i % n_imgs_per_shard
            acts_fpath = os.path.join(self.root, f"acts{shard:06}.bin")
            acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
            return torch.from_numpy(acts[pos].copy()), self.labels[i]
        else:
            shards = i // n_imgs_per_shard
            pos = i % n_imgs_per_shard
            batch = []
            for shard, p in zip(shards, pos):
                acts_fpath = os.path.join(self.root, f"acts{shard.item():06}.bin")
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
                batch.append(torch.from_numpy(acts[p].copy()))
            return torch.stack(batch), self.labels[i]

    @property
    def d_vit(self) -> int:
        return self.metadata.d_vit

    def __len__(self) -> int:
        return self.metadata.n_imgs


@beartype.beartype
def make_dataloader(
    cfg: config.ImagenetEvaluate, *, is_train: bool
) -> tuple[torch.utils.data.DataLoader, int]:
    """
    Make a new dataloader for the shards.
    """
    shard_root = cfg.train_shard_root if is_train else cfg.val_shard_root
    split = "train" if is_train else "validation"
    labels = torch.tensor(
        datasets.load_dataset("ILSVRC/imagenet-1k", split=split)["label"]
    )

    dataload_cfg = config.DataLoad(shard_root, patches="meanpool", layer=-2)
    acts = activations.Dataset(dataload_cfg)
    dataset = torch.utils.data.StackDataset(acts, labels)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.sgd_batch_size,
        num_workers=cfg.n_workers,
        shuffle=is_train,
    )

    return dataloader, acts.d_vit


############
# Evaluate #
############


@beartype.beartype
def evaluate(cfg: config.ImagenetEvaluate) -> float:
    """
    Fit linear ImageNet-1K classifiers to both the original model activations and the SAE reconstructions.
    """
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logger = logging.getLogger("imagenet1k")
    train_dataloader, dim = make_dataloader(cfg, is_train=True)
    val_dataloader, _ = make_dataloader(cfg, is_train=False)

    train_dataloader = BatchLimiter(train_dataloader, cfg.n_steps)

    classifiers, param_groups = make_classifiers(dim, 1_000)
    classifiers = classifiers.to(cfg.device)
    classifiers = torch.compile(classifiers)

    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0, fused=True)

    # 1. Fit classifier to original model activations.
    for global_step, ((vit_acts, i_img, i_p), labels) in enumerate(
        helpers.progress(train_dataloader, every=cfg.log_every)
    ):
        vit_acts = vit_acts.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        breakpoint()

        outputs = classifiers(vit_acts)

        losses = {k: F.cross_entropy(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        if global_step % cfg.log_every == 0:
            with torch.no_grad():
                best_loss, best_key = min((v.item(), k) for k, v in losses.items())
                best_acc = (outputs[best_key].argmax(axis=1) == labels).float().mean()
                mean_loss = sum(losses.values()).item() / len(losses)
            logger.info(
                "step: %d, mean loss: %.5f, best loss: %.5f, best key: %s, best acc: %.1f%%",
                global_step,
                mean_loss,
                best_loss,
                best_key,
                best_acc * 100,
            )

    # 2. Evaluate classifier on original model activations (for val split)
    for acts, labels in helpers.progress(val_dataloader):
        acts = acts.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)

        outputs = classifiers(acts)

        breakpoint()

        # Probably something like (outputs.argmax(axis=1) == labels).mean()


class BatchLimiter:
    """
    Repeats a dataloader until we return `n_batches` batches.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, n_batches: int):
        self.dataloader = dataloader
        self.n_batches = n_batches

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        self.n_seen = 0
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += 1
                if self.n_seen >= self.n_batches:
                    return
