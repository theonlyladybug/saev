import abc
import concurrent.futures
import csv
import dataclasses
import logging
import os.path
import random
import threading
import typing

import beartype
import numpy as np
import torch
from jaxtyping import Float, Int, Shaped, jaxtyped
from PIL import Image
from torch import Tensor

from . import activations, config, helpers, imaging, nn

logger = logging.getLogger("broden")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Record:
    image: str
    is_train: bool
    height: int
    width: int
    segment_height: int
    segment_width: int
    colors: list[str] = dataclasses.field(default_factory=list)
    objects: list[str] = dataclasses.field(default_factory=list)
    parts: list[str] = dataclasses.field(default_factory=list)
    materials: list[str] = dataclasses.field(default_factory=list)
    textures: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert self.segment_width * 2 == self.width
        assert self.segment_height * 2 == self.height

    @classmethod
    def from_row_dict(cls, row: dict[str, str]) -> "Record":
        return cls(
            row["image"],
            row["split"] == "train",
            int(row["ih"]),
            int(row["iw"]),
            int(row["sh"]),
            int(row["sw"]),
            row["color"].split(";") if row["color"] else [],
            row["object"].split(";") if row["object"] else [],
            row["part"].split(";") if row["part"] else [],
            row["material"].split(";") if row["material"] else [],
            [int(i) for i in row["texture"].split(";")] if row["texture"] else [],
        )

    @property
    def dataset(self) -> str:
        dataset, *_ = self.image.split("/")
        return dataset


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Category(abc.ABC):
    """
    Represents something like a material, texture, etc.
    """

    code: int
    number: int
    name: str
    frequency: int

    @classmethod
    def from_row_dict(cls, row: dict[str, str]) -> typing.Self:
        return cls(
            int(row["code"]), int(row["number"]), row["name"], int(row["frequency"])
        )

    @property
    @abc.abstractmethod
    def category(self) -> str: ...


T = typing.TypeVar("T", bound=Category)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Texture(Category):
    category: str = "texture"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Material(Category):
    category: str = "material"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Color(Category):
    category: str = "color"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Object(Category):
    category: str = "object"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Part(Category):
    category: str = "part"


@beartype.beartype
class TextureDataset(torch.utils.data.Dataset):
    cfg: config.BrodenEvaluate
    texture: Texture

    def __init__(self, cfg: config.BrodenEvaluate, texture: Texture, *, is_train: bool):
        self.cfg = cfg
        self.texture = texture
        self.is_train = is_train
        self.vit_acts = activations.Dataset(cfg.data)

        # Load samples
        pos_samples = []
        neg_samples = []

        with open(os.path.join(cfg.root, "index.csv")) as fd:
            for i, row in enumerate(csv.DictReader(fd)):
                if not row["texture"]:
                    continue

                if (is_train and row["split"] == "val") or (
                    not is_train and row["split"] == "train"
                ):
                    continue

                texture_numbers = [int(t) for t in row["texture"].split(";")]

                if self.texture.number == texture_numbers[0]:
                    pos_samples.append((i, 1))
                elif self.texture.number not in texture_numbers:
                    neg_samples.append((i, 0))
                else:
                    # Texture is present, but not primary focus.
                    continue

        random.seed(cfg.seed)
        random.shuffle(neg_samples)
        assert len(neg_samples) >= len(pos_samples)
        neg_samples = neg_samples[: len(pos_samples)]
        self.samples = pos_samples + neg_samples

    def __getitem__(self, i):
        orig_i, label = self.samples[i]
        start = orig_i * self.vit_acts.metadata.n_patches_per_img
        end = start + self.vit_acts.metadata.n_patches_per_img
        acts, i_im, _ = zip(*[self.vit_acts[i] for i in range(start, end)])
        i_im = torch.unique(torch.tensor(i_im))
        assert len(i_im) == 1
        return {
            "activation": torch.stack(acts),
            "label": label,
            "index": i,
            "index_im": i_im.item(),
        }

    def __len__(self) -> int:
        return len(self.samples)


@beartype.beartype
def get_texture_datasets(
    cfg: config.BrodenEvaluate,
) -> list[tuple[Texture, TextureDataset, TextureDataset]]:
    datasets = []
    with open(os.path.join(cfg.root, "c_texture.csv")) as fd:
        for row in csv.DictReader(fd):
            try:
                texture = Texture.from_row_dict(row)
            except Exception as err:
                breakpoint()
                print(err)
            datasets.append((
                texture,
                TextureDataset(cfg, texture, is_train=True),
                TextureDataset(cfg, texture, is_train=False),
            ))

    return datasets


class DummyExecutor(concurrent.futures.Executor):
    def __init__(self):
        self._shutdown = False
        self._shutdownLock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = concurrent.futures.Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Stats:
    p_value: float

    def significant(self) -> bool:
        return self.p_value < 0.025


@jaxtyped(typechecker=beartype.beartype)
def check(
    scores: Float[np.ndarray, " d_sae"],
    *,
    latent: int,
    n_bootstraps: int = 500,
) -> Stats:
    rng = np.random.default_rng(seed=0)

    # If the max feature score is not statistically significant, then we would expect that the difference between feature_scores.max() and feature_scores.mean() is unlike any other random sample.
    observed_diff = scores[latent] - scores.mean()

    bootstrap_diffs = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        bootstrap_sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_choice = rng.choice(bootstrap_sample)
        bootstrap_diffs[i] = bootstrap_choice - bootstrap_sample.mean()

    p_value = np.mean(bootstrap_diffs >= observed_diff)

    return Stats(p_value)


@jaxtyped(typechecker=beartype.beartype)
def plot(scores: Float[np.ndarray, " d_sae"], *, latent: int) -> tuple[object, object]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(scores, bins=200)
    ax.set_yscale("log")
    ax.set_ylabel("Feature Count")
    ax.set_xlabel("Mean difference between positive and negative examples")

    ax.axvline(scores[latent], color="red")

    return fig, ax


@jaxtyped(typechecker=beartype.beartype)
def get_scores(
    sae_acts: Float[Tensor, "n d_sae"], labels: Int[Tensor, " n"]
) -> tuple[Float[np.ndarray, " d_sae"], int]:
    mean_pos_score = sae_acts[labels == 1].mean(axis=0)
    mean_neg_score = sae_acts[labels == 0].mean(axis=0)
    scores = (mean_pos_score - mean_neg_score).cpu().numpy()
    return scores, scores.argmax().item()


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Report:
    name: str
    """Broden sub-task name."""
    score: float
    """Score, out of 1."""
    splits: dict[str, float]
    """Other splits."""


@beartype.beartype
@torch.inference_mode()
def evaluate_textures(cfg: config.BrodenEvaluate) -> Report:
    """
    Try to find a sparse autoencoder latent that accurately detects the presence/absence of a particular texture.

    Textures are easy because they have one or more integer labels.
    """
    sae = nn.load(cfg.ckpt).to(cfg.device)
    sae.eval()

    successes, failures, splits = [], [], {}
    for texture, train_dataset, val_dataset in helpers.progress(
        get_texture_datasets(cfg), every=1
    ):
        max_train_sae_acts, train_labels = [], []
        for example in train_dataset:
            _, f_x, *_ = sae(example["activation"].to(cfg.device))
            max_train_sae_acts.append(f_x.max(axis=0).values)
            train_labels.append(example["label"])

        max_train_sae_acts = torch.stack(max_train_sae_acts)
        train_labels = torch.tensor(train_labels)

        max_val_sae_acts, val_labels = [], []
        for example in val_dataset:
            _, f_x, *_ = sae(example["activation"].to(cfg.device))
            max_val_sae_acts.append(f_x.max(axis=0).values)
            val_labels.append(example["label"])

        max_val_sae_acts = torch.stack(max_val_sae_acts)
        val_labels = torch.tensor(val_labels)

        train_scores, train_latent = get_scores(max_train_sae_acts, train_labels)
        train_stats = check(train_scores, latent=train_latent)
        val_scores, val_latent = get_scores(max_val_sae_acts, val_labels)
        val_stats = check(val_scores, latent=train_latent)

        # Plot and save histogram.
        fig, ax = plot(val_scores, latent=train_latent)
        ax.set_title(f"{texture.name.capitalize()} Validation Scores")
        fig_fpath = os.path.join(
            get_chart_dir(cfg, texture), f"{texture.name}-val-hist.png"
        )
        fig.savefig(fig_fpath)
        logger.info("Saved histogram to '%s'.", fig_fpath)

        if (
            train_stats.significant
            and val_stats.significant
            and train_latent == val_latent
        ):
            successes.append(texture)
            splits[texture.name] = 1.0
        else:
            failures.append(texture)
            splits[texture.name] = 0.0

    return Report("textures", len(successes) / (len(successes) + len(failures)), splits)


@jaxtyped(typechecker=beartype.beartype)
def double(x: Shaped[np.ndarray, "w h"]) -> Shaped[np.ndarray, "w*2 h*2"]:
    w, h = x.shape
    return np.repeat(np.repeat(x, np.full((w,), 2), axis=0), np.full((h,), 2), axis=1)


@jaxtyped(typechecker=beartype.beartype)
def make_patch_lookup(
    *,
    patch_size_px: tuple[int, int],
    im_size_px: tuple[int, int] = (224, 224),
) -> Int[np.ndarray, "w_px h_px"]:
    im_w_px, im_h_px = im_size_px
    p_w_px, p_h_px = patch_size_px

    xv, yv = np.meshgrid(np.arange(im_w_px), np.arange(im_h_px))

    patch_lookup = (xv // p_w_px) + (yv // p_h_px) * (im_h_px // p_h_px)
    return patch_lookup


@jaxtyped(typechecker=beartype.beartype)
def get_patches(
    cfg: config.BrodenEvaluate,
    pixel_file: str,
    number: int,
    patch_lookup: Int[np.ndarray, "w_px h_px"],
    *,
    threshold: float = 0.5,
) -> Int[np.ndarray, " n"]:
    """
    Gets a list of patches that contain more than `threshold` fraction pixels containing the category with number `number`.
    """
    img = Image.open(os.path.join(cfg.root, "images", pixel_file))
    raw = np.array(img).astype(np.uint32)
    # 256 is hardcoded as a constant in the broden dataset because image files use 8 bits per color channel.
    nums = raw[:, :, 1] * 256 + raw[:, :, 0]
    nums = double(nums)

    x, y = np.where(nums == number)
    patches, counts = np.unique(patch_lookup[x, y], return_counts=True)

    n_pixels = cfg.patch_size[0] * cfg.patch_size[1] * threshold
    return patches[counts > n_pixels]


@beartype.beartype
class PixelDataset(torch.utils.data.Dataset, typing.Generic[T]):
    """
    Gets a list
    """

    def __init__(
        self,
        cfg: config.BrodenEvaluate,
        category: T,
        others: frozenset[T],
        *,
        is_train: bool,
    ):
        err_msg = "Doesn't make sense to use non-patch activations with a pixel-level dataset."
        assert cfg.data.patches == "patches", err_msg

        assert category not in others
        for other in others:
            assert category.category == other.category

        self.cfg = cfg
        self.category = category

        self.vit_acts = activations.Dataset(cfg.data)

        with open(os.path.join(cfg.root, "index.csv")) as fd:
            records = [
                (i_im, Record.from_row_dict(row))
                for i_im, row in enumerate(csv.DictReader(fd))
            ]

        # Filter examples from wrong split.
        records = [
            (i_im, record) for i_im, record in records if is_train == record.is_train
        ]

        # Shuffle the remaining records.
        random.seed(cfg.seed)
        random.shuffle(records)

        split_frac = 0.7 if is_train else 0.3
        max_samples_per_label = min(
            cfg.sample_range[1], int(category.frequency * split_frac)
        )

        patch_lookup = make_patch_lookup(patch_size_px=cfg.patch_size)

        pos_samples, neg_samples = [], []
        for i_im, record in records:
            for pixel_file in getattr(record, category.category + "s"):
                # Positive patches
                patches = get_patches(
                    self.cfg, pixel_file, category.number, patch_lookup
                )
                if patches.size:
                    pos_samples.append(
                        i_im * self.vit_acts.metadata.n_patches_per_img + patches
                    )

                # Negative patches
                for other in others:
                    patches = get_patches(
                        self.cfg, pixel_file, other.number, patch_lookup
                    )
                    if patches.size:
                        neg_samples.append(
                            i_im * self.vit_acts.metadata.n_patches_per_img + patches
                        )

            n_positive = sum(p.size for p in pos_samples)
            n_negative = sum(p.size for p in neg_samples)

            if (
                n_positive >= max_samples_per_label
                and n_negative >= max_samples_per_label
            ):
                break

        pos_samples = np.concatenate(pos_samples)
        neg_samples = np.concatenate(neg_samples)

        rng = np.random.default_rng(seed=cfg.seed)
        rng.shuffle(pos_samples)
        rng.shuffle(neg_samples)

        self.indices = np.concatenate((
            pos_samples[:max_samples_per_label],
            neg_samples[:max_samples_per_label],
        ))
        self.labels = np.concatenate((
            np.ones(max_samples_per_label, dtype=int),
            np.zeros(max_samples_per_label, dtype=int),
        ))

    def __getitem__(self, i: int) -> dict[str, object]:
        act_i = self.indices[i].item()
        label = self.labels[i].item()
        vit_act, i_im, i_p = self.vit_acts[act_i]
        return {
            "activation": vit_act,
            "i_im": i_im,
            "i_p": i_p,
            "label": torch.tensor(label),
            "index": torch.tensor(i),
        }

    def __len__(self) -> int:
        return len(self.labels)


@jaxtyped(typechecker=beartype.beartype)
@torch.inference_mode()
def get_sae_acts(
    cfg: config.BrodenEvaluate,
    sae: nn.SparseAutoencoder,
    dataset: PixelDataset[T],
) -> tuple[Float[Tensor, "n d_sae"], Int[Tensor, " n"]]:
    sae_acts, labels = [], []
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_workers,
        pin_memory=True,
    )
    for batch in dataloader:
        vit_acts = batch["activation"].to(cfg.device, non_blocking=True)
        _, f_x, *_ = sae(vit_acts)

        sae_acts.append(f_x)
        labels.append(batch["label"])

    return torch.cat(sae_acts), torch.cat(labels)


@beartype.beartype
def get_chart_dir(cfg: config.BrodenEvaluate, category: T) -> str:
    ckpt_name = os.path.basename(os.path.dirname(cfg.ckpt))
    chart_dir = os.path.join(cfg.dump_to, ckpt_name, category.category + "s", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    return chart_dir


@beartype.beartype
def get_im_dir(cfg: config.BrodenEvaluate, category: T) -> str:
    ckpt_name = os.path.basename(os.path.dirname(cfg.ckpt))
    im_dir = os.path.join(
        cfg.dump_to, ckpt_name, category.category + "s", "images", category.name
    )
    os.makedirs(im_dir, exist_ok=True)
    return im_dir


@beartype.beartype
@torch.inference_mode()
def evaluate_category(
    cfg: config.BrodenEvaluate, category: T, others: frozenset[T]
) -> tuple[str, float]:
    logger.info("Evaluating %s '%s'.", category.category, category.name)

    assert category not in others

    sae = nn.load(cfg.ckpt).to(cfg.device)
    sae.eval()

    train_dataset = PixelDataset(cfg, category, others, is_train=True)
    val_dataset = PixelDataset(cfg, category, others, is_train=False)
    logger.info("Loaded datasets for %s '%s'.", category.category, category.name)

    train_acts, train_labels = get_sae_acts(cfg, sae, train_dataset)
    val_acts, val_labels = get_sae_acts(cfg, sae, val_dataset)

    train_scores, train_latent = get_scores(train_acts, train_labels)
    train_stats = check(train_scores, latent=train_latent)
    val_scores, val_latent = get_scores(val_acts, val_labels)
    val_stats = check(val_scores, latent=train_latent)

    # Plot and save histogram.
    fig, ax = plot(val_scores, latent=train_latent)
    ax.set_title(f"{category.name.capitalize()} Validation Scores")
    fig_fpath = os.path.join(
        get_chart_dir(cfg, category), f"{category.name}-val-hist.png"
    )
    fig.savefig(fig_fpath)
    logger.info("Saved histogram to '%s'.", fig_fpath)

    if train_stats.significant and val_stats.significant and train_latent == val_latent:
        score = 1.0
    else:
        score = 0.0

    torch.cuda.empty_cache()
    logger.info("Identified best latent for %s '%s'.", category.category, category.name)

    # It's also important that we save some examples of images to qualitatively demonstrate that our code works.
    # Specifically, we want:
    # 1. Positive images that activate train_latent (true positive)
    # 2. Negative images that do not activate train_latent (true negative)
    # 3. Positive images that do not activate train_latent (false negatives)
    # 4. Negative images that do activation train_latent (false positives)
    true_pos_im_i = (
        val_acts[:, train_latent][val_labels == 1].argsort(descending=True).tolist()
    )
    dump_k_comparisons(cfg, val_dataset, sae, true_pos_im_i, train_latent)

    return (category.name, score)


@beartype.beartype
@torch.inference_mode()
def dump_k_comparisons(
    cfg: config.BrodenEvaluate,
    dataset: PixelDataset,
    sae: nn.SparseAutoencoder,
    samples: list[int],
    latent: int,
):
    n_patches = dataset.vit_acts.metadata.n_patches_per_img
    all_patches = np.arange(n_patches)

    patch_lookup = make_patch_lookup(patch_size_px=cfg.patch_size)

    with open(os.path.join(cfg.root, "index.csv")) as fd:
        records = [Record.from_row_dict(row) for row in csv.DictReader(fd)]

    seen_i_im = set()

    for i in samples:
        i_im = dataset[i]["i_im"].item()
        if i_im in seen_i_im:
            logger.debug("Already dumped image %d; skipping.", i_im)
            continue

        vit_i = i_im * n_patches + all_patches
        vit_acts = torch.stack([dataset.vit_acts[v_i.item()].vit_acts for v_i in vit_i])
        _, sae_acts, *_ = sae(vit_acts.to(cfg.device))
        # logger.info("Saved example image to '%s'.", im_fpath)

        dst = Image.new("RGB", (224 * 3, 224), (255, 255, 255))

        # Original image
        orig_img = Image.open(os.path.join(cfg.root, "images", records[i_im].image))
        dst.paste(orig_img, (0, 0))

        # True image
        true_patches = np.zeros(n_patches)
        for pixel_file in getattr(records[i_im], dataset.category.category + "s"):
            true_i_p = get_patches(
                cfg, pixel_file, dataset.category.number, patch_lookup
            )
            true_patches[true_i_p] = 1
        true_img = imaging.add_highlights(orig_img, true_patches, upper=1.0)
        dst.paste(true_img, (224, 0))

        # Predicted image
        pred_patches = sae_acts[:, latent].cpu().numpy()
        pred_img = imaging.add_highlights(
            orig_img, pred_patches, upper=pred_patches.max().item()
        )
        dst.paste(pred_img, (448, 0))

        im_fpath = os.path.join(get_im_dir(cfg, dataset.category), f"{i_im}.png")
        dst.save(im_fpath)
        logger.info("Saved example image to '%s'.", im_fpath)

        seen_i_im.add(i_im)
        if len(seen_i_im) >= cfg.k:
            return


@beartype.beartype
def evaluate_group(cfg: config.BrodenEvaluate, group_cls: type[T]) -> Report:
    with open(os.path.join(cfg.root, f"c_{group_cls.category}.csv")) as fd:
        categories = frozenset(
            group_cls.from_row_dict(row) for row in csv.DictReader(fd)
        )

    splits: dict[str, float] = {}

    if cfg.debug:
        pool_cls, pool_kwargs = DummyExecutor, {}
    else:
        pool_cls = concurrent.futures.ProcessPoolExecutor
        pool_kwargs = dict(max_tasks_per_child=1)

    with pool_cls(**pool_kwargs) as pool:
        futures = []
        for category in categories:
            n_val_samples = int(category.frequency * 0.3)
            if n_val_samples < cfg.sample_range[0]:
                logger.info(
                    "Skipping %s '%s' because it only has %d validation samples (less than mininimum %d).",
                    group_cls.category,
                    category.name,
                    n_val_samples,
                    cfg.sample_range[0],
                )
                continue

            others = categories - {category}
            assert isinstance(others, frozenset)
            futures.append(pool.submit(evaluate_category, cfg, category, others))

        for future in concurrent.futures.as_completed(futures):
            name, score = future.result()
            splits[name] = score

    return Report(group_cls.category, np.mean(list(splits.values())).item(), splits)


@beartype.beartype
@torch.inference_mode()
def evaluate(cfg: config.BrodenEvaluate):
    reports = []
    for group_cls in (Material, Part, Object, Color):
        reports.append(evaluate_group(cfg, group_cls))
    breakpoint()
