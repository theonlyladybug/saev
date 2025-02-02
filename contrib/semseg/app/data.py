import base64
import dataclasses
import functools
import io
import logging
import os.path
import random
import typing

import beartype
import einops.layers.torch
import numpy as np
import torchvision.datasets.folder
from jaxtyping import UInt8, jaxtyped
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

logger = logging.getLogger("app.data")


@beartype.beartype
class Ade20k:
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Sample:
        img_path: str
        seg_path: str
        label: str
        target: int

    samples: list[Sample]

    def __init__(
        self,
        root: str,
        split: str,
        *,
        img_transform: typing.Callable | None = None,
        seg_transform: typing.Callable | None = lambda x: None,
    ):
        self.logger = logging.getLogger("ade20k")
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, "images")
        self.seg_dir = os.path.join(root, "annotations")
        self.img_transform = img_transform
        self.seg_transform = seg_transform

        # Check that we have the right path.
        for subdir in ("images", "annotations"):
            if not os.path.isdir(os.path.join(root, subdir)):
                # Something is missing.
                if os.path.realpath(root).endswith(subdir):
                    self.logger.warning(
                        "The ADE20K root should contain 'images/' and 'annotations/' directories."
                    )
                raise ValueError(f"Can't find path '{os.path.join(root, subdir)}'.")

        _, split_mapping = torchvision.datasets.folder.find_classes(self.img_dir)
        split_lookup: dict[int, str] = {
            value: key for key, value in split_mapping.items()
        }
        self.loader = torchvision.datasets.folder.default_loader

        err_msg = f"Split '{split}' not in '{set(split_lookup.values())}'."
        assert split in set(split_lookup.values()), err_msg

        # Load all the image paths.
        imgs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.img_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == split
        ]

        segs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.seg_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == split
        ]

        # Load all the targets, classes and mappings
        with open(os.path.join(root, "sceneCategories.txt")) as fd:
            img_labels: list[str] = [line.split()[1] for line in fd.readlines()]

        label_set = sorted(set(img_labels))
        label_to_idx = {label: i for i, label in enumerate(label_set)}

        self.samples = [
            self.Sample(img_path, seg_path, label, label_to_idx[label])
            for img_path, seg_path, label in zip(imgs, segs, img_labels)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        # Convert to dict.
        sample = dataclasses.asdict(self.samples[index])

        sample["image"] = self.loader(sample.pop("img_path"))
        if self.img_transform is not None:
            image = self.img_transform(sample.pop("image"))
            if image is not None:
                sample["image"] = image

        sample["segmentation"] = Image.open(sample.pop("seg_path")).convert("L")
        if self.seg_transform is not None:
            segmentation = self.seg_transform(sample.pop("segmentation"))
            if segmentation is not None:
                sample["segmentation"] = segmentation

        sample["index"] = index

        return sample

    def __len__(self) -> int:
        return len(self.samples)


@functools.cache
def get_dataset() -> Ade20k:
    img_transform = v2.Compose([
        v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((448, 448)),
    ])

    seg_transform = v2.Compose([
        v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((448, 448)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("() width height -> width height"),
    ])

    return Ade20k(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/",
        split="validation",
        img_transform=img_transform,
        seg_transform=seg_transform,
    )


@beartype.beartype
def get_sample(i: int) -> tuple[Image.Image, Image.Image]:
    dataset = get_dataset()
    sample = dataset[i]
    return sample["image"], seg_to_img(sample["segmentation"])


@jaxtyped(typechecker=beartype.beartype)
def seg_to_img(map: UInt8[Tensor, "width height"]) -> Image.Image:
    map = map.cpu().numpy()
    if map.ndim == 3:
        map = einops.rearrange(map, "w h () -> w h")
    colored = np.zeros((448, 448, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored[map == i + 1, :] = color

    return Image.fromarray(colored)


@jaxtyped(typechecker=beartype.beartype)
def make_colors() -> UInt8[np.ndarray, "n 3"]:
    values = (0, 51, 102, 153, 204, 255)
    colors = []
    for r in values:
        for g in values:
            for b in values:
                colors.append((r, g, b))
    # Fixed seed
    random.Random(42).shuffle(colors)
    colors = np.array(colors, dtype=np.uint8)

    # Fixed colors for example 3122
    colors[2] = np.array([201, 249, 255], dtype=np.uint8)
    colors[4] = np.array([151, 204, 4], dtype=np.uint8)
    colors[13] = np.array([104, 139, 88], dtype=np.uint8)
    colors[16] = np.array([54, 48, 32], dtype=np.uint8)
    colors[26] = np.array([45, 125, 210], dtype=np.uint8)
    colors[46] = np.array([238, 185, 2], dtype=np.uint8)
    colors[52] = np.array([88, 91, 86], dtype=np.uint8)
    colors[72] = np.array([76, 46, 5], dtype=np.uint8)
    colors[94] = np.array([12, 15, 10], dtype=np.uint8)

    return colors


colors = make_colors()


@beartype.beartype
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="webp")
    b64 = base64.b64encode(buf.getvalue())
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64
