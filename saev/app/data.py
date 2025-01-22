import torchvision.datasets
import base64
import typing
import torch
import pyvips
import beartype

from .. import activations, config


class VipsImageFolder(torchvision.datasets.ImageFolder):
    """
    Clone of ImageFolder that returns pyvips.Image instead of PIL.Image.Image.
    """

    def __init__(
        self,
        root: str,
        transform: typing.Callable | None = None,
        target_transform: typing.Callable | None = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=self._vips_loader,
        )

    @staticmethod
    def _vips_loader(path: str) -> torch.Tensor:
        """Load and convert image to tensor using pyvips."""
        image = pyvips.Image.new_from_file(path, access="random")
        return image

    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": sample,
            "target": target,
            "label": self.classes[target],
            "index": index,
        }


@beartype.beartype
class VipsImagenet(activations.Imagenet):
    def __getitem__(self, i):
        example = self.hf_dataset[i]
        example["index"] = i

        example["image"] = example["image"].convert("RGB")
        # Convert to pyvips
        example["image"] = pyvips.Image.new_from_memory(
            example["image"].tobytes(),
            example["image"].width,
            example["image"].height,
            3,  # bands (RGB)
            "uchar",
        )
        if self.img_transform:
            example["image"] = self.img_transform(example["image"])
        example["target"] = example.pop("label")
        example["label"] = self.labels[example["target"]]

        return example


datasets = {
    "inat21__train_mini": VipsImageFolder(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini/"
    ),
    "imagenet__train": VipsImagenet(config.ImagenetDataset()),
}


@beartype.beartype
def get_img_v_raw(key: str, i: int) -> tuple[pyvips.Image, str]:
    """
    Get raw image and processed label from dataset.

    Returns:
        Tuple of pyvips.Image and classname.
    """
    dataset = datasets[key]
    sample = dataset[i]
    # iNat21 specific: Remove taxonomy prefix
    label = " ".join(sample["label"].split("_")[1:])
    return sample["image"], label


def to_sized(
    img_v_raw: pyvips.Image, max_px: int, crop_px: tuple[int, int]
) -> pyvips.Image:
    """Convert raw vips image to standard model input size (resize + crop)."""
    # Calculate scaling factors to reach RESIZE_SIZE
    breakpoint()
    # Scale the min side to max_px, keep the aspect ratio, then center crop to crop_px. You will need to calculate crop_coords based on the updated width/height. AI!
    hscale = RESIZE_SIZE[0] / img_v_raw.width
    vscale = RESIZE_SIZE[1] / img_v_raw.height

    # Resize then crop to CROP_COORDS
    resized = img_v_raw.resize(hscale, vscale=vscale)
    return resized.crop(*CROP_COORDS)


@beartype.beartype
def vips_to_base64(img_v: pyvips.Image) -> str:
    buf = img_v.write_to_buffer(".webp")
    b64 = base64.b64encode(buf)
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64
