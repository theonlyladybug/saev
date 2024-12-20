# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beartype",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///

import dataclasses
import os
import random
import shutil
import tarfile
import zipfile

import beartype
import requests
import tqdm
import tyro

url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

IMG_EXTS = (".jpg", ".jpeg", ".png")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Args:
    """Configure download options."""

    dir: str = "."
    """Where to save data."""

    chunk_size_kb: int = 1
    """How many KB to download at a time before writing to file."""

    seed: int = 42
    """Random seed used to generate split."""


def main(args: Args):
    """Download CUB-200-2011."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    tgz_path = os.path.join(args.dir, "CUB_200_2011.tgz")

    # Download dataset.
    r = requests.get(url, stream=True)
    r.raise_for_status()

    n_bytes = int(r.headers["content-length"])

    with open(tgz_path, "wb") as fd:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=n_bytes / chunk_size,
            unit="b",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading dataset",
        ):
            fd.write(chunk)
    print(f"Downloaded dataset: {tgz_path}.")

    with tarfile.open(tgz_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting data"):
            tar.extract(member, path=args.dir, filter="data")
    print("Extracted data.")

    # Clean up and organize files for a torchvision.datasets.ImageFolder.
    # Some notes on dataset structure:
    #
    # * args.dir/CUB_200_2011/image_class_labels.txt has space-separated pairs of numbers, with the IMG_NUM first and the CLASS_NUM next.
    # * args.dir/CUB_200_2011/images.txt has space-separated (number, string) pairs, where the number is the IMG_NUM above and the string is the relative path from args.dir/CUB_200_2011/images to the actual image.
    # * args.dir/CUB_200_2011/classes.txt has space-separated (number, string) pairs, where the number is the CLASS_NUM above and the string is the classname.
    # * args.dir/CUB_200_2011/classes.txt has space-separated (number, number) pairs, where the first number is the IMG_NUM above and the second number is either 0 or 1. 1 indicates train, 0 indicates test.
    #
    # The next block of code creates the train/test splits in args.dir/train and args.dir/test folders.

    # Create output directories
    train_dir = os.path.join(args.dir, "train")
    test_dir = os.path.join(args.dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Fill out the rest of this code. AI!


if __name__ == "__main__":
    main(tyro.cli(Args))
