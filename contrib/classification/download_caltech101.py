# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beartype",
#     "requests",
#     "scipy",
#     "tqdm",
#     "tyro",
# ]
# ///
"""
A script to download the Caltech101 dataset for use as an saev.activations.ImageFolderDataset.

```sh
uv run contrib/classification/download_flowers.py --help
```
"""

import shutil
import dataclasses
import os
import tarfile
import zipfile

import beartype
import requests
import tqdm
import tyro

url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"


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
    """Download NeWT."""
    os.makedirs(args.dir, exist_ok=True)
    chunk_size = int(args.chunk_size_kb * 1024)
    zip_path = os.path.join(args.dir, "caltech-101.zip")
    tgz_path = os.path.join(args.dir, "caltech-101", "101_ObjectCategories.tar.gz")

    # Download dataset.
    r = requests.get(url, stream=True)
    r.raise_for_status()

    n_bytes = int(r.headers["content-length"])

    with open(zip_path, "wb") as fd:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=n_bytes / chunk_size,
            unit="b",
            unit_scale=1,
            unit_divisor=1024,
            desc="Downloading dataset",
        ):
            fd.write(chunk)
    print(f"Downloaded dataset: {zip_path}.")

    zip = zipfile.ZipFile(zip_path)
    zip.extract("caltech-101/101_ObjectCategories.tar.gz", args.dir)
    print("Unzipped file.")

    with tarfile.open(tgz_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images"):
            tar.extract(member, path=args.dir, filter="data")
    print("Extracted images.")

    # Clean up and organize files

    # Remove the temporary caltech-101 directory
    dpath = os.path.join(args.dir, "caltech-101")
    shutil.rmtree(dpath)
    print(f"Removed temporary directory: {dpath}")

    # Move 101_ObjectCategories to caltech-101
    os.rename(os.path.join(args.dir, "101_ObjectCategories"), dpath)
    print(f"Moved dataset to: {dpath}")

    # Remove the BACKGROUND_Google folder
    shutil.rmtree(os.path.join(dpath, "BACKGROUND_Google"))
    print("Removed BACKGROUND_Google folder")

    # Using a fixed seed, generate a train/test split with 30 images per class for training and at most 50 images per class for testing.
    # Then move the files on disk from args.dir/caltech-101 into a args.dir/train and an args.dir/test directory.
    # AI!


if __name__ == "__main__":
    main(tyro.cli(Args))
