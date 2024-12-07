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

    with tarfile.open(tgz_path, "r") as tar:
        for member in tqdm.tqdm(tar, desc="Extracting images"):
            tar.extract(member, path=args.dir, filter="data")

    # Make these changes: AI!
    # * Remove args.dir/caltech-101/
    # * Move args.dir/101_ObjectCategories/ to args.dir/caltech-101
    # * Remove the args.dir/caltech-101/BACKGROUND_Google folder and its contents.
    # Add a print() statement after each of these actions.
    breakpoint()


if __name__ == "__main__":
    main(tyro.cli(Args))
