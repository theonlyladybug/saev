import marimo

__generated_with = "0.9.14"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    import collections
    import csv
    import dataclasses
    import math
    import os.path
    import typing

    import beartype
    import einops
    import marimo as mo
    import numpy as np
    import torch
    from jaxtyping import Float, Shaped, jaxtyped
    from PIL import Image, ImageDraw

    return (
        Float,
        Image,
        ImageDraw,
        Shaped,
        beartype,
        collections,
        csv,
        dataclasses,
        einops,
        jaxtyped,
        math,
        mo,
        np,
        os,
        torch,
        typing,
    )


@app.cell
def __():
    import sys

    if ".." not in sys.path:
        sys.path.append("..")
    return (sys,)


@app.cell
def __():
    from saev.saev import nn
    from saev.saev.broden import (
        Material,
        PixelDataset,
        Record,
        get_patches,
        make_patch_lookup,
    )
    from saev.saev.config import BrodenEvaluate, DataLoad

    return (
        BrodenEvaluate,
        DataLoad,
        Material,
        PixelDataset,
        Record,
        get_patches,
        make_patch_lookup,
        nn,
    )


@app.cell
def __():
    from saev.saev import imaging

    return (imaging,)


@app.cell
def __(BrodenEvaluate, DataLoad):
    cfg = BrodenEvaluate(
        ckpt="checkpoints/xb9bph3r/sae.pt",
        patch_size=(14, 14),
        root="/research/nfs_su_809/workspace/stevens.994/datasets/broden",
        data=DataLoad(
            shard_root="/local/scratch/stevens.994/cache/saev/484e6a98e98421b3f18c7647d62f8aab152ee4fa4d7752cd4ae3f0b8a7fa9091/",
            patches="patches",
            layer=-2,
        ),
        n_workers=8,
        batch_size=1024,
        sample_range=(200, 1000),
        dump_to="./logs/broden",
        device="cuda",
        log_every=10,
        seed=42,
        debug=True,
    )
    return (cfg,)


@app.cell
def __(cfg, nn):
    sae = nn.load(cfg.ckpt)
    sae.eval()
    return (sae,)


@app.cell
def __(mo):
    mo.md(
        r"""
        I want to look at many different examples of materials. 
        There are lots of different datasets in `index.csv` and I want to see at least 5-10 from each.
        I also want to see 5-10 examples of every material.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""Given the segmentation masks at 112x112, I want to know which patches in a 224x224 image (no center crop because images are already 224x224) are more than 50% filled with a given material (eventually not just material, but any "category")."""
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        I think I understand how to get the patch number for a given `(x, y)` pair in pixel space.
        Now I need to know which patches match a given material.

        For example, given `opensurfaces/100963.jpg` (which is `records[96]`), which patches are 50% or more "painted" (code 2, number 20)?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Now I would like to find 10K patches with a given material *m* and 10K patches with many different *other* materials.
        Then I will use that as a specific probe to check for the existence of a feature for material *m*.
        """
    )
    return


@app.cell
def __():
    i = 965
    return (i,)


@app.cell
def __(Record, cfg, csv, os):
    with open(os.path.join(cfg.root, "index.csv")) as fd:
        records = [Record.from_row_dict(row) for row in csv.DictReader(fd)]
    return fd, records


@app.cell
def __(Material, PixelDataset, cfg, csv, os):
    def load_materials() -> frozenset[Material]:
        with open(os.path.join(cfg.root, "c_material.csv")) as fd:
            materials = [Material.from_row_dict(row) for row in csv.DictReader(fd)]
        return frozenset(materials)

    materials = load_materials()

    material = Material(
        code=6, number=49, name="tile", frequency=3409, category="material"
    )
    others = materials - {material}

    dataset = PixelDataset(cfg, material, others, is_train=False)
    return dataset, load_materials, material, materials, others


@app.cell
def __(dataset):
    latent = 7178

    dataset.category
    return (latent,)


@app.cell
def __(cfg, dataset, make_patch_lookup, np):
    n_patches = dataset.vit_acts.metadata.n_patches_per_img
    all_patches = np.arange(n_patches)

    patch_lookup = make_patch_lookup(patch_size_px=cfg.patch_size)
    patch_lookup
    return all_patches, n_patches, patch_lookup


@app.cell
def __(dataset, i):
    i_im = dataset[i]["i_im"].item()
    i_im
    return (i_im,)


@app.cell
def __(all_patches, i_im, n_patches):
    vit_i = i_im * n_patches + all_patches
    vit_i
    return (vit_i,)


@app.cell
def __(Image, cfg, i_im, os, records):
    orig_img = Image.open(os.path.join(cfg.root, "images", records[i_im].image))
    orig_img
    return (orig_img,)


@app.cell
def __(
    cfg,
    dataset,
    get_patches,
    i_im,
    imaging,
    n_patches,
    np,
    orig_img,
    patch_lookup,
    records,
):
    true_patches = np.zeros(n_patches)
    for pixel_file in getattr(records[i_im], dataset.category.category + "s"):
        true_i_p = get_patches(cfg, pixel_file, dataset.category.number, patch_lookup)
        true_patches[true_i_p] = 1
    true_img = imaging.add_highlights(orig_img, true_patches, upper=1.0)
    true_img
    return pixel_file, true_i_p, true_img, true_patches


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
