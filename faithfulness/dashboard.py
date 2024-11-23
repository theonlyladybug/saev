import marimo

__generated_with = "0.9.20"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    import sys

    pkg_root = "/home/stevens.994/projects/saev-live"
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)

    import os
    import csv
    import einops

    import marimo as mo
    import sklearn.decomposition

    import numpy as np
    import datasets
    import beartype
    from PIL import Image, ImageDraw
    from jaxtyping import jaxtyped, Int

    import torch
    import matplotlib.pyplot as plt
    import polars as pl
    import altair as alt

    import saev.activations
    import saev.config
    import saev.helpers
    return (
        Image,
        ImageDraw,
        Int,
        alt,
        beartype,
        csv,
        datasets,
        einops,
        jaxtyped,
        mo,
        np,
        os,
        pkg_root,
        pl,
        plt,
        saev,
        sklearn,
        sys,
        torch,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        This dashboard lets you play with the feature steering using a bunch of different techniques.

        # To Do

        1. Record activations for ADE20K using DINOv2.
        2. Select the subset of activations that are \(>t\) of some semantic class.
        3. Embed a subset of *those* activations using PCA.
        4. ... Sliders, example images, movement through the PCA `transform()` method.

        ## Other Stuff

        1. Embed patches with PCA using (1) pixel values (2) DINOv2 activations and (3) SAE decompositions. Expect that the more disentangled values are more seperable.
        """
    )
    return


@app.cell
def __(saev):
    act_dataset = saev.activations.Dataset(
        saev.config.DataLoad(
            shard_root="/local/scratch/stevens.994/cache/saev/e20bbda1b6b011896dc6f49a698597a7ec000390d73cd7197b0fb243a1e13273/",
            patches="patches",
            layer=-2,
            scale_norm=True,
            scale_mean=True,
        )
    )

    img_dataset = saev.activations.TransformedAde20k(
        saev.config.Ade20kDataset(
            root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k"
        ),
        # transform=make_img_transform(),
        # seg_transform=make_seg_transform(),
    )
    return act_dataset, img_dataset


@app.cell
def __(
    cls_lookup,
    get_patches,
    img_dataset,
    np,
    pl,
    saev,
    seg_transform,
    torch,
):
    def make_df():
        max_seen_p = -1
        df = []
        for sample in saev.helpers.progress(
            torch.utils.data.Subset(
                img_dataset,
                np.random.default_rng(seed=1).permutation(len(img_dataset))[:5000].tolist(),
            ),
            every=100,
        ):
            rows = [
                {
                    "i_act": sample["index"] * 196 + i,
                    "i_im": sample["index"],
                    "i_p": i,
                    "obj_cls": 0,
                }
                for i in range(196)
            ]
            seg = seg_transform(sample["segmentation"]).squeeze().numpy()
            for cls in cls_lookup.keys():
                for p in get_patches(seg, cls):
                    rows[p]["obj_cls"] = cls

                    if p > max_seen_p:
                        max_seen_p = p
            df.extend(rows)
        print(max_seen_p)
        return pl.DataFrame(df)


    df = make_df()
    return df, make_df


@app.cell
def __(df):
    df
    return


@app.cell
def __():
    obj_classes = (29, 21, 75, 38)
    return (obj_classes,)


@app.cell
def __(act_dataset, cls_lookup, df, np, obj_classes, pl, torch):
    activations, labels, i_ims, i_ps = [], [], [], []
    for obj_cls in obj_classes:
        for i_act, i_im, i_p, obj_cls in (
            df.filter(pl.col("obj_cls") == obj_cls).sample(50).iter_rows()
        ):
            activations.append(act_dataset[i_act].vit_acts)
            labels.append(cls_lookup[obj_cls])
            i_ims.append(i_im)
            i_ps.append(i_p)


    activations = torch.stack(activations).numpy()
    labels = np.array(labels)
    i_im = np.array(i_ims)
    i_p = np.array(i_ps)
    activations.shape
    return activations, i_act, i_im, i_ims, i_p, i_ps, labels, obj_cls


@app.cell
def __(activations, sklearn):
    pca = sklearn.decomposition.IncrementalPCA(n_components=2)
    pca.fit(activations)
    x_r = pca.transform(activations)
    return pca, x_r


@app.cell
def __(alt, directions, i_im, i_p, labels, mo, np, pca, pl, sliders, x_r):
    x_shift, y_shift = (
        pca.transform((np.array(sliders.value) @ directions).reshape(1, -1))
        .reshape(-1)
        .astype(np.float32)
    )

    chart = mo.ui.altair_chart(
        alt.Chart(
            pl.concat(
                (
                    pl.from_numpy(x_r, ("x", "y")),
                    pl.from_numpy(i_im, ("i_im",)),
                    pl.from_numpy(i_p, ("i_p",)),
                    pl.from_numpy(labels, ("label",)),
                ),
                how="horizontal",
            ).vstack(
                pl.DataFrame(
                    {
                        "x": 0 + x_shift,
                        "y": 0 + y_shift,
                        "i_im": 0,
                        "i_p": 0,
                        "label": "manipulated",
                    }
                )
            )
        )
        .mark_point(opacity=0.6)
        .encode(
            x=alt.X("x"),
            y=alt.Y("y"),
            tooltip=["i_im"],
            color="label:N",
            shape="label:N",
        )
    )
    return chart, x_shift, y_shift


@app.cell
def __(chart):
    chart
    return


@app.cell
def __(chart, highlight_patches, img_dataset, mo):
    sample = chart.value
    if len(sample) > 6:
        sample = sample.sample(6)

    imgs = [
        highlight_patches(img_dataset[i_im]["image"], [i_p])
        for i_im, i_p in sample.select("i_im", "i_p").iter_rows()
    ]

    mo.hstack(imgs, justify="start")
    return imgs, sample


@app.cell
def __(mo, np):
    # Instead of random unit-norm directions, we should be using the sparse autoencoder to choose the directions.
    # Specifically, we can get f_x for the manipulated patch, then pick the dimensions that have maximal value.
    # We can pick out the columns of W_dec and move the patch in those directions.
    directions = np.random.default_rng(seed=3).random((2, 768))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sliders = mo.ui.array(
        [
            mo.ui.slider(-500, 500, step=10.0, label=f"Direction {i+1}", value=0)
            for i in range(2)
        ]
    )
    return directions, sliders


@app.cell
def __(mo, sliders):
    mo.vstack(sliders)
    return


@app.cell
def __(chart, dataset, mo, pl):
    mo.stop(
        len(chart.value) != 1,
        f"Select exactly 1 instance ({len(chart.value)} selected now) to manipulate dimensions.",
    )

    select_vit_act, _, _ = dataset[
        chart.value.select(pl.col("i_im") * 196 + pl.col("i_p")).item()
    ]
    select_vit_act.numpy()
    return (select_vit_act,)


@app.cell
def __(Image, ImageDraw, beartype):
    @beartype.beartype
    def highlight_patches(img: Image.Image, patches: list[int]) -> Image.Image:
        """
        Resizes and crops image, then highlights the specific patch.

        Assumes 256x256 resize -> 224x224 center crop -> 16x16 patches
        """
        resize_size_px = (256, 256)
        resize_w_px, resize_h_px = resize_size_px
        crop_size_px = (224, 224)
        crop_w_px, crop_h_px = crop_size_px
        crop_coords_px = (
            (resize_w_px - crop_w_px) // 2,
            (resize_h_px - crop_h_px) // 2,
            (resize_w_px + crop_w_px) // 2,
            (resize_h_px + crop_h_px) // 2,
        )
        iw_np, ih_np = 14, 14
        pw_px, ph_px = 16, 16

        img = img.resize(resize_size_px).crop(crop_coords_px)

        # Create a transparent red overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for patch in patches:
            x_np, y_np = patch % iw_np, patch // ih_np
            draw.rectangle(
                [
                    (x_np * pw_px, y_np * ph_px),
                    (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
                ],
                fill=(255, 0, 0, 128),
            )

        # Composite the original image and the overlay
        return Image.alpha_composite(img.convert("RGBA"), overlay)
    return (highlight_patches,)


@app.cell
def __(csv, img_dataset, os):
    def make_cls_lookup():
        cls_lookup = {}
        with open(os.path.join(img_dataset.cfg.root, "objectInfo150.txt")) as fd:
            for row in csv.DictReader(fd, delimiter="\t"):
                cls_lookup[int(row["Idx"])] = row["Name"]
        return cls_lookup


    cls_lookup = make_cls_lookup()
    cls_lookup
    return cls_lookup, make_cls_lookup


@app.cell
def __(Int, Uint8, beartype, jaxtyped, np):
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
        seg: Uint8[np.ndarray, "224 224"], cls: int, *, threshold: float = 0.9
    ) -> Int[np.ndarray, " n"]:
        """
        Gets a list of patches that contain more than `threshold` fraction pixels containing the category with number `number`.
        """
        x, y = np.where(seg == cls)
        patches, counts = np.unique(patch_lookup[x, y], return_counts=True)
        n_pixels = 16 * 16 * threshold
        return patches[counts > n_pixels]


    patch_lookup = make_patch_lookup(patch_size_px=(16, 16), im_size_px=(224, 224))
    return get_patches, make_patch_lookup, patch_lookup


@app.cell
def __(Image):
    def make_img_transform():
        from torchvision.transforms import v2

        return v2.Compose(
            [v2.Resize(size=(256, 256)), v2.CenterCrop(size=(224, 224)), v2.ToImage()]
        )


    def make_seg_transform():
        from torchvision.transforms import v2

        return v2.Compose(
            [
                v2.Resize(size=(256, 256), interpolation=Image.Resampling.NEAREST),
                v2.CenterCrop(size=(224, 224)),
                v2.ToImage(),
            ]
        )


    img_transform = make_img_transform()
    seg_transform = make_seg_transform()
    return (
        img_transform,
        make_img_transform,
        make_seg_transform,
        seg_transform,
    )


if __name__ == "__main__":
    app.run()
