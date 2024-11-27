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

    import csv
    import os

    import altair as alt
    import beartype
    import datasets
    import einops
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import sklearn.decomposition
    import torch
    from jaxtyping import Int, jaxtyped
    from PIL import Image, ImageDraw

    import saev.activations
    import saev.config
    import saev.helpers
    import saev.nn
    import saev.visuals

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
        # Manipulate SAE Features in Vision Models

        This dashboard lets you play with the feature steering using a bunch of different techniques.
        """
    )
    return


@app.cell
def __(cls_lookup, cls_select, mo):
    selected_classes_md = mo.md(
        "\n".join(
            f"* **'{cls_lookup[i]}'** (ADE20K Class {i})" for i in cls_select.value
        )
    )

    mo.vstack([cls_select, selected_classes_md])
    return (selected_classes_md,)


@app.cell
def __(saev):
    act_dataset = saev.activations.Dataset(
        saev.config.DataLoad(
            shard_root="/local/scratch/stevens.994/cache/saev/a860104bf29d6093dd18b8e2dccd2e7efdfcd9fac35dceb932795af05187cb9f/",
            patches="patches",
            layer=-2,
            scale_norm=False,
            scale_mean=False,
        )
    )

    img_dataset = saev.activations.Ade20k(
        saev.config.Ade20kDataset(
            root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k",
        ),
        seg_transform=lambda x: x,
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
    def random_subset_of(dataset, *, n_subset=5000, seed=1):
        n_original = len(dataset)
        indices = (
            np.random.default_rng(seed=seed).permutation(n_original)[:n_subset].tolist()
        )

        return torch.utils.data.Subset(dataset, indices)

    def make_df():
        df = []
        for sample in saev.helpers.progress(random_subset_of(img_dataset), every=100):
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

            df.extend(rows)

        return pl.DataFrame(df)

    df = make_df()
    return df, make_df, random_subset_of


@app.cell
def __(mo):
    n_examples_per_class_slider = mo.ui.slider(10, 100, value=20, debounce=True)
    return (n_examples_per_class_slider,)


@app.cell
def __(mo, n_examples_per_class_slider):
    mo.hstack(
        [
            mo.md(f"{n_examples_per_class_slider.value} samples per class"),
            n_examples_per_class_slider,
        ],
        justify="start",
    )
    return


@app.cell
def __(
    act_dataset,
    cls_lookup,
    cls_select,
    df,
    mo,
    n_examples_per_class_slider,
    np,
    pl,
    torch,
):
    mo.stop(len(cls_select.value) <= 1, "Choose at least two (2) classes.")

    activations, labels, i_ims, i_ps, i_acts = [], [], [], [], []
    for obj_cls in cls_select.value:
        obj_df = df.filter(pl.col("obj_cls") == obj_cls)
        if len(obj_df) > n_examples_per_class_slider.value:
            obj_df = obj_df.sample(n_examples_per_class_slider.value, seed=42)

        for i_act, i_im, i_p, obj_cls in obj_df.iter_rows():
            activations.append(act_dataset[i_act].vit_acts)
            labels.append(cls_lookup[obj_cls])
            i_ims.append(i_im)
            i_ps.append(i_p)
            i_acts.append(i_act)

    activations = torch.stack(activations).numpy()
    labels = np.array(labels)
    i_im = np.array(i_ims)
    i_p = np.array(i_ps)
    i_act = np.array(i_acts)
    return (
        activations,
        i_act,
        i_acts,
        i_im,
        i_ims,
        i_p,
        i_ps,
        labels,
        obj_cls,
        obj_df,
    )


@app.cell
def __(activations, np, sklearn):
    x_proj_init = sklearn.decomposition.IncrementalPCA(n_components=2).fit_transform(
        activations
    )
    without_outliers = np.nonzero(x_proj_init[:, 0] < 10)[0]

    pca = sklearn.decomposition.IncrementalPCA(n_components=2)
    pca.fit(activations[without_outliers])
    x_proj = pca.transform(activations[without_outliers])
    return pca, without_outliers, x_proj, x_proj_init


@app.cell
def __(
    alt,
    feature_dirs,
    i_im,
    i_p,
    labels,
    mo,
    np,
    pca,
    pl,
    sliders,
    x_proj,
):
    x_shift, y_shift = (
        (
            pca.transform(
                (np.array(sliders.value) @ feature_dirs.reshape(-1, 768)).reshape(1, -1)
            )
            - pca.transform(np.zeros((1, 768)))
        )
        .reshape(-1)
        .astype(np.float32)
    )

    chart = mo.ui.altair_chart(
        alt.Chart(
            pl.concat(
                (
                    pl.from_numpy(x_proj, ("x", "y")),
                    pl.from_numpy(i_im, ("i_im",)),
                    pl.from_numpy(i_p, ("i_p",)),
                    pl.from_numpy(labels, ("label",)),
                    pl.from_numpy(np.arange(len(i_im)), ("example_index",)),
                ),
                how="horizontal",
            ).vstack(
                pl.DataFrame({
                    "x": x_proj[0, 0] + x_shift,
                    "y": x_proj[0, 1] + y_shift,
                    "i_im": i_im[0].item(),
                    "i_p": i_p[0].item(),
                    "label": f"(CHANGEABLE) {labels[0].item()}",
                    "example_index": 0,
                })
            )
        )
        .mark_point(opacity=0.6)
        .encode(
            x=alt.X("x"),
            y=alt.Y("y"),
            tooltip=["example_index", "i_im"],
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
def __(cls_lookup, cls_select, feature_idxs, mo, slider_images):
    def make_sliders():
        sliders = []
        for obj_idxs, obj_cls_ in zip(feature_idxs, cls_select.value):
            for i, j in enumerate(obj_idxs):
                sliders.append(
                    mo.ui.slider(
                        -50,
                        50,
                        step=2.0,
                        label=f"'{cls_lookup[obj_cls_]}' #{i + 1} (SAE {j})",
                        value=0,
                    )
                )

        return mo.ui.array(sliders)

    sliders = make_sliders()

    def make_slider_ui():
        rows = []
        for slider, (orig, highlighted) in zip(sliders, slider_images):
            row = mo.vstack([
                slider,
                mo.accordion(
                    {
                        "Example Images": mo.vstack([
                            mo.hstack(orig[:4], justify="start"),
                            mo.hstack(highlighted[:4], justify="start"),
                        ])
                    },
                    multiple=True,
                ),
            ])
            rows.append(row)
        return mo.vstack(rows)

    make_slider_ui()
    return make_slider_ui, make_sliders, sliders


@app.cell
def __(Image, beartype, feature_idxs, resize_and_crop, saev, torch):
    @beartype.beartype
    def make_images() -> list[tuple[list[Image.Image], list[Image.Image]]]:
        top_i = torch.load(
            "/research/nfs_su_809/workspace/stevens.994/saev/webapp/ercgckr1/sort_by_patch/top_img_i.pt",
            weights_only=True,
            map_location="cpu",
        )
        top_values_p = torch.load(
            "/research/nfs_su_809/workspace/stevens.994/saev/webapp/ercgckr1/sort_by_patch/top_values.pt",
            weights_only=True,
            map_location="cpu",
        )

        in1k_dataset = saev.activations.get_dataset(
            saev.config.ImagenetDataset(), transform=None
        )

        images = []

        # Flatten feature_idxs because it's nested with respect the number of object classes
        for idx in saev.helpers.progress(feature_idxs.reshape(-1), every=1):
            seen_i_im = set()

            elems = []
            for i_im, values_p in zip(top_i[idx].tolist(), top_values_p[idx]):
                if i_im in seen_i_im:
                    continue

                example = in1k_dataset[i_im]
                elem = saev.visuals.GridElement(
                    example["image"], example["label"], values_p
                )
                elems.append(elem)

                seen_i_im.add(i_im)

            upper = None
            if top_values_p[idx].numel() > 0:
                upper = top_values_p[idx].max().item()

            orig = [
                resize_and_crop(elem.img, (256, 256), (224, 224)).resize((196, 196))
                for elem in elems
            ]
            highlighted = [
                saev.visuals.make_img(elem, upper=upper).resize((196, 196))
                for elem in elems
            ]
            images.append((orig, highlighted))

        return images

    slider_images = make_images()
    return make_images, slider_images


@app.cell
def __(Image, ImageDraw, beartype):
    @beartype.beartype
    def resize_and_crop(
        img: Image.Image,
        resize_px: tuple[int, int] = (256, 256),
        crop_px: tuple[int, int] = (224, 224),
    ) -> Image.Image:
        resize_w_px, resize_h_px = resize_px
        crop_w_px, crop_h_px = crop_px
        crop_coords_px = (
            (resize_w_px - crop_w_px) // 2,
            (resize_h_px - crop_h_px) // 2,
            (resize_w_px + crop_w_px) // 2,
            (resize_h_px + crop_h_px) // 2,
        )
        return img.resize(resize_px).crop(crop_coords_px)

    @beartype.beartype
    def highlight_patches(img: Image.Image, patches: list[int]) -> Image.Image:
        """
        Resizes and crops image, then highlights the specific patch.

        Assumes 256x256 resize -> 224x224 center crop -> 16x16 patches
        """
        resize_size_px = (256, 256)
        crop_size_px = (224, 224)
        iw_np, ih_np = 14, 14
        pw_px, ph_px = 16, 16

        img = resize_and_crop(img, resize_size_px, crop_size_px)

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

    return highlight_patches, resize_and_crop


@app.cell
def __(beartype, csv, img_dataset, mo, os):
    @beartype.beartype
    def make_cls_lookup() -> dict[int, str]:
        cls_lookup = {}
        with open(os.path.join(img_dataset.cfg.root, "objectInfo150.txt")) as fd:
            for row in csv.DictReader(fd, delimiter="\t"):
                cls_lookup[int(row["Idx"])] = row["Name"]
        return cls_lookup

    cls_lookup = make_cls_lookup()

    cls_select = mo.ui.multiselect(
        {name: value for value, name in cls_lookup.items()}, label="Classes"
    )
    return cls_lookup, cls_select, make_cls_lookup


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

        return v2.Compose([
            v2.Resize(size=(256, 256)),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
        ])

    def make_seg_transform():
        from torchvision.transforms import v2

        return v2.Compose([
            v2.Resize(size=(256, 256), interpolation=Image.Resampling.NEAREST),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
        ])

    img_transform = make_img_transform()
    seg_transform = make_seg_transform()
    return (
        img_transform,
        make_img_transform,
        make_seg_transform,
        seg_transform,
    )


@app.cell
def __(
    Float,
    Int,
    activations,
    beartype,
    cls_lookup,
    cls_select,
    jaxtyped,
    labels,
    np,
    saev,
    torch,
    without_outliers,
):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.no_grad
    def get_feature_directions(
        acts: Float[np.ndarray, "n d_vit"], labels: Int[np.ndarray, " n"], *, k: int = 2
    ) -> tuple[Int[np.ndarray, "n_unique k"], Float[np.ndarray, "n_unique k d_vit"]]:
        """
        Get the k most meaningful features for each unique obj_cls.
        """
        sae = saev.nn.load(
            "/home/stevens.994/projects/saev-live/checkpoints/ercgckr1/sae.pt"
        )

        acts_pt = torch.from_numpy(acts)
        _, f_x, loss = sae(acts_pt)

        print("MSE Loss:", loss.mse.item())

        out_idxs, out_dirs = [], []
        for obj_cls in cls_select.value:
            i = labels == cls_lookup[obj_cls]
            vals, idxs = f_x[i].topk(32)
            idxs, counts = np.unique(idxs.numpy(), return_counts=True)
            top = list(reversed(np.argsort(counts)[-k:]))
            # print(len(top), idxs.shape, idxs[top])

            out_idxs.append(idxs[top])
            out_dirs.append(sae.W_dec[idxs[top]].numpy())
        # print(out_dirs)
        return np.array(out_idxs), np.array(out_dirs)

    feature_idxs, feature_dirs = get_feature_directions(
        activations[without_outliers], labels[without_outliers]
    )
    return feature_dirs, feature_idxs, get_feature_directions


@app.cell
def __(mo):
    mo.md(
        r"""
        # Roadmap

        What do I still want to do with this?

        *  Embed patches with PCA using (1) pixel values (2) DINOv2 activations and (3) SAE decompositions. Expect that the more disentangled values are more seperable.
        """
    )
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
