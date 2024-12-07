import marimo

__generated_with = "0.9.20"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    n_images_per_feature = 5
    n_features = 3
    return n_features, n_images_per_feature


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        This is a dashboard to explore how you can use sparse autoencoders to manipulate model classifications. It presents a couple different ways to automatically propose features that you might want to manipulate.

        How do we propose features?

        1. Pick the 3 features that maximally activate on this image's class. This demonstrates that we can influence decisions for *this* class.
        2. Pick the 3 features that maximally activate for a different class. This demonstrates that we can influence decisions for another class.
        3. Pick 3 random features. This demonstrates that arbitrary, unrelated directions don't really matter for downstream predictions (precision).

        For all of these features, show the `n_images_per_feature` highest activating examples from ImageNet-1K.
        We also don't show features that activate on all classes.
        """
    )
    return


@app.cell(hide_code=True)
def __(contrib, saev):
    clf_ckpt_fpath = "checkpoints/contrib/classification/lr_0_01__wd_0_3/model.pt"
    clf = contrib.classification.training.load_model(clf_ckpt_fpath)
    clf.eval()

    sae_ckpt_fpath = "checkpoints/bd97z80b/sae.pt"
    sae = saev.nn.load(sae_ckpt_fpath)
    sae.eval()

    imgs_test_dpath = "/research/nfs_su_809/workspace/stevens.994/datasets/caltech101/test"
    imgs_cfg = saev.config.ImageFolderDataset(root=imgs_test_dpath)
    imgs_dataset = saev.activations.ImageFolder(imgs_cfg.root)

    acts_test_dpath = "/local/scratch/stevens.994/cache/saev/f03ba457ce4247cbdff55999540ca3dc97e4c333ca8cf6a6a66939629b2f36a5"
    acts_cfg = saev.config.DataLoad(shard_root=acts_test_dpath, patches="cls")

    acts_ND = contrib.classification.training.load_acts(acts_cfg)
    targets_N = contrib.classification.training.load_targets(imgs_cfg)

    assert len(imgs_dataset) == len(acts_ND)
    return (
        acts_ND,
        acts_cfg,
        acts_test_dpath,
        clf,
        clf_ckpt_fpath,
        imgs_cfg,
        imgs_dataset,
        imgs_test_dpath,
        sae,
        sae_ckpt_fpath,
        targets_N,
    )


@app.cell(hide_code=True)
def __(os, torch):
    sae_data_root = (
        "/research/nfs_su_809/workspace/stevens.994/saev/features/bd97z80b/sort_by_img"
    )

    top_img_i = torch.load(
        os.path.join(sae_data_root, "top_img_i.pt"), weights_only=True, map_location="cpu"
    )
    top_values = torch.load(
        os.path.join(sae_data_root, "top_values.pt"), weights_only=True, map_location="cpu"
    )
    sparsity = torch.load(
        os.path.join(sae_data_root, "sparsity.pt"), weights_only=True, map_location="cpu"
    )
    return sae_data_root, sparsity, top_img_i, top_values


@app.cell
def __(mo):
    example_getter, example_setter = mo.state(1)
    return example_getter, example_setter


@app.cell
def __(example_setter, imgs_dataset, mo, random):
    random_example_btn = mo.ui.button(
        label="Random Example",
        on_change=lambda _: example_setter(random.randrange(len(imgs_dataset)) + 1),
    )
    return (random_example_btn,)


@app.cell
def __(example_getter, example_setter, imgs_dataset, mo):
    example_num = mo.ui.number(
        start=1,
        stop=len(imgs_dataset),
        step=1,
        value=example_getter(),
        label="Example:",
        on_change=example_setter,
    )
    return (example_num,)


@app.cell
def __(imgs_dataset, mo):
    cls_dropdown = mo.ui.dropdown(
        options={
            name: i
            for i, name in sorted(enumerate(imgs_dataset.classes), key=lambda pair: pair[1])
        },
        label="Alternative class:",
        value=imgs_dataset.classes[0],
    )
    return (cls_dropdown,)


@app.cell
def __(features, mo):
    getter, setter = mo.state([feature.default for feature in features])
    return getter, setter


@app.cell
def __(beartype, getter, setter):
    @beartype.beartype
    def indexed_setter(i: int, v: float):
        setter(getter()[:i] + [v] + getter()[i + 1 :])
    return (indexed_setter,)


@app.cell
def __(indexed_setter):
    def make_indexed_reset(i: int, default: float):
        return lambda _: indexed_setter(i, default)
    return (make_indexed_reset,)


@app.cell
def __(features, functools, getter, indexed_setter, mo):
    sliders = [
        mo.ui.slider(
            start=-1,
            stop=1,
            step=0.05,
            value=getter()[i],
            on_change=functools.partial(indexed_setter, i),
        )
        for i, _ in enumerate(features)
    ]
    sliders = mo.ui.array(sliders)
    return (sliders,)


@app.cell
def __(features, mo, setter):
    reset_btn = mo.ui.button(
        label="Reset Sliders",
        kind="danger",
        on_change=lambda _: setter([feature.default for feature in features]),
    )
    return (reset_btn,)


@app.cell
def __(Feature, beartype, mo):
    @beartype.beartype
    def make_slider(feature: Feature) -> tuple[object, object, object]:
        getter, setter = mo.state(feature.default)
        slider = mo.ui.slider(start=-1, stop=1, step=0.1, value=getter(), on_change=setter)
        return getter, setter, slider
    return (make_slider,)


@app.cell
def __(Int, Tensor, beartype, imgs_dataset, jaxtyped, torch):
    @jaxtyped(typechecker=beartype.beartype)
    def load_targets() -> Int[Tensor, " n"]:
        targets = torch.tensor([tgt for sample, tgt in imgs_dataset.samples])
        return targets


    targets = load_targets()
    return load_targets, targets


@app.cell
def __(acts_ND, example_num, imgs_dataset, targets):
    target = targets[example_num.value - 1].item()
    act_D = acts_ND[example_num.value - 1]

    pil_image = imgs_dataset[example_num.value - 1]["image"]
    return act_D, pil_image, target


@app.cell
def __(sae, sparsity, torch):
    mask = torch.ones((sae.cfg.d_sae), dtype=bool)

    # No need for masking dataset-specific latents for caltech-101.
    # popular_latents = torch.tensor([f.latent for f in get_agg_features(acts_ND, n=100)])
    # mask[popular_latents] = False

    mask = mask & (sparsity < 1e-2)
    return (mask,)


@app.cell
def __(
    acts_ND,
    cls_dropdown,
    example_num,
    get_agg_features,
    get_random_features,
    mask,
    targets,
):
    features = (
        get_agg_features(acts_ND[targets == targets[example_num.value - 1]], mask=mask)
        + get_agg_features(acts_ND[targets == cls_dropdown.value], mask=mask)
        + get_random_features()
    )
    return (features,)


@app.cell
def __(
    cls_dropdown,
    example_num,
    imgs_dataset,
    mo,
    random_example_btn,
    target,
):
    mo.hstack(
        [
            mo.hstack(
                [
                    random_example_btn,
                    example_num,
                    mo.md(f"'{imgs_dataset.classes[target]}'"),
                ],
                justify="start",
            ),
            cls_dropdown,
        ]
    )
    return


@app.cell
def __(reset_btn):
    reset_btn
    return


@app.cell(hide_code=True)
def __(
    cls_dropdown,
    features,
    imgs_dataset,
    in1k_dataset,
    mo,
    n_features,
    n_images_per_feature,
    sliders,
    target,
    top_img_i,
):
    def make_sliders_ui():
        rows = []
        for slider, feature in zip(sliders, features):
            img_i = top_img_i[feature.latent, :n_images_per_feature].tolist()
            imgs = [in1k_dataset[i]["image"] for i in img_i]
            row = [
                mo.hstack([slider, f"{slider.value:.3f}"], justify="start"),
                mo.hstack(imgs, justify="start", gap=0.1),
            ]

            rows.append(mo.vstack(row))

        err_msg = f"len(rows) == {len(rows)} != n_features * 3 == {n_features * 3}"
        assert len(rows) == n_features * 3, err_msg

        return mo.hstack(
            [
                mo.vstack(
                    [mo.md(f"Features for '{imgs_dataset.classes[target]}'")]
                    + rows[n_features * 0 : n_features * 1]
                ),
                mo.vstack(
                    [mo.md(f"Features for '{imgs_dataset.classes[cls_dropdown.value]}'")]
                    + rows[n_features * 1 : n_features * 2]
                ),
                mo.vstack(
                    [mo.md("Random Features")] + rows[n_features * 2 : n_features * 3]
                ),
            ],
            justify="start",
            gap=1.0,
        )


    make_sliders_ui()
    return (make_sliders_ui,)


@app.cell(hide_code=True)
def __(act_D, clf, modify, torch):
    with torch.inference_mode():
        modified_act_D = modify(act_D)
        logits_C = clf(act_D)
        modified_logits_C = clf(modified_act_D)

        probs = torch.nn.functional.softmax(logits_C, dim=0)
        modified_probs = torch.nn.functional.softmax(modified_logits_C, dim=0)

    # mo.md(f"Mean difference: {(modified_probs - probs).abs().mean().item():.3f}")
    return (
        logits_C,
        modified_act_D,
        modified_logits_C,
        modified_probs,
        probs,
    )


@app.cell
def __(
    imgs_dataset,
    mo,
    modified_probs,
    pil_image,
    plot_probs,
    probs,
    target,
):
    mo.hstack(
        [
            mo.vstack(
                [pil_image, mo.md(f"True class: {imgs_dataset.classes[target]}")],
                align="center",
            ),
            plot_probs(probs),
            plot_probs(modified_probs),
            # plot_probs(modified_probs - probs),
        ]
    )
    return


@app.cell
def __(imgs_dataset, pl, plt, torch):
    def plot_probs(probabilities):
        df = pl.from_dicts(
            [
                {"probability": value, "class": cls, "rank": i}
                for i, (value, cls) in enumerate(zip(*torch.topk(probabilities, k=5)))
            ],
            schema={"probability": float, "class": int, "rank": int},
        )

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.barh(-df["rank"], df["probability"])

        # Customize appearance
        ax.set_xlabel("Probability")
        ax.set_yticks(-df["rank"], [imgs_dataset.classes[i] for i in df["class"]])

        ax.set_xlim(0, 1.0)
        ax.spines[["right", "top"]].set_visible(False)

        fig.tight_layout()
        return fig
    return (plot_probs,)


@app.cell
def __(
    Float,
    Tensor,
    beartype,
    einops,
    features,
    jaxtyped,
    sae,
    sliders,
    torch,
):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.inference_mode
    def modify(act_D: Float[Tensor, " d_vit"]) -> Float[Tensor, " d_vit"]:
        x_hat_BD, f_x_BS, _ = sae(act_D[None, :])
        x_hat_D = x_hat_BD[0]
        f_x_S = f_x_BS[0]

        err_D = act_D - x_hat_D

        latents = [f.latent for f in features]
        values = torch.tensor(
            [f.unscaled(sliders.value[i]) for i, f in enumerate(features)]
        )
        modified_f_x_S = f_x_S.clone()
        modified_f_x_S[..., latents] = values

        # Reproduce the SAE forward pass after f_x
        modified_x_hat_D = (
            einops.einsum(modified_f_x_S, sae.W_dec, "d_sae, d_sae d_vit -> d_vit")
            + sae.b_dec
        )
        modified_D = modified_x_hat_D + err_D

        return modified_D
    return (modify,)


@app.cell
def __(
    Bool,
    Feature,
    Float,
    Tensor,
    act_D,
    beartype,
    n_features,
    sae,
    torch,
):
    @beartype.beartype
    @torch.inference_mode
    def get_agg_features(
        acts_MD: Float[Tensor, "m d_vit"],
        mask: Bool[Tensor, " d_sae"] = torch.ones(sae.cfg.d_sae, dtype=bool),
        n: int = n_features,
    ) -> list[Feature]:
        _, f_x_MS, _ = sae(acts_MD)

        f_x_S = f_x_MS.sum(axis=0)
        latents = torch.argsort(f_x_S, descending=True).cpu()

        latents = latents[mask[latents]][:n]

        _, f_x_1S, _ = sae(act_D[None, :])
        f_x_S = f_x_1S[0]
        vals = f_x_S[latents]

        # TODO: pick a good value for this.
        max_obs = 100.0

        return [
            Feature(latent.item(), max_obs, obs.item())
            for obs, latent in zip(vals, latents)
        ]
    return (get_agg_features,)


@app.cell
def __(Feature, act_D, beartype, n_features, sae, sparsity, torch):
    @beartype.beartype
    @torch.inference_mode
    def get_random_features() -> list[Feature]:
        _, f_x_1S, _ = sae(act_D[None, :])
        f_x_S = f_x_1S[0]

        max_obs = 100.0

        latents = torch.randperm(len(f_x_S))
        mask = sparsity < 1e-2
        latents = latents[mask[latents]][:n_features]

        values = f_x_S[latents]

        return [
            Feature(latent.item(), max_obs, obs.item())
            for obs, latent in zip(values, latents)
        ]
    return (get_random_features,)


@app.cell
def __(saev, v2):
    in1k_dataset = saev.activations.get_dataset(
        saev.config.ImagenetDataset(),
        img_transform=v2.Compose(
            [v2.Resize(size=(128, 128)), v2.CenterCrop(size=(112, 112))]
        ),
    )
    return (in1k_dataset,)


@app.cell
def __(beartype, dataclasses):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Feature:
        latent: int
        max_obs: float
        raw_obs: float

        @property
        def default(self) -> float:
            """Return raw_obs, scaled from [-max_obs, max_obs] to [-1, 1]."""
            return self.scaled(self.raw_obs)

        def unscaled(self, x: float) -> float:
            """Scale from [-1, 1] to [-max_obs, max_obs]."""
            return -self.max_obs + (x + 1) * (2 * self.max_obs) / 2

        def scaled(self, x: float) -> float:
            """Return raw_obs, scaled from [-max_obs, max_obs] to [-1, 1]."""
            return -1 + (x + self.max_obs) * 2 / (2 * self.max_obs)
    return (Feature,)


@app.cell
def __():
    import sys

    pkg_root = "/home/stevens.994/projects/saev"
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)
    import random
    import os.path
    import beartype
    import dataclasses
    from jaxtyping import jaxtyped, Float, Bool
    import marimo as mo
    import torch
    import contrib.classification.training
    import saev.nn
    import polars as pl
    import einops
    from torch import Tensor
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    import functools
    return (
        Bool,
        Float,
        Tensor,
        beartype,
        contrib,
        dataclasses,
        einops,
        functools,
        jaxtyped,
        mo,
        os,
        pkg_root,
        pl,
        plt,
        random,
        saev,
        sys,
        torch,
        v2,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
