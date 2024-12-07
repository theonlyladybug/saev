import marimo

__generated_with = "0.9.20"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    import sys

    pkg_root = "/home/stevens.994/projects/saev"
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)

    import os.path
    import beartype
    import dataclasses
    from jaxtyping import jaxtyped, Float
    import marimo as mo
    import torch
    import contrib.classification.training
    import saev.nn
    import polars as pl
    import einops
    import matplotlib.pyplot as plt
    from torchvision.transforms import v2
    return (
        Float,
        beartype,
        contrib,
        dataclasses,
        einops,
        jaxtyped,
        mo,
        os,
        pkg_root,
        pl,
        plt,
        saev,
        sys,
        torch,
        v2,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        This is a dashboard to explore how you can use sparse autoencoders to manipulate model classifications. It presents a couple different ways to automatically propose features that you might want to manipulate.

        How do we propose features?

        1. Pick the 3 features that maximally activate on this image.
        2. Pick the 3 features that maximally activate on all images of this class.
        3. Pick 3 random features.

        For all of these features, show the 8 highest activating examples from ImageNet-1K.
        """
    )
    return


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
def __(contrib, saev):
    clf_ckpt_fpath = "checkpoints/contrib/classification/lr_0_001__wd_0_001/model.pt"
    clf = contrib.classification.training.load_model(clf_ckpt_fpath)
    clf.eval()

    sae_ckpt_fpath = "checkpoints/bd97z80b/sae.pt"
    sae = saev.nn.load(sae_ckpt_fpath)
    sae.eval()

    imgs_test_dpath = "/research/nfs_su_809/workspace/stevens.994/datasets/flowers102/test"
    imgs_dataset = saev.activations.ImageFolder(imgs_test_dpath)

    acts_test_dpath = "/local/scratch/stevens.994/cache/saev/00a598c97730b46d48aa632804c0c72e236aca07f33492d3d1deaca68d26dfb8"
    acts_cfg = saev.config.DataLoad(shard_root=acts_test_dpath, patches="cls")
    acts_dataset = saev.activations.Dataset(acts_cfg)


    assert len(imgs_dataset) == len(acts_dataset)
    return (
        acts_cfg,
        acts_dataset,
        acts_test_dpath,
        clf,
        clf_ckpt_fpath,
        imgs_dataset,
        imgs_test_dpath,
        sae,
        sae_ckpt_fpath,
    )


@app.cell
def __(os, torch):
    sae_data_root = (
        "/research/nfs_su_809/workspace/stevens.994/saev/features/bd97z80b/sort_by_img"
    )

    top_img_i = torch.load(os.path.join(sae_data_root, "top_img_i.pt"), weights_only=True)
    top_values = torch.load(os.path.join(sae_data_root, "top_values.pt"), weights_only=True)
    return sae_data_root, top_img_i, top_values


@app.cell
def __(imgs_dataset, mo):
    example_num = mo.ui.number(start=1, stop=len(imgs_dataset), step=1, label="Example")
    return (example_num,)


@app.cell
def __(example_num):
    example_num
    return


@app.cell
def __(Feature, beartype, mo):
    @beartype.beartype
    def make_slider(feature: Feature) -> tuple[object, object]:
        slider = mo.ui.slider(start=-1, stop=1, step=0.1, value=feature.default)
        btn = mo.ui.button(label="Reset", kind="danger")
        return slider, btn
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
def __(
    Float,
    Tensor,
    acts_dataset,
    beartype,
    imgs_dataset,
    jaxtyped,
    targets,
    torch,
):
    @jaxtyped(typechecker=beartype.beartype)
    def get_cls_acts(i) -> Float[Tensor, "n d_vit"]:
        target = imgs_dataset[i]["target"]
        js = torch.nonzero(targets == target).view(-1).tolist()
        examples = torch.stack([acts_dataset[j]["act"] for j in js])
        return examples
    return (get_cls_acts,)


@app.cell
def __(acts_dataset, example_num, get_cls_acts):
    act_D = acts_dataset[example_num.value - 1]["act"]
    acts_ND = get_cls_acts(example_num.value - 1)
    return act_D, acts_ND


@app.cell
def __(
    get_cls_specific_features,
    get_img_specific_features,
    get_random_features,
):
    features = (
        get_img_specific_features() + get_cls_specific_features() + get_random_features()
    )
    return (features,)


@app.cell
def __(features, make_slider, mo):
    sliders, btns = zip(*[make_slider(f) for f in features])
    sliders = mo.ui.array(sliders)
    btns = mo.ui.array(btns)
    return btns, sliders


@app.cell
def __(btns, features, in1k_dataset, mo, sliders, top_img_i):
    def make_sliders_ui():
        rows = []
        for slider, btn, feature in zip(sliders, btns, features):
            img_i = top_img_i[feature.latent, :4].tolist()
            imgs = [in1k_dataset[i]["image"] for i in img_i]
            items = {"Example Images": mo.hstack(imgs, justify="start")}
            row = [
                mo.hstack([slider, btn, f"{slider.value:.1f}"], justify="start"),
                mo.accordion(items, multiple=True),
            ]

            rows.append(mo.vstack(row))

        return mo.hstack(
            [mo.vstack(rows[0:3]), mo.vstack(rows[3:6]), mo.vstack(rows[6:9])],
            justify="start",
        )


    make_sliders_ui()
    return (make_sliders_ui,)


@app.cell
def __(act_D, clf, mo, modify, torch):
    with torch.inference_mode():
        modified_act_D = modify(act_D)
        logits_C = clf(act_D)
        modified_logits_C = clf(modified_act_D)

        probs = torch.nn.functional.softmax(logits_C, dim=0)
        modified_probs = torch.nn.functional.softmax(modified_logits_C, dim=0)

    mo.md(f"Mean difference: {(modified_probs - probs).abs().mean().item():.3f}")
    return (
        logits_C,
        modified_act_D,
        modified_logits_C,
        modified_probs,
        probs,
    )


@app.cell
def __(example_num, imgs_dataset, mo, modified_probs, plot_probs, probs):
    mo.hstack(
        [
            imgs_dataset[example_num.value - 1]["image"],
            plot_probs(probs),
            plot_probs(modified_probs),
            plot_probs(modified_probs - probs),
        ]
    )
    return


@app.cell
def __(classnames, pl, plt, torch):
    def plot_probs(probabilities):
        df = pl.from_dicts(
            [
                {"probability": value, "class": cls, "rank": i}
                for i, (value, cls) in enumerate(zip(*torch.topk(probabilities, k=10)))
            ],
            schema={"probability": float, "class": int, "rank": int},
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.barh(-df["rank"], df["probability"])

        # Customize appearance
        ax.set_xlabel("Probability")
        ax.set_yticks(-df["rank"], [classnames[i] for i in df["class"]])

        ax.set_xlim(0, 1.0)
        ax.spines[["right", "top"]].set_visible(False)

        fig.tight_layout()
        return fig
    return (plot_probs,)


@app.cell
def __(act_D, features, sae, sliders, torch):
    @torch.inference_mode
    def fn():
        _, f_x_BS, _ = sae(act_D[None, :])
        f_x_S = f_x_BS[0]

        for i, f in enumerate(features):
            print(sliders.value[i], f.unscaled(sliders.value[i]), f_x_S[f.latent].item())


    fn()
    return (fn,)


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
def __(Feature, act_D, beartype, sae, torch):
    @beartype.beartype
    @torch.inference_mode
    def get_img_specific_features() -> list[Feature]:
        _, f_x_1S, _ = sae(act_D[None, :])
        f_x_S = f_x_1S[0]

        max_obs = 100.0

        return [
            Feature(latent.item(), max_obs, obs.item())
            for obs, latent in zip(*torch.topk(f_x_S, k=3))
        ]
    return (get_img_specific_features,)


@app.cell
def __(Feature, act_D, acts_ND, beartype, sae, torch):
    @beartype.beartype
    @torch.inference_mode
    def get_cls_specific_features() -> list[Feature]:
        _, f_x_NS, _ = sae(acts_ND)

        f_x_S = f_x_NS.sum(axis=0)
        _, latents = torch.topk(f_x_S, k=3)

        _, f_x_1S, _ = sae(act_D[None, :])
        f_x_S = f_x_1S[0]
        vals = f_x_S[latents]

        max_obs = 100.0

        return [
            Feature(latent.item(), max_obs, obs.item())
            for obs, latent in zip(vals, latents)
        ]
    return (get_cls_specific_features,)


@app.cell
def __(Feature, act_D, beartype, sae, torch):
    @beartype.beartype
    @torch.inference_mode
    def get_random_features() -> list[Feature]:
        _, f_x_1S, _ = sae(act_D[None, :])
        f_x_S = f_x_1S[0]

        max_obs = 100.0

        latents = torch.randperm(len(f_x_S))[:3]
        values = f_x_S[latents]

        return [
            Feature(latent.item(), max_obs, obs.item())
            for obs, latent in zip(values, latents)
        ]
    return (get_random_features,)


@app.cell
def __():
    classnames = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink and yellow dahlia",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "air plant",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]
    return (classnames,)


@app.cell
def __(saev, v2):
    in1k_dataset = saev.activations.get_dataset(
        saev.config.ImagenetDataset(),
        img_transform=v2.Compose(
            [v2.Resize(size=(128, 128)), v2.CenterCrop(size=(112, 112))]
        ),
    )
    return (in1k_dataset,)


if __name__ == "__main__":
    app.run()
