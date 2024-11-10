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
    import marimo as mo
    import numpy as np
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
        jaxtyped,
        math,
        mo,
        np,
        os,
        typing,
    )


@app.cell
def __():
    root = "/research/nfs_su_809/workspace/stevens.994/datasets/broden"
    return (root,)


@app.cell
def __(np):
    colors = np.array(
        [
            [0, 0, 0],
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [127, 0, 0],
            [0, 127, 0],
            [0, 0, 127],
            [63, 0, 0],
            [0, 63, 0],
            [0, 0, 63],
            [191, 0, 0],
            [0, 191, 0],
            [0, 0, 191],
            [63, 191, 0],
            [191, 0, 63],
            [0, 63, 191],
            [191, 63, 0],
            [63, 0, 191],
            [0, 191, 63],
            [191, 0, 0],
            [0, 191, 0],
            [0, 0, 191],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 127, 0],
            [255, 0, 127],
            [0, 255, 127],
            [127, 255, 0],
            [127, 0, 255],
            [0, 127, 255],
            [127, 127, 0],
            [127, 0, 127],
            [0, 127, 127],
            [127, 127, 127],
        ]
    )
    colors.shape
    return (colors,)


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
def __(beartype, csv, dataclasses, os, root):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Material:
        code: int
        number: int
        name: str
        frequency: int

        @classmethod
        def from_row_dict(cls, row: dict[str, str]) -> "Material":
            return cls(
                int(row["code"]), int(row["number"]), row["name"], int(row["frequency"])
            )


    def load_materials() -> list[Material]:
        with open(os.path.join(root, "c_material.csv")) as fd:
            materials = [Material.from_row_dict(row) for row in csv.DictReader(fd)]
        return materials


    materials = load_materials()
    material_lookup = {material.code: material for material in materials}
    materials[:2]
    return Material, load_materials, material_lookup, materials


@app.cell
def __(Image, beartype, dataclasses, np, os, root, typing):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Record:
        image: str
        is_train: bool
        height: int
        width: int
        segment_height: int
        segment_width: int
        colors: list[str] = dataclasses.field(default_factory=list)
        objects: list[str] = dataclasses.field(default_factory=list)
        parts: list[str] = dataclasses.field(default_factory=list)
        materials: list[str] = dataclasses.field(default_factory=list)
        textures: list[int] = dataclasses.field(default_factory=list)

        @classmethod
        def from_row_dict(cls, row: dict[str, str]) -> "Record":
            if row["texture"]:
                textures = [int(i) for i in row["texture"].split(";")]
            else:
                textures = []

            return cls(
                row["image"],
                row["split"] == "train",
                int(row["ih"]),
                int(row["iw"]),
                int(row["sh"]),
                int(row["sw"]),
                row["color"].split(";") if row["color"] else [],
                row["object"].split(";") if row["object"] else [],
                row["part"].split(";") if row["part"] else [],
                row["material"].split(";") if row["material"] else [],
                [int(i) for i in row["texture"].split(";")] if row["texture"] else [],
            )

        @property
        def dataset(self) -> str:
            dataset, *rest = self.image.split("/")
            return dataset

        def to_image(self, field: str, lookup: dict[int, typing.Any]):
            assert field in ("colors", "objects", "parts", "materials")

            fnames = getattr(self, field)

            dst = Image.new(
                "RGB",
                (self.segment_width * (len(fnames) + 1), self.segment_height),
                (255, 255, 255),
            )
            dst.paste(
                Image.open(os.path.join(root, "images", self.image)).resize(
                    (
                        self.segment_width,
                        self.segment_height,
                    )
                ),
                (0, 0),
            )

            for i, fname in enumerate(fnames):
                fpath = os.path.join(root, "images", fname)
                raw = np.asarray(Image.open(fpath)).copy().astype(np.uint32)

                if (raw == 0).all():
                    continue

                for j, color in enumerate(colors):
                    if j not in lookup:
                        continue
                    raw[(raw[:, :, 0] + raw[:, :, 1] * 256) == lookup[j].number] = color

                dst.paste(
                    Image.fromarray(raw.astype(np.uint8)),
                    ((i + 1) * self.segment_width, 0),
                )

            return dst
    return (Record,)


@app.cell
def __(Record, csv, os, root):
    with open(os.path.join(root, "index.csv")) as fd:
        records = [Record.from_row_dict(row) for row in csv.DictReader(fd)]

    # np.random.default_rng(seed=42).shuffle(records)
    records[:5]
    return fd, records


@app.cell
def __(material_lookup, records):
    [
        (record.to_image("materials", material_lookup), record.image)
        for record in records[:100]
        if record.materials
    ]
    return


@app.cell
def __(mo):
    mo.md(r"""Given the segmentation masks at 112x112, I want to know which patches in a 224x224 image (no center crop because images are already 224x224) are more than 50% filled with a given material (eventually not just material, but any "category").""")
    return


@app.cell
def __():
    import einops
    return (einops,)


@app.cell
def __(Image, einops, np, os, records, root):
    im = np.array(Image.open(os.path.join(root, "images", records[0].image)))
    einops.rearrange(im, "(w pw) (h ph) c -> (w h) (c pw ph)", pw=16, ph=16)
    return (im,)


@app.cell
def __(np):
    w, h = 224, 224
    pw, ph = 14, 14

    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    return h, ph, pw, w, xv, yv


@app.cell
def __(pw, xv):
    xv // pw
    return


@app.cell
def __(ph, yv):
    yv // ph
    return


@app.cell
def __(h, ph, pw, xv, yv):
    patch_lookup = (xv // pw) + (yv // ph) * (h // ph)
    patch_lookup
    return (patch_lookup,)


@app.cell
def __(patch_lookup):
    patch_lookup.shape
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
def __(Shaped, np):
    def double(x: Shaped[np.ndarray, "w h"]) -> Shaped[np.ndarray, "w*2 h*2"]:
        w, h = x.shape
        return np.repeat(np.repeat(x, np.full((w,), 2), axis=0), np.full((h,), 2), axis=1)
    return (double,)


@app.cell
def __(Image, Record, double, np, os, patch_lookup, ph, pw, records, root):
    def get_patches(record: Record, material_number: int) -> list[int]:
        raw = np.array(
            Image.open(os.path.join(root, "images", record.materials[0]))
        ).astype(np.uint32)
        nums = raw[:, :, 1] * 256 + raw[:, :, 0]
        nums = double(nums)

        x, y = np.where(nums == material_number)
        patches, counts = np.unique(patch_lookup[x, y], return_counts=True)
        return patches[counts > (pw * ph / 2)]


    get_patches(records[0], 202)
    return (get_patches,)


@app.cell
def __(
    Float,
    Image,
    ImageDraw,
    beartype,
    get_patches,
    jaxtyped,
    math,
    np,
    os,
    records,
    root,
):
    @jaxtyped(typechecker=beartype.beartype)
    def add_highlights(
        img: Image.Image,
        patches: Float[np.ndarray, " n_patches"],
        *,
        upper: float | None = None,
    ) -> Image.Image:
        iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
        iw_px, ih_px = img.size
        pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
        assert iw_np * ih_np == len(patches)

        # Create a transparent red overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Using semi-transparent red (255, 0, 0, alpha)
        for p, val in enumerate(patches):
            assert upper is not None
            alpha = int(val / upper * 128)
            x_np, y_np = p % iw_np, p // ih_np
            draw.rectangle(
                [
                    (x_np * pw_px, y_np * ph_px),
                    (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
                ],
                fill=(255, 0, 0, alpha),
            )

        # Composite the original image and the overlay
        return Image.alpha_composite(img, overlay)


    patches = np.zeros((256,), dtype=float)
    patches[get_patches(records[28794], 69)] = 1.0

    add_highlights(
        Image.open(os.path.join(root, "images", records[28794].image)).convert("RGBA"),
        patches,
        upper=1.0,
    )
    return add_highlights, patches


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
def __(records):
    records[29864]
    return


@app.cell
def __(Image, add_highlights, np, os, records, root):
    my_patches = np.zeros(256)
    my_patches[18] = 1

    add_highlights(
        Image.open(os.path.join(root, "images", records[28794].image)).convert("RGBA"),
        my_patches,
        upper=1.0,
    )
    return (my_patches,)


@app.cell
def __(mo):
    mo.md(
        r"""
        So given the validation images, pick the top 16 patches that the best latent activates the most for, then get those images.

        Then get all the true patches and highlight them in blue.

        Then get all the SAE scores and highlight them in red (varying intensity) (on a different image)

        Then save the pair of those images under "true_positives" or something.

        Then repeat this for true_negatives, false_positives, false_negatives.
        """
    )
    return


if __name__ == "__main__":
    app.run()
