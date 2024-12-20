import math
import os

import beartype
import einops.layers.torch
import tyro
from PIL import Image, ImageDraw
from torchvision.transforms import v2

default_highlighted_i = [
    202,
    203,
    204,
    205,
    206,
    207,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
]


@beartype.beartype
def add_highlights(img: Image.Image, patches: list[bool]) -> Image.Image:
    if not len(patches):
        return img

    iw_np, ih_np = int(math.sqrt(len(patches))), int(math.sqrt(len(patches)))
    iw_px, ih_px = img.size
    pw_px, ph_px = iw_px // iw_np, ih_px // ih_np
    assert iw_np * ih_np == len(patches)

    # Create a transparent overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for p, highlight in enumerate(patches):
        if not highlight:
            continue
        x_np, y_np = p % iw_np, p // ih_np

        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(225, 29, 72, 128),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)


@beartype.beartype
def make_figure_semseg(
    example_i: int = 3122,
    patchified_i: list[int] = [0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255],
    highlighted_i: list[int] = default_highlighted_i,
    out: str = os.path.join(".", "logs", "figures"),
    ade20k: str = "/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/",
):
    """
    Parts of the figure that need to be programmatically generated:

    * Split image into image patches. They should be saved as patch1.png, patch2.png, etc. Only patches 1, 2, 3, and 254, 255 are required. If at all possible, they should be larger than 16x16 (resize to 512x512, crop to 448x448, then patchify to 32x32).

    * Image with specific patches highlighted. Use highlighted_i to choose the patches. Again, should be saved at a 448x448 resolution.

    * Semantic segmentation prediction before modification. Classes should be specific colors rather than random colors.
    """
    os.makedirs(out, exist_ok=True)

    import saev.activations
    import saev.config

    to_array = v2.Compose([
        v2.Resize(512, interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((448, 448)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("channels width height -> width height channels"),
    ])

    ade20k_dataset = saev.activations.Ade20k(
        saev.config.Ade20kDataset(root=ade20k),
        img_transform=to_array,
        seg_transform=to_array,
    )

    img_arr = ade20k_dataset[example_i]["image"]

    # Split into 28x28 pixel image patches and save patches `patchified_i` as ade20k_example{i}_patch{p} to out
    patch_size = 28
    n_patch_per_side = 16
    for p in patchified_i:
        row = (p // n_patch_per_side) * patch_size
        col = (p % n_patch_per_side) * patch_size
        patch = img_arr[row : row + patch_size, col : col + patch_size]
        patch_img = Image.fromarray(patch.numpy())

        # Highlight some patches
        if p in highlighted_i:
            overlay = Image.new("RGBA", patch_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([(0, 0), patch_img.size], fill=(225, 29, 72, 128))
            patch_img = Image.alpha_composite(patch_img.convert("RGBA"), overlay)

        patch_img.save(os.path.join(out, f"ade20k_example{example_i}_patch{p}.png"))

    # Save image with highlighted_i patches highlighted.
    bool_patches = [i in highlighted_i for i in range(256)]
    highlighted_img = add_highlights(Image.fromarray(img_arr.numpy()), bool_patches)
    highlighted_img.save(os.path.join(out, f"ade20k_highlighted_img{example_i}.png"))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "semseg": make_figure_semseg,
        "no-op": lambda: print("no op"),
    })
