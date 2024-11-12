import math

import beartype
import numpy as np
from jaxtyping import Float, jaxtyped
from PIL import Image, ImageDraw


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
        alpha = int(val / (upper + 1e-9) * 128)
        x_np, y_np = p % iw_np, p // ih_np
        draw.rectangle(
            [
                (x_np * pw_px, y_np * ph_px),
                (x_np * pw_px + pw_px, y_np * ph_px + ph_px),
            ],
            fill=(255, 0, 0, alpha),
        )

    # Composite the original image and the overlay
    return Image.alpha_composite(img.convert("RGBA"), overlay)
