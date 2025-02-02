import base64

import beartype
from PIL import Image


@beartype.beartype
def vips_to_base64(img_v: Image.Image) -> str:
    buf = img_v.write_to_buffer(".webp")
    b64 = base64.b64encode(buf)
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64
