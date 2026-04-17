from __future__ import annotations

from PIL import Image


def ensure_mask_is_single_channel(mask: Image.Image) -> Image.Image:
    return mask.convert("L")


def apply_binary_mask_to_image(image: Image.Image, mask: Image.Image, fill_value: int = 0) -> Image.Image:
    """
    Keep only the unmasked region from `image`.

    Assumption for the scaffold:
    - white mask pixels indicate the editable/inpaint region
    - editable region is removed from the source image

    TODO:
    - verify mask polarity on the final local dataset
    - optionally support inverted mask logic from config
    """

    image = image.convert("RGB")
    mask = ensure_mask_is_single_channel(mask)
    output = Image.new("RGB", image.size, (fill_value, fill_value, fill_value))
    keep_mask = Image.eval(mask, lambda px: 255 - px)
    output.paste(image, mask=keep_mask)
    return output
