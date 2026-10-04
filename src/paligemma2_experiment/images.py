from __future__ import annotations

from PIL import Image


def degrade_then_restore(
    image: Image.Image, low_resolution: int = 224, target_resolution: int = 448
) -> Image.Image:
    """Remove fine detail while preserving the final input dimensions."""
    square = image.convert("RGB").resize(
        (low_resolution, low_resolution), Image.Resampling.BICUBIC
    )
    return square.resize(
        (target_resolution, target_resolution), Image.Resampling.BICUBIC
    )
