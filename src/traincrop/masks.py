"""Face masks for masked (loss-weighted) LoRA training.

A body or outfit LoRA should not learn the subject's face, but painting over
the face teaches the model faceless people. The face stays in the crop
instead, and a mask beside it tells the trainer to ignore those pixels when
scoring its output: white is learned, black is not.

The file naming follows OneTrainer's convention -- ``0001.png`` pairs with
``0001-masklabel.png`` -- which OneTrainer picks up with masked training on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFilter

from .geometry import Rect

MASK_SUFFIX = "-masklabel"

#: How far the ellipse's centre sits above the face box's, as a fraction of
#: the box height. YuNet boxes brow-to-chin, so the head (hair included)
#: extends further above the box than below it.
_LIFT = 0.12


def mask_name(image_file: str) -> str:
    """``3/0001.jpg`` -> ``3/0001-masklabel.png``. Always PNG: a mask must be
    lossless, or JPEG ringing leaks grey weight into the ignored area."""
    path = Path(image_file)
    return str(path.with_name(f"{path.stem}{MASK_SUFFIX}.png"))


def build_mask(size: int, faces: Sequence[Rect], margin: float) -> Image.Image:
    """A ``size`` x ``size`` greyscale mask, white except over each face.

    Each face becomes an ellipse around its box, grown by ``margin`` of the
    box side on every side so hair, ears and jaw are covered too -- they carry
    identity as much as the eyes do. The edge is feathered outward, so the
    face itself stays fully black and the falloff eats into the surroundings
    rather than the other way round.
    """
    mask = Image.new("L", (size, size), 255)
    if not faces:
        return mask

    draw = ImageDraw.Draw(mask)
    feather = 0.0
    for box in faces:
        side = max(box.w, box.h)
        half_w = box.w / 2.0 + margin * side
        half_h = box.h / 2.0 + margin * side
        cx, cy = box.cx, box.cy - _LIFT * box.h
        # Grow by the feather radius before blurring, so the blur's falloff
        # sits outside the intended ellipse and everything inside stays solid.
        radius = 0.08 * side
        feather = max(feather, radius)
        draw.ellipse(
            (cx - half_w - radius, cy - half_h - radius,
             cx + half_w + radius, cy + half_h + radius),
            fill=0,
        )

    return mask.filter(ImageFilter.GaussianBlur(feather / 2.0)) if feather >= 1 else mask
