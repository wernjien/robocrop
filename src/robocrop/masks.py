"""Loss masks for masked (loss-weighted) training.

A model of a body or outfit should not learn the subject's face, and hardly
any model should learn the backgrounds its photos happened to be taken against.
Painting over either teaches the model the paint. The pixels stay in the
crop instead, and a mask beside it tells the trainer how much each one
counts when scoring its output: white is learned in full, black not at all,
grey in proportion -- OneTrainer multiplies the loss by the mask value.

The file naming follows OneTrainer's convention -- ``0001.png`` pairs with
``0001-masklabel.png`` -- which OneTrainer picks up with masked training on.
``--mask-dir`` writes them to a separate folder instead, named like the crop,
for ai-toolkit and kohya.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
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
    return path.with_name(f"{path.stem}{MASK_SUFFIX}.png").as_posix()


def isolate_person(matte: np.ndarray, box: Rect) -> np.ndarray | None:
    """Keep only the part of ``matte`` that belongs to the detected person.

    The matte marks every foreground thing in the crop -- a bystander, a
    ball -- so only the blobs that reach into the detection ``box`` are kept.
    Kept blobs are grown by a hair before being cut out, so the soft edge of
    the matte survives instead of being clipped at the 50% line.

    Returns None when nothing in the matte touches the box.
    """
    h, w = matte.shape
    solid = (matte >= 0.5).astype(np.uint8)
    count, labels = cv2.connectedComponents(solid, connectivity=8)
    if count <= 1:
        return None

    x0, y0 = max(0, int(box.x)), max(0, int(box.y))
    x1, y1 = min(w, int(np.ceil(box.x2))), min(h, int(np.ceil(box.y2)))
    touching = np.unique(labels[y0:y1, x0:x1])
    touching = touching[touching != 0]
    if touching.size == 0:
        return None

    keep = np.isin(labels, touching).astype(np.uint8)
    grow = max(1, min(h, w) // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    keep = cv2.dilate(keep, kernel)
    return np.where(keep > 0, matte, 0.0).astype(np.float32)


def box_person(size: int | tuple[int, int], box: Rect) -> np.ndarray:
    """A hard-edged stand-in for a person matte: 1 inside the box, else 0."""
    w, h = _dims(size)
    person = np.zeros((h, w), dtype=np.float32)
    x0, y0 = max(0, int(box.x)), max(0, int(box.y))
    x1, y1 = min(w, int(np.ceil(box.x2))), min(h, int(np.ceil(box.y2)))
    person[y0:y1, x0:x1] = 1.0
    return person


def build_mask(
    size: int | tuple[int, int],
    faces: Sequence[Rect],
    margin: float,
    *,
    person: np.ndarray | None = None,
    background: float = 0.1,
    clothing: np.ndarray | None = None,
    clothing_weight: float = 0.0,
    face: np.ndarray | None = None,
) -> Image.Image:
    """A greyscale mask, ``size`` square or (width, height): white where the crop is learned.

    With ``person`` (a matte of the mask's shape, in [0, 1]), the background
    drops to the ``background`` weight and the person stays white, blending
    along the matte's soft edge. Without it, the whole crop starts white.
    With ``clothing`` (a matte of the same shape), clothing drops to
    ``clothing_weight`` the same way, and with ``face`` (a SegFace matte)
    the face drops to 0 along its own outline.

    Boxes in ``faces`` are explicit oval masks or parser fallbacks. Each
    becomes a black ellipse around its box, grown by
    ``margin`` of the box side on every side so hair, ears and jaw are
    covered too -- they carry identity as much as the eyes do. The edge is
    feathered outward, so the face itself stays fully black and the falloff
    eats into the surroundings rather than the other way round.
    """
    faces_mask = _face_mask(size, faces, margin)
    if person is None and clothing is None and face is None:
        return faces_mask

    weights = np.asarray(faces_mask, dtype=np.float32) / 255.0
    if person is not None:
        weights = np.minimum(weights, background + (1.0 - background) * np.clip(person, 0.0, 1.0))
    if clothing is not None:
        weights = np.minimum(weights, 1.0 - (1.0 - clothing_weight) * np.clip(clothing, 0.0, 1.0))
    if face is not None:
        weights = np.minimum(weights, 1.0 - np.clip(face, 0.0, 1.0))
    return Image.fromarray(np.round(weights * 255.0).astype(np.uint8))


def _dims(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else size


def _face_mask(size: int | tuple[int, int], faces: Sequence[Rect], margin: float) -> Image.Image:
    mask = Image.new("L", _dims(size), 255)
    if not faces:
        return mask

    draw = ImageDraw.Draw(mask)
    # One blur for the whole mask, sized to the largest face. Every ellipse
    # is grown by that same radius, so the blur's falloff sits outside each
    # intended ellipse and everything inside stays solid -- a bystander's
    # small face included, which a per-face radius would leave half grey.
    radius = 0.08 * max(max(box.w, box.h) for box in faces)
    for box in faces:
        side = max(box.w, box.h)
        half_w = box.w / 2.0 + margin * side
        half_h = box.h / 2.0 + margin * side
        cx, cy = box.cx, box.cy - _LIFT * box.h
        draw.ellipse(
            (cx - half_w - radius, cy - half_h - radius,
             cx + half_w + radius, cy + half_h + radius),
            fill=0,
        )

    return mask.filter(ImageFilter.GaussianBlur(radius / 2.0)) if radius >= 1 else mask
