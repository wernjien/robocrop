"""Image discovery, loading and crop extraction."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .geometry import CropPlan

# Large scans of camera originals trip Pillow's decompression-bomb guard; these
# are the user's own files, so raise the ceiling rather than skip them.
Image.MAX_IMAGE_PIXELS = None

#: Extensions understood without optional plugins.
BASE_EXTENSIONS = (
    ".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".ppm",
)

#: Added when pillow-heif is installed (iPhone libraries need these).
HEIF_EXTENSIONS = (".heic", ".heif", ".avif")

FillMode = Literal["edge", "blur", "reflect", "color"]

_heif_ready = False


def register_plugins() -> tuple[str, ...]:
    """Enable optional format plugins. Returns every supported extension."""
    global _heif_ready
    if not _heif_ready:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            try:
                pillow_heif.register_avif_opener()
            except AttributeError:
                pass  # older pillow-heif has no AVIF support
            _heif_ready = True
        except ImportError:
            pass
    return BASE_EXTENSIONS + (HEIF_EXTENSIONS if _heif_ready else ())


def iter_images(
    root: Path,
    *,
    extensions: tuple[str, ...] | None = None,
    exclude: tuple[str, ...] = (),
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Walk ``root`` recursively, yielding image paths in a stable order.

    Sorted at every level so output numbering is reproducible across runs and
    across machines, which ``os.walk`` alone does not guarantee.
    """
    exts = tuple(e.lower() for e in (extensions or register_plugins()))
    root = root.expanduser().resolve()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        here = Path(dirpath)
        # Skip dot-directories and anything excluded, and prune in place so
        # os.walk never descends into them.
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and not _matches(here / d, root, exclude)
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue  # dotfiles and macOS ._ resource forks
            path = here / name
            if path.suffix.lower() not in exts:
                continue
            if _matches(path, root, exclude):
                continue
            yield path


def _matches(path: Path, root: Path, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    name = path.name
    return any(
        fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(name, p) for p in patterns
    )


def load_image(path: Path) -> Image.Image:
    """Open an image upright and in RGB.

    EXIF rotation is applied first: a portrait photo stored as landscape plus
    an orientation tag would otherwise be detected sideways, and faces would be
    missed entirely.
    """
    with Image.open(path) as handle:
        handle.load()
        upright = ImageOps.exif_transpose(handle)
        return upright.convert("RGB") if upright.mode != "RGB" else upright.copy()


def to_bgr(image: Image.Image) -> np.ndarray:
    """PIL RGB -> contiguous OpenCV BGR array."""
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[:, :, ::-1])


def measure_sharpness(image: Image.Image, *, inner_fraction: float = 1.0) -> float:
    """Variance of the Laplacian over the central ``inner_fraction`` of the
    image: a standard blur heuristic (higher is sharper). Restricting to a
    central window avoids padding/background dragging the score down on an
    otherwise-sharp crop.
    """
    w, h = image.size
    if inner_fraction < 1.0:
        margin_w = int(w * (1.0 - inner_fraction) / 2.0)
        margin_h = int(h * (1.0 - inner_fraction) / 2.0)
        image = image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    gray = np.array(image.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(
    image: Image.Image,
    plan: CropPlan,
    *,
    fill: FillMode = "blur",
    fill_color: tuple[int, int, int] = (0, 0, 0),
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Cut the planned window out of ``image`` and resize it to the tier.

    When the window extends past the edge (``edge='extend'``), the missing area
    is manufactured according to ``fill`` rather than left transparent.
    """
    left, top, right, bottom = plan.rect.rounded()
    width, height = image.size

    inside = left >= 0 and top >= 0 and right <= width and bottom <= height
    if inside:
        crop = image.crop((left, top, right, bottom))
    else:
        crop = _crop_extended(
            image, (left, top, right, bottom), fill=fill, fill_color=fill_color
        )

    if crop.size != (plan.tier, plan.tier):
        crop = crop.resize((plan.tier, plan.tier), resample)
    return crop


def _crop_extended(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: FillMode,
    fill_color: tuple[int, int, int],
) -> Image.Image:
    left, top, right, bottom = box
    out_w, out_h = right - left, bottom - top

    if fill == "color":
        canvas = Image.new("RGB", (out_w, out_h), fill_color)
        # Paste whatever part of the window really exists, at its offset.
        src = image.crop(
            (max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height))
        )
        canvas.paste(src, (max(-left, 0), max(-top, 0)))
        return canvas

    if fill == "reflect":
        padded, origin = _pad_array(image, box, mode="reflect")
    elif fill == "edge":
        padded, origin = _pad_array(image, box, mode="edge")
    else:  # blur: replicate the edge, then blur so the invention is not sharp
        padded, _ = _pad_array(image, box, mode="edge")
        padded = _blur_outside(padded, image, box)

    return padded


def _pad_array(
    image: Image.Image, box: tuple[int, int, int, int], *, mode: str
) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = box
    pad_l = max(-left, 0)
    pad_t = max(-top, 0)
    pad_r = max(right - image.width, 0)
    pad_b = max(bottom - image.height, 0)

    arr = np.asarray(image, dtype=np.uint8)
    # numpy's reflect needs at least one real row/column to mirror; fall back
    # to edge replication on degenerate inputs rather than raising.
    if mode == "reflect" and (
        pad_l >= image.width or pad_r >= image.width
        or pad_t >= image.height or pad_b >= image.height
    ):
        mode = "edge"
    padded = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)

    full = Image.fromarray(padded)
    # Coordinates inside the padded canvas where the requested window starts.
    ox, oy = left + pad_l, top + pad_t
    return full.crop((ox, oy, ox + (right - left), oy + (bottom - top))), (pad_l, pad_t)


def _blur_outside(
    crop: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Blur the invented border, keeping the real pixels sharp."""
    left, top, right, bottom = box
    radius = max(4, min(crop.size) // 24)
    blurred = crop.filter(ImageFilter.GaussianBlur(radius))

    # Where the genuine image sits within the crop.
    inner = (
        max(0, -left),
        max(0, -top),
        max(0, -left) + min(right, image.width) - max(left, 0),
        max(0, -top) + min(bottom, image.height) - max(top, 0),
    )
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return blurred
    blurred.paste(crop.crop(inner), (inner[0], inner[1]))
    return blurred
