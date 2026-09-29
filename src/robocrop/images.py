"""Image discovery, loading and crop extraction."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import superres
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

#: Stems that mark training side-files, never photos: OneTrainer's loss masks
#: and conditioning images. Pointing --input at a dataset must not crop them.
_SIDE_FILE_SUFFIXES = ("-masklabel", "-condlabel")

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
    skip_dirs: tuple[Path, ...] = (),
    skip_marker: str = "",
) -> Iterator[Path]:
    """Walk ``root`` recursively, yielding image paths in a stable order.

    Sorted at every level so output numbering is reproducible across runs and
    across machines, which ``os.walk`` alone does not guarantee. Directories
    in ``skip_dirs``, and subdirectories holding a file named ``skip_marker``,
    are never entered.
    """
    exts = tuple(e.lower() for e in (extensions or register_plugins()))
    root = root.expanduser().resolve()
    skipped = {Path(d).expanduser().resolve() for d in skip_dirs}
    visited: set[tuple[int, int]] = set()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        here = Path(dirpath)
        if follow_symlinks:
            # A symlink back up the tree would otherwise loop forever.
            st = os.stat(here)
            if (st.st_dev, st.st_ino) in visited:
                dirnames[:] = []
                continue
            visited.add((st.st_dev, st.st_ino))
        # Skip dot-directories and anything excluded, and prune in place so
        # os.walk never descends into them.
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".")
            and not _matches(here / d, root, exclude)
            and not (skipped and (here / d).resolve() in skipped)
            and not (skip_marker and (here / d / skip_marker).is_file())
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue  # dotfiles and macOS ._ resource forks
            path = here / name
            if path.suffix.lower() not in exts:
                continue
            if path.stem.endswith(_SIDE_FILE_SUFFIXES):
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
        # exif_transpose already returns a new image, detached from the file,
        # so no further copy is needed -- on a 40 MP photo that is 120 MB.
        upright = ImageOps.exif_transpose(handle)
        return upright.convert("RGB") if upright.mode != "RGB" else upright


def to_bgr(image: Image.Image) -> np.ndarray:
    """PIL RGB -> contiguous OpenCV BGR array, in one SIMD pass."""
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)


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
    upscale: bool = False,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Cut the planned window out of ``image`` and resize it to the tier.

    When the window extends past the edge (``edge='extend'``), the missing area
    is manufactured according to ``fill`` rather than left transparent. A crop
    smaller than the tier is enlarged with AI upscaling when
    ``upscale`` is set, and with a plain resize otherwise; a crop that
    already meets or exceeds the tier is always just resized down to it.
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

    size = plan.output_size
    if max(crop.size) < plan.tier and upscale:
        crop = _upscale_to(crop, size)
    elif crop.size != size:
        # reducing_gap box-reduces a large downscale (6x and up, a 3900 px
        # crop to 512) before the Lanczos pass: ~2.6x faster, and no pixel
        # moves by more than 1/255. Smaller reductions are unaffected.
        crop = crop.resize(size, resample, reducing_gap=3.0)
    return crop


def _upscale_to(crop: Image.Image, size: tuple[int, int]) -> Image.Image:
    enlarged = superres.upscale(to_bgr(crop))
    result = Image.fromarray(enlarged[:, :, ::-1], mode="RGB")
    if result.size != size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


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
    width, height = image.size
    pad_l = max(-left, 0)
    pad_t = max(-top, 0)
    pad_r = max(right - width, 0)
    pad_b = max(bottom - height, 0)

    # numpy's reflect needs at least one real row/column to mirror; fall back
    # to edge replication on degenerate inputs rather than raising.
    if mode == "reflect" and (
        pad_l >= width or pad_r >= width or pad_t >= height or pad_b >= height
    ):
        mode = "edge"

    # Pad only the slice of the image the window needs, not the whole photo:
    # on a 40 MP source, padding everything copied ~120 MB per crop. The
    # slice keeps the true image edge on every padded side, and for reflect
    # also the pixels the mirror reads back from, so the result is identical.
    x0, x1 = _source_span(left, right, width, pad_l, pad_r, mode)
    y0, y1 = _source_span(top, bottom, height, pad_t, pad_b, mode)
    arr = np.asarray(image.crop((x0, y0, x1, y1)), dtype=np.uint8)
    padded = np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)

    full = Image.fromarray(padded)
    # Coordinates inside the padded canvas where the requested window starts.
    ox, oy = left - x0 + pad_l, top - y0 + pad_t
    return full.crop((ox, oy, ox + (right - left), oy + (bottom - top))), (pad_l, pad_t)


def _source_span(
    start: int, stop: int, size: int, pad_lo: int, pad_hi: int, mode: str
) -> tuple[int, int]:
    """The ``[lo, hi)`` range of real pixels along one axis that padding the
    window ``[start, stop)`` reads from. Mirroring about an edge reads up to
    ``pad`` pixels back in from it, beyond what the window itself covers."""
    lo, hi = max(start, 0), min(stop, size)
    if mode == "reflect":
        if pad_lo:
            hi = max(hi, min(size, pad_lo + 1))
        if pad_hi:
            lo = min(lo, max(0, size - 1 - pad_hi))
    # A window wholly off one side still needs the edge pixel it replicates.
    if hi <= lo:
        lo, hi = (0, 1) if stop <= 0 else (size - 1, size)
    return lo, hi


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
