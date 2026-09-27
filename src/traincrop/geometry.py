"""Crop geometry: turning a detected region into a square, padded, tiered crop.

All maths here is pure and side-effect free so it can be unit tested without
images.  Coordinates are floats in source-image pixel space; the origin is the
top-left corner, +x right, +y down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

BaseSideMode = Literal["max", "mean", "width", "height", "diag"]
EdgeMode = Literal["shift", "extend", "skip"]


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in source-image pixel space."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @classmethod
    def from_center(cls, cx: float, cy: float, side: float) -> "Rect":
        return cls(cx - side / 2.0, cy - side / 2.0, side, side)

    def rounded(self) -> tuple[int, int, int, int]:
        """Integer (left, top, right, bottom) suitable for PIL.Image.crop.

        Rounds the edges rather than the origin+size so the box never drifts,
        then guarantees at least one pixel in each axis.
        """
        left = int(round(self.x))
        top = int(round(self.y))
        right = int(round(self.x2))
        bottom = int(round(self.y2))
        return left, top, max(right, left + 1), max(bottom, top + 1)


@dataclass(frozen=True)
class CropPlan:
    """The resolved plan for one crop, plus the numbers behind the decision."""

    rect: Rect
    """Final crop window in source coordinates. May fall outside the image
    bounds when ``edge='extend'``; never does otherwise."""

    tier: int
    """Output edge length in pixels (one of the configured sizes)."""

    base_side: float
    """Square side derived from the detection, *before* padding. This is the
    value the minimum-size rule is tested against."""

    padded_side: float
    """``base_side`` grown by the padding fraction on every side."""

    fitted_side: float
    """``padded_side`` after being shrunk to fit inside the image, if needed."""

    clamped: bool
    """True when the window had to be moved or shrunk to stay in bounds."""

    extends: bool
    """True when the window falls outside the image and needs canvas fill."""

    @property
    def scale(self) -> float:
        """Resample factor. <1 downscales (good), >1 needs enlarging (a plain
        resize by default, or AI upscaling with --upscale)."""
        return self.tier / self.fitted_side


class CropRejected(Exception):
    """Raised when no configured size can be satisfied for a detection."""

    def __init__(self, reason: str, base_side: float, required: float) -> None:
        super().__init__(reason)
        self.reason = reason
        self.base_side = base_side
        self.required = required


def base_side_from(w: float, h: float, mode: BaseSideMode) -> float:
    """Collapse a detection's width/height into a single square side."""
    if mode == "max":
        return max(w, h)
    if mode == "mean":
        return (w + h) / 2.0
    if mode == "width":
        return w
    if mode == "height":
        return h
    if mode == "diag":
        return math.hypot(w, h) / math.sqrt(2.0)
    raise ValueError(f"unknown base-side mode: {mode!r}")


def choose_tier(measure: float, sizes: Sequence[int], min_ratio: float) -> int | None:
    """Largest configured size whose minimum-size rule ``measure`` satisfies.

    The rule is ``measure >= min_ratio * size``. Returns ``None`` when even
    the smallest configured size is out of reach.
    """
    for size in sorted(sizes, reverse=True):
        if measure >= min_ratio * size:
            return size
    return None


def plan_crop(
    region: Rect,
    image_w: int,
    image_h: int,
    *,
    padding: float,
    sizes: Sequence[int],
    min_ratio: float = 0.8,
    base_mode: BaseSideMode = "max",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    edge: EdgeMode = "shift",
) -> CropPlan:
    """Resolve one detection into a square crop window and an output size.

    Args:
        region: Detection box in source pixels.
        image_w, image_h: Source image dimensions.
        padding: Fraction of ``base_side`` added to *each* side.
        sizes: Candidate output edge lengths.
        min_ratio: Fraction of a size the base needs to reach to qualify for it.
        base_mode: How to collapse the detection box into a square side.
        offset_x, offset_y: Recentring nudge, as a fraction of ``base_side``.
            Negative ``offset_y`` moves the window up, which helps when a
            detector reports a tight face box that excludes hair and forehead.
        edge: What to do when the window leaves the image. ``shift`` slides it
            back in (shrinking only if the image is smaller than the window),
            ``extend`` lets it hang outside for the caller to fill, and ``skip``
            raises rather than produce an off-image crop.

    Raises:
        CropRejected: No configured size meets the minimum-size rule, or
            ``edge='skip'`` and the window does not fit.
    """
    if padding < 0:
        raise ValueError("padding must be >= 0")
    if not sizes:
        raise ValueError("at least one output size is required")

    base_side = base_side_from(region.w, region.h, base_mode)
    if base_side <= 0:
        raise CropRejected("degenerate detection box", base_side, 0.0)

    # The minimum-size rule is judged on the bare detection, before padding
    # inflates it -- padding adds context, not detail.
    tier = choose_tier(base_side, sizes, min_ratio)
    if tier is None:
        smallest = min(sizes)
        raise CropRejected(
            f"detection is {base_side:.0f}px, needs >={min_ratio * smallest:.0f}px "
            f"for the {smallest} tier",
            base_side,
            min_ratio * smallest,
        )

    cx = region.cx + offset_x * base_side
    cy = region.cy + offset_y * base_side
    padded_side = base_side * (1.0 + 2.0 * padding)

    if edge == "extend":
        rect = Rect.from_center(cx, cy, padded_side)
        extends = (
            rect.x < 0 or rect.y < 0 or rect.x2 > image_w or rect.y2 > image_h
        )
        return CropPlan(
            rect=rect,
            tier=tier,
            base_side=base_side,
            padded_side=padded_side,
            fitted_side=padded_side,
            clamped=False,
            extends=extends,
        )

    # 'shift' and 'skip' both need the window to live inside the image.
    fitted_side = min(padded_side, float(min(image_w, image_h)))
    x = _clamp(cx - fitted_side / 2.0, 0.0, image_w - fitted_side)
    y = _clamp(cy - fitted_side / 2.0, 0.0, image_h - fitted_side)
    rect = Rect(x, y, fitted_side, fitted_side)
    clamped = fitted_side < padded_side - 1e-6 or not _centred(rect, cx, cy)

    if edge == "skip" and clamped:
        raise CropRejected(
            "padded crop does not fit inside the image", base_side, padded_side
        )

    # Sliding the window inward loses padding, not face pixels, so the tier
    # normally still stands. It only needs revisiting when the image is so
    # small that the window is now tighter than the face itself.
    if fitted_side < base_side:
        retier = choose_tier(fitted_side, sizes, min_ratio)
        if retier is None:
            raise CropRejected(
                f"crop shrank to {fitted_side:.0f}px to fit the image",
                fitted_side,
                min_ratio * min(sizes),
            )
        tier = retier

    return CropPlan(
        rect=rect,
        tier=tier,
        base_side=base_side,
        padded_side=padded_side,
        fitted_side=fitted_side,
        clamped=clamped,
        extends=False,
    )


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp into ``[low, high]``, tolerating an inverted range.

    When the window is as large as the image ``high`` can land just below
    ``low`` through float error; pinning to ``low`` keeps the crop on-image.
    """
    if high < low:
        return low
    return max(low, min(high, value))


def _centred(rect: Rect, cx: float, cy: float, tol: float = 1e-6) -> bool:
    return abs(rect.cx - cx) <= tol and abs(rect.cy - cy) <= tol
