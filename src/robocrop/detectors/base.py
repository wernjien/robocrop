"""Detector interface.

A detector turns an image into zero or more :class:`Region` boxes. Everything
downstream -- tier selection, padding, cropping, captioning -- works off
``Region`` alone and never asks what kind of thing was detected, so adding a
body, head or object detector later is a matter of dropping a new module in
this package and registering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry import Rect


@dataclass(frozen=True)
class Region:
    """One detected thing."""

    rect: Rect
    """Bounding box in source-image pixels."""

    score: float
    """Detector confidence in ``[0, 1]``."""

    label: str = "face"
    """What was detected: ``face``, ``person``, ``head``, ... Used only for
    reporting and caption hints; the crop pipeline treats all labels alike."""

    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Named points in source pixels, when the detector provides them.
    YuNet gives ``right_eye``, ``left_eye``, ``nose``, ``right_mouth``,
    ``left_mouth``; used to estimate head pose for the caption."""

    meta: dict[str, object] = field(default_factory=dict)
    """Detector-specific extras, passed through to the manifest."""

    @property
    def area(self) -> float:
        return self.rect.w * self.rect.h


@runtime_checkable
class Detector(Protocol):
    """What the pipeline needs from a detector."""

    name: str

    def detect(self, image: np.ndarray) -> list[Region]:
        """Find regions in a BGR uint8 array (OpenCV's channel order)."""
        ...

    def close(self) -> None:
        """Release any native handles. Safe to call more than once."""
        ...


class BaseDetector:
    """Shared plumbing: confidence filtering, ordering and a no-op close."""

    name = "base"

    recommended_offset_y = 0.0
    """Vertical recentring this detector's boxes want, as a fraction of the
    box side, applied when the user has not set ``--offset-y``.

    It belongs to the detector because it corrects for how that detector
    frames things, not for what the user prefers: a face detector boxes the
    face tightly and leaves hair and forehead outside it, so its crops want
    lifting. A whole-body box has no such bias and wants no shift.
    """

    def __init__(self, min_score: float = 0.6) -> None:
        self.min_score = min_score

    def detect(self, image: np.ndarray) -> list[Region]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "BaseDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _finalize(self, regions: list[Region]) -> list[Region]:
        """Drop low-confidence hits and order the rest largest-first.

        Largest-first makes output numbering stable and lets ``--multi-face
        largest`` simply take the head of the list.
        """
        kept = [r for r in regions if r.score >= self.min_score]
        kept.sort(key=lambda r: r.area, reverse=True)
        return kept
