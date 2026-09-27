"""Haar cascade face detector.

A low-accuracy fallback for environments where the YuNet ONNX cannot be
fetched or loaded. It has no landmarks and no real confidence score, so hits
are reported at a flat confidence derived from neighbour count.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models
from ..geometry import Rect
from .base import BaseDetector, Region


class HaarDetector(BaseDetector):
    name = "haar"

    # Haar boxes the face tightly, much like YuNet.
    # Lifting the window by 8% of the box keeps the whole head in frame.
    recommended_offset_y = -0.08

    def __init__(
        self,
        min_score: float = 0.6,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        detect_max_side: int = 1600,
        quiet: bool = False,
    ) -> None:
        super().__init__(min_score)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.detect_max_side = detect_max_side
        path = models.ensure("haar_frontalface", quiet=quiet)
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade from {path}")

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        scale = 1.0
        work = image
        longest = max(h, w)
        if self.detect_max_side and longest > self.detect_max_side:
            scale = self.detect_max_side / longest
            work = cv2.resize(
                image,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes, _, weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )

        inv = 1.0 / scale
        regions: list[Region] = []
        for i, (x, y, bw, bh) in enumerate(boxes):
            # Cascades emit an unbounded "reject level", not a probability.
            # Squash it into [0, 1] so --min-score behaves consistently across
            # detectors; 5.0 is roughly the level of a confident hit.
            weight = float(weights[i]) if i < len(weights) else 0.0
            score = max(0.0, min(1.0, weight / 5.0))
            regions.append(
                Region(
                    rect=Rect(float(x) * inv, float(y) * inv, float(bw) * inv, float(bh) * inv),
                    score=score,
                    label="face",
                    meta={"detector": self.name, "reject_level": weight},
                )
            )
        return self._finalize(regions)
