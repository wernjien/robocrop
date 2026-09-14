"""YuNet face detector (OpenCV DNN).

Fast, accurate on non-frontal and small faces, and returns five landmarks that
we reuse to estimate head pose for the caption. This is the default detector.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models
from ..geometry import Rect
from .base import BaseDetector, Region

#: YuNet's landmark columns, in the order it emits them.
_LANDMARKS = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


class YuNetDetector(BaseDetector):
    name = "yunet"

    # YuNet boxes brow-to-chin, excluding hair and forehead.
    # Lifting the window by 8% of the box keeps the whole head in frame.
    recommended_offset_y = -0.08

    def __init__(
        self,
        min_score: float = 0.6,
        *,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        detect_max_side: int = 1600,
        quiet: bool = False,
    ) -> None:
        super().__init__(min_score)
        self.detect_max_side = detect_max_side
        path = models.ensure("yunet", quiet=quiet)
        # Input size is a placeholder; it is set per image in detect().
        self._net = cv2.FaceDetectorYN.create(
            model=str(path),
            config="",
            input_size=(320, 320),
            score_threshold=float(min_score),
            nms_threshold=float(nms_threshold),
            top_k=int(top_k),
        )

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        # YuNet is trained around modest input sizes and slows down sharply on
        # 40-megapixel photos, so detect on a downscaled copy and map the boxes
        # back. Detail lost here costs nothing: the crop is taken from the
        # full-resolution original.
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

        wh, ww = work.shape[:2]
        self._net.setInputSize((ww, wh))
        _, raw = self._net.detect(work)
        if raw is None:
            return []

        inv = 1.0 / scale
        regions: list[Region] = []
        for row in raw:
            x, y, bw, bh = (float(v) * inv for v in row[:4])
            score = float(row[14])
            marks = {
                name: (float(row[4 + i * 2]) * inv, float(row[5 + i * 2]) * inv)
                for i, name in enumerate(_LANDMARKS)
            }
            regions.append(
                Region(
                    rect=Rect(x, y, bw, bh),
                    score=score,
                    label="face",
                    landmarks=marks,
                    meta={"detector": self.name},
                )
            )
        return self._finalize(regions)
