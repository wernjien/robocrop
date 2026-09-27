"""Person matting (MODNet, OpenCV DNN), for background masks.

MODNet predicts a soft alpha matte -- 1 on the person, 0 on the background,
in between along hair and edges -- which is exactly the shape a loss mask
wants. It is a portrait model, but holds up on half- and full-body crops,
and it keeps hair that coarser person segmenters cut away.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models

_INPUT = 512


class PersonSegmenter:
    def __init__(self, *, quiet: bool = False) -> None:
        path = models.ensure("modnet", quiet=quiet)
        self._net = cv2.dnn.readNet(str(path))

    def matte(self, image: np.ndarray) -> np.ndarray:
        """Alpha matte for a BGR uint8 image, float32 in [0, 1], same size."""
        h, w = image.shape[:2]
        # MODNet was trained on RGB scaled to [-1, 1].
        blob = cv2.dnn.blobFromImage(
            image, 1 / 127.5, (_INPUT, _INPUT), (127.5, 127.5, 127.5), swapRB=True
        )
        self._net.setInput(blob)
        out = self._net.forward().reshape(_INPUT, _INPUT)
        matte = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(matte, 0.0, 1.0).astype(np.float32)

    def close(self) -> None:
        return None


def create(*, quiet: bool = False) -> PersonSegmenter:
    return PersonSegmenter(quiet=quiet)
