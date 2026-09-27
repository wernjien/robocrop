"""Clothes parsing (SegFormer-B2 on ATR, OpenCV DNN), for clothing masks."""

from __future__ import annotations

import cv2
import numpy as np

from . import models

_INPUT = 512
_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1) * 255

#: ATR labels counted as clothing: hat, sunglasses, upper-clothes, skirt,
#: pants, dress, belt, left/right shoe, bag, scarf. Hair, face and limbs are not.
CLOTHING_CLASSES = (1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17)


class ClothingSegmenter:
    def __init__(self, *, quiet: bool = False) -> None:
        path = models.ensure("segformer_clothes", quiet=quiet)
        self._net = cv2.dnn.readNet(str(path))

    def matte(self, image: np.ndarray) -> np.ndarray:
        """Clothing probability for a BGR uint8 image, float32 in [0, 1], same size."""
        h, w = image.shape[:2]
        # mean is given in RGB order because swapRB runs first.
        blob = cv2.dnn.blobFromImage(image, 1.0, (_INPUT, _INPUT), _MEAN, swapRB=True)
        self._net.setInput(blob / _STD)
        logits = self._net.forward()[0]
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        clothing = exp[list(CLOTHING_CLASSES)].sum(axis=0) / exp.sum(axis=0)
        matte = cv2.resize(clothing, (w, h), interpolation=cv2.INTER_LINEAR)
        # Logits are 1/4 scale, so edges blur over one cell; grow past it or a rim is learned.
        grow = int(np.ceil(max(h, w) / logits.shape[-1]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        return np.clip(cv2.dilate(matte, kernel), 0.0, 1.0).astype(np.float32)

    def close(self) -> None:
        return None


def create(*, quiet: bool = False) -> ClothingSegmenter:
    return ClothingSegmenter(quiet=quiet)
