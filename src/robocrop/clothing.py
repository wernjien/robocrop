"""FASHN Human Parser (SegFormer-B4) clothing loss masks."""

from __future__ import annotations

import cv2
import numpy as np

from .mask_inference import OnnxMaskModel, normalize

_WIDTH, _HEIGHT = 384, 576

# FASHN labels: top, dress, skirt, pants, belt, bag, hat, scarf, glasses,
# jewelry. Feet are excluded: the model has no separate shoe class,
# and bare skin must stay learned in clothing-only masks.
CLOTHING_CLASSES = (3, 4, 5, 6, 7, 8, 9, 10, 11, 17)


class ClothingSegmenter(OnnxMaskModel):
    def __init__(self, *, quiet: bool = False) -> None:
        super().__init__("fashn_clothes", quiet=quiet)

    def matte(self, image: np.ndarray) -> np.ndarray:
        """Clothing probability for a BGR image, float32 in [0, 1]."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # INTER_AREA is the exact resize used by the official FASHN package.
        resized = cv2.resize(rgb, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_AREA)
        logits = self._run(normalize(resized))[0]
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probs = exp / exp.sum(axis=0, keepdims=True)
        matte = cv2.resize(
            probs[list(CLOTHING_CLASSES)].sum(axis=0),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        # Class confidence is not garment opacity. Small residual scores on
        # skin must not lower its training weight, and confident garments
        # must reach the requested clothing weight rather than leave a rim
        # of residual loss. Keep the uncertain boundary soft.
        matte = np.clip((matte - 0.3) / 0.4, 0.0, 1.0)
        # Cover a logit cell's soft rim, so garment edges are not learned.
        grow = max(1, int(np.ceil(max(h / logits.shape[1], w / logits.shape[2]))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)
        )
        return np.clip(cv2.dilate(matte, kernel), 0.0, 1.0).astype(np.float32)


def create(*, quiet: bool = False) -> ClothingSegmenter:
    return ClothingSegmenter(quiet=quiet)
