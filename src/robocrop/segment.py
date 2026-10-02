"""BiRefNet-matting alpha mattes for background loss masks."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .mask_inference import OnnxMaskModel, normalize, sigmoid

_INPUT = 1024


class PersonSegmenter(OnnxMaskModel):
    def __init__(self, *, quiet: bool = False) -> None:
        super().__init__("birefnet_matting", quiet=quiet)

    def matte(self, image: np.ndarray) -> np.ndarray:
        """Alpha matte for a BGR uint8 image, float32 in [0, 1], same size."""
        h, w = image.shape[:2]
        # Match upstream's PIL RGB resize and ImageNet normalization. Resize
        # the whole image: a centre crop would discard edge subjects.
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resized = np.asarray(rgb.resize((_INPUT, _INPUT), Image.Resampling.BILINEAR))
        out = self._run(normalize(resized))[0, 0]
        # This pinned export returns logits, even when all values happen to
        # lie between zero and one. Do not guess output semantics from range.
        matte = cv2.resize(sigmoid(out), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(matte, 0.0, 1.0).astype(np.float32)


def create(*, quiet: bool = False) -> PersonSegmenter:
    return PersonSegmenter(quiet=quiet)
