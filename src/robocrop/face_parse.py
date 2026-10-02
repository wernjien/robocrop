"""SegFace facial-part masks on separate, padded detector crops."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image

from . import models
from .detectors.base import Region
from .geometry import Rect
from .mask_inference import normalize

_INPUT = 512
# CelebAMask-HQ checkpoint order: skin, ears, brows, eyes, nose, mouth,
# lips and glasses. Background, neck, hair, clothing and jewelry stay out.
FACE_CLASSES = (2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15)


class FaceParser:
    def __init__(self, *, quiet: bool = False) -> None:
        # Cropping and oval masks remain usable without importing PyTorch.
        import torch

        from ._segface import SegFace

        path = models.ensure("segface", quiet=quiet)
        self._torch = torch
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        # The published training checkpoint also contains optimizer state.
        # mmap avoids copying that unused state into RAM during loading.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict_backbone"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self._model = SegFace()
        self._model.load_state_dict(state, strict=True)
        self._model.to(self._device).eval()
        self._lock = threading.Lock()

    def _parse(self, crop: np.ndarray) -> np.ndarray:
        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        resized = np.asarray(rgb.resize((_INPUT, _INPUT), Image.Resampling.BICUBIC))
        tensor = self._torch.from_numpy(normalize(resized)).to(self._device)
        with self._lock, self._torch.inference_mode():
            logits = self._model(tensor)
            # Upsample logits before selecting classes, as in upstream inference.
            # Bound the 19-channel tensor for large --keep-size photographs.
            size = tuple(min(side, 1024) for side in crop.shape[:2])
            logits = self._torch.nn.functional.interpolate(
                logits,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            labels = logits.argmax(dim=1)[0].cpu().numpy()
        matte = np.isin(labels, FACE_CLASSES).astype(np.float32)
        if matte.shape != crop.shape[:2]:
            matte = cv2.resize(
                matte,
                (crop.shape[1], crop.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return matte

    def outline(
        self,
        image: np.ndarray,
        faces: Sequence[Region],
        *,
        min_cover: float = 0.08,
    ) -> tuple[np.ndarray, list[Rect]]:
        """Face-only matte and detector boxes needing the oval fallback."""
        h, w = image.shape[:2]
        outline = np.zeros((h, w), dtype=np.float32)
        missed: list[Rect] = []
        for face in faces:
            box = face.rect
            # A square around the whole head, with enough neck context for
            # SegFace to separate the chin. Even tiny faces get 512px inference.
            side = max(2, int(np.ceil(2.0 * max(box.w, box.h))))
            left = int(np.floor(box.cx - side / 2))
            top = int(np.floor(box.cy - 0.1 * box.h - side / 2))
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(w, left + side), min(h, top + side)
            if x1 <= x0 or y1 <= y0:
                missed.append(box)
                continue
            # Preserve square geometry at image edges; trim the padding after
            # inference so no invented pixels reach the output mask.
            crop = cv2.copyMakeBorder(
                image[y0:y1, x0:x1],
                y0 - top,
                top + side - y1,
                x0 - left,
                left + side - x1,
                cv2.BORDER_REPLICATE,
            )
            local = self._parse(crop)
            local = local[y0 - top : y1 - top, x0 - left : x1 - left]
            kept = _detected_face(local, box, x0, y0, min_cover)
            if kept is None:
                missed.append(box)
                continue
            # Close internal gaps; an outward, one-pixel feather covers the
            # edge while limiting spill into hair and neck.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel)
            kept = np.maximum(
                kept, cv2.GaussianBlur(cv2.dilate(kept, kernel), (0, 0), 0.6)
            )
            outline[y0:y1, x0:x1] = np.maximum(outline[y0:y1, x0:x1], kept)
        return outline, missed

    def close(self) -> None:
        self._model = None


def _detected_face(
    matte: np.ndarray,
    box: Rect,
    x: int,
    y: int,
    min_cover: float,
) -> np.ndarray | None:
    """Keep facial components intersecting this face, excluding neighbours."""
    h, w = matte.shape
    x0, y0 = max(0, int(box.x - x)), max(0, int(box.y - y))
    x1, y1 = min(w, int(np.ceil(box.x2 - x))), min(h, int(np.ceil(box.y2 - y)))
    if x1 <= x0 or y1 <= y0:
        return None
    count, labels = cv2.connectedComponents(
        (matte >= 0.5).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    touching = np.unique(labels[y0:y1, x0:x1])
    touching = touching[touching != 0]
    kept = np.isin(labels, touching).astype(np.float32)
    if kept[y0:y1, x0:x1].sum() < min_cover * (x1 - x0) * (y1 - y0):
        return None
    return kept


def create(*, quiet: bool = False) -> FaceParser:
    return FaceParser(quiet=quiet)
