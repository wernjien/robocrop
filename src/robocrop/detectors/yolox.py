"""YOLOX object detector (OpenCV DNN, COCO classes).

This is the "something other than a face" backend: it detects any of the 80
COCO classes and hands them to the same cropping pipeline a face goes through.
Pick what to crop with ``--detector-opt classes=...``::

    --detector yolox --detector-opt classes=person
    --detector yolox --detector-opt classes=cat,dog

Because the pipeline only ever sees a :class:`~.base.Region`, nothing
downstream needs to know a body was detected rather than a face.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models
from ..geometry import Rect
from .base import BaseDetector, Region

#: COCO class names, in the order YOLOX emits them.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

_INPUT = 640
_STRIDES = (8, 16, 32)
_PAD_VALUE = 114.0  # YOLOX's letterbox grey


class YoloxDetector(BaseDetector):
    name = "yolox"

    # A whole-body or object box already contains the subject; shifting it up
    # would only cut off feet.
    recommended_offset_y = 0.0

    def __init__(
        self,
        min_score: float = 0.6,
        *,
        classes: str | list[str] = "person",
        nms_threshold: float = 0.45,
        quiet: bool = False,
    ) -> None:
        super().__init__(min_score)
        self.nms_threshold = nms_threshold
        self.wanted = self._resolve_classes(classes)
        path = models.ensure("yolox", quiet=quiet)
        self._net = cv2.dnn.readNet(str(path))
        self._grids, self._expanded = _build_grid()

    @staticmethod
    def _resolve_classes(classes: str | list[str]) -> set[int] | None:
        """Turn a class spec into class indices. ``all`` means no filtering."""
        if isinstance(classes, str):
            names = [c.strip() for c in classes.split(",") if c.strip()]
        else:
            names = [str(c).strip() for c in classes]
        if not names or any(n.lower() == "all" for n in names):
            return None

        lookup = {name: i for i, name in enumerate(COCO_CLASSES)}
        out: set[int] = set()
        for name in names:
            key = name.lower().replace("_", " ")
            if key not in lookup:
                raise ValueError(
                    f"unknown COCO class {name!r}; valid classes are: "
                    + ", ".join(COCO_CLASSES)
                )
            out.add(lookup[key])
        return out

    def detect(self, image: np.ndarray) -> list[Region]:
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []

        blob, ratio = _letterbox(image)
        self._net.setInput(blob)
        raw = self._net.forward(self._net.getUnconnectedOutLayersNames())[0][0]

        # Decode YOLOX's grid-relative predictions into letterbox pixels.
        boxes_xy = (raw[:, 0:2] + self._grids) * self._expanded
        boxes_wh = np.exp(np.clip(raw[:, 2:4], -20, 20)) * self._expanded
        scores = raw[:, 4:5] * raw[:, 5:]

        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = confidences >= self.min_score
        if self.wanted is not None:
            keep &= np.isin(class_ids, list(self.wanted))
        if not keep.any():
            return []

        boxes_xy, boxes_wh = boxes_xy[keep], boxes_wh[keep]
        class_ids, confidences = class_ids[keep], confidences[keep]

        # Undo the letterbox scale to land back in source pixels.
        tl = (boxes_xy - boxes_wh / 2.0) / ratio
        sizes = boxes_wh / ratio
        rects = np.concatenate([tl, sizes], axis=1)

        # Per class: a cat on a person's lap overlaps them heavily, and
        # class-blind suppression would throw one of the two away.
        indices = cv2.dnn.NMSBoxesBatched(
            rects.tolist(),
            confidences.astype(float).tolist(),
            class_ids.astype(int).tolist(),
            float(self.min_score),
            float(self.nms_threshold),
        )
        if len(indices) == 0:
            return []

        regions: list[Region] = []
        for i in np.array(indices).flatten():
            x, y, bw, bh = rects[i]
            # Clip to the frame: YOLOX happily predicts past the edge for a
            # partly visible body, and a box outside the image breaks cropping.
            x0, y0 = max(0.0, float(x)), max(0.0, float(y))
            x1, y1 = min(float(w), float(x + bw)), min(float(h), float(y + bh))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            label = COCO_CLASSES[int(class_ids[i])]
            regions.append(
                Region(
                    rect=Rect(x0, y0, x1 - x0, y1 - y0),
                    score=float(confidences[i]),
                    label=label,
                    meta={"detector": self.name, "class": label},
                )
            )
        return self._finalize(regions)


def _build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute YOLOX's anchor grid and per-cell stride.

    Fixed for a fixed input size, so it is built once per detector rather than
    per image.
    """
    grids, expanded = [], []
    for stride in _STRIDES:
        cells = _INPUT // stride
        ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grid = np.stack((xs, ys), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride))
    return (
        np.concatenate(grids, axis=0).astype(np.float32),
        np.concatenate(expanded, axis=0).astype(np.float32),
    )


def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize into a 640x640 grey-padded square, preserving aspect ratio."""
    h, w = image.shape[:2]
    ratio = min(_INPUT / h, _INPUT / w)
    new_h, new_w = max(1, int(h * ratio)), max(1, int(w * ratio))

    canvas = np.full((_INPUT, _INPUT, 3), _PAD_VALUE, dtype=np.float32)
    # INTER_AREA when shrinking: a 40 MP photo is reduced ~10x here, and a
    # linear resize would sample a fraction of the pixels and alias badly.
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    # The model was trained on RGB; OpenCV hands us BGR.
    canvas[:new_h, :new_w] = resized[:, :, ::-1].astype(np.float32)

    blob = canvas.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(blob), ratio
