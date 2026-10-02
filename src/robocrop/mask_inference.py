"""Shared preprocessing and bounded-memory ONNX inference for loss masks."""

from __future__ import annotations

import os
import threading

import numpy as np
import onnxruntime as ort

from . import models

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 to ImageNet-normalized float32 NCHW."""
    pixels = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))


class OnnxMaskModel:
    """One session shared by crop workers, with one inference at a time."""

    def __init__(self, model: str, *, quiet: bool = False) -> None:
        path = models.ensure(model, quiet=quiet)
        options = ort.SessionOptions()
        # There is one shared model, rather than a thread pool per worker.
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def _run(self, blob: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._session.run(None, {self._input_name: blob})[0]

    def close(self) -> None:
        self._session = None
