"""Optional super-resolution, used instead of a plain resize when a crop
needs enlarging (``--upscale``).

FSRCNN is small and fast enough to run per-crop on CPU, but its module lives
in ``opencv-contrib-python``; that contrib build is included in RoboCrop's
standard requirements and the module is only checked when requested.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from . import models

#: The bundled model's fixed enlargement factor. Crops are never upscaled
#: beyond this in one pass; anything past it is corrected by a final resize.
FACTOR = 2

#: One engine per crop worker. An OpenCV DNN net is not safe to run from two
#: threads at once, and the crop pass calls this from several.
_local = threading.local()


def _get_engine():
    engine = getattr(_local, "engine", None)
    if engine is not None:
        return engine
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "--upscale needs opencv-contrib-python; reinstall RoboCrop's "
            "standard requirements with ./robocrop --setup."
        )
    path = models.ensure("fsrcnn_x2")
    engine = cv2.dnn_superres.DnnSuperResImpl_create()
    engine.readModel(str(path))
    engine.setModel("fsrcnn", FACTOR)
    _local.engine = engine
    return engine


def upscale(bgr: np.ndarray) -> np.ndarray:
    """Enlarge a BGR array by `FACTOR` with FSRCNN."""
    return _get_engine().upsample(bgr)
