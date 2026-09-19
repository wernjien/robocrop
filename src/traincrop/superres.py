"""Optional super-resolution, used instead of a plain resize when a crop
needs enlarging (``--upscale``).

FSRCNN is small and fast enough to run per-crop on CPU, but its module lives
in ``opencv-contrib-python``, not the plain ``opencv-python`` most installs
use -- so it is only imported when actually requested.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models

#: The bundled model's fixed enlargement factor. Crops are never upscaled
#: beyond this in one pass; anything past it is corrected by a final resize.
FACTOR = 2

_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "--upscale needs the dnn_superres module, which plain opencv-python "
            "does not ship. Install opencv-contrib-python instead (pip uninstall "
            "opencv-python && pip install opencv-contrib-python)."
        )
    path = models.ensure("fsrcnn_x2")
    engine = cv2.dnn_superres.DnnSuperResImpl_create()
    engine.readModel(str(path))
    engine.setModel("fsrcnn", FACTOR)
    _engine = engine
    return _engine


def upscale(bgr: np.ndarray) -> np.ndarray:
    """Enlarge a BGR array by `FACTOR` with FSRCNN."""
    return _get_engine().upsample(bgr)
