"""Detector registry.

Backends are imported lazily: MediaPipe costs a couple of seconds to import
and torch-based backends far more, so nothing is loaded until it is asked for.
To add a detector, write a module exposing a :class:`~.base.BaseDetector`
subclass and add one line to ``_BACKENDS``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseDetector, Detector, Region

#: Public name -> "module:class" within this package.
_BACKENDS: dict[str, str] = {
    "yunet": "yunet:YuNetDetector",
    "haar": "haar:HaarDetector",
    "yolox": "yolox:YoloxDetector",
}

#: Shown in --help so the choices explain themselves.
DESCRIPTIONS: dict[str, str] = {
    "yunet": "OpenCV YuNet face detector - accurate, fast, 5 landmarks (default)",
    "haar": "Haar cascade - last-resort fallback, frontal faces only",
    "yolox": "YOLOX - bodies and objects, 80 COCO classes; "
             "see --detector-opt classes=...",
}


def available() -> list[str]:
    return list(_BACKENDS)


def create(name: str, **kwargs: Any) -> BaseDetector:
    """Instantiate a detector backend by name."""
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown detector {name!r}; choose one of {', '.join(_BACKENDS)}"
        ) from None

    module_name, class_name = target.split(":")
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:
        raise RuntimeError(
            f"detector {name!r} needs a dependency that is not installed: {exc}"
        ) from exc
    return getattr(module, class_name)(**kwargs)


__all__ = ["BaseDetector", "Detector", "Region", "available", "create", "DESCRIPTIONS"]
