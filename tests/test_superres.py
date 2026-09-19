"""Tests for the optional AI super-resolution helper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from traincrop import superres  # noqa: E402


def test_upscale_without_dnn_superres_raises_a_friendly_error(monkeypatch):
    import cv2

    monkeypatch.setattr(superres, "_engine", None)
    monkeypatch.delattr(cv2, "dnn_superres", raising=False)

    with pytest.raises(RuntimeError, match="opencv-contrib-python"):
        superres.upscale(object())
