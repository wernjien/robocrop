"""Tests for the optional AI super-resolution helper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from robocrop import superres  # noqa: E402


def test_upscale_without_dnn_superres_raises_a_friendly_error(monkeypatch):
    import cv2

    monkeypatch.setattr(superres, "_local", __import__("threading").local())
    monkeypatch.delattr(cv2, "dnn_superres", raising=False)

    with pytest.raises(RuntimeError, match="opencv-contrib-python"):
        superres.upscale(object())


def test_each_thread_gets_its_own_engine(monkeypatch):
    """An OpenCV DNN net must not be shared across the crop workers."""
    import threading

    import cv2

    made = []

    class FakeEngine:
        def readModel(self, path): pass
        def setModel(self, name, factor): pass

    class FakeModule:
        @staticmethod
        def DnnSuperResImpl_create():
            engine = FakeEngine()
            made.append(engine)
            return engine

    monkeypatch.setattr(superres, "_local", threading.local())
    monkeypatch.setattr(cv2, "dnn_superres", FakeModule, raising=False)
    monkeypatch.setattr(superres.models, "ensure", lambda name, **kw: Path("fsrcnn.pb"))

    seen = []
    threads = [threading.Thread(target=lambda: seen.append(superres._get_engine())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(made) == 4 and len({id(e) for e in seen}) == 4
    assert superres._get_engine() is superres._get_engine()
