"""BiRefNet preprocessing and alpha semantics, without downloading weights."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from robocrop import mask_inference, segment


class Input:
    name = "input"


class StubSession:
    def __init__(self):
        self.feed = None

    def get_inputs(self):
        return [Input()]

    def run(self, output_names, feed):
        self.feed = feed
        # A valid logit inside [0,1] must still go through sigmoid.
        return [np.full((1, 1, 1024, 1024), 0.75, dtype=np.float32)]


def test_birefnet_normalizes_rgb_and_always_converts_logits(monkeypatch, tmp_path):
    session = StubSession()
    created = {}

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    def make_session(path, *, sess_options, providers):
        created.update(path=path, options=sess_options, providers=providers)
        return session

    def ensure(name, **kwargs):
        assert name == "birefnet_matting"
        return tmp_path / "birefnet.onnx"

    monkeypatch.setattr(mask_inference.models, "ensure", ensure)
    monkeypatch.setattr(mask_inference.ort, "SessionOptions", Options)
    monkeypatch.setattr(mask_inference.ort, "InferenceSession", make_session)

    parser = segment.PersonSegmenter(quiet=True)
    image = np.full((120, 80, 3), (10, 20, 30), dtype=np.uint8)
    matte = parser.matte(image)

    assert matte.shape == (120, 80) and matte.dtype == np.float32
    assert np.allclose(matte, 1.0 / (1.0 + np.exp(-0.75)))
    blob = session.feed["input"]
    assert blob.shape == (1, 3, 1024, 1024) and blob.dtype == np.float32
    assert blob[0, :, 0, 0] == pytest.approx(
        (np.array([30, 20, 10]) / 255.0 - [0.485, 0.456, 0.406])
        / [0.229, 0.224, 0.225],
    )
    assert created["providers"] == ["CPUExecutionProvider"]
    assert created["options"].intra_op_num_threads == min(4, os.cpu_count() or 1)
    assert created["options"].inter_op_num_threads == 1


def test_sigmoid_stays_finite_on_extreme_logits():
    out = mask_inference.sigmoid(np.array([-1000, 0, 1000], np.float32))
    assert np.isfinite(out).all()
    assert out == pytest.approx([0, 0.5, 1], abs=1e-6)
