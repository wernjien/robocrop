"""Tests for image extraction helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from traincrop.images import measure_sharpness  # noqa: E402


def _noisy_with_flat_border(size=200, border=40, seed=0):
    """A flat canvas with real high-frequency noise only in the center."""
    rng = np.random.default_rng(seed)
    arr = np.full((size, size, 3), 120, dtype=np.uint8)
    noise = rng.integers(0, 256, (size - 2 * border, size - 2 * border, 3), dtype=np.uint8)
    arr[border:size - border, border:size - border] = noise
    return Image.fromarray(arr)


def test_flat_image_scores_exactly_zero():
    flat = Image.new("RGB", (200, 200), (120, 140, 160))
    assert measure_sharpness(flat) == 0.0
    assert measure_sharpness(flat, inner_fraction=0.5) == 0.0


def test_central_window_excludes_a_flat_border():
    """A plain/blurred border must not drag down a sharp subject's score."""
    img = _noisy_with_flat_border()

    whole_image = measure_sharpness(img, inner_fraction=1.0)
    central_only = measure_sharpness(img, inner_fraction=0.5)

    assert central_only > whole_image
