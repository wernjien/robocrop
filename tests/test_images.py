"""Tests for image extraction helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from traincrop.geometry import Rect, plan_crop  # noqa: E402
from traincrop.images import extract, measure_sharpness  # noqa: E402


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


def test_extract_upscale_calls_superres_for_an_undersized_crop(monkeypatch):
    calls = []

    def fake_upscale(bgr):
        calls.append(bgr.shape[:2])
        return np.repeat(np.repeat(bgr, 2, axis=0), 2, axis=1)

    monkeypatch.setattr("traincrop.images.superres.upscale", fake_upscale)

    image = Image.new("RGB", (100, 100), (10, 20, 30))
    plan = plan_crop(Rect(0, 0, 100, 100), 100, 100, padding=0.0, sizes=(512,), min_ratio=0.1)

    crop = extract(image, plan, upscale=True)

    assert crop.size == (512, 512)
    assert calls == [(100, 100)]


def test_extract_without_upscale_flag_uses_plain_resize(monkeypatch):
    def fail_upscale(bgr):
        raise AssertionError("superres.upscale should not run without --upscale")

    monkeypatch.setattr("traincrop.images.superres.upscale", fail_upscale)

    image = Image.new("RGB", (100, 100), (10, 20, 30))
    plan = plan_crop(Rect(0, 0, 100, 100), 100, 100, padding=0.0, sizes=(512,), min_ratio=0.1)

    crop = extract(image, plan, upscale=False)

    assert crop.size == (512, 512)


def test_extract_downscale_ignores_upscale_flag(monkeypatch):
    def fail_upscale(bgr):
        raise AssertionError("superres.upscale should not run for a crop that already fits")

    monkeypatch.setattr("traincrop.images.superres.upscale", fail_upscale)

    image = Image.new("RGB", (1000, 1000), (10, 20, 30))
    plan = plan_crop(Rect(0, 0, 1000, 1000), 1000, 1000, padding=0.0, sizes=(512,), min_ratio=0.1)

    crop = extract(image, plan, upscale=True)

    assert crop.size == (512, 512)


# -- extend-fill pads only the region it needs ---------------------------------
def _reference_pad(image, box, mode):
    """The original implementation: pad the whole image, then crop."""
    left, top, right, bottom = box
    pad_l, pad_t = max(-left, 0), max(-top, 0)
    pad_r, pad_b = max(right - image.width, 0), max(bottom - image.height, 0)
    if mode == "reflect" and (
        pad_l >= image.width or pad_r >= image.width
        or pad_t >= image.height or pad_b >= image.height
    ):
        mode = "edge"
    arr = np.asarray(image, dtype=np.uint8)
    padded = Image.fromarray(np.pad(arr, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode))
    ox, oy = left + pad_l, top + pad_t
    return padded.crop((ox, oy, ox + right - left, oy + bottom - top))


@pytest.mark.parametrize("mode", ["edge", "reflect"])
def test_padding_a_slice_matches_padding_the_whole_image(mode):
    from traincrop.images import _pad_array

    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    boxes = [(-30, -20, 60, 70), (80, 50, 170, 140), (-10, 10, 40, 60),
             (100, -40, 130, -10), (-200, -5, -150, 45), (-50, -50, 170, 140)]
    for _ in range(200):
        x, y = int(rng.integers(-150, 150)), int(rng.integers(-120, 120))
        side = int(rng.integers(1, 200))
        boxes.append((x, y, x + side, y + side))

    for box in boxes:
        got, _ = _pad_array(image, box, mode=mode)
        want = _reference_pad(image, box, mode)
        assert got.size == want.size, box
        assert np.array_equal(np.asarray(got), np.asarray(want)), box
