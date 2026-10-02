"""--no-crop: each whole photo kept at its own shape, resized to a size."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.detectors.base import Region  # noqa: E402
from robocrop.geometry import CropRejected, Rect, plan_whole  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, Pipeline  # noqa: E402

from test_pipeline import base_config, make_photo, stub  # noqa: E402,F401

PERSON = Region(rect=Rect(1000, 500, 1000, 2000), score=0.9, label="person")


def rows(tmp_path):
    lines = (tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_plan_whole_keeps_the_shape_and_picks_the_largest_size():
    plan = plan_whole(4000, 3000, sizes=(512, 768, 1024))
    assert plan.tier == 1024 and plan.output_size == (1024, 768)

    plan = plan_whole(600, 900, sizes=(512, 768, 1024))
    assert plan.tier == 1024 and plan.output_size == (683, 1024)


def test_plan_whole_rejects_a_photo_too_small_for_any_size():
    with pytest.raises(CropRejected):
        plan_whole(300, 200, sizes=(512, 768, 1024))


def test_default_no_crop_keeps_a_small_photo_at_the_256_tier(tmp_path, stub):
    stub._current = [Region(rect=Rect(50, 50, 100, 100), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png", size=(300, 200))

    stats = Pipeline(base_config(tmp_path, no_crop=True)).run()

    assert stats.written == 1
    with Image.open(tmp_path / "out" / "0001.png") as image:
        assert image.size == (256, 171)
    assert rows(tmp_path)[0]["tier"] == 256


def test_one_whole_photo_per_image_with_several_detections(tmp_path, stub):
    stub._current = [PERSON, Region(rect=Rect(2500, 500, 800, 1600), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(base_config(tmp_path, no_crop=True)).run()

    assert stats.written == 1
    assert Image.open(tmp_path / "out" / "0001.png").size == (1024, 768)
    assert rows(tmp_path)[0]["crop"] == [0.0, 0.0, 4000.0, 3000.0]


def test_photos_without_a_detection_are_still_skipped(tmp_path, stub):
    stub._current = []
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(base_config(tmp_path, no_crop=True)).run()

    assert stats.written == 0


def test_mask_faces_is_allowed_with_a_face_detector():
    Config(detector="yunet", mask_faces=True, no_crop=True).validate()


def test_background_mask_matches_the_photo_shape(tmp_path, stub, monkeypatch):
    stub._current = [PERSON]
    make_photo(tmp_path / "photos" / "a.png")

    class Segmenter:
        def matte(self, image):
            matte = np.zeros(image.shape[:2], dtype=np.float32)
            matte[150:700, 280:480] = 1.0
            return matte

        def close(self):
            pass

    monkeypatch.setattr(pipeline_module.segment, "create", lambda **kwargs: Segmenter())

    Pipeline(base_config(tmp_path, no_crop=True, mask_background=True)).run()

    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.size == (1024, 768)
    assert mask.getpixel((380, 400)) == 255      # the person
    assert mask.getpixel((900, 100)) == round(0.1 * 255)  # the background


# -- --keep-size -----------------------------------------------------------------
def test_keep_size_leaves_a_photo_in_range_untouched(tmp_path, stub):
    stub._current = [Region(rect=Rect(300, 200, 300, 600), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png", size=(1200, 900))

    Pipeline(base_config(tmp_path, keep_size=True)).run()

    assert Image.open(tmp_path / "out" / "0001.png").size == (1200, 900)


def test_keep_size_shrinks_a_photo_over_the_maximum_whole(tmp_path, stub):
    stub._current = [Region(rect=Rect(3200, 1000, 400, 800), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png", size=(4000, 3000))

    Pipeline(base_config(tmp_path, keep_size=True)).run()

    assert Image.open(tmp_path / "out" / "0001.png").size == (1536, 1152)
    assert rows(tmp_path)[0]["crop"] == [0.0, 0.0, 4000.0, 3000.0]


def test_keep_size_skips_a_photo_under_the_minimum(tmp_path, stub):
    stub._current = [Region(rect=Rect(50, 50, 100, 100), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png", size=(200, 150))

    stats = Pipeline(base_config(tmp_path, keep_size=True)).run()

    assert stats.written == 0 and stats.skipped_too_small == 1


def test_keep_size_uses_the_min_and_max_given(tmp_path, stub):
    stub._current = [Region(rect=Rect(300, 200, 300, 600), score=0.9, label="person")]
    make_photo(tmp_path / "photos" / "a.png", size=(1200, 900))

    Pipeline(base_config(tmp_path, keep_size=True, max_side=800)).run()

    assert Image.open(tmp_path / "out" / "0001.png").size == (800, 600)


def test_keep_size_and_no_crop_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Config(no_crop=True, keep_size=True).validate()
