"""--face-mask outline: faces masked along their own shape, from the clothes parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.detectors.base import Region  # noqa: E402
from robocrop.geometry import Rect  # noqa: E402
from robocrop.masks import build_mask, face_outline  # noqa: E402
from robocrop.pipeline import Pipeline  # noqa: E402

from test_face_mask import head, stubs  # noqa: E402,F401
from test_pipeline import base_config, make_photo  # noqa: E402


def face_blob(size=1024):
    """A face-shaped patch around the stub's default face box (462, 100, 100)."""
    matte = np.zeros((size, size), dtype=np.float32)
    matte[105:195, 470:555] = 1.0
    return matte


class StubParser:
    def __init__(self, face):
        self._face = face

    def face_matte(self, image):
        return self._face

    def close(self):
        pass


@pytest.fixture
def parser(monkeypatch):
    holder = {"face": face_blob()}
    monkeypatch.setattr(
        pipeline_module.clothing, "create", lambda **kwargs: StubParser(holder["face"])
    )
    return holder


def test_outline_keeps_only_the_face_pixels_near_each_face():
    matte = face_blob()
    matte[900:1000, 900:1000] = 1.0   # something face-like far from any detected face

    outline, missed = face_outline(matte, [head()])

    assert not missed
    assert outline[150, 500] == 1.0
    assert outline[950, 950] == 0.0


def test_a_face_the_parser_misses_falls_back_to_an_oval():
    outline, missed = face_outline(np.zeros((1024, 1024), np.float32), [head()])

    assert missed == [Rect(462, 100, 100, 100)]
    assert outline.max() == 0.0


def with_neck(size=1024):
    """The face blob, with the neck and chest below it that the parser also calls face."""
    matte = face_blob(size)
    matte[195:330, 480:545] = 1.0
    return matte


def test_outline_stops_at_the_chin_found_from_the_landmarks():
    face = Region(rect=Rect(462, 100, 100, 100), score=0.9, landmarks={
        "left_eye": (530, 135), "right_eye": (490, 135),
        "left_mouth": (525, 165), "right_mouth": (495, 165),
    })   # chin at 165 + 0.7 * 30 = 186

    outline, _ = face_outline(with_neck(), [face])

    assert outline[180, 510] == 1.0
    assert outline[200, 510] == 0.0
    assert outline[300, 510] == 0.0


def test_the_chin_line_follows_a_tilted_head():
    # Head tilted so "down" runs along (1, 1): the chin line is a diagonal.
    face = Region(rect=Rect(462, 100, 100, 100), score=0.9, landmarks={
        "left_eye": (500, 120), "right_eye": (480, 140),
        "left_mouth": (520, 140), "right_mouth": (500, 160),
    })   # chin at (524, 164)

    outline, _ = face_outline(with_neck(), [face])

    assert outline[190, 485] == 1.0   # below a flat cut through the chin, but above the tilted one
    assert outline[190, 540] == 0.0


def test_without_landmarks_the_outline_stops_at_the_box_bottom():
    outline, _ = face_outline(with_neck(), [head()])   # box bottom at 200

    assert outline[190, 510] == 1.0
    assert outline[220, 510] == 0.0


def test_build_mask_blacks_out_the_face_shape_only():
    mask = build_mask(1024, [], 0.35, face=face_blob())

    assert mask.getpixel((500, 150)) == 0      # the face
    assert mask.getpixel((500, 70)) == 255     # hair above it stays learned
    assert mask.getpixel((440, 150)) == 255    # and beside it


def test_outline_is_the_default_and_masks_the_face_shape(tmp_path, stubs, parser):
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(base_config(tmp_path, mask_faces=True)).run()

    assert stats.masked == 1
    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.getpixel((500, 150)) == 0
    assert mask.getpixel((500, 70)) == 255     # an oval would have covered this


def test_bad_face_mask_value_is_rejected():
    with pytest.raises(ValueError, match="face-mask"):
        Config(detector="yolox", mask_faces=True, face_mask="square").validate()
