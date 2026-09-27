"""--mask-background: the background weighted down in the loss mask."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from traincrop import models  # noqa: E402
from traincrop import pipeline as pipeline_module  # noqa: E402
from traincrop.config import Config  # noqa: E402
from traincrop.geometry import Rect  # noqa: E402
from traincrop.masks import build_mask, isolate_person  # noqa: E402
from traincrop.pipeline import MANIFEST_NAME, TRAINING_CONFIG_NAME, Pipeline  # noqa: E402

from test_face_mask import head, stubs  # noqa: E402,F401
from test_pipeline import base_config, make_photo  # noqa: E402

# Where test_face_mask's BODY box lands inside its 1024 crop: the 2800px
# window starts at (100, 100) and is scaled by 1024 / 2800.
BOX = Rect(329, 146, 366, 731)
BG = round(0.1 * 255)


def person_matte(size=1024):
    """A standing figure inside BOX, plus a separate blob off to the side."""
    matte = np.zeros((size, size), dtype=np.float32)
    matte[150:870, 380:640] = 1.0     # the person
    matte[150:870, 375:380] = 0.3     # a soft edge
    matte[900:1000, 900:1000] = 1.0   # a bystander / a ball, outside BOX
    return matte


class StubSegmenter:
    def __init__(self, matte):
        self._matte = matte

    def matte(self, image):
        return self._matte

    def close(self):
        pass


@pytest.fixture
def segmenter(monkeypatch):
    holder = {"matte": person_matte(), "created": 0}

    def fake_create(**kwargs):
        holder["created"] += 1
        return StubSegmenter(holder["matte"])

    monkeypatch.setattr(pipeline_module.segment, "create", fake_create)
    return holder


def bg_config(tmp_path, **overrides):
    return base_config(tmp_path, mask_background=True, **overrides)


def row(tmp_path):
    return json.loads((tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()[0])


def mask_of(tmp_path):
    return Image.open(tmp_path / "out" / "0001-masklabel.png")


# -- isolating the subject ---------------------------------------------------
def test_isolate_person_keeps_only_what_touches_the_box():
    person = isolate_person(person_matte(), BOX)

    assert person[500, 500] == 1.0
    assert person[950, 950] == 0.0          # the separate blob is dropped
    assert person[500, 377] == pytest.approx(0.3)  # the soft edge survives


def test_isolate_person_is_none_when_nothing_touches_the_box():
    matte = np.zeros((1024, 1024), dtype=np.float32)
    matte[900:1000, 900:1000] = 1.0
    assert isolate_person(matte, BOX) is None
    assert isolate_person(np.zeros((64, 64), dtype=np.float32), Rect(0, 0, 10, 10)) is None


def test_build_mask_weights_background_person_and_face():
    person = isolate_person(person_matte(), BOX)
    mask = build_mask(1024, [Rect(462, 150, 60, 60)], 0.35, person=person, background=0.1)

    assert mask.getpixel((500, 600)) == 255   # body
    assert mask.getpixel((100, 100)) == BG    # background
    assert mask.getpixel((490, 180)) == 0     # face, inside the person


# -- the run -------------------------------------------------------------------
def test_background_is_weighted_down_and_the_person_kept(tmp_path, stubs, segmenter):
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(bg_config(tmp_path)).run()

    assert stats.written == 1 and stats.background_masked == 1 and stats.background_box == 0
    mask = mask_of(tmp_path)
    assert mask.getpixel((500, 500)) == 255
    assert mask.getpixel((50, 50)) == BG
    assert mask.getpixel((950, 950)) == BG    # the blob outside the box
    assert row(tmp_path)["background_mask"] == "outline"
    assert row(tmp_path)["masked_faces"] == 0


def test_background_weight_is_configurable(tmp_path, stubs, segmenter):
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(bg_config(tmp_path, background_weight=0.0)).run()

    assert mask_of(tmp_path).getpixel((50, 50)) == 0


def test_no_person_outline_falls_back_to_the_detection_box(tmp_path, stubs, segmenter):
    segmenter["matte"] = np.zeros((1024, 1024), dtype=np.float32)
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(bg_config(tmp_path)).run()

    assert stats.written == 1 and stats.background_box == 1
    mask = mask_of(tmp_path)
    assert mask.getpixel((340, 160)) == 255   # just inside the box corner
    assert mask.getpixel((300, 160)) == BG    # just outside it
    assert row(tmp_path)["background_mask"] == "box"


def test_faces_and_background_combine_in_one_mask(tmp_path, stubs, segmenter):
    _, faces, _ = stubs
    faces._current = [head(462, 150, side=60)]
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(bg_config(tmp_path, mask_faces=True)).run()

    assert stats.masked == 1 and stats.background_masked == 1
    mask = mask_of(tmp_path)
    assert mask.getpixel((490, 180)) == 0
    assert mask.getpixel((500, 600)) == 255
    assert mask.getpixel((50, 50)) == BG
    assert not (tmp_path / "out" / "0002-masklabel.png").exists()


def test_background_only_does_not_run_the_face_detector(tmp_path, stubs, segmenter):
    _, _, created = stubs
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(bg_config(tmp_path)).run()

    assert "yunet" not in dict(created)
    assert stats.unmasked_kept == 0


def test_no_segmenter_without_mask_background(tmp_path, stubs, segmenter):
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(base_config(tmp_path, mask_faces=True)).run()

    assert segmenter["created"] == 0
    assert row(tmp_path)["background_mask"] is None


def test_training_config_turns_masked_training_on(tmp_path, stubs, segmenter, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text(json.dumps({
        "batch_size": 1, "masked_training": False,
        "unmasked_probability": 0.1, "unmasked_weight": 0.1,
    }))
    monkeypatch.setattr(pipeline_module, "ONETRAINER_TEMPLATE_PATH", template)
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(bg_config(tmp_path, training_config=True)).run()

    written = json.loads((tmp_path / "out" / TRAINING_CONFIG_NAME).read_text())
    assert written["masked_training"] is True
    # A 0.1 floor would lift the background to 0.1 whatever the mask says.
    assert written["unmasked_weight"] == 0.0


# -- configuration -------------------------------------------------------------
@pytest.mark.parametrize("detector, opts", [
    ("yunet", {}), ("haar", {}), ("yolox", {}),
    ("yolox", {"classes": "person"}), ("yolox", {"classes": "dog,person"}),
    ("yolox", {"classes": "all"}), ("yolox", {"classes": ""}),
])
def test_background_masking_accepts_people_detectors(detector, opts):
    Config(detector=detector, detector_opts=opts, mask_background=True).validate()


@pytest.mark.parametrize("classes", ["dog", "cat,bird", ["car"]])
def test_background_masking_rejects_non_person_classes(classes):
    cfg = Config(detector="yolox", detector_opts={"classes": classes}, mask_background=True)
    with pytest.raises(ValueError, match="mask-background"):
        cfg.validate()


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_background_weight_must_be_a_fraction(weight):
    with pytest.raises(ValueError, match="background-weight"):
        Config(background_weight=weight).validate()


def test_modnet_caches_under_its_own_name():
    assert models.REGISTRY["modnet"].cache_name == "modnet_photographic_portrait_matting.onnx"
    assert models.REGISTRY["yolox"].cache_name == "object_detection_yolox_2022nov.onnx"
