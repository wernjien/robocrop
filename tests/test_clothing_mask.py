"""--mask-clothing: clothing and accessories weighted down in the loss mask."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import models  # noqa: E402
from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.cli import build_config  # noqa: E402
from robocrop.clothing import CLOTHING_CLASSES, ClothingSegmenter  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.geometry import Rect  # noqa: E402
from robocrop.masks import build_mask  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, Pipeline  # noqa: E402

from test_background_mask import BG, segmenter  # noqa: E402,F401
from test_face_mask import head, stubs  # noqa: E402,F401
from test_pipeline import base_config, make_photo  # noqa: E402


def outfit(size=1024):
    """A shirt and trousers down the middle of the crop, with a soft hem."""
    matte = np.zeros((size, size), dtype=np.float32)
    matte[300:870, 400:620] = 1.0
    matte[300:870, 395:400] = 0.5
    return matte


class StubParser:
    def __init__(self, matte):
        self._matte = matte

    def matte(self, image):
        return self._matte

    def close(self):
        pass


@pytest.fixture
def parser(monkeypatch):
    holder = {"matte": outfit(), "created": 0}

    def fake_create(**kwargs):
        holder["created"] += 1
        return StubParser(holder["matte"])

    monkeypatch.setattr(pipeline_module.clothing, "create", fake_create)
    return holder


def clothing_config(tmp_path, **overrides):
    return base_config(tmp_path, mask_clothing=True, **overrides)


def row(tmp_path):
    return json.loads((tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()[0])


def mask_of(tmp_path):
    return Image.open(tmp_path / "out" / "0001-masklabel.png")


# -- the parser ------------------------------------------------------------------
class StubNet:
    """Upper-clothes wins the top half of the logits, face the bottom half."""

    def setInput(self, blob):
        assert blob.shape == (1, 3, 512, 512) and blob.dtype == np.float32

    def forward(self):
        logits = np.zeros((1, 18, 128, 128), dtype=np.float32)
        logits[0, 4, :64] = 10.0
        logits[0, 11, 64:] = 10.0
        return logits


def test_parser_sums_the_clothing_classes_into_one_matte():
    parser = ClothingSegmenter.__new__(ClothingSegmenter)
    parser._net = StubNet()

    matte = parser.matte(np.zeros((300, 200, 3), dtype=np.uint8))

    assert matte.shape == (300, 200) and matte.dtype == np.float32
    assert matte[10, 100] > 0.99    # upper-clothes
    assert matte[290, 100] < 0.01   # face


def test_clothing_classes_leave_the_body_out():
    face, hair, limbs = 11, 2, (12, 13, 14, 15)
    assert not {face, hair, *limbs} & set(CLOTHING_CLASSES)


# -- the mask image ----------------------------------------------------------------
def test_build_mask_drops_clothing_to_its_weight():
    mask = build_mask(1024, [], 0.35, clothing=outfit(), clothing_weight=0.0)

    assert mask.getpixel((500, 600)) == 0     # clothing
    assert mask.getpixel((397, 600)) == 128   # the soft hem, half clothing
    assert mask.getpixel((500, 100)) == 255   # everything else


def test_build_mask_combines_clothing_with_person_and_faces():
    person = np.zeros((1024, 1024), dtype=np.float32)
    person[100:900, 380:640] = 1.0

    mask = build_mask(
        1024, [Rect(462, 150, 60, 60)], 0.35,
        person=person, background=0.1, clothing=outfit(), clothing_weight=0.25,
    )

    assert mask.getpixel((490, 180)) == 0              # face
    assert mask.getpixel((500, 600)) == round(0.25 * 255)  # clothing
    assert mask.getpixel((630, 200)) == 255            # bare arm
    assert mask.getpixel((50, 50)) == BG               # background


# -- the run -------------------------------------------------------------------------
def test_clothing_is_weighted_down(tmp_path, stubs, parser):
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(clothing_config(tmp_path)).run()

    assert stats.written == 1 and stats.clothing_masked == 1
    mask = mask_of(tmp_path)
    assert mask.getpixel((500, 600)) == 0
    assert mask.getpixel((50, 50)) == 255
    assert row(tmp_path)["masked_clothing"] == pytest.approx((570 * 225) / 1024 ** 2, abs=1e-3)
    assert row(tmp_path)["mask_file"] == "0001-masklabel.png"


def test_clothing_weight_is_configurable(tmp_path, stubs, parser):
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(clothing_config(tmp_path, clothing_weight=0.5)).run()

    assert mask_of(tmp_path).getpixel((500, 600)) == 128


def test_no_clothing_found_keeps_the_crop_with_a_white_mask(tmp_path, stubs, parser):
    parser["matte"] = np.zeros((1024, 1024), dtype=np.float32)
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(clothing_config(tmp_path)).run()

    assert stats.written == 1 and stats.clothing_masked == 0
    assert mask_of(tmp_path).getextrema() == (255, 255)
    assert row(tmp_path)["masked_clothing"] == 0.0


def test_all_three_masks_combine_in_one_file(tmp_path, stubs, parser, segmenter):
    _, faces, _ = stubs
    faces._current = [head(462, 150, side=60)]
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(clothing_config(tmp_path, mask_faces=True, mask_background=True)).run()

    assert stats.masked == 1 and stats.background_masked == 1 and stats.clothing_masked == 1
    mask = mask_of(tmp_path)
    assert mask.getpixel((490, 180)) == 0     # face
    assert mask.getpixel((500, 600)) == 0     # clothing
    assert mask.getpixel((630, 200)) == 255   # bare arm, inside the person
    assert mask.getpixel((50, 50)) == BG      # background


def test_clothing_only_runs_neither_face_detector_nor_matting(tmp_path, stubs, parser, segmenter):
    _, _, created = stubs
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(clothing_config(tmp_path)).run()

    assert "yunet" not in dict(created)
    assert segmenter["created"] == 0


def test_no_parser_without_mask_clothing(tmp_path, stubs, parser):
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(base_config(tmp_path, mask_faces=True)).run()

    assert parser["created"] == 0
    assert row(tmp_path)["masked_clothing"] == 0.0


# -- configuration -------------------------------------------------------------------
@pytest.mark.parametrize("classes", ["dog", "cat,bird", ["car"]])
def test_clothing_masking_rejects_non_person_classes(classes):
    cfg = Config(detector="yolox", detector_opts={"classes": classes}, mask_clothing=True)
    with pytest.raises(ValueError, match="mask-clothing"):
        cfg.validate()


@pytest.mark.parametrize("detector", ["yunet", "haar", "yolox"])
def test_clothing_masking_accepts_people_detectors(detector):
    Config(detector=detector, mask_clothing=True).validate()


@pytest.mark.parametrize("weight", [-0.1, 1.5])
def test_clothing_weight_must_be_a_fraction(weight):
    with pytest.raises(ValueError, match="clothing-weight"):
        Config(clothing_weight=weight).validate()


def test_clothing_flags_parse(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = build_config(["-i", str(tmp_path), "--mask-clothing", "--clothing-weight", "0.2"])

    assert cfg.mask_clothing and cfg.clothing_weight == 0.2 and cfg.writes_masks


def test_segformer_caches_under_its_own_name():
    assert models.REGISTRY["segformer_clothes"].cache_name == "segformer_b2_clothes.onnx"
