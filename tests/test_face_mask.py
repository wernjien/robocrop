"""--mask-faces: body crops with the faces masked out of training."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.cli import build_config  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.detectors.base import Region  # noqa: E402
from robocrop.geometry import Rect  # noqa: E402
from robocrop.masks import build_mask, mask_name  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, SUMMARY_NAME, TRAINING_CONFIG_NAME, Pipeline  # noqa: E402

from test_pipeline import StubDetector, base_config, make_photo  # noqa: E402

# A standing person in a 4000x3000 photo: base side 2000, so a 1024 crop.
BODY = Region(rect=Rect(1000, 500, 1000, 2000), score=0.9, label="person")


def head(x=462, y=100, side=100, score=0.9):
    """A face box in *crop* pixels -- the face detector runs on the crop."""
    return Region(rect=Rect(x, y, side, side), score=score, label="face")


@pytest.fixture
def stubs(monkeypatch):
    """A body detector and a face detector, told apart by name."""
    body, faces = StubDetector(), StubDetector()
    body._current, faces._current = [BODY], [head()]
    created = []

    def fake_create(name, **kwargs):
        created.append((name, kwargs))
        return faces if name == "yunet" else body

    monkeypatch.setattr(pipeline_module.detectors, "create", fake_create)
    return body, faces, created


def masked_config(tmp_path, **overrides):
    return base_config(tmp_path, mask_faces=True, **overrides)


def row(tmp_path, n=0):
    lines = (tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()
    return json.loads(lines[n])


# -- the mask image ----------------------------------------------------------
def test_mask_name_sits_beside_the_crop_and_is_always_png():
    assert mask_name("0001.png") == "0001-masklabel.png"
    assert mask_name("768/body_0042.jpg") == "768/body_0042-masklabel.png"


def test_mask_is_black_over_the_face_and_white_elsewhere():
    mask = build_mask(1024, [Rect(462, 100, 100, 100)], margin=0.35)

    assert mask.mode == "L" and mask.size == (1024, 1024)
    assert mask.getpixel((512, 150)) == 0       # the face itself
    assert mask.getpixel((512, 80)) == 0        # hair, above the box
    assert mask.getpixel((440, 150)) == 0       # ears, beside the box
    assert mask.getpixel((512, 600)) == 255     # the body
    assert mask.getpixel((0, 0)) == 255


def test_mask_margin_widens_the_covered_area():
    tight = build_mask(1024, [Rect(462, 100, 100, 100)], margin=0.0)
    loose = build_mask(1024, [Rect(462, 100, 100, 100)], margin=0.5)

    assert tight.getpixel((420, 150)) == 255
    assert loose.getpixel((420, 150)) == 0


def test_no_faces_gives_an_all_white_mask():
    assert build_mask(512, [], margin=0.35).getextrema() == (255, 255)


# -- the run -------------------------------------------------------------------
def test_mask_is_written_beside_each_crop(tmp_path, stubs):
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(masked_config(tmp_path)).run()

    assert stats.written == 1 and stats.masked == 1
    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.size == (1024, 1024)
    assert mask.getpixel((512, 150)) == 0
    assert mask.getpixel((512, 600)) == 255
    assert row(tmp_path)["mask_file"] == "0001-masklabel.png"
    assert row(tmp_path)["masked_faces"] == 1


def test_the_crop_itself_is_left_untouched(tmp_path, stubs):
    make_photo(tmp_path / "photos" / "a.png", colour=(120, 140, 160))

    Pipeline(masked_config(tmp_path)).run()

    crop = Image.open(tmp_path / "out" / "0001.png")
    assert crop.getpixel((512, 150)) == (120, 140, 160)


def test_every_face_in_the_crop_is_masked(tmp_path, stubs):
    _, faces, _ = stubs
    faces._current = [head(462, 100), head(100, 300, side=60)]
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(masked_config(tmp_path)).run()

    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.getpixel((512, 150)) == 0
    assert mask.getpixel((130, 330)) == 0
    assert row(tmp_path)["masked_faces"] == 2


def test_a_crop_with_no_face_is_skipped_by_default(tmp_path, stubs):
    _, faces, _ = stubs
    faces._current = []
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(masked_config(tmp_path)).run()

    assert stats.written == 0 and stats.skipped_no_face == 1
    assert not list((tmp_path / "out").glob("*.png"))
    summary = json.loads((tmp_path / "out" / SUMMARY_NAME).read_text())
    assert summary["skipped"][0]["reason"] == "no_face"


def test_keep_uses_a_faceless_crop_with_an_all_white_mask(tmp_path, stubs):
    _, faces, _ = stubs
    faces._current = []
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(masked_config(tmp_path, mask_missing="keep")).run()

    assert stats.written == 1 and stats.masked == 0 and stats.unmasked_kept == 1
    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.getextrema() == (255, 255)
    assert row(tmp_path)["masked_faces"] == 0


def test_masks_follow_per_size_dirs_and_stay_png(tmp_path, stubs):
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(masked_config(tmp_path, per_size_dirs=True, format="jpg")).run()

    assert (tmp_path / "out" / "1024" / "0001.jpg").exists()
    assert (tmp_path / "out" / "1024" / "0001-masklabel.png").exists()


def test_face_detector_uses_the_mask_score(tmp_path, stubs):
    _, _, created = stubs
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(masked_config(tmp_path, mask_min_score=0.3)).run()

    kwargs = dict(created)["yunet"]
    assert kwargs["min_score"] == 0.3


def test_no_face_detector_without_mask_faces(tmp_path, stubs):
    _, _, created = stubs
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(base_config(tmp_path)).run()

    assert "yunet" not in dict(created)
    assert not list((tmp_path / "out").glob("*-masklabel.png"))
    assert row(tmp_path)["mask_file"] is None
    assert stats.masked == 0


def test_dry_run_writes_no_masks(tmp_path, stubs):
    make_photo(tmp_path / "photos" / "a.png")

    stats = Pipeline(masked_config(tmp_path, dry_run=True)).run()

    assert stats.masked == 1
    assert not (tmp_path / "out").exists()


# -- training config ---------------------------------------------------------
def _template(tmp_path, monkeypatch):
    path = tmp_path / "template.json"
    path.write_text(json.dumps({
        "batch_size": 1, "resolution": "", "epochs": 0,
        "masked_training": False, "unmasked_probability": 0.1, "unmasked_weight": 0.1,
    }))
    monkeypatch.setattr(pipeline_module, "ONETRAINER_TEMPLATE_PATH", path)


def test_training_config_turns_masked_training_on(tmp_path, stubs, monkeypatch):
    _template(tmp_path, monkeypatch)
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(masked_config(tmp_path, training_config=True)).run()

    written = json.loads((tmp_path / "out" / TRAINING_CONFIG_NAME).read_text())
    assert written["masked_training"] is True
    assert written["unmasked_probability"] == 0.0
    assert written["unmasked_weight"] == 0.0


def test_training_config_leaves_masking_off_without_masks(tmp_path, stubs, monkeypatch):
    _template(tmp_path, monkeypatch)
    make_photo(tmp_path / "photos" / "a.png")

    Pipeline(base_config(tmp_path, training_config=True)).run()

    written = json.loads((tmp_path / "out" / TRAINING_CONFIG_NAME).read_text())
    assert written["masked_training"] is False


def test_training_config_only_detects_masks_from_the_manifest(tmp_path, stubs, monkeypatch):
    _template(tmp_path, monkeypatch)
    make_photo(tmp_path / "photos" / "a.png")
    Pipeline(masked_config(tmp_path)).run()

    Pipeline(base_config(tmp_path, training_config_only=True)).run()

    written = json.loads((tmp_path / "out" / TRAINING_CONFIG_NAME).read_text())
    assert written["masked_training"] is True


# -- configuration -------------------------------------------------------------
@pytest.mark.parametrize("detector", ["yunet", "haar"])
def test_mask_faces_rejects_a_face_detector(detector):
    with pytest.raises(ValueError, match="mask-faces"):
        Config(detector=detector, mask_faces=True).validate()


@pytest.mark.parametrize("field, value", [
    ("mask_margin", -0.1), ("mask_missing", "estimate"), ("mask_min_score", 1.5),
])
def test_bad_mask_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        Config(detector="yolox", **{field: value}).validate()


def test_mask_margin_flag_is_a_percentage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep a stray ./robocrop.toml out of it
    cfg = build_config([
        "-i", str(tmp_path), "-d", "yolox", "--mask-faces", "--mask-margin", "50",
    ])
    assert cfg.mask_faces and cfg.mask_margin == 0.5


def test_mask_margin_in_toml_is_a_percentage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    toml = tmp_path / "run.toml"
    toml.write_text('detector = "yolox"\nmask_faces = true\nmask_margin = 20\n')

    cfg = build_config(["-i", str(tmp_path), "--config", str(toml)])

    assert cfg.mask_margin == pytest.approx(0.2)


def test_a_small_face_beside_a_large_one_is_still_fully_masked():
    """One blur covers the whole mask; the small face's oval must survive it."""
    mask = build_mask(1024, [Rect(100, 100, 300, 300), Rect(700, 300, 30, 30)], margin=0.35)

    assert mask.getpixel((250, 250)) == 0
    assert mask.getpixel((715, 315)) == 0
