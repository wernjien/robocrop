"""Crop-only, caption-only, and training-config-only run modes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from traincrop.config import Config  # noqa: E402
from traincrop.pipeline import Pipeline, MANIFEST_NAME, TRAINING_CONFIG_NAME  # noqa: E402

from test_pipeline import StubDetector, base_config, face, make_photo, stub  # noqa: E402,F401


def test_captioner_none_is_crop_only(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 900)]

    stats = Pipeline(base_config(tmp_path, captioner="none")).run()

    assert stats.written == 1
    assert stats.captioned == 0
    assert not list((tmp_path / "out").glob("*.txt"))


def test_caption_only_captions_an_existing_dataset(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 900)]
    Pipeline(base_config(tmp_path, captioner="none")).run()
    assert not list((tmp_path / "out").glob("*.txt"))

    cfg = base_config(tmp_path, captioner="template", caption_only=True)
    stats = Pipeline(cfg).run()

    assert stats.written == 1
    assert stats.captioned == 1
    txt = tmp_path / "out" / "0001.txt"
    assert txt.exists() and txt.read_text().strip()

    manifest_lines = (tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()
    row = json.loads(manifest_lines[0])
    assert row["caption_file"] == "0001.txt"
    assert row["caption"]


def test_caption_only_with_resume_skips_already_captioned(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 900), face(2500, 1200, 900)]
    Pipeline(base_config(tmp_path, captioner="none", multi="all")).run()

    cfg = base_config(tmp_path, captioner="template", caption_only=True)
    Pipeline(cfg).run()
    (tmp_path / "out" / "0001.txt").unlink()  # pretend only crop 2 was captioned

    cfg2 = base_config(tmp_path, captioner="template", caption_only=True, resume=True)
    stats = Pipeline(cfg2).run()

    assert stats.written == 1
    assert stats.skipped_existing == 1
    assert (tmp_path / "out" / "0001.txt").exists()


def test_caption_only_requires_an_existing_manifest(tmp_path):
    cfg = base_config(tmp_path, captioner="template", caption_only=True)
    with pytest.raises(FileExistsError):
        Pipeline(cfg).run()


def test_caption_only_rejects_captioner_none():
    cfg = Config(input=Path("."), output=Path("./out"), caption_only=True, captioner="none")
    with pytest.raises(ValueError):
        cfg.validate()


def test_training_config_only_generates_from_existing_dataset(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 900)]
    Pipeline(base_config(tmp_path, captioner="none")).run()
    assert not (tmp_path / "out" / TRAINING_CONFIG_NAME).exists()

    cfg = base_config(tmp_path, captioner="none", training_config_only=True)
    stats = Pipeline(cfg).run()

    assert stats.written == 1
    config_path = tmp_path / "out" / TRAINING_CONFIG_NAME
    assert config_path.exists()
    written = json.loads(config_path.read_text())
    assert written["resolution"] == "1024"


def test_training_config_only_requires_an_existing_manifest(tmp_path):
    cfg = base_config(tmp_path, captioner="none", training_config_only=True)
    with pytest.raises(FileExistsError):
        Pipeline(cfg).run()


def test_caption_only_and_training_config_only_are_exclusive():
    cfg = Config(caption_only=True, training_config_only=True)
    with pytest.raises(ValueError):
        cfg.validate()


# -- config checks ------------------------------------------------------------
@pytest.mark.parametrize("field", ["limit", "max_per_image", "workers", "start_index"])
def test_negative_counts_are_rejected(field):
    with pytest.raises(ValueError, match="cannot be negative"):
        Config(**{field: -1}).validate()


@pytest.mark.parametrize("template, message", [
    ("{shot}, {mood}", "unknown token"),
    ("{shot}, {}", "not a valid format string"),
    ("{shot", "not a valid format string"),
])
def test_a_bad_caption_template_is_caught_before_cropping(template, message):
    with pytest.raises(ValueError, match=message):
        Config(captioner="template", caption_template=template).validate()


def test_every_template_token_is_filled():
    from traincrop.captioners.template import TOKENS, TemplateCaptioner
    from traincrop.captioners.base import CaptionRequest
    from traincrop.detectors.base import Region
    from traincrop.geometry import Rect

    captioner = TemplateCaptioner(" ".join("{%s}" % t for t in TOKENS))
    request = CaptionRequest(Image.new("RGB", (64, 64)), "a.png", Region(Rect(0, 0, 8, 8), 0.9), 512, 1)
    assert captioner.caption(request)


def test_output_cannot_be_the_input(tmp_path, monkeypatch):
    from traincrop.cli import build_config

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="different directory"):
        build_config(["-i", str(tmp_path), "-o", str(tmp_path)])
