"""Selective recaptioning preserves the rest of an existing dataset."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from test_pipeline import base_config, make_photo

from robocrop import pipeline as pipeline_module
from robocrop.cli import build_config
from robocrop.config import Config
from robocrop.pipeline import MANIFEST_NAME, Pipeline


def make_dataset(tmp_path, **options):
    for name in ("a.png", "b.png", "c.png"):
        make_photo(tmp_path / "photos" / name, size=(512, 384))
    Pipeline(base_config(tmp_path, skip_detection=True, **options)).run()
    return tmp_path / "out"


def rows(out):
    return [json.loads(line) for line in (out / MANIFEST_NAME).read_text().splitlines()]


@pytest.mark.parametrize("per_size_dirs", [False, True])
@pytest.mark.parametrize(
    "caption_model", ["smolvlm", "joycaption", "joycaption-nf4", "huihui-qwen3-vl-4b"]
)
def test_selected_images_receive_prompt_and_preserve_other_records(
    tmp_path, monkeypatch, per_size_dirs, caption_model
):
    out = make_dataset(tmp_path, captioner="template", per_size_dirs=per_size_dirs)
    before_rows = rows(out)
    untouched = out / before_rows[1]["caption_file"]
    untouched.write_text("My manually edited caption\n")
    before_files = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    calls = []
    seen = []

    class Captioner:
        def caption_batch(self, requests):
            seen.extend(r.index for r in requests)
            return [f"new caption {r.index}" for r in requests]

        def close(self):
            pass

    def create(name, **kwargs):
        calls.append((name, kwargs))
        return Captioner()

    def forbidden(*args, **kwargs):
        raise AssertionError("recaptioning must not detect or crop")

    monkeypatch.setattr(pipeline_module.captioners, "create", create)
    monkeypatch.setattr(pipeline_module.detectors, "create", forbidden)
    first, third = before_rows[0]["image_file"], before_rows[2]["image_file"]
    stats = Pipeline(
        base_config(
            tmp_path,
            caption_only=True,
            captioner="vlm",
            caption_model=caption_model,
            caption_images=(first, str(out / third), first),
            caption_prompt="Describe the clothing.",
            caption_batch=1,
        )
    ).run()

    assert stats.captioned == stats.scanned == stats.written == 2
    assert seen == [1, 3]
    assert calls[0][0] == "vlm"
    assert calls[0][1]["model"] == caption_model
    assert calls[0][1]["prompt"] == "Describe the clothing."
    after_rows = rows(out)
    assert after_rows[1] == before_rows[1]
    assert after_rows[0]["caption"] == "new caption 1"
    assert after_rows[2]["caption"] == "new caption 3"
    changed = {
        out / MANIFEST_NAME,
        out / before_rows[0]["caption_file"],
        out / before_rows[2]["caption_file"],
    }
    for path, content in before_files.items():
        if path not in changed:
            assert path.read_bytes() == content


def test_selection_from_crop_only_dataset_preserves_unselected_rows(tmp_path):
    out = make_dataset(tmp_path, captioner="none")
    before = rows(out)
    stats = Pipeline(
        base_config(
            tmp_path,
            caption_only=True,
            captioner="template",
            caption_images=("0002.png",),
        )
    ).run()
    assert stats.captioned == 1
    after = rows(out)
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1]["caption_file"] == "0002.txt"
    assert not (out / "0001.txt").exists()
    assert not (out / "0003.txt").exists()


@pytest.mark.parametrize("unknown", ["typo.png", "../outside.png"])
def test_unknown_selection_fails_before_any_caption_or_manifest_changes(
    tmp_path, monkeypatch, unknown
):
    out = make_dataset(tmp_path, captioner="template")
    before = {p: p.read_bytes() for p in out.iterdir() if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selection must not load a captioner")

    monkeypatch.setattr(pipeline_module.captioners, "create", forbidden)
    with pytest.raises(ValueError, match="not a dataset image"):
        Pipeline(
            base_config(
                tmp_path,
                caption_only=True,
                captioner="template",
                caption_images=("0001.png", unknown),
            )
        ).run()
    assert {p: p.read_bytes() for p in out.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_selection_resume_retries_only_selected_missing_or_empty_captions(
    tmp_path, dry_run
):
    out = make_dataset(tmp_path, captioner="template")
    (out / "0002.txt").write_text("")
    (out / "0003.txt").unlink()
    before = {p: p.read_bytes() for p in out.iterdir() if p.is_file()}
    stats = Pipeline(
        base_config(
            tmp_path,
            caption_only=True,
            captioner="template",
            resume=True,
            caption_images=("0001.png", "0002.png"),
            dry_run=dry_run,
        )
    ).run()
    assert stats.scanned == stats.written == 1
    assert stats.skipped_existing == 1
    assert stats.captioned == (0 if dry_run else 1)
    assert not (out / "0003.txt").exists()
    if dry_run:
        assert {p: p.read_bytes() for p in out.iterdir() if p.is_file()} == before
    else:
        assert (out / "0002.txt").read_text().strip()


def test_cli_multiple_and_repeated_selections_override_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robocrop.toml").write_text('caption_images = ["old.png"]\n')
    cfg = build_config(
        [
            "--caption-only",
            "--caption-images",
            "0001.png",
            "0002.png",
            "--caption-images",
            "0003.png",
            "--caption-prompt",
            "Describe the pose.",
        ]
    )
    assert cfg.caption_images == ("0001.png", "0002.png", "0003.png")
    assert cfg.caption_prompt == "Describe the pose."
    assert build_config(["--caption-only"]).caption_images == ("old.png",)


def test_selection_requires_caption_only():
    with pytest.raises(ValueError, match="requires --caption-only"):
        Config(caption_images=("0001.png",)).validate()


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        Config(caption_only=True, caption_images=(" ",)).validate()
