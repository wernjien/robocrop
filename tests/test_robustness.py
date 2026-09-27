"""Regression tests: resume after a crash, --overwrite, re-scanning our own
output, and manifests that point somewhere they should not."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.images import iter_images  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, TRAINING_CONFIG_NAME, Pipeline  # noqa: E402

from test_pipeline import base_config, face, make_photo, stub  # noqa: E402,F401


def manifest_rows(tmp_path):
    text = (tmp_path / "out" / MANIFEST_NAME).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# -- caption-only --resume ---------------------------------------------------
def test_caption_only_resume_keeps_every_manifest_row(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900), face(2000, 1000, 900)]
    Pipeline(base_config(tmp_path, captioner="none")).run()
    Pipeline(base_config(tmp_path, captioner="template", caption_only=True)).run()
    (tmp_path / "out" / "0001.txt").unlink()

    Pipeline(base_config(tmp_path, captioner="template", caption_only=True, resume=True)).run()

    rows = manifest_rows(tmp_path)
    assert [r["index"] for r in rows] == [1, 2]
    assert all(r["caption"] for r in rows)


# -- --overwrite ------------------------------------------------------------
def test_overwrite_removes_the_previous_runs_files(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900), face(2000, 1000, 900)]
    Pipeline(base_config(tmp_path, captioner="template")).run()
    stranger = tmp_path / "out" / "notes.txt"
    stranger.write_text("mine")

    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path, captioner="none", overwrite=True)).run()

    out = tmp_path / "out"
    assert (out / "0001.png").exists()
    assert not (out / "0002.png").exists()       # the old run's extra crop
    assert not (out / "0001.txt").exists()       # its caption, beside a new crop
    assert stranger.read_text() == "mine"        # files it never made are kept


def test_overwrite_warns_about_a_stale_training_config(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path)).run()
    (tmp_path / "out" / TRAINING_CONFIG_NAME).write_text("{}")
    events = []

    Pipeline(base_config(tmp_path, overwrite=True),
             on_event=lambda level, msg: events.append((level, msg))).run()

    assert any(level == "warn" and TRAINING_CONFIG_NAME in msg for level, msg in events)


def test_overwrite_dry_run_deletes_nothing(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900), face(2000, 1000, 900)]
    Pipeline(base_config(tmp_path)).run()

    Pipeline(base_config(tmp_path, overwrite=True, dry_run=True)).run()

    assert (tmp_path / "out" / "0002.png").exists()


def test_overwrite_never_deletes_outside_the_output(tmp_path, stub):
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"keep me")
    out = tmp_path / "out"
    out.mkdir()
    (out / MANIFEST_NAME).write_text(json.dumps({"index": 1, "image_file": "../victim.png"}) + "\n")
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]

    Pipeline(base_config(tmp_path, overwrite=True)).run()

    assert victim.read_bytes() == b"keep me"


def test_caption_only_refuses_a_manifest_path_outside_the_output(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path)).run()
    manifest = tmp_path / "out" / MANIFEST_NAME
    row = json.loads(manifest.read_text())
    row["image_file"] = "../../escaped.png"
    manifest.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="outside"):
        Pipeline(base_config(tmp_path, captioner="template", caption_only=True)).run()


# -- resuming after a crash ------------------------------------------------
def test_resume_after_a_crash_replaces_the_orphan_and_the_torn_row(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path)).run()

    # Killed after saving crop 2 but mid-way through writing its row.
    out = tmp_path / "out"
    Image.new("RGB", (8, 8)).save(out / "0002.png")
    with (out / MANIFEST_NAME).open("a") as handle:
        handle.write('{"index": 2, "sou')
    make_photo(tmp_path / "photos" / "b.png")

    stats = Pipeline(base_config(tmp_path, resume=True)).run()

    assert stats.written == 1
    assert [r["index"] for r in manifest_rows(tmp_path)] == [1, 2]
    assert Image.open(out / "0002.png").size == (1024, 1024)


def test_resume_without_a_manifest_does_not_overwrite_strangers(tmp_path, stub):
    out = tmp_path / "out"
    out.mkdir()
    Image.new("RGB", (8, 8)).save(out / "0001.png")
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]

    with pytest.raises(FileExistsError):
        Pipeline(base_config(tmp_path, resume=True)).run()
    assert Image.open(out / "0001.png").size == (8, 8)


def test_resume_mends_a_last_row_missing_its_newline(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path)).run()
    manifest = tmp_path / "out" / MANIFEST_NAME
    manifest.write_text(manifest.read_text().rstrip("\n"))
    make_photo(tmp_path / "photos" / "b.png")

    Pipeline(base_config(tmp_path, resume=True)).run()

    assert [r["index"] for r in manifest_rows(tmp_path)] == [1, 2]


def test_resume_training_config_counts_the_whole_dataset(tmp_path, stub, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"batch_size": 1}))
    monkeypatch.setattr(pipeline_module, "ONETRAINER_TEMPLATE_PATH", template)

    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 700) for _ in range(40)]      # 40 crops at 768
    Pipeline(base_config(tmp_path)).run()
    make_photo(tmp_path / "photos" / "b.png")
    stub._current = [face(0, 0, 1000) for _ in range(10)]     # 10 more at 1024

    Pipeline(base_config(tmp_path, resume=True, training_config=True)).run()

    written = json.loads((tmp_path / "out" / TRAINING_CONFIG_NAME).read_text())
    assert written["resolution"] == "768"   # not 1024, from the new batch alone
    assert written["epochs"] == 80          # 4000 / 50 crops, not 4000 / 10


def test_resume_with_nothing_new_still_writes_the_training_config(tmp_path, stub, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"batch_size": 1}))
    monkeypatch.setattr(pipeline_module, "ONETRAINER_TEMPLATE_PATH", template)
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 700)]
    Pipeline(base_config(tmp_path)).run()

    Pipeline(base_config(tmp_path, resume=True, training_config=True)).run()

    assert (tmp_path / "out" / TRAINING_CONFIG_NAME).exists()


def test_an_interrupted_manifest_rewrite_leaves_the_old_one(tmp_path, stub, monkeypatch):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path)).run()
    before = (tmp_path / "out" / MANIFEST_NAME).read_text()

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        Pipeline(base_config(tmp_path, captioner="template", caption_only=True)).run()

    assert (tmp_path / "out" / MANIFEST_NAME).read_text() == before
    assert not list((tmp_path / "out").glob(".manifest-*"))


# -- scanning ------------------------------------------------------------------
def test_an_output_inside_the_input_is_not_scanned(tmp_path, stub):
    photos = tmp_path / "photos"
    make_photo(photos / "a.png")
    stub._current = [face(500, 500, 900)]
    Pipeline(base_config(tmp_path, output=photos / "dataset")).run()

    again = Pipeline(base_config(tmp_path, output=photos / "dataset", overwrite=True)).run()
    elsewhere = Pipeline(base_config(tmp_path, output=photos / "dataset2")).run()

    assert again.scanned == 1       # its own output
    assert elsewhere.scanned == 1   # another robocrop dataset under the input


def test_mask_side_files_are_never_scanned_as_photos(tmp_path):
    for name in ("a.png", "a-masklabel.png", "b-condlabel.png"):
        make_photo(tmp_path / name, size=(8, 8))

    assert [p.name for p in iter_images(tmp_path)] == ["a.png"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_a_symlink_loop_does_not_hang_the_scan(tmp_path):
    make_photo(tmp_path / "sub" / "a.png", size=(8, 8))
    (tmp_path / "sub" / "loop").symlink_to(tmp_path, target_is_directory=True)

    found = list(iter_images(tmp_path, follow_symlinks=True))

    assert [p.name for p in found] == ["a.png"]
