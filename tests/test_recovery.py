"""Interrupted crop/caption runs, manifest integrity and CLI failures."""

# Pytest discovers the imported fixture; parameters intentionally reuse its name.
# ruff: noqa: F811

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from robocrop import cli, pipeline as pipeline_module  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, Pipeline, Stats  # noqa: E402
from test_pipeline import base_config, face, make_photo, stub  # noqa: E402,F401


def rows(tmp_path):
    return [json.loads(line) for line in
            (tmp_path / "out" / MANIFEST_NAME).read_text().splitlines()]



def two_crops(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png", size=(1500, 1000))
    stub._current = [face(50, 50, 900), face(900, 50, 500)]



def test_resume_finishes_a_partially_saved_multi_face_photo(tmp_path, stub, monkeypatch):
    two_crops(tmp_path, stub)
    original_save = Pipeline._save

    def interrupted(self, crop, record):
        if record.face_index == 1:
            raise KeyboardInterrupt
        original_save(self, crop, record)

    with monkeypatch.context() as patch:
        patch.setattr(Pipeline, "_save", interrupted)
        with pytest.raises(KeyboardInterrupt):
            Pipeline(base_config(tmp_path)).run()
    first = (tmp_path / "out" / "0001.png").read_bytes()

    stats = Pipeline(base_config(tmp_path, resume=True)).run()

    assert stats.written == 1
    assert [(r["index"], r["face_index"]) for r in rows(tmp_path)] == [(1, 0), (2, 1)]
    assert (tmp_path / "out" / "0001.png").read_bytes() == first
    assert (tmp_path / "out" / "0002.png").exists()



def test_normal_resume_recovers_the_interrupted_caption_pass(tmp_path, stub, monkeypatch):
    two_crops(tmp_path, stub)

    def interrupted(self):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(Pipeline, "_caption_pass", interrupted)
        with pytest.raises(KeyboardInterrupt):
            Pipeline(base_config(tmp_path, captioner="template")).run()

    stats = Pipeline(base_config(tmp_path, captioner="template", resume=True)).run()

    assert stats.written == 0
    assert stats.captioned == 2
    assert all(r["caption"] for r in rows(tmp_path))
    assert len(list((tmp_path / "out").glob("*.txt"))) == 2



def test_resume_recovers_saved_caption_text_without_replacing_it(tmp_path, stub):
    two_crops(tmp_path, stub)
    Pipeline(base_config(tmp_path, captioner="template")).run()
    text_file = tmp_path / "out" / "0001.txt"
    text_file.write_text("manually corrected caption\n")
    (tmp_path / "out" / "0002.txt").write_text(" \n")

    stats = Pipeline(base_config(tmp_path, captioner="template", resume=True)).run()

    assert stats.captioned == 1
    assert text_file.read_text() == "manually corrected caption\n"
    assert rows(tmp_path)[0]["caption"] == "manually corrected caption"
    assert rows(tmp_path)[1]["caption"]



@pytest.mark.parametrize("failure", ["short", "exception"])
def test_caption_batch_failure_is_explicit_and_closes_images(tmp_path, stub, monkeypatch, failure):
    two_crops(tmp_path, stub)
    Pipeline(base_config(tmp_path)).run()

    class Captioner:
        closed = False
        requests = []

        def caption_batch(self, requests):
            self.requests = requests
            if failure == "exception":
                raise RuntimeError("inference failed")
            return ["only one caption"]

        def close(self):
            self.closed = True

    captioner = Captioner()
    monkeypatch.setattr(pipeline_module.captioners, "create", lambda *a, **k: captioner)
    error = RuntimeError if failure == "exception" else ValueError
    with pytest.raises(error):
        Pipeline(base_config(tmp_path, captioner="template", caption_only=True)).run()

    assert captioner.closed
    for request in captioner.requests:
        with pytest.raises(ValueError, match="closed"):
            request.image.getpixel((0, 0))
    assert not list((tmp_path / "out").glob("*.txt"))



@pytest.mark.parametrize("mode", ["resume", "overwrite", "caption_only"])
def test_interior_manifest_corruption_is_rejected_before_any_changes(tmp_path, stub, mode):
    two_crops(tmp_path, stub)
    Pipeline(base_config(tmp_path)).run()
    manifest = tmp_path / "out" / MANIFEST_NAME
    lines = manifest.read_text().splitlines()
    damaged = lines[0] + "\n{invalid json}\n" + lines[1] + "\n"
    manifest.write_text(damaged)
    original = (tmp_path / "out" / "0001.png").read_bytes()
    options = {mode: True}
    if mode == "caption_only":
        options["captioner"] = "template"

    with pytest.raises(ValueError, match="line 2"):
        Pipeline(base_config(tmp_path, **options)).run()

    assert manifest.read_text() == damaged
    assert (tmp_path / "out" / "0001.png").read_bytes() == original



def test_summary_contains_actual_elapsed_time(tmp_path, stub, monkeypatch):
    two_crops(tmp_path, stub)
    ticks = iter([100.0, 105.0])
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: next(ticks))
    stats = Pipeline(base_config(tmp_path)).run()
    summary = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert stats.seconds == summary["stats"]["seconds"] == 5.0



def test_cli_reports_a_partial_run_as_failed(monkeypatch):
    monkeypatch.setattr(cli, "build_config", lambda argv: Config(quiet=True))
    monkeypatch.setattr(Pipeline, "run", lambda self: Stats(scanned=2, written=1, errors=1))
    assert cli.main([]) == 1



def test_cli_reports_disk_errors_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_config", lambda argv: Config(quiet=True))

    def failed(self):
        raise OSError("disk is full")

    monkeypatch.setattr(Pipeline, "run", failed)
    assert cli.main([]) == 1
    assert capsys.readouterr().err == "error: disk is full\n"



def test_a_later_caption_batch_failure_preserves_completed_manifest_rows(tmp_path, stub, monkeypatch):
    two_crops(tmp_path, stub)
    Pipeline(base_config(tmp_path)).run()

    class Captioner:
        calls = 0

        def caption_batch(self, requests):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second batch failed")
            return ["finished caption"]

        def close(self):
            pass

    monkeypatch.setattr(pipeline_module.captioners, "create", lambda *a, **k: Captioner())
    with pytest.raises(RuntimeError, match="second batch"):
        Pipeline(base_config(tmp_path, captioner="template", caption_only=True, caption_batch=1)).run()
    assert rows(tmp_path)[0]["caption"] == "finished caption"
    assert (tmp_path / "out" / "0001.txt").read_text() == "finished caption\n"
    assert not (tmp_path / "out" / "0002.txt").exists()
