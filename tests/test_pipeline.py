"""End-to-end pipeline tests.

A stub detector stands in for the real ones so these run in milliseconds and
assert on the pipeline's own behaviour -- numbering, pairing, sizing, the
manifest and resume -- rather than on detector accuracy.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import pipeline as pipeline_module  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.detectors.base import BaseDetector, Region  # noqa: E402
from robocrop.geometry import Rect  # noqa: E402
from robocrop.pipeline import Pipeline  # noqa: E402


class StubDetector(BaseDetector):
    """Returns boxes planted in the image filename: ``x_y_w_h``, comma separated."""

    name = "stub"

    def __init__(self, min_score=0.6, **kwargs):
        super().__init__(min_score)
        self.boxes_by_name: dict[str, list[tuple[float, ...]]] = {}

    def detect(self, image):
        return self._finalize(self._current)


@pytest.fixture
def stub(monkeypatch):
    """Install a detector whose output the test controls per image."""
    holder = StubDetector()
    holder._current = []

    def fake_create(name, **kwargs):
        return holder

    monkeypatch.setattr(pipeline_module.detectors, "create", fake_create)
    return holder


def make_photo(path: Path, size=(4000, 3000), colour=(120, 140, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def make_noisy_photo(path: Path, size=(4000, 3000), seed=0):
    """A photo with real high-frequency content, for sharpness tests."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


def face(x, y, side, score=0.9):
    return Region(rect=Rect(x, y, side, side), score=score, label="face")


def base_config(tmp_path, **overrides):
    settings = {
        "input": tmp_path / "photos",
        "output": tmp_path / "out",
        "detector": "stub",
        "captioner": "none",
        "workers": 1,
        "min_sharpness": 0,
    }
    settings.update(overrides)
    cfg = Config(**settings)
    cfg.validate()
    return cfg


# -- sizing ---------------------------------------------------------------
@pytest.mark.parametrize(
    "side, expected_size",
    [(1200, 1024), (820, 1024), (700, 768), (615, 768), (500, 512), (410, 512)],
)
def test_crop_is_written_at_the_largest_qualifying_size(tmp_path, stub, side, expected_size):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, side)]

    stats = Pipeline(base_config(tmp_path)).run()

    assert stats.written == 1
    out = tmp_path / "out" / "0001.png"
    assert Image.open(out).size == (expected_size, expected_size)


def test_faces_below_the_floor_produce_nothing(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(100, 100, 300)]     # under 80% of 512

    stats = Pipeline(base_config(tmp_path)).run()

    assert stats.written == 0
    assert stats.skipped_too_small == 1
    assert not list((tmp_path / "out").glob("*.png"))


def test_skip_reason_is_recorded_in_the_summary(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(100, 100, 300)]

    Pipeline(base_config(tmp_path)).run()

    summary = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert summary["skipped"][0]["reason"] == "too_small"


# -- multiple faces -------------------------------------------------------
def test_every_face_becomes_its_own_numbered_crop(tmp_path, stub):
    make_photo(tmp_path / "photos" / "group.png")
    stub._current = [face(0, 0, 1000), face(1500, 500, 800), face(2500, 1200, 600)]

    stats = Pipeline(base_config(tmp_path, multi="all")).run()

    assert stats.written == 3
    names = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
    assert names == ["0001.png", "0002.png", "0003.png"]


def test_largest_mode_keeps_only_the_biggest_face(tmp_path, stub):
    make_photo(tmp_path / "photos" / "group.png")
    stub._current = [face(0, 0, 1000), face(1500, 500, 800)]

    stats = Pipeline(base_config(tmp_path, multi="largest")).run()

    assert stats.written == 1
    assert Image.open(tmp_path / "out" / "0001.png").size == (1024, 1024)


def test_skip_mode_drops_multi_face_photos(tmp_path, stub):
    make_photo(tmp_path / "photos" / "group.png")
    stub._current = [face(0, 0, 1000), face(1500, 500, 800)]

    stats = Pipeline(base_config(tmp_path, multi="skip")).run()

    assert stats.written == 0
    assert stats.skipped_multi == 1


def test_max_per_image_caps_the_crops(tmp_path, stub):
    make_photo(tmp_path / "photos" / "group.png")
    stub._current = [face(0, 0, 1000), face(1500, 500, 900), face(2500, 1200, 800)]

    stats = Pipeline(base_config(tmp_path, multi="all", max_per_image=2)).run()

    assert stats.written == 2


# -- recursion and output layout -----------------------------------------
def test_subdirectories_are_scanned(tmp_path, stub):
    for name in ["a.png", "sub/b.png", "sub/deeper/c.png"]:
        make_photo(tmp_path / "photos" / name)
    stub._current = [face(1000, 800, 1000)]

    stats = Pipeline(base_config(tmp_path)).run()

    assert stats.scanned == 3
    assert stats.written == 3


def test_captions_share_the_image_filename(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path, captioner="template", trigger="my subject")).run()

    out = tmp_path / "out"
    assert (out / "0001.png").exists()
    assert (out / "0001.txt").exists()
    assert (out / "0001.txt").read_text().startswith("my subject,")


def test_prefix_digits_and_format_are_honoured(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(
        tmp_path, prefix="subj_", digits=3, start_index=7, format="jpg",
    )).run()

    assert (tmp_path / "out" / "subj_007.jpg").exists()


def test_per_size_dirs_split_the_output(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000), face(0, 0, 600)]

    Pipeline(base_config(tmp_path, per_size_dirs=True)).run()

    assert (tmp_path / "out" / "1024" / "0001.png").exists()
    assert (tmp_path / "out" / "512" / "0002.png").exists()


def test_excluded_paths_are_not_scanned(tmp_path, stub):
    make_photo(tmp_path / "photos" / "keep.png")
    make_photo(tmp_path / "photos" / "junk" / "drop.png")
    stub._current = [face(1000, 800, 1000)]

    stats = Pipeline(base_config(tmp_path, exclude=("junk",))).run()

    assert stats.scanned == 1


# -- manifest and resume --------------------------------------------------
def test_manifest_row_describes_the_crop(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path)).run()

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == 1
    assert row["tier"] == 1024
    assert row["base_side"] == 1000
    assert row["source"].endswith("a.png")


def test_a_second_run_into_the_same_output_is_refused(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]
    Pipeline(base_config(tmp_path)).run()

    with pytest.raises(FileExistsError):
        Pipeline(base_config(tmp_path)).run()


def test_resume_continues_numbering_without_redoing_work(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]
    Pipeline(base_config(tmp_path)).run()

    make_photo(tmp_path / "photos" / "b.png")
    stats = Pipeline(base_config(tmp_path, resume=True)).run()

    assert stats.written == 1           # only the new photo
    assert stats.skipped_existing == 1
    assert (tmp_path / "out" / "0002.png").exists()


def test_dry_run_writes_nothing(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]

    stats = Pipeline(base_config(tmp_path, dry_run=True)).run()

    assert stats.written == 1           # reported
    assert not (tmp_path / "out").exists()


# -- robustness -----------------------------------------------------------
def test_an_unreadable_file_does_not_stop_the_run(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    (tmp_path / "photos" / "broken.png").write_bytes(b"not an image")
    make_photo(tmp_path / "photos" / "z.png")
    stub._current = [face(1000, 800, 1000)]

    stats = Pipeline(base_config(tmp_path)).run()

    assert stats.errors == 1
    assert stats.written == 2


def test_photos_without_faces_are_counted_not_crashed(tmp_path, stub):
    make_photo(tmp_path / "photos" / "empty.png")
    stub._current = []

    stats = Pipeline(base_config(tmp_path)).run()

    assert stats.written == 0
    assert stats.skipped_no_detection == 1


# -- manifest integrity across runs ---------------------------------------
def test_resume_with_captions_keeps_earlier_manifest_rows(tmp_path, stub):
    """The caption pass rewrites the manifest; prior rows must survive it."""
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]
    Pipeline(base_config(tmp_path, captioner="template")).run()

    make_photo(tmp_path / "photos" / "b.png")
    Pipeline(base_config(tmp_path, captioner="template", resume=True)).run()

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    ]
    assert [r["index"] for r in rows] == [1, 2]
    assert sorted(Path(r["source"]).name for r in rows) == ["a.png", "b.png"]
    assert all(r["caption"] for r in rows)


def test_overwrite_starts_the_manifest_afresh(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(1000, 800, 1000)]
    Pipeline(base_config(tmp_path)).run()

    Pipeline(base_config(tmp_path, overwrite=True)).run()

    rows = (tmp_path / "out" / "manifest.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1


# -- detector-supplied framing correction ---------------------------------
def _crop_top(tmp_path):
    """Top edge of the crop window, read back from the manifest."""
    row = json.loads(
        (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()[0]
    )
    return row["crop"][1], row["offset_y"]


def test_face_detector_lifts_the_crop_by_default(tmp_path, stub):
    """A face box excludes hair, so its crops shift up unless told otherwise."""
    make_photo(tmp_path / "photos" / "a.png")
    stub.recommended_offset_y = -0.08
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path)).run()

    top, offset = _crop_top(tmp_path)
    assert offset == -0.08
    # 8% of the 1000px box, above where an unshifted window would sit.
    assert top == pytest.approx(800 - 200 - 80)


def test_a_detector_without_a_recommendation_is_not_shifted(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub.recommended_offset_y = 0.0
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path)).run()

    top, offset = _crop_top(tmp_path)
    assert offset == 0.0
    assert top == pytest.approx(800 - 200)


def test_explicit_offset_overrides_the_detector(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")
    stub.recommended_offset_y = -0.08
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path, offset_y=-0.2)).run()

    top, offset = _crop_top(tmp_path)
    assert offset == -0.2
    assert top == pytest.approx(800 - 200 - 200)


def test_explicit_zero_disables_the_recommendation(tmp_path, stub):
    """0.0 must mean 'no shift', not 'unset'."""
    make_photo(tmp_path / "photos" / "a.png")
    stub.recommended_offset_y = -0.08
    stub._current = [face(1000, 800, 1000)]

    Pipeline(base_config(tmp_path, offset_y=0.0)).run()

    top, offset = _crop_top(tmp_path)
    assert offset == 0.0
    assert top == pytest.approx(800 - 200)


# -- OneTrainer config -----------------------------------------------------
def _write_onetrainer_template(tmp_path, **overrides):
    data = {"batch_size": 2, "resolution": "", "epochs": 0}
    data.update(overrides)
    path = tmp_path / "template.json"
    path.write_text(json.dumps(data))
    return path


def test_training_config_off_by_default(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH", _write_onetrainer_template(tmp_path)
    )
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 1000)]

    Pipeline(base_config(tmp_path)).run()

    assert not (tmp_path / "out" / "training_config.json").exists()


def test_training_config_uses_smallest_tier_and_computed_epochs(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH",
        _write_onetrainer_template(tmp_path, batch_size=2),
    )
    make_photo(tmp_path / "photos" / "a.png")
    # 15 crops land at tier 1024, 15 at tier 768 -> 30 images total.
    stub._current = [face(0, 0, 1000) for _ in range(15)] + [face(0, 0, 700) for _ in range(15)]

    stats = Pipeline(base_config(tmp_path, training_config=True)).run()

    assert stats.written == 30
    written = json.loads((tmp_path / "out" / "training_config.json").read_text())
    assert written["resolution"] == "768"          # the smaller of the two tiers produced
    assert written["epochs"] == 267                # round(4000 * 2 / 30)


def test_training_config_epochs_clamp_to_300_for_a_tiny_dataset(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH",
        _write_onetrainer_template(tmp_path, batch_size=2),
    )
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 1000)]

    Pipeline(base_config(tmp_path, training_config=True)).run()

    written = json.loads((tmp_path / "out" / "training_config.json").read_text())
    assert written["epochs"] == 300                # raw 4000*2/1=8000, clamped down


def test_training_config_epochs_clamp_to_10_for_a_large_dataset(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH",
        _write_onetrainer_template(tmp_path, batch_size=1),
    )
    make_photo(tmp_path / "photos" / "a.png", size=(900, 900))
    stub._current = [face(0, 0, 500) for _ in range(500)]

    stats = Pipeline(base_config(tmp_path, training_config=True)).run()

    assert stats.written == 500
    written = json.loads((tmp_path / "out" / "training_config.json").read_text())
    assert written["epochs"] == 10                 # raw 4000*1/500=8, clamped up


def test_training_config_respects_dry_run(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH", _write_onetrainer_template(tmp_path)
    )
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 1000)]

    Pipeline(base_config(tmp_path, training_config=True, dry_run=True)).run()

    assert not (tmp_path / "out" / "training_config.json").exists()


def test_training_config_skipped_when_nothing_is_written(tmp_path, stub, monkeypatch):
    monkeypatch.setattr(
        pipeline_module, "ONETRAINER_TEMPLATE_PATH", _write_onetrainer_template(tmp_path)
    )
    make_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(100, 100, 300)]           # below the floor for every tier

    stats = Pipeline(base_config(tmp_path, training_config=True)).run()

    assert stats.written == 0
    assert not (tmp_path / "out" / "training_config.json").exists()


# -- blur / sharpness -------------------------------------------------------
def test_blurry_crop_is_dropped(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")          # flat colour -> 0 variance
    stub._current = [face(0, 0, 1000)]

    stats = Pipeline(base_config(tmp_path, min_sharpness=10)).run()

    assert stats.written == 0
    assert stats.skipped_blurry == 1
    summary = json.loads((tmp_path / "out" / "manifest.json").read_text())
    skipped = summary["skipped"][0]
    assert skipped["reason"] == "blurry"
    assert skipped["sharpness"] == 0.0


def test_sharp_crop_is_kept(tmp_path, stub):
    make_noisy_photo(tmp_path / "photos" / "a.png")
    stub._current = [face(0, 0, 1000)]

    stats = Pipeline(base_config(tmp_path, min_sharpness=10)).run()

    assert stats.written == 1
    assert stats.skipped_blurry == 0


def test_sharpness_is_recorded_even_when_the_check_is_disabled(tmp_path, stub):
    make_photo(tmp_path / "photos" / "a.png")          # flat colour -> 0 variance
    stub._current = [face(0, 0, 1000)]

    Pipeline(base_config(tmp_path, min_sharpness=0)).run()

    manifest = json.loads(
        (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()[0]
    )
    assert manifest["sharpness"] == 0.0
