"""Manifest recovery preserves existing dataset files and supports later captioning."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from robocrop import cli, pipeline as pipeline_module  # noqa: E402
from robocrop.config import Config  # noqa: E402
from robocrop.pipeline import MANIFEST_NAME, Pipeline  # noqa: E402


def photo(path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 140, 160)).save(path)
    return path


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def rows(root):
    return [
        json.loads(line) for line in (root / MANIFEST_NAME).read_text().splitlines()
    ]


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("manifest recovery must not load models")

    monkeypatch.setattr(pipeline_module.detectors, "create", forbidden)
    monkeypatch.setattr(pipeline_module.captioners, "create", forbidden)


@pytest.mark.parametrize("dry_run", [False, True])
def test_recovery_keeps_all_files_and_pairs_nested_captions(tmp_path, dry_run):
    out = tmp_path / "dataset"
    photo(out / "0007.png")
    photo(out / "512" / "0009.jpg", (512, 512))
    photo(out / "512" / "0009-masklabel.png", (512, 512))
    photo(out / "512" / "0009-condlabel.png")
    photo(out / "uncaptioned.webp")
    photo(out / "empty.png")
    (out / "0007.txt").write_bytes("  Manually edited café caption\r\n".encode())
    (out / "512" / "0009.txt").write_bytes(b"custom nested caption\n")
    (out / "empty.txt").write_bytes(b" \r\n")
    (out / "orphan.txt").write_bytes(b"a caption without an image\n")
    (out / "manifest.json").write_bytes(b'{"old": "summary"}\n')
    (out / "training_config.json").write_bytes(b'{"old": "config"}\n')
    before = snapshot(out)
    pipeline = Pipeline(Config(output=out, rebuild_manifest=True, dry_run=dry_run))

    stats = pipeline.run()

    assert stats.written == stats.scanned == 4
    assert stats.skipped_existing == 3
    assert stats.captioned == 0
    after = snapshot(out)
    assert {p: after[p] for p in before} == before
    assert set(after) - set(before) == (set() if dry_run else {Path(MANIFEST_NAME)})
    records = [vars(record) for record in pipeline.records] if dry_run else rows(out)
    by_image = {r["image_file"]: r for r in records}
    assert by_image["0007.png"]["caption"] == "Manually edited café caption"
    assert by_image["512/0009.jpg"]["caption_file"] == "512/0009.txt"
    assert by_image["512/0009.jpg"]["mask_file"] == "512/0009-masklabel.png"
    assert by_image["uncaptioned.webp"]["caption_file"] is None
    assert by_image["empty.png"]["caption_file"] == "empty.txt"
    assert by_image["empty.png"]["caption"] == ""
    assert all(r["recovered"] and not r["landmarks"] for r in records)
    assert by_image["0007.png"]["crop"] == [0, 0, 64, 48]
    assert by_image["512/0009.jpg"]["tier"] == 512


def test_cli_recovers_default_dataset_without_needing_an_input(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    photo(tmp_path / "dataset" / "0001.png")
    (tmp_path / "robocrop.toml").write_text('input = "missing-originals"\n')
    assert cli.main(["--rebuild-manifest"]) == 0
    assert (tmp_path / "dataset" / MANIFEST_NAME).is_file()
    output = capsys.readouterr()
    assert "manifest rows    1" in output.out
    assert "cannot be recovered" in output.err


def test_cli_output_can_equal_input_in_recovery_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    photo(tmp_path / "a.png")
    assert cli.main(["--rebuild-manifest", "-o", "."]) == 0


@pytest.mark.parametrize("content", [b"", b"existing manifest\n"])
def test_existing_manifest_is_refused_and_everything_is_preserved(tmp_path, content):
    photo(tmp_path / "a.png")
    (tmp_path / MANIFEST_NAME).write_bytes(content)
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        Pipeline(Config(output=tmp_path, rebuild_manifest=True)).run()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("failure", ["image", "caption", "shared_caption", "mask"])
def test_invalid_files_fail_without_writing_a_partial_manifest(tmp_path, failure):
    photo(tmp_path / "a.png")
    (tmp_path / "a.txt").write_bytes(b"preserve this caption\n")
    if failure == "image":
        (tmp_path / "z.png").write_bytes(b"broken image")
    elif failure == "caption":
        photo(tmp_path / "z.png")
        (tmp_path / "z.txt").write_bytes(b"\xff")
    elif failure == "shared_caption":
        photo(tmp_path / "a.jpg")
    else:
        photo(tmp_path / "a-masklabel.png", (32, 32))
    before = snapshot(tmp_path)
    with pytest.raises((OSError, ValueError)):
        Pipeline(Config(output=tmp_path, rebuild_manifest=True)).run()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_empty_dataset_is_an_error(tmp_path, missing):
    out = tmp_path / "missing" if missing else tmp_path
    with pytest.raises(
        (FileNotFoundError, ValueError), match="not found|no dataset images"
    ):
        Pipeline(Config(output=out, rebuild_manifest=True)).run()
    assert not (out / MANIFEST_NAME).exists()


def test_recovery_records_external_masks_and_keeps_them_unchanged(tmp_path):
    out, mask_dir = tmp_path / "dataset", tmp_path / "masks"
    photo(out / "512" / "0001.png")
    photo(mask_dir / "0001.png")
    before = snapshot(tmp_path)
    config = Config(output=out, mask_dir=mask_dir, rebuild_manifest=True)
    config.validate()
    Pipeline(config).run()
    record = rows(out)[0]
    assert record["mask_file"] == "0001.png"
    assert record["mask_dir"] == str(mask_dir.resolve())
    after = snapshot(tmp_path)
    assert {p: after[p] for p in before} == before


@pytest.mark.parametrize("kind", ["image", "caption", "mask"])
def test_symlinks_outside_dataset_are_rejected(tmp_path, kind):
    out = tmp_path / "dataset"
    photo(out / "a.png")
    outside = photo(tmp_path / "outside.png")
    name = {"image": "z.png", "caption": "a.txt", "mask": "a-masklabel.png"}[kind]
    try:
        (out / name).symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        Pipeline(Config(output=out, rebuild_manifest=True)).run()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "option",
    [
        "caption_only",
        "training_config_only",
        "training_config",
        "skip_detection",
        "resume",
        "overwrite",
        "mask_background",
        "limit",
    ],
)
def test_conflicting_recovery_modes_are_rejected(option):
    with pytest.raises(ValueError, match="--rebuild-manifest"):
        Config(
            rebuild_manifest=True, **{option: True if option != "limit" else 1}
        ).validate()


def test_rebuilt_manifest_supports_missing_caption_recovery_and_training_config(
    tmp_path, monkeypatch
):
    photo(tmp_path / "a.png", (256, 256))
    photo(tmp_path / "b.png", (256, 256))
    photo(tmp_path / "c.png", (256, 256))
    (tmp_path / "a.txt").write_bytes(b"my custom caption\r\n")
    (tmp_path / "b.txt").write_bytes(b" \n")
    Pipeline(Config(output=tmp_path, rebuild_manifest=True)).run()
    before = snapshot(tmp_path)

    # A real offline captioner exercises the full caption-only path.
    from robocrop.captioners.template import TemplateCaptioner

    monkeypatch.setattr(
        pipeline_module.captioners,
        "create",
        lambda name, **kwargs: TemplateCaptioner(**kwargs),
    )
    stats = Pipeline(
        Config(output=tmp_path, caption_only=True, captioner="template", resume=True)
    ).run()
    assert stats.captioned == 2 and stats.skipped_existing == 1
    assert (tmp_path / "a.txt").read_bytes() == before[Path("a.txt")]
    assert all(r["caption"] for r in rows(tmp_path))
    assert all(r["recovered"] for r in rows(tmp_path))

    Pipeline(Config(output=tmp_path, training_config_only=True)).run()
    assert (
        json.loads((tmp_path / "training_config.json").read_text())["resolution"]
        == "256"
    )
    for name in ("a.png", "b.png", "c.png"):
        assert (tmp_path / name).read_bytes() == before[Path(name)]


def test_rebuilt_manifest_cannot_resume_the_original_crop_run(tmp_path):
    out, originals = tmp_path / "dataset", tmp_path / "photos"
    photo(out / "0001.png")
    photo(originals / "original.png")
    Pipeline(Config(output=out, rebuild_manifest=True)).run()
    before = snapshot(out)
    with pytest.raises(ValueError, match="cannot resume the original crop run"):
        Pipeline(Config(input=originals, output=out, resume=True)).run()
    assert snapshot(out) == before
