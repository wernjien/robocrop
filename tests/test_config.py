"""Configuration type checks, validation and command-line overrides."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402
from robocrop.cli import build_config  # noqa: E402
from robocrop.config import Config  # noqa: E402


def test_cli_sizes_override_an_older_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robocrop.toml").write_text('sizes = [512, 768, 1024]\n')
    assert build_config([]).sizes == (512, 768, 1024)
    assert build_config(["--sizes", "256,512,768,1024"]).sizes == (256, 512, 768, 1024)


@pytest.mark.parametrize("field", ["padding", "mask_margin", "min_sharpness", "offset_x", "offset_y"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_geometry_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        Config(**{field: value}).validate()



@pytest.mark.parametrize("setting", [
    'workers = "four"', 'sizes = [512.5]', 'min-score = true',
    'mask-faces = "false"', 'fill-color = [0, 0]', 'padding = "lots"',
    'base-mode = "wide"', 'captioner = "typo"', 'detector-opts = "person"',
])
def test_bad_toml_settings_have_a_configuration_error(tmp_path, monkeypatch, setting):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robocrop.toml").write_text(setting + "\n")
    with pytest.raises(SystemExit, match="(?:invalid|must|needs)"):
        build_config([])



def test_invalid_caption_filter_is_rejected_before_cropping():
    with pytest.raises(ValueError, match="invalid regular expression"):
        Config(caption_drop=("[",)).validate()



def test_invalid_template_attribute_is_a_configuration_error():
    with pytest.raises(ValueError, match="not a valid format string"):
        Config(captioner="template", caption_template="{tier.missing}").validate()



def test_cli_detector_options_merge_with_toml_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robocrop.toml").write_text(
        'detector = "yolox"\n[detector-opts]\nclasses = "dog"\nnms_threshold = 0.45\n'
    )
    cfg = build_config(["--detector-opt", "nms_threshold=0.2"])
    assert cfg.detector_opts == {"classes": "dog", "nms_threshold": 0.2}



@pytest.mark.parametrize("name", ["min_score", "quiet"])
def test_reserved_detector_options_are_rejected(name):
    with pytest.raises(ValueError, match="instead of --detector-opt"):
        Config(detector_opts={name: 0.5}).validate()



def test_bad_clothing_detector_classes_are_a_configuration_error():
    with pytest.raises(ValueError, match="classes must be"):
        Config(detector="yolox", detector_opts={"classes": True}, mask_clothing=True).validate()
