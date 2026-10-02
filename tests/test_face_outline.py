"""SegFace masks preserve hair and neck, map crops back, and fall back safely."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from PIL import Image

from robocrop import face_parse, pipeline as pipeline_module
from robocrop.config import Config
from robocrop.detectors.base import Region
from robocrop.face_parse import FACE_CLASSES, FaceParser, _detected_face
from robocrop.geometry import Rect
from robocrop.masks import build_mask
from robocrop.pipeline import Pipeline

from test_face_mask import head, stubs
from test_pipeline import base_config, make_photo


def face_blob(size=1024):
    matte = np.zeros((size, size), dtype=np.float32)
    matte[105:195, 470:555] = 1.0
    return matte


class StubParser:
    def __init__(self, face):
        self._face = face

    def outline(self, image, faces):
        missed = [f.rect for f in faces] if not self._face.any() else []
        return self._face, missed

    def close(self):
        pass


@pytest.fixture
def parser(monkeypatch):
    holder = {"face": face_blob(), "created": 0}

    def create(**kwargs):
        holder["created"] += 1
        return StubParser(holder["face"])

    monkeypatch.setattr(pipeline_module.face_parse, "create", create)
    return holder


def test_face_class_union_excludes_neck_hair_clothing_and_accessories():
    assert 2 in FACE_CLASSES and 11 in FACE_CLASSES and 15 in FACE_CLASSES
    assert not set(FACE_CLASSES) & {0, 1, 3, 14, 16, 17, 18}


def test_face_selection_ignores_separate_neighbour():
    matte = np.zeros((200, 200), np.float32)
    matte[50:130, 50:120] = 1
    matte[140:190, 140:190] = 1
    kept = _detected_face(matte, Rect(50, 50, 80, 80), 0, 0, 0.08)
    assert kept[80, 80] == 1 and kept[150, 150] == 0
    assert (
        _detected_face(np.zeros_like(matte), Rect(50, 50, 80, 80), 0, 0, 0.08) is None
    )


def test_outline_projects_each_face_crop_and_preserves_neck():
    parser = FaceParser.__new__(FaceParser)
    crops = []

    def parse(crop):
        crops.append(crop)
        matte = np.zeros(crop.shape[:2], np.float32)
        # Detector box (60,50,40,40): square is (40,26) to (120,106).
        matte[24:64, 20:60] = 1
        return matte

    parser._parse = parse
    outline, missed = parser.outline(
        np.zeros((140, 180, 3), np.uint8), [Region(Rect(60, 50, 40, 40), 1.0)]
    )
    assert not missed and crops[0].shape == (80, 80, 3)
    assert outline[70, 80] == 1
    assert outline[100, 80] == 0  # neck
    assert outline[30, 80] == 0  # hair


def test_edge_face_is_padded_without_distorting_or_shifting_the_mask():
    parser = FaceParser.__new__(FaceParser)
    seen = []

    def parse(crop):
        seen.append(crop)
        matte = np.zeros(crop.shape[:2], np.float32)
        matte[14:34, 10:30] = 1
        return matte

    parser._parse = parse
    image = np.full((60, 80, 3), 123, np.uint8)
    outline, missed = parser.outline(image, [Region(Rect(0, 0, 20, 20), 1.0)])
    assert not missed
    assert seen[0].shape == (40, 40, 3) and np.all(seen[0] == 123)
    assert outline.shape == (60, 80)
    assert outline[10, 10] == 1 and outline[35, 10] == 0


def test_empty_or_outside_face_uses_oval_fallback():
    parser = FaceParser.__new__(FaceParser)
    parser._parse = lambda crop: np.zeros(crop.shape[:2], np.float32)
    faces = [Region(Rect(20, 20, 20, 20), 1), Region(Rect(200, 200, 20, 20), 1)]
    outline, missed = parser.outline(np.zeros((100, 100, 3), np.uint8), faces)
    assert missed == [f.rect for f in faces] and not outline.any()


def test_build_mask_blacks_out_the_face_shape_only():
    mask = build_mask(1024, [], 0.35, face=face_blob())
    assert mask.getpixel((500, 150)) == 0
    assert mask.getpixel((500, 70)) == 255
    assert mask.getpixel((440, 150)) == 255


def test_outline_is_default_and_does_not_load_clothing_parser(
    tmp_path, stubs, parser, monkeypatch
):
    monkeypatch.setattr(
        pipeline_module.clothing,
        "create",
        lambda **kwargs: pytest.fail("clothing loaded"),
    )
    make_photo(tmp_path / "photos" / "a.png")
    stats = Pipeline(base_config(tmp_path, mask_faces=True)).run()
    assert stats.masked == 1 and parser["created"] == 1
    mask = Image.open(tmp_path / "out" / "0001-masklabel.png")
    assert mask.getpixel((500, 150)) == 0
    assert mask.getpixel((500, 70)) == 255


def test_pipeline_uses_oval_when_segface_cannot_parse_face(tmp_path, stubs, parser):
    parser["face"] = np.zeros((1024, 1024), np.float32)
    make_photo(tmp_path / "photos" / "a.png")
    stats = Pipeline(base_config(tmp_path, mask_faces=True)).run()
    assert stats.masked == 1
    assert Image.open(tmp_path / "out" / "0001-masklabel.png").getpixel((512, 150)) == 0


def test_bad_face_mask_value_is_rejected():
    with pytest.raises(ValueError, match="face-mask"):
        Config(detector="yolox", mask_faces=True, face_mask="square").validate()


def test_mask_model_is_shared_across_workers_and_closed_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    created = []

    class Model:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    model = Model()
    ready = threading.Barrier(4)

    def factory(**kwargs):
        time.sleep(0.01)  # give simultaneous callers an opportunity to race
        created.append(model)
        return model

    monkeypatch.setattr(pipeline_module.face_parse, "create", factory)
    pipe = Pipeline(Config())

    def get(_):
        ready.wait()
        return pipe._face_parser()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(get, range(4)))
    assert len(created) == 1 and all(m is model for m in results)
    pipe._close_detectors()
    assert model.closed == 1 and not pipe._mask_models
    pipe._close_detectors()
    assert model.closed == 1


def test_segface_preprocessing_and_semantic_union():
    import threading
    import torch

    parser = FaceParser.__new__(FaceParser)
    parser._torch, parser._device, parser._lock = torch, "cpu", threading.Lock()
    feeds = []

    def model(blob):
        feeds.append(blob)
        logits = torch.full((1, 19, 512, 512), -10.0)
        logits[0, 2, :256] = 10  # skin
        logits[0, 1, 256:] = 10  # neck
        logits[0, 11, 50:100, 50:100] = 20  # mouth is also masked
        return logits

    parser._model = model
    crop = np.full((100, 100, 3), (10, 20, 30), np.uint8)
    mask = parser._parse(crop)
    assert mask[10, 10] == 1 and mask[90, 50] == 0
    assert feeds[0].shape == (1, 3, 512, 512)
    assert feeds[0][0, :, 0, 0].numpy() == pytest.approx(
        (np.array([30, 20, 10]) / 255.0 - [0.485, 0.456, 0.406])
        / [0.229, 0.224, 0.225],
    )


def test_large_face_keeps_logit_memory_bounded(monkeypatch):
    import threading
    import torch

    parser = FaceParser.__new__(FaceParser)
    parser._torch, parser._device, parser._lock = torch, "cpu", threading.Lock()
    parser._model = lambda blob: torch.zeros((1, 19, 32, 32))
    sizes = []
    original = torch.nn.functional.interpolate

    def interpolate(logits, *, size, **kwargs):
        sizes.append(size)
        return original(logits, size=size, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", interpolate)
    matte = parser._parse(np.zeros((2048, 2048, 3), np.uint8))
    assert sizes == [(1024, 1024)]
    assert matte.shape == (2048, 2048) and not matte.any()
