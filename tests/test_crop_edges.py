"""Crop dimensions, coordinate projection and off-image mask bounds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from robocrop import masks, pipeline as pipeline_module  # noqa: E402
from robocrop.geometry import Rect, plan_crop, plan_native, plan_whole  # noqa: E402
from robocrop.images import extract  # noqa: E402


@pytest.mark.parametrize("shape", [(1, 10000), (10000, 1)])
@pytest.mark.parametrize("native", [False, True])
def test_extremely_thin_images_still_have_valid_output_dimensions(shape, native):
    plan = (plan_native(*shape, min_side=1, max_side=512) if native
            else plan_whole(*shape, sizes=(512,)))
    with Image.new("RGB", shape) as image:
        with extract(image, plan) as crop:
            assert crop.size == ((1, 512) if shape[0] == 1 else (512, 1))



def test_detection_boxes_use_actual_rounded_crop_pixels():
    plan = plan_native(2000, 1001, min_side=1, max_side=512)
    result = pipeline_module._project_box(Rect(100, 100, 500, 500), plan, (512, 256))
    assert result.x == pytest.approx(25.6)
    assert result.y == pytest.approx(100 * 256 / 1001)
    assert result.h == pytest.approx(500 * 256 / 1001)



def test_detection_projection_accounts_for_subpixel_crop_rounding():
    plan = plan_crop(Rect(10.3, 12.7, 50.5, 50.5), 100, 100, padding=0,
                     sizes=(64,), min_ratio=0.5, offset_y=0)
    box = pipeline_module._project_box(Rect(10, 13, 51, 50), plan, (64, 64))
    assert box == Rect(0, 0, 64, 64)



@pytest.mark.parametrize("box", [Rect(-30, 0, 10, 10), Rect(0, -30, 10, 10),
                                 Rect(30, 0, 10, 10), Rect(0, 30, 10, 10)])
def test_wholly_outside_boxes_never_select_foreground(box):
    matte = np.ones((20, 20), dtype=np.float32)
    assert masks.isolate_person(matte, box) is None
    assert not masks.box_person((20, 20), box).any()



@pytest.mark.parametrize("offset", [-5, 5])
def test_color_extension_can_fill_a_window_wholly_outside_the_image(offset):
    plan = plan_crop(Rect(30, 30, 40, 40), 100, 100, padding=0,
                     sizes=(32,), edge="extend", offset_x=offset)
    with Image.new("RGB", (100, 100), "white") as image:
        with extract(image, plan, fill="color", fill_color=(12, 34, 56)) as crop:
            assert crop.size == (32, 32)
            assert crop.getextrema() == ((12, 12), (34, 34), (56, 56))
