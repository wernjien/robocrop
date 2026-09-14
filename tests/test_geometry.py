"""Tests for the crop rules: tier selection, padding, and edge fitting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from traincrop.geometry import (  # noqa: E402
    CropRejected, Rect, base_side_from, choose_tier, plan_crop,
)

SIZES = (512, 768, 1024)


# -- the 90% rule ---------------------------------------------------------
@pytest.mark.parametrize(
    "measure, expected",
    [
        (2000, 1024),   # comfortably above every threshold
        (922, 1024),    # just over 90% of 1024 (921.6)
        (921, 768),     # just under -> drops a tier
        (692, 768),     # just over 90% of 768 (691.2)
        (691, 512),
        (461, 512),     # just over 90% of 512 (460.8)
        (460, None),    # below the floor -> no crop at all
        (10, None),
    ],
)
def test_tier_thresholds(measure, expected):
    assert choose_tier(measure, SIZES, 0.9) == expected


def test_min_ratio_is_configurable():
    assert choose_tier(800, SIZES, 0.9) == 768
    # A looser ratio lets the same detection earn the larger tier.
    assert choose_tier(800, SIZES, 0.75) == 1024


def test_detection_below_floor_is_rejected():
    with pytest.raises(CropRejected) as err:
        plan_crop(Rect(0, 0, 400, 400), 4000, 4000, padding=0.25, sizes=SIZES)
    assert "410px" in str(err.value) or "410" in err.value.reason


# -- padding --------------------------------------------------------------
def test_padding_is_added_to_every_side():
    # 25% of a 1000px base is 250px per side -> 1500px window.
    plan = plan_crop(Rect(1000, 1000, 1000, 1000), 6000, 6000, padding=0.25, sizes=SIZES)
    assert plan.base_side == 1000
    assert plan.padded_side == 1500
    assert plan.rect.w == plan.rect.h == 1500


def test_zero_padding_keeps_the_detection_box():
    plan = plan_crop(Rect(100, 100, 1000, 1000), 4000, 4000, padding=0.0, sizes=SIZES)
    assert plan.rect.w == 1000
    assert (plan.rect.x, plan.rect.y) == (100, 100)


def test_padding_does_not_change_the_tier():
    """Tier comes from the detection, so context never inflates the size."""
    tight = plan_crop(Rect(0, 0, 700, 700), 8000, 8000, padding=0.0, sizes=SIZES)
    loose = plan_crop(Rect(0, 0, 700, 700), 8000, 8000, padding=1.0, sizes=SIZES)
    assert tight.tier == loose.tier == 768
    assert loose.padded_side == 2100


# -- squareness -----------------------------------------------------------
@pytest.mark.parametrize("w, h", [(800, 1000), (1000, 800), (950, 950)])
def test_crop_is_always_square(w, h):
    plan = plan_crop(Rect(500, 500, w, h), 5000, 5000, padding=0.2, sizes=SIZES)
    assert plan.rect.w == plan.rect.h


def test_base_side_modes():
    assert base_side_from(800, 1000, "max") == 1000
    assert base_side_from(800, 1000, "mean") == 900
    assert base_side_from(800, 1000, "width") == 800
    assert base_side_from(800, 1000, "height") == 1000


def test_crop_stays_centred_on_the_detection_when_it_fits():
    region = Rect(2000, 1500, 1000, 1000)
    plan = plan_crop(region, 6000, 6000, padding=0.3, sizes=SIZES)
    assert plan.rect.cx == region.cx
    assert plan.rect.cy == region.cy
    assert not plan.clamped


# -- offsets --------------------------------------------------------------
def test_offset_y_moves_the_window_up():
    region = Rect(2000, 2000, 1000, 1000)
    plain = plan_crop(region, 6000, 6000, padding=0.25, sizes=SIZES)
    raised = plan_crop(region, 6000, 6000, padding=0.25, sizes=SIZES, offset_y=-0.1)
    assert raised.rect.cy == plain.rect.cy - 100
    assert raised.rect.cx == plain.rect.cx


# -- edges ----------------------------------------------------------------
def test_shift_keeps_the_window_inside_the_image():
    # Detection hard against the top-left corner.
    plan = plan_crop(Rect(0, 0, 1000, 1000), 3000, 3000, padding=0.5, sizes=SIZES, edge="shift")
    assert plan.rect.x >= 0 and plan.rect.y >= 0
    assert plan.rect.x2 <= 3000 and plan.rect.y2 <= 3000
    assert plan.clamped


def test_shift_shrinks_when_the_image_is_smaller_than_the_window():
    plan = plan_crop(Rect(100, 100, 1000, 1000), 1200, 1200, padding=1.0, sizes=SIZES, edge="shift")
    assert plan.rect.w == 1200            # capped by the short edge
    assert (plan.rect.x, plan.rect.y) == (0, 0)
    assert plan.tier == 1024              # face is still 1000px, tier stands


def test_tier_drops_only_when_the_window_cuts_into_the_face():
    # A 1000px face in a 700px-tall image: the window is now tighter than the
    # face itself, so the tier must be recomputed from what survives.
    plan = plan_crop(Rect(0, 0, 1000, 1000), 2000, 700, padding=0.25, sizes=SIZES, edge="shift")
    assert plan.fitted_side == 700
    assert plan.tier == 768               # 700 >= 0.8 * 768 = 614.4


def test_extend_allows_the_window_to_leave_the_image():
    plan = plan_crop(Rect(0, 0, 1000, 1000), 3000, 3000, padding=0.5, sizes=SIZES, edge="extend")
    assert plan.extends
    assert plan.rect.x < 0 and plan.rect.y < 0
    assert plan.rect.w == 2000


def test_skip_refuses_a_crop_that_does_not_fit():
    with pytest.raises(CropRejected):
        plan_crop(Rect(0, 0, 1000, 1000), 3000, 3000, padding=0.5, sizes=SIZES, edge="skip")


def test_skip_allows_a_crop_that_does_fit():
    plan = plan_crop(Rect(1000, 1000, 1000, 1000), 4000, 4000, padding=0.2, sizes=SIZES, edge="skip")
    assert not plan.clamped


# -- misc -----------------------------------------------------------------
def test_scale_reports_downscaling():
    plan = plan_crop(Rect(0, 0, 2000, 2000), 8000, 8000, padding=0.25, sizes=SIZES)
    assert plan.tier == 1024
    assert plan.scale < 1.0               # 1024 from a 3000px window


def test_single_configured_size():
    plan = plan_crop(Rect(0, 0, 800, 800), 4000, 4000, padding=0.1, sizes=(768,))
    assert plan.tier == 768


def test_rounded_box_never_collapses():
    left, top, right, bottom = Rect(10.4, 10.4, 0.2, 0.2).rounded()
    assert right > left and bottom > top


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError):
        plan_crop(Rect(0, 0, 900, 900), 4000, 4000, padding=-0.1, sizes=SIZES)
