"""Tests for caption assembly: boilerplate removal, trigger, and framing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from traincrop.captioners.base import BaseCaptioner, clean_caption  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("A photo of a woman smiling", "a woman smiling"),
        ("This image shows a man in a hat", "a man in a hat"),
        ("The shot type is a frontal close-up", "frontal close-up"),
        ("Cropped photo of a woman, neutral", "a woman, neutral"),
        ("a close-up photo of a person", "a person"),
        ("In this image, a dog runs", "a dog runs"),
        ("There is a child by a window", "a child by a window"),
        ("Sure! A portrait of a man", "a man"),
        ("this image shows a photo of a cat", "a cat"),  # stacked openers
        ("a woman with red hair", "a woman with red hair"),  # nothing to strip
    ],
)
def test_boilerplate_is_stripped(raw, expected):
    assert clean_caption(raw) == expected


def test_acronyms_keep_their_capitals():
    assert clean_caption("NASA engineer at a console") == "NASA engineer at a console"


def test_sentence_capital_is_lowered():
    assert clean_caption("Blonde hair and a denim jacket") == "blonde hair and a denim jacket"


class _Fixed(BaseCaptioner):
    """Captioner returning a canned string, to test assembly alone."""

    def __init__(self, text, **kw):
        super().__init__(**kw)
        self._text = text

    def _describe(self, request):
        return self._text


def _caption(text, **kw):
    return _Fixed(text, **kw).caption(None)


def test_trigger_comes_first():
    assert _caption("a woman smiling", trigger="my subject") == (
        "my subject, a woman smiling"
    )


def test_prefix_and_suffix_wrap_the_description():
    got = _caption(
        "a woman smiling",
        trigger="my subject",
        prefix="portrait",
        suffix="studio lighting",
    )
    assert got == "my subject, portrait, a woman smiling, studio lighting"


def test_no_trigger_leaves_a_bare_description():
    assert _caption("a woman smiling") == "a woman smiling"


def test_drop_patterns_remove_text():
    got = _caption(
        "a woman smiling, wearing a hat",
        drop=[r",?\s*wearing a hat"],
    )
    assert got == "a woman smiling"


def test_trailing_period_and_stray_commas_are_tidied():
    assert _caption("a woman smiling,, .") == "a woman smiling"


def test_max_chars_trims_at_a_comma():
    got = _caption("aaaa, bbbb, cccc, dddd", max_chars=14)
    assert got == "aaaa, bbbb"
    assert len(got) <= 14


def test_empty_description_still_yields_the_trigger():
    assert _caption("", trigger="my subject") == "my subject"


# -- repetition guards ----------------------------------------------------
def test_repeated_phrases_are_collapsed():
    """Small VLMs stutter; a looped tag must not reach the caption file."""
    got = _caption("close-up, blue eyes, blue eyes, blue eyes, denim jacket")
    assert got == "close-up, blue eyes, denim jacket"


def test_deduplication_ignores_case_and_trailing_stops():
    assert _caption("smiling, Smiling., smiling") == "smiling"


def test_deduplication_keeps_first_occurrence_order():
    assert _caption("a, b, a, c, b, d") == "a, b, c, d"


def test_distinct_phrases_survive():
    got = _caption("close-up, blonde hair, denim jacket, soft lighting")
    assert got == "close-up, blonde hair, denim jacket, soft lighting"
