"""Captioner interface and the shared caption clean-up rules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image

from ..detectors.base import Region

#: Openers VLMs habitually emit. A training caption should describe the subject,
#: not announce that it is a picture, and these phrases waste tokens and teach
#: the model nothing.
_BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"this\s+(?:image|photo|picture|crop)\s+(?:is\s+|shows?\s+|depicts?\s+|features?\s+)?|"
    r"the\s+(?:image|photo|picture|crop)\s+(?:is\s+|shows?\s+|depicts?\s+|features?\s+)?|"
    r"the\s+shot\s+type\s+is\s+(?:an?\s+)?|"
    r"an?\s+(?:cropped\s+|close[\s-]?up\s+|colou?r\s+|black[\s-]and[\s-]white\s+)*"
    r"(?:image|photo(?:graph)?|picture|portrait|shot|crop)\s+of\s+|"
    r"(?:cropped|close[\s-]?up)\s+(?:image|photo(?:graph)?|picture|portrait|shot)\s+of\s+|"
    r"there\s+(?:is|are)\s+|"
    r"it\s+(?:is|shows)\s+|"
    r"we\s+(?:can\s+)?see\s+|"
    r"in\s+this\s+(?:image|photo|picture|crop)\s*,?\s*|"
    r"here\s+(?:is|we\s+see)\s+|"
    r"sure[,!.]?\s*|"
    r"caption\s*:\s*|"
    r"description\s*:\s*"
    r")+",
    re.IGNORECASE,
)

#: Words that must keep their capital when a caption is lower-cased.
_KEEP_CAPITAL = re.compile(r"^(?:[A-Z]{2,}|[A-Z][a-z]+(?=\s+[A-Z]))")

_WHITESPACE = re.compile(r"\s+")
_REPEATED_COMMAS = re.compile(r"\s*,(?:\s*,)+")


@dataclass
class CaptionRequest:
    """Everything a captioner may use to describe one crop."""

    image: Image.Image
    """The finished square crop, exactly as written to disk."""

    source_path: str
    """Original file, for template tokens and debugging."""

    region: Region
    """The detection this crop came from; landmarks drive the pose hint."""

    tier: int
    """Output edge length in pixels."""

    index: int
    """1-based output number."""

    extras: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Captioner(Protocol):
    name: str

    def caption(self, request: CaptionRequest) -> str: ...

    def caption_batch(self, requests: list[CaptionRequest]) -> list[str]: ...

    def close(self) -> None: ...


class BaseCaptioner:
    """Shared caption assembly: clean the raw text, then frame it.

    Subclasses implement :meth:`_describe`; everything about trigger words,
    prefixes and blocked phrases is handled once, here.
    """

    name = "base"

    def __init__(
        self,
        *,
        trigger: str = "",
        prefix: str = "",
        suffix: str = "",
        drop: list[str] | None = None,
        max_chars: int = 0,
    ) -> None:
        self.trigger = trigger.strip()
        self.prefix = prefix.strip()
        self.suffix = suffix.strip()
        self.max_chars = max_chars
        self._drop = [re.compile(p, re.IGNORECASE) for p in (drop or [])]

    # -- to implement -----------------------------------------------------
    def _describe(self, request: CaptionRequest) -> str:  # pragma: no cover
        raise NotImplementedError

    def _describe_batch(self, requests: list[CaptionRequest]) -> list[str]:
        return [self._describe(r) for r in requests]

    # -- public API -------------------------------------------------------
    def caption(self, request: CaptionRequest) -> str:
        return self._assemble(self._describe(request))

    def caption_batch(self, requests: list[CaptionRequest]) -> list[str]:
        return [self._assemble(text) for text in self._describe_batch(requests)]

    def close(self) -> None:
        return None

    def __enter__(self) -> "BaseCaptioner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- assembly ---------------------------------------------------------
    def _assemble(self, described: str) -> str:
        body = clean_caption(described)
        for pattern in self._drop:
            body = pattern.sub("", body)
        body = tidy_punctuation(body)

        body = dedupe_phrases(body)

        parts = [p for p in (self.trigger, self.prefix, body, self.suffix) if p]
        caption = ", ".join(parts)
        caption = tidy_punctuation(caption)
        if self.max_chars and len(caption) > self.max_chars:
            caption = caption[: self.max_chars].rsplit(",", 1)[0].strip().rstrip(",")
        return caption


def clean_caption(text: str) -> str:
    """Strip VLM boilerplate, normalise whitespace and case."""
    out = _WHITESPACE.sub(" ", (text or "").strip())
    # Models stack openers ("this image shows a cropped photo of ..."), so peel
    # until nothing matches rather than removing a single prefix.
    while True:
        stripped = _BOILERPLATE.sub("", out, count=1).lstrip()
        if stripped == out:
            break
        out = stripped
    return _lower_first(out.strip())


def _lower_first(text: str) -> str:
    """Lower the opening capital so the caption reads as one clause.

    A caption is assembled as ``trigger, description``, and a stray sentence
    capital mid-phrase ("my subject, The shot is...") is noise. Acronyms and
    what look like proper names are left alone.
    """
    if not text or not text[0].isupper():
        return text
    if _KEEP_CAPITAL.match(text):
        return text
    return text[0].lower() + text[1:]


def dedupe_phrases(text: str) -> str:
    """Drop repeated comma-separated phrases, keeping the first of each.

    Guards against two things at once: a model that stutters a tag, and a
    caption that simply says the same thing twice. Duplicate tags carry no
    extra signal for training and dilute the ones that matter.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for phrase in text.split(","):
        stripped = phrase.strip()
        if not stripped:
            continue
        key = stripped.lower().rstrip(".")
        if key in seen:
            continue
        seen.add(key)
        kept.append(stripped)
    return ", ".join(kept)


def tidy_punctuation(text: str) -> str:
    out = _REPEATED_COMMAS.sub(",", text or "")
    out = _WHITESPACE.sub(" ", out).strip()
    out = out.strip(" ,;")
    return out.rstrip(".").strip()
