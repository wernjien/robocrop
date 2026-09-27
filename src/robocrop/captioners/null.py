"""Captioner that writes nothing; selected by ``--captioner none``."""

from __future__ import annotations

from .base import BaseCaptioner, CaptionRequest


class NullCaptioner(BaseCaptioner):
    name = "none"

    def __init__(self, **caption_kwargs) -> None:
        super().__init__(**caption_kwargs)

    def _describe(self, request: CaptionRequest) -> str:
        return ""

    def _assemble(self, described: str) -> str:
        # Deliberately skips trigger/prefix/suffix: "none" must produce no file
        # at all, not a file containing only the trigger word.
        return ""
