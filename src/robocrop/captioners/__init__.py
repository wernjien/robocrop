"""Captioner registry. Backends load lazily; torch is never imported unless a
VLM backend is actually selected."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseCaptioner, CaptionRequest, Captioner

_BACKENDS: dict[str, str] = {
    "vlm": "vlm:VLMCaptioner",
    "template": "template:TemplateCaptioner",
    "none": "null:NullCaptioner",
}

DESCRIPTIONS: dict[str, str] = {
    "vlm": "local vision-language model, offline after first download",
    "template": "measured attributes filled into a format string, no model",
    "none": "write crops only, no .txt files",
}


def available() -> list[str]:
    return list(_BACKENDS)


def create(name: str, **kwargs: Any) -> BaseCaptioner:
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown captioner {name!r}; choose one of {', '.join(_BACKENDS)}"
        ) from None
    module_name, class_name = target.split(":")
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:
        raise RuntimeError(
            f"captioner {name!r} needs a dependency that is not installed: {exc}\n"
            f"install the captioning extras, or pass --captioner template"
        ) from exc
    return getattr(module, class_name)(**kwargs)


__all__ = ["BaseCaptioner", "CaptionRequest", "Captioner", "available", "create", "DESCRIPTIONS"]
