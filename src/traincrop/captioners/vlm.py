"""Local vision-language captioner (transformers, fully offline after download).

Handles two model families behind one interface:

* **caption models** (BLIP) - take an image and emit a caption directly.
* **instruction models** (SmolVLM, Qwen2.5-VL, LLaVA, ...) - take an image plus
  a prompt, so the caption can be steered toward LoRA-training style.

Weights are downloaded once into the Hugging Face cache and reused offline.
"""

from __future__ import annotations

import os
from typing import Any

from .base import BaseCaptioner, CaptionRequest

#: Short name -> Hugging Face model id, approximate download, and a note.
PRESETS: dict[str, tuple[str, str, str]] = {
    "smolvlm": (
        "HuggingFaceTB/SmolVLM-Instruct",
        "~4.5 GB",
        "instruction-following, balanced quality and memory (default)",
    ),
    "smolvlm-small": (
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "~1.0 GB",
        "instruction-following, lightest option",
    ),
    "qwen": (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "~7.5 GB",
        "best detail; wants 16 GB+ of free memory",
    ),
    "blip": (
        "Salesforce/blip-image-captioning-large",
        "~1.9 GB",
        "fastest, short generic captions, not steerable",
    ),
}

DEFAULT_PROMPT = (
    "Write a single training caption for this cropped photo. Give, in order "
    "and separated by commas: which way the subject faces, the facial "
    "expression, the hair, the visible clothing or lack of it, the body "
    "pose, the lighting, and the background. Be literal, concrete and clinical "
    "- describe exactly what is visible, including nudity or sexual content, "
    "using plain anatomical terms. Do not soften, censor, or euphemise anything "
    "that is visible. Write plain comma-separated phrases with no field names, "
    "labels or colons. Do not start with 'a photo of' or 'this image'. Do not "
    "moralise or comment on appropriateness. Do not name or identify anyone."
)
"""The default instruction sent to instruction-following models (SmolVLM,
Qwen, LLaVA, ...). It asks for literal, uncensored, tag-style description
because that is the standard captioning convention for LoRA/embedding
training datasets - including adult ones - and a caption that hedges or
omits visible content teaches the model the wrong thing just as surely as
one that is factually wrong.

This is a prompt instruction, not a jailbreak: it cannot override safety
training baked into a model's weights. General chat-tuned VLMs (all of the
presets below) retain some of that tuning regardless of what they are asked,
and may still hedge, refuse, or soften explicit description on some images.
If you need reliably explicit tag captions, see the NSFW captioning note in
README.md for the actual limitation and the community-standard alternative
(a booru-style tagger, e.g. WD14/DeepDanbooru, trained specifically on
explicit tags and with no chat safety tuning to work around)."""


def resolve_model(name: str) -> str:
    """Map a preset name to a model id, or pass a Hugging Face id straight through."""
    if name in PRESETS:
        return PRESETS[name][0]
    return name


class VLMCaptioner(BaseCaptioner):
    name = "vlm"

    def __init__(
        self,
        model: str = "smolvlm",
        *,
        device: str = "auto",
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 96,
        batch_size: int = 4,
        num_beams: int = 1,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 4,
        quiet: bool = False,
        **caption_kwargs: Any,
    ) -> None:
        super().__init__(**caption_kwargs)
        self.model_id = resolve_model(model)
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        self.num_beams = max(1, int(num_beams))
        # Small VLMs fall into loops on low-detail crops ("blue eyes, blue
        # eyes, ..."). These two make that vanishingly unlikely, and a looped
        # caption is far more damaging to training than a terse one.
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self._quiet = quiet

        # Imported here, not at module scope: torch costs seconds to import and
        # must not be paid by runs that use --captioner template or none.
        import torch
        from transformers import AutoConfig, AutoProcessor

        _silence_stale_pad_token_warning()

        self._torch = torch
        self.device = _pick_device(torch, device)
        # bfloat16 halves memory and is native on Apple silicon and modern GPUs;
        # CPU inference stays in float32 where bf16 is emulated and slower.
        self.dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        if not quiet:
            print(
                f"  loading caption model {self.model_id} on {self.device} "
                f"({str(self.dtype).replace('torch.', '')}) ...",
                flush=True,
            )

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=False)
        self.family = "blip" if getattr(config, "model_type", "") == "blip" else "chat"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = self._load_model()
        self._model.eval()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            # Decoder-only generation needs left padding, or short sequences in
            # a batch generate from pad tokens and come back as gibberish.
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    def _load_model(self):
        from transformers import AutoModelForImageTextToText, BlipForConditionalGeneration

        cls = BlipForConditionalGeneration if self.family == "blip" else AutoModelForImageTextToText
        try:
            model = cls.from_pretrained(self.model_id, dtype=self.dtype)
        except TypeError:
            # transformers < 5 spelled this argument torch_dtype.
            model = cls.from_pretrained(self.model_id, torch_dtype=self.dtype)
        return model.to(self.device)

    # -- captioning -------------------------------------------------------
    def _describe(self, request: CaptionRequest) -> str:
        return self._describe_batch([request])[0]

    def _describe_batch(self, requests: list[CaptionRequest]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            try:
                out.extend(self._run(chunk))
            except Exception as exc:  # noqa: BLE001
                if len(chunk) == 1:
                    # One bad image should cost one caption, not the whole run.
                    if not self._quiet:
                        print(f"    caption failed for {chunk[0].source_path}: {exc}")
                    out.append("")
                else:
                    # Most batch failures are memory pressure; retry one by one.
                    for single in chunk:
                        out.extend(self._describe_batch([single]))
        return out

    def _run(self, chunk: list[CaptionRequest]) -> list[str]:
        torch = self._torch
        images = [r.image.convert("RGB") for r in chunk]

        if self.family == "blip":
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                ids = self._model.generate(**inputs, **self._generate_kwargs())
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
            return [t.strip() for t in texts]

        messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
            for _ in chunk
        ]
        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages
        ]
        inputs = self.processor(
            text=prompts, images=images, return_tensors="pt", padding=True
        )
        inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs, do_sample=False, **self._generate_kwargs()
            )
        # Slice off the prompt so only the model's own words are decoded.
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = ids[:, prompt_len:]
        texts = self.processor.batch_decode(new_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
        }
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            # Passed explicitly so batched generation pads with a real token
            # rather than warning and guessing.
            kwargs["pad_token_id"] = pad_id
        return kwargs

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is None:
            return
        self._model = None
        del model
        torch = self._torch
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


_pad_token_warning_silenced = False


def _silence_stale_pad_token_warning() -> None:
    """Quiet a known-benign transformers warning some Hub configs trigger.

    Several vision-language configs on the Hub (SmolVLM included) carry a
    stale top-level ``pad_token_id`` left over from whatever base checkpoint
    they were derived from - SmolVLM's says 128002 (a Llama-3 special-token
    id), while its own vocabulary is 49155 wide. transformers validates this
    at config-load time and logs "may result in unexpected behavior" every
    time, which reads as an active problem.

    It is not one here: ``model.generation_config.pad_token_id`` is built
    from the model's real ``text_config`` (a valid ``2``, matching the
    tokenizer), and every ``generate()`` call in this module passes that
    value explicitly via ``_generate_kwargs()``. The stale field is never
    actually used for generation - only logged about. transformers' own
    source acknowledges this is a known issue with configs on the Hub, not
    a transformers bug to fix upstream, so this silences the specific
    validator's logger rather than leaving alarming, inactionable noise in
    every run, without touching any other transformers logging.
    """
    global _pad_token_warning_silenced
    if _pad_token_warning_silenced:
        return
    import logging

    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    _pad_token_warning_silenced = True


def _pick_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        # Lets unsupported ops fall back to CPU instead of raising mid-run.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
