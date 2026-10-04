"""Prompt formats and batched generation for local instruction captioners."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from robocrop.captioners import vlm
from robocrop.captioners.vlm import VLMCaptioner


@pytest.mark.parametrize("family", ["joycaption", "chat"])
def test_prompt_uses_the_models_expected_content_format(family):
    captioner = object.__new__(VLMCaptioner)
    captioner.family = family
    captioner.prompt = "Describe the visible clothing."

    class Processor:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            if family == "joycaption":
                assert messages[0] == {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                }
                # JoyCaption calls string methods and inserts its own image token.
                text = messages[-1]["content"].replace("<image>", "").lstrip()
            else:
                assert messages[0]["content"][0] == {"type": "image"}
                text = messages[0]["content"][1]["text"]
            return "<image>" + text + "<assistant>"

    captioner.processor = Processor()
    assert captioner._format_prompt() == (
        "<image>Describe the visible clothing.<assistant>"
    )


@pytest.mark.parametrize("family", ["joycaption", "chat"])
def test_batch_casts_image_tensors_and_decodes_only_generated_tokens(family):
    torch = pytest.importorskip("torch")
    from PIL import Image

    captioner = object.__new__(VLMCaptioner)
    captioner.family = family
    captioner.prompt = "Describe the objects."
    captioner.device = "cpu"
    captioner.dtype = torch.bfloat16
    captioner._torch = torch
    captioner.max_new_tokens = 8
    captioner.num_beams = 1
    captioner.repetition_penalty = 1.15
    captioner.no_repeat_ngram_size = 4

    class Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, *, text, images, return_tensors, padding):
            assert text == ["prompt", "prompt"]
            assert all(image.mode == "RGB" for image in images)
            assert padding is True
            return {
                "input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
                "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                "pixel_values": torch.zeros(2, 3, 16, 16),
            }

        def batch_decode(self, ids, *, skip_special_tokens):
            assert ids.tolist() == [[10, 11], [12, 13]]
            assert skip_special_tokens is True
            return [" caption one ", " caption two "]

    class Model:
        def generate(self, *, input_ids, attention_mask, pixel_values, **kwargs):
            assert input_ids.dtype == attention_mask.dtype == torch.int64
            assert pixel_values.dtype == torch.bfloat16
            assert kwargs["do_sample"] is False
            assert kwargs["pad_token_id"] == 0
            return torch.cat([input_ids, torch.tensor([[10, 11], [12, 13]])], dim=1)

    captioner.processor = Processor()
    captioner._model = Model()
    requests = [SimpleNamespace(image=Image.new("RGBA", (32, 32))) for _ in range(2)]
    assert captioner._run(requests) == ["caption one", "caption two"]


@pytest.fixture
def fake_cuda_loader(monkeypatch):
    """Exercise initialization without optional packages or weight downloads."""
    calls = []
    gpu = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
    )
    torch = SimpleNamespace(
        cuda=gpu,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
    )

    class Model:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))

    class Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("load", model_id, kwargs))
            return Model()

    def config(model_id, **kwargs):
        calls.append(("config", model_id, kwargs))
        return SimpleNamespace(model_type="llava")

    def quantization(**kwargs):
        calls.append(("quantization", kwargs))
        return SimpleNamespace(**kwargs)

    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=config),
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda *a, **kw: SimpleNamespace(tokenizer=None)
        ),
        AutoModelForImageTextToText=Loader,
        BlipForConditionalGeneration=Loader,
        BitsAndBytesConfig=quantization,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(vlm, "find_spec", lambda name: object())
    return torch, calls


@pytest.mark.parametrize("bf16_supported", [False, True])
def test_nf4_loads_directly_on_cuda_and_preserves_vision_weights(
    fake_cuda_loader, bf16_supported
):
    torch, calls = fake_cuda_loader
    torch.cuda.is_bf16_supported = lambda: bf16_supported
    captioner = VLMCaptioner("joycaption-nf4", quiet=True)
    assert captioner.family == "joycaption"
    assert captioner.dtype == ("bfloat16" if bf16_supported else "float16")
    settings = next(call[1] for call in calls if call[0] == "quantization")
    assert settings == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": captioner.dtype,
        "bnb_4bit_use_double_quant": True,
        "llm_int8_skip_modules": ["vision_tower", "multi_modal_projector"],
    }
    load = next(call for call in calls if call[0] == "load")
    assert load[1] == vlm.resolve_model("joycaption")
    assert load[2]["device_map"] == {"": "cuda"}
    assert load[2]["quantization_config"] is captioner._quantization_config
    assert load[2]["torch_dtype"] == captioner.dtype
    assert load[2]["trust_remote_code"] is False
    assert not any(call[0] == "to" for call in calls)


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda"])
def test_nf4_rejects_unavailable_cuda_before_downloading(fake_cuda_loader, device):
    torch, calls = fake_cuda_loader
    torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="requires an available NVIDIA CUDA GPU"):
        VLMCaptioner("joycaption-nf4", device=device, quiet=True)
    assert calls == []


def test_nf4_missing_dependency_has_install_help_before_download(
    fake_cuda_loader, monkeypatch
):
    _, calls = fake_cuda_loader
    monkeypatch.setattr(vlm, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="requirements-caption-nf4.txt"):
        VLMCaptioner("joycaption-nf4", quiet=True)
    assert calls == []


@pytest.mark.parametrize("model", ["joycaption", vlm.resolve_model("joycaption")])
def test_normal_joycaption_still_loads_without_bitsandbytes(
    fake_cuda_loader, monkeypatch, model
):
    _, calls = fake_cuda_loader
    monkeypatch.setattr(vlm, "find_spec", lambda name: None)
    captioner = VLMCaptioner(model, device="cpu", quiet=True)
    assert captioner.family == "joycaption"
    assert captioner._quantization_config is None
    assert not any(call[0] == "quantization" for call in calls)
    assert ("to", "cpu") in calls
