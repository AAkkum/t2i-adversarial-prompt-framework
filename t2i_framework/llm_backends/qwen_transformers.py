from __future__ import annotations

import gc
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3-14B"


class QwenTransformersBackend:
    """Lazy in-process Qwen backend powered by Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.model = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a response, loading the model on the first request."""
        self._load()
        chat = [{"role": "user", "content": prompt}]
        try:
            rendered = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(rendered, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        generation_options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            generation_options.update(do_sample=True, temperature=temperature)
        else:
            generation_options["do_sample"] = False

        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generation_options)
        generated = output[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def unload(self) -> None:
        """Release model references and return cached CUDA memory to PyTorch."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Transformers Qwen provider requires torch, transformers, "
                "accelerate, and safetensors. Install them with: "
                'pip install -e ".[models]"'
            ) from exc

        device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_dtype = _resolve_dtype(torch, self.dtype, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        self._model.eval()
        self._torch = torch


def _resolve_dtype(torch: Any, dtype: str | None, device: str) -> Any:
    normalized = (dtype or ("bfloat16" if device.startswith("cuda") else "float32")).lower()
    aliases = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported Transformers dtype {dtype!r}. "
            "Use auto, bfloat16, float16, or float32."
        )
    return aliases[normalized]
