"""Frozen SD 1.4 CLIP features and paired concept extraction for Ring-A-Bell."""

from __future__ import annotations

import gc
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

ENCODER_ID = "CompVis/stable-diffusion-v1-4"


def normalize_concept_key(concept: str) -> str:
    """Normalize lookup keys only; never rewrite the framework target or prompt."""
    if not isinstance(concept, str):
        raise TypeError("Concept key must be text.")
    normalized = " ".join(unicodedata.normalize("NFKC", concept).casefold().split())
    if not normalized:
        raise ValueError("Concept key must be non-empty text.")
    return normalized


def load_pairs(path: Path, concept: str) -> tuple[list[tuple[str, str]], str]:
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("concepts"), dict):
        raise TypeError("Concept data must contain a 'concepts' object.")
    lookup = {}
    for key, entries in data["concepts"].items():
        normalized = normalize_concept_key(key)
        if normalized in lookup:
            raise ValueError(f"Ambiguous concept keys after normalization: {key!r}.")
        lookup[normalized] = entries
    entries = lookup.get(normalize_concept_key(concept))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No positive/negative pairs configured for concept {concept!r}.")
    pairs = []
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip()
            for key in ("positive", "negative")
        ):
            raise ValueError("Every concept pair needs non-empty positive and negative text.")
        pairs.append((entry["positive"], entry["negative"]))
    return pairs, hashlib.sha256(payload).hexdigest()


def extract_concept(encoder: Any, pairs: list[tuple[str, str]]) -> Any:
    import numpy as np
    import torch

    # Equation (3); the notebook repeats every prompt five times before np.mean.
    positive = np.concatenate([encoder.embed_texts([p] * 5).cpu().numpy() for p, _ in pairs])
    negative = np.concatenate([encoder.embed_texts([n] * 5).cpu().numpy() for _, n in pairs])
    vector = np.mean(positive - negative, axis=0)
    if vector.shape != (77, 768) or not np.isfinite(vector).all():
        raise ValueError("Concept vector must be finite and have shape (77, 768).")
    return torch.from_numpy(vector)


class RingABellEncoder:
    """Load only tokenizer/text_encoder, never the SD 1.4 image pipeline."""

    def __init__(self, device: str | None, batch_size: int, revision: str | None) -> None:
        self.device = device
        self.batch_size = batch_size
        self.revision = revision
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            ENCODER_ID, subfolder="tokenizer", revision=self.revision
        )
        self.model = CLIPTextModel.from_pretrained(
            ENCODER_ID,
            subfolder="text_encoder",
            revision=self.revision,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        if (
            self.model.config.hidden_size != 768
            or self.model.config.max_position_embeddings != 77
            or self.tokenizer.bos_token_id != 49406
            or self.tokenizer.eos_token_id != 49407
            or self.tokenizer.vocab_size != 49408
        ):
            raise ValueError("Loaded encoder/tokenizer does not match the authors' SD 1.4 CLIP.")

    def embed_ids(self, ids: Any) -> Any:
        import torch

        with torch.inference_mode():
            # Full last_hidden_state, including padding; no attention_mask or pooling.
            return self.model(ids.to(self.device), return_dict=True)[0]

    def embed_texts(self, texts: list[str]) -> Any:
        ids = self.tokenizer(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        return self.embed_ids(ids)

    def losses(self, population: list[Any], target: Any) -> Any:
        import numpy as np
        import torch

        from t2i_framework.attacks.ring_a_bell_search import fitness

        target = target.to(self.device)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(population), self.batch_size):
                ids = torch.cat(population[start : start + self.batch_size], dim=0)
                chunks.append(fitness(self.embed_ids(ids), target).cpu().numpy())
        return np.concatenate(chunks)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def metadata(self) -> dict[str, Any]:
        import numpy as np
        import torch
        import transformers

        return {
            "device": self.device,
            "dtype": "float32",
            "attention_implementation": "eager",
            "encoder_revision": getattr(self.model.config, "_commit_hash", None),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "transformers_version": transformers.__version__,
        }

    def close(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
