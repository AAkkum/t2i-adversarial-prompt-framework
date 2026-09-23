from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import DefenseDecision
from t2i_framework.defenses.base import Defense


LatentGuardScorer = Callable[[str, list[str]], list[float]]


class LatentGuardLiteDefense(Defense):
    """Pre-generation defense inspired by LatentGuard's learned latent checker.

    This adapter reuses the public LatentGuard Embedding Mapping Layer idea while
    fitting the framework's standard `check_prompt` API. It expects the released
    LatentGuard state dict to be present locally and compares a prompt against
    runtime/restricted concepts.
    """

    name = "latent_guard_lite"

    def __init__(
        self,
        concepts_path: Path | None = None,
        weights_path: Path | None = None,
        model_id: str = "openai/clip-vit-large-patch14",
        threshold: float = 9.0131,
        device: str | None = None,
        include_aliases: bool = True,
        use_target_concept: bool = True,
        fail_on_error: bool = False,
        expose_score: bool = True,
        log_score: bool = True,
        num_heads: int = 16,
        head_dim: int = 32,
        out_dim: int = 128,
        input_dim: int = 768,
        batch_size: int = 64,
        scorer: LatentGuardScorer | None = None,
    ) -> None:
        self.concepts_path = concepts_path or Path("data/latent_guard/restricted_concepts.yaml")
        self.weights_path = weights_path or Path("data/latent_guard/model_parameters.pth")
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self.include_aliases = include_aliases
        self.use_target_concept = use_target_concept
        self.fail_on_error = fail_on_error
        self.expose_score = expose_score
        self.log_score = log_score
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.scorer = scorer
        self.concepts = self._load_concepts(self.concepts_path)
        self._latent_scorer: _LatentGuardLiteScorer | None = None
        self._warning_printed = False

    def check_prompt(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> DefenseDecision:
        """Check whether a prompt contains any protected concept in latent space."""
        context = context or {}
        self._apply_context_config(context)
        terms = self._restricted_terms(target_concept)
        if not terms:
            return DefenseDecision(
                allowed=True,
                reason="no restricted concepts configured",
                score=0.0 if self.expose_score else None,
                metadata=self._decision_metadata(None, 0.0, 0),
            )

        scores = self._score_many(prompt, terms)
        if not scores:
            return DefenseDecision(
                allowed=True,
                reason="LatentGuard unavailable",
                score=None,
                metadata={
                    "threshold": self.threshold,
                    "checked_terms": len(terms),
                    "available": False,
                }
                if self.expose_score
                else {"score_exposed": False, "details_exposed": False},
            )

        best_index, best_score = max(enumerate(scores), key=lambda item: item[1])
        matched_term = terms[best_index]
        allowed = best_score < self.threshold

        if self.log_score:
            console.print(
                "[latent_guard_lite] "
                f'prompt="{prompt}" matched="{matched_term}" '
                f"score={best_score:.3f} threshold={self.threshold:.3f} "
                f"allowed={allowed}",
                markup=False,
            )

        return DefenseDecision(
            allowed=allowed,
            reason=self._decision_reason(allowed, best_score, matched_term),
            score=best_score if self.expose_score else None,
            metadata=self._decision_metadata(matched_term, best_score, len(terms)),
        )

    @staticmethod
    def _load_concepts(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("concepts", {})

    def _restricted_terms(self, target_concept: str | None) -> list[str]:
        terms: list[str] = []
        if target_concept and self.use_target_concept:
            terms.append(target_concept)

        for concept, config in self.concepts.items():
            terms.append(concept)
            if self.include_aliases:
                terms.extend(config.get("aliases", []))

        unique_terms = []
        seen = set()
        for term in terms:
            key = str(term).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique_terms.append(str(term).strip())
        return unique_terms

    def _score_many(self, prompt: str, terms: list[str]) -> list[float]:
        if self.scorer is not None:
            return [float(value) for value in self.scorer(prompt, terms)]

        try:
            if self._latent_scorer is None:
                self._latent_scorer = _LatentGuardLiteScorer(
                    weights_path=self.weights_path,
                    model_id=self.model_id,
                    device=self.device,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    out_dim=self.out_dim,
                    input_dim=self.input_dim,
                    batch_size=self.batch_size,
                )
            return self._latent_scorer.score_many(prompt, terms)
        except RuntimeError as exc:
            if self.fail_on_error:
                raise
            if not self._warning_printed:
                console.print(f"[latent_guard_lite] disabled: {exc}", markup=False)
                self._warning_printed = True
            return []

    def _apply_context_config(self, context: dict[str, Any]) -> None:
        defense_config = dict((context.get("config") or {}).get("defense", {}))
        defense_config.pop("name", None)

        if "concepts_path" in defense_config:
            concepts_path = Path(defense_config["concepts_path"])
            if concepts_path != self.concepts_path:
                self.concepts_path = concepts_path
                self.concepts = self._load_concepts(self.concepts_path)

        if "weights_path" in defense_config:
            weights_path = Path(defense_config["weights_path"])
            if weights_path != self.weights_path:
                self.weights_path = weights_path
                self._latent_scorer = None

        for key in [
            "model_id",
            "threshold",
            "device",
            "include_aliases",
            "use_target_concept",
            "fail_on_error",
            "expose_score",
            "log_score",
            "num_heads",
            "head_dim",
            "out_dim",
            "input_dim",
            "batch_size",
        ]:
            if key in defense_config:
                new_value = defense_config[key]
                if getattr(self, key) == new_value:
                    continue
                setattr(self, key, new_value)
                if key in {
                    "model_id",
                    "device",
                    "num_heads",
                    "head_dim",
                    "out_dim",
                    "input_dim",
                    "batch_size",
                }:
                    self._latent_scorer = None

    def _decision_reason(self, allowed: bool, best_score: float, matched_term: str) -> str:
        if self.expose_score:
            return (
                f"LatentGuard score {best_score:.3f} >= threshold "
                f"{self.threshold:.3f}: {matched_term}"
                if not allowed
                else f"max LatentGuard score {best_score:.3f} below threshold "
                f"{self.threshold:.3f}"
            )
        return "LatentGuard score exceeded threshold" if not allowed else "LatentGuard allowed"

    def _decision_metadata(
        self,
        matched_term: str | None,
        best_score: float,
        checked_terms: int,
    ) -> dict[str, Any]:
        if not self.expose_score:
            return {
                "score_exposed": False,
                "details_exposed": False,
            }
        return {
            "matched_term": matched_term,
            "score": best_score,
            "threshold": self.threshold,
            "checked_terms": checked_terms,
            "score_exposed": True,
            "method": self.name,
            "model_id": self.model_id,
            "weights_path": str(self.weights_path),
        }


class _LatentGuardLiteScorer:
    """Lazy scorer compatible with the released LatentGuard state dict."""

    def __init__(
        self,
        weights_path: Path,
        model_id: str,
        device: str | None,
        num_heads: int,
        head_dim: int,
        out_dim: int,
        input_dim: int,
        batch_size: int,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.model_id = model_id
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.batch_size = max(1, int(batch_size))
        self._torch = None
        self._functional = None
        self._model = None
        self._clip_model = None
        self._tokenizer = None
        self._device = None
        self._concept_embedding_cache: dict[str, Any] = {}

    def score_many(self, prompt: str, terms: list[str]) -> list[float]:
        self._load()
        if not terms:
            return []

        scores: list[float] = []
        with self._torch.inference_mode():
            prompt_emb = self._embed_texts([prompt])
            for start in range(0, len(terms), self.batch_size):
                batch_terms = terms[start : start + self.batch_size]
                concept_emb = self._concept_embeddings(batch_terms)
                repeated_prompt = prompt_emb.repeat(len(batch_terms), 1, 1)
                output = self._model(repeated_prompt, concept_emb)
                scores.extend(self._forward_contrastive(output).detach().cpu().tolist())
        return [float(score) for score in scores]

    def _concept_embeddings(self, terms: list[str]):
        missing_terms = list(
            dict.fromkeys(
                term for term in terms if term not in self._concept_embedding_cache
            )
        )
        if missing_terms:
            embeddings = self._embed_texts(missing_terms)[:, 0, :]
            for term, embedding in zip(missing_terms, embeddings):
                self._concept_embedding_cache[term] = embedding.detach()

        return self._torch.stack(
            [self._concept_embedding_cache[term] for term in terms],
            dim=0,
        )

    def _load(self) -> None:
        if self._model is not None and self._clip_model is not None:
            return

        if not self.weights_path.exists():
            raise RuntimeError(
                "LatentGuard weights not found. Download the released "
                f"model_parameters.pth to {self.weights_path}."
            )

        try:
            import torch
            import torch.nn as nn
            from torch.nn import functional
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LatentGuard-lite requires torch and transformers. "
                'Install model dependencies with: pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._functional = functional
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        mapping_layer_class = self._build_mapping_layer_class(nn, functional)
        self._model = mapping_layer_class(
            self.num_heads,
            self.head_dim,
            self.out_dim,
            self.input_dim,
        ).to(self._device)
        try:
            state = torch.load(
                self.weights_path,
                map_location=self._device,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(self.weights_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state)
        self._model.eval()

        try:
            self._clip_model = CLIPModel.from_pretrained(self.model_id).to(self._device)
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "LatentGuard-lite could not load the CLIP text encoder. "
                "Run once with internet access so Hugging Face can cache "
                f"{self.model_id}."
            ) from exc
        self._clip_model.eval()

    def _build_mapping_layer_class(self, nn: Any, functional: Any) -> type:
        class EmbeddingMappingLayer(nn.Module):
            def __init__(
                self,
                num_heads: int,
                head_dim: int,
                out_dim: int = 128,
                input_dim: int = 768,
            ) -> None:
                super().__init__()
                self.num_heads = num_heads
                self.head_dim = head_dim
                self.key_d = self.head_dim * self.num_heads
                self.out_dim = out_dim
                self.x1_to_key = nn.Linear(input_dim, self.key_d)
                self.x2_to_query = nn.Linear(input_dim, self.key_d)
                self.x1_to_value = nn.Linear(input_dim, self.key_d)
                self.final_mlp = nn.Linear(self.key_d, self.out_dim)
                self.mlp_query1 = nn.Linear(self.key_d, self.out_dim)
                self.tempr = nn.Parameter(self._temperature_tensor(), requires_grad=True)

            def _temperature_tensor(self):
                return self.x1_to_key.weight.new_tensor(1 / 0.07)

            def forward(self, x1, x2):
                batch_size, seq_len, _ = x1.shape
                key = self.x1_to_key(x1).view(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                )
                key = key.transpose(1, 2)
                value = self.x1_to_value(x1).view(
                    batch_size,
                    seq_len,
                    self.num_heads,
                    self.head_dim,
                )
                value = value.transpose(1, 2)
                query = self.x2_to_query(x2).view(
                    batch_size,
                    1,
                    self.num_heads,
                    self.head_dim,
                )
                query = query.transpose(1, 2)
                attention_scores = self._torch_einsum(query, key) / (self.head_dim**0.5)
                attention_weights = functional.softmax(attention_scores, dim=-1)
                value = self._torch_value(attention_weights, value)
                value = value.view(batch_size, -1)
                value = self.final_mlp(value)
                query = query.reshape(batch_size, -1)
                query = self.mlp_query1(query)
                return value, query

            @staticmethod
            def _torch_einsum(query, key):
                import torch

                return torch.einsum("bnqd,bnkd->bnqk", query, key)

            @staticmethod
            def _torch_value(attention_weights, value):
                import torch

                return torch.einsum("bnqk,bnkd->bnqd", attention_weights, value)

        return EmbeddingMappingLayer

    def _embed_texts(self, texts: list[str]):
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        outputs = self._clip_model.text_model(input_ids=input_ids)
        hidden = outputs.last_hidden_state
        if hidden.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"LatentGuard weights expect input_dim={self.input_dim}, but "
                f"{self.model_id} returned hidden size {hidden.shape[-1]}."
            )
        eos_indices = input_ids.argmax(dim=-1)
        pooled = hidden[self._torch.arange(hidden.shape[0], device=self._device), eos_indices]
        return self._torch.cat([pooled.unsqueeze(1), hidden], dim=1)

    def _forward_contrastive(self, output: tuple[Any, Any]):
        value, query = output
        value = self._functional.normalize(value, p=2, dim=1)
        query = self._functional.normalize(query, p=2, dim=1)
        return self._torch.sum(value * query, dim=1) * self._model.tempr
