from __future__ import annotations


class CLIPTextSimilarityScorer:
    """Lazy CLIP text-embedding cosine similarity scorer."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._torch = None

    def score(self, source: str, candidate: str) -> float:
        scores = self.score_many(source, [candidate])
        return scores[0] if scores else 0.0

    def score_many(self, source: str, candidates: list[str]) -> list[float]:
        self._load()
        if not candidates:
            return []

        inputs = self._tokenizer(
            [source, *candidates],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
            features = self._extract_tensor(features)
            features = features / features.norm(dim=-1, keepdim=True)
            similarities = (features[0].unsqueeze(0) * features[1:]).sum(dim=-1)

        return [float(value) for value in similarities.detach().cpu().tolist()]

    def _extract_tensor(self, features: object):
        if isinstance(features, self._torch.Tensor):
            return features

        for attr_name in ("text_embeds", "pooler_output"):
            value = getattr(features, attr_name, None)
            if isinstance(value, self._torch.Tensor):
                return value

        if isinstance(features, (tuple, list)):
            for value in features:
                if isinstance(value, self._torch.Tensor) and value.ndim == 2:
                    return value

        raise RuntimeError(
            "CLIP similarity could not read text embeddings from the installed "
            "transformers CLIP output."
        )

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "CLIP similarity requires torch and transformers. "
                'Install model dependencies with: pip install -e ".[models]"'
            ) from exc

        self._torch = torch
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self._device)
        except (OSError, RuntimeError):
            try:
                self._tokenizer = CLIPTokenizer.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                )
                self._model = CLIPModel.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                ).to(self._device)
            except (OSError, RuntimeError) as local_exc:
                raise RuntimeError(
                    "CLIP similarity could not load the CLIP model. "
                    "Run once with internet access so Hugging Face can cache "
                    f"{self.model_id}, or disable it with use_clip_similarity: false."
                ) from local_exc
        self._model.eval()
