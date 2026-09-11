from __future__ import annotations


class SentenceTextSimilarityScorer:
    """Lazy sentence-transformer text-embedding cosine similarity scorer."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def score(self, source: str, candidate: str) -> float:
        scores = self.score_many(source, [candidate])
        return scores[0] if scores else 0.0

    def score_many(self, source: str, candidates: list[str]) -> list[float]:
        self._load()
        if not candidates:
            return []

        texts = [source, *candidates]
        embeddings = self._model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )
        similarities = embeddings[1:] @ embeddings[0]
        return [float(value) for value in similarities.detach().cpu().tolist()]

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Prompt-prompt similarity requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        try:
            self._model = SentenceTransformer(self.model_id, device=self.device)
        except (OSError, RuntimeError):
            try:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    local_files_only=True,
                )
            except (OSError, RuntimeError) as local_exc:
                raise RuntimeError(
                    "Sentence-transformer similarity could not load the model. "
                    "Run once with internet access so Hugging Face can cache "
                    f"{self.model_id}, or disable it with "
                    "evaluation.prompt_similarity.enabled: false."
                ) from local_exc
