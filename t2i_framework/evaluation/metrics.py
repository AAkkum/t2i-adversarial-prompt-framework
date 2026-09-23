from __future__ import annotations

import re


def text_similarity(original: str, candidate: str) -> float:
    """Lightweight Jaccard token overlap, not a research-grade semantic metric."""

    original_tokens = set(re.findall(r"[a-z0-9]+", original.lower()))
    candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    union = original_tokens | candidate_tokens
    return len(original_tokens & candidate_tokens) / len(union) if union else 0.0
