from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AttackCandidate:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseDecision:
    allowed: bool
    reason: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    prompt: str
    image_path: Path | None
    seed: int
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    run_id: str
    model_name: str
    attack_name: str
    defense_name: str
    original_prompt: str
    attacked_prompt: str
    target_concept: str | None
    seed: int
    prompt_blocked: bool
    image_blocked: bool
    generated_image_path: str | None
    success: bool
    query_count: int
    runtime_seconds: float
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
