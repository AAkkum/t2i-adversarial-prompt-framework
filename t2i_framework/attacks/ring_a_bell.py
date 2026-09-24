"""Author-code-based Ring-A-Bell attack; see docs/ring_a_bell.md."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import AttackCandidate

AUTHOR_REVISION = "e4585ada0a5eb185fbef89bb94147c97f8d2f79a"


@dataclass(frozen=True)
class RingABellSettings:
    population_size: int = 200
    generations: int = 3000
    mutation_rate: float = 0.25
    crossover_rate: float = 0.5
    prompt_length: int = 16
    coefficient: float = 3.0
    concept_pairs_path: str = "data/ring_a_bell/concept_pairs.json"
    device: str | None = None
    encoder_revision: str | None = None
    batch_size: int = 32
    log_interval: int = 50

    def __post_init__(self) -> None:
        for key, minimum in (
            ("population_size", 2),
            ("generations", 1),
            ("prompt_length", 1),
            ("batch_size", 1),
            ("log_interval", 1),
        ):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}.")
        if self.prompt_length > 75:
            raise ValueError("prompt_length must be <= 75 (BOS/EOS occupy two positions).")
        for key in ("mutation_rate", "crossover_rate", "coefficient"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be numeric.")
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
            if key != "coefficient" and not 0 <= value <= 1:
                raise ValueError(f"{key} must be between 0 and 1.")


class RingABellAttack(Attack):
    name = "ring_a_bell"

    def __init__(self) -> None:
        self._encoder: Any = None

    def generate(
        self, prompt: str, target_concept: str | None = None, context: dict[str, Any] | None = None
    ) -> list[AttackCandidate]:
        context = context if context is not None else {}
        context["attack_error_stage"] = "attack_configuration"
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Ring-A-Bell requires a non-empty base prompt.")
        if not isinstance(target_concept, str) or not target_concept.strip():
            raise ValueError("Ring-A-Bell requires --target matching a concept in the pairs file.")
        config = dict((context.get("config") or {}).get("attack", {}))
        config.pop("name", None)
        settings = RingABellSettings(**config)
        seed = context.get("seed", 42)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Ring-A-Bell seed must be an integer in [0, 2**32).")

        # Optional ML imports remain lazy so registry and mock CLI need no torch.
        from t2i_framework.attacks.ring_a_bell_encoder import (
            ENCODER_ID,
            RingABellEncoder,
            extract_concept,
            load_pairs,
        )
        from t2i_framework.attacks.ring_a_bell_search import discover

        context["attack_error_stage"] = "concept_data"
        pairs, data_hash = load_pairs(Path(settings.concept_pairs_path), target_concept)
        console.print(
            f"Ring-A-Bell | seed={seed} concept={target_concept} encoder={ENCODER_ID}\n"
            f"Positive examples: {len(pairs)}; Negative examples: {len(pairs)}\n"
            f"Population={settings.population_size}; Generations={settings.generations}; "
            f"Mutation={settings.mutation_rate}; Crossover={settings.crossover_rate}; "
            f"Coefficient={settings.coefficient}; Prompt length={settings.prompt_length}",
            markup=False,
        )
        context["attack_error_stage"] = "encoder_load"
        self._encoder = RingABellEncoder(
            settings.device, settings.batch_size, settings.encoder_revision
        )
        self._encoder.load()
        context["attack_error_stage"] = "concept_extraction"
        vector = extract_concept(self._encoder, pairs)
        console.print("Concept vector: created (77, 768)")
        context["attack_error_stage"] = "target_representation"
        # Equation (4): add the unnormalised concept direction to the base embedding.
        target = self._encoder.embed_texts([prompt]).cpu() + settings.coefficient * vector
        context["attack_error_stage"] = "prompt_discovery"

        def report(record: dict[str, Any]) -> None:
            console.print(
                f"Generation {record['generation']} / {settings.generations} | "
                f"Best fitness (lower is better): {record['best_fitness']:.6f}",
                markup=False,
            )

        result = discover(self._encoder, target, settings, seed, report)
        text = self._encoder.decode(result.token_ids[1 : settings.prompt_length + 1])
        if not text.strip():
            raise ValueError("Ring-A-Bell decoded an empty prompt.")
        metadata = {
            "method": self.name,
            "seed": seed,
            "concept": target_concept,
            "encoder": ENCODER_ID,
            "author_revision": AUTHOR_REVISION,
            "parameters": asdict(settings),
            "positive_examples": len(pairs),
            "negative_examples": len(pairs),
            "concept_pairs_sha256": data_hash,
            "concept_vector_shape": list(vector.shape),
            "concept_vector_sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
            "final_fitness": result.fitness,
            "fitness_function": "sum_squared_l2_all_77x768_hidden_states",
            "fitness_evaluations": result.evaluations,
            "query_count": 0,
            "token_ids": result.token_ids,
            "progress": result.progress,
            "runtime": self._encoder.metadata(),
        }
        console.print(
            f"Final fitness: {result.fitness:.6f}\nDiscovered prompt: {text}", markup=False
        )
        context.pop("attack_error_stage", None)
        return [AttackCandidate(text=text, metadata=metadata)]

    def cleanup(self, context: dict[str, Any] | None = None) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
