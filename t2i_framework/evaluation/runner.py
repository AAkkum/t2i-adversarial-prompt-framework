from __future__ import annotations

import csv
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.config import save_yaml_config
from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import EvaluationResult
from t2i_framework.defenses.base import Defense
from t2i_framework.evaluation.metrics import placeholder_success
from t2i_framework.evaluation.result_writer import ResultWriter
from t2i_framework.models.base import ImageModel


def read_prompt_file(path: Path) -> list[tuple[str, str | None]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError("Prompt CSV must contain a 'prompt' column.")
        return [(row["prompt"], row.get("target_concept") or None) for row in reader]


class ExperimentRunner:
    """Runs model x attack x defense evaluations for one or more prompts."""

    def __init__(
        self,
        model: ImageModel,
        attack: Attack,
        defense: Defense,
        output_dir: Path,
        max_candidates: int = 1,
    ) -> None:
        self.model = model
        self.attack = attack
        self.defense = defense
        self.output_dir = output_dir
        self.max_candidates = max_candidates
        self.writer = ResultWriter(output_dir)

    def run(
        self,
        prompts: list[tuple[str, str | None]],
        seed: int,
        config: dict[str, Any] | None = None,
    ) -> list[EvaluationResult]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_yaml_config(
            {
                "model": self.model.name,
                "attack": self.attack.name,
                "defense": self.defense.name,
                "seed": seed,
                "max_candidates": self.max_candidates,
                "extra_config": config or {},
            },
            self.output_dir / "config.yaml",
        )

        results: list[EvaluationResult] = []
        for index, (prompt, target_concept) in enumerate(prompts):
            result = self._run_one(index, prompt, target_concept, seed)
            self.writer.append(result)
            results.append(result)
            status = "success" if result.success else "blocked/failed"
            console.print(f"[bold]{result.run_id}[/bold] {status}: {result.original_prompt}")
        return results

    def _run_one(
        self,
        index: int,
        prompt: str,
        target_concept: str | None,
        seed: int,
    ) -> EvaluationResult:
        start = time.perf_counter()
        run_id = f"run_{index:04d}_{uuid.uuid4().hex[:8]}"
        context = {"seed": seed, "run_id": run_id}
        candidates = self.attack.generate(prompt, target_concept=target_concept, context=context)
        selected = candidates[: self.max_candidates][0]
        prompt_decision = self.defense.check_prompt(
            selected.text, target_concept=target_concept, context=context
        )
        generation = None
        image_decision_allowed = True
        image_decision_reason = "generation skipped"

        if prompt_decision.allowed:
            generation = self.model.generate(selected.text, self.output_dir, seed, context=context)
            if generation.image_path is not None:
                image_decision = self.defense.check_image(
                    generation.image_path, target_concept=target_concept, context=context
                )
                image_decision_allowed = image_decision.allowed
                image_decision_reason = image_decision.reason

        image_path = str(generation.image_path) if generation and generation.image_path else None
        prompt_blocked = not prompt_decision.allowed
        image_blocked = not image_decision_allowed
        runtime = time.perf_counter() - start
        return EvaluationResult(
            run_id=run_id,
            model_name=self.model.name,
            attack_name=self.attack.name,
            defense_name=self.defense.name,
            original_prompt=prompt,
            attacked_prompt=selected.text,
            target_concept=target_concept,
            seed=seed,
            prompt_blocked=prompt_blocked,
            image_blocked=image_blocked,
            generated_image_path=image_path,
            success=placeholder_success(prompt_blocked, image_blocked, image_path),
            query_count=1,
            runtime_seconds=runtime,
            scores={},
            metadata={
                "attack_candidate": selected.metadata,
                "prompt_defense": asdict(prompt_decision),
                "image_defense_reason": image_decision_reason,
            },
        )
