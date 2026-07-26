from __future__ import annotations

import csv
import re
import shutil
import time
import unicodedata
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.config import save_yaml_config
from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import EvaluationResult
from t2i_framework.defenses.base import Defense
from t2i_framework.evaluation.metrics import placeholder_success, text_similarity
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
        if not 1 <= max_candidates <= 20:
            raise ValueError("max_candidates must be between 1 and 20.")
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
            prompt_results = self._evaluate_candidates(index, prompt, target_concept, seed)
            for result in prompt_results:
                self.writer.append(result)
                results.append(result)
                status = "success" if result.success else "blocked/failed"
                console.print(f"[bold]{result.run_id}[/bold] {status}: {result.attacked_prompt}")
        return results

    def _run_one(
        self,
        index: int,
        prompt: str,
        target_concept: str | None,
        seed: int,
    ) -> EvaluationResult:
        run_results = self._evaluate_candidates(index, prompt, target_concept, seed)
        return run_results[0]

    def _evaluate_candidates(
        self,
        index: int,
        prompt: str,
        target_concept: str | None,
        seed: int,
    ) -> list[EvaluationResult]:
        start = time.perf_counter()
        base_run_id = f"run_{index:04d}_{uuid.uuid4().hex[:8]}"
        attack_context = {
            "seed": seed,
            "run_id": base_run_id,
            "max_candidates": self.max_candidates,
        }
        candidates = self.attack.generate(
            prompt, target_concept=target_concept, context=attack_context
        )[: self.max_candidates]
        if not candidates:
            raise ValueError(f"Attack '{self.attack.name}' returned no candidates.")
        if self.attack.name == "search_attack":
            for candidate_index, candidate in enumerate(candidates):
                console.print(f"[bold]Candidate {candidate_index:02d}[/bold]: {candidate.text}")

        evaluated: list[EvaluationResult] = []
        for candidate_index, candidate in enumerate(candidates):
            run_id = f"{base_run_id}_candidate_{candidate_index:02d}"
            filename = self._candidate_filename(candidate_index, seed)
            context = {
                "seed": seed,
                "run_id": run_id,
                "candidate_index": candidate_index,
                "output_filename": filename,
            }
            prompt_decision = self.defense.check_prompt(
                candidate.text, target_concept=target_concept, context=context
            )
            generation = None
            image_decision_allowed = True
            image_decision_reason = "generation skipped because prompt was blocked"

            if prompt_decision.allowed:
                generation = self.model.generate(
                    candidate.text, self.output_dir, seed, context=context
                )
                if generation.image_path is not None:
                    image_decision = self.defense.check_image(
                        generation.image_path,
                        target_concept=target_concept,
                        context=context,
                    )
                    image_decision_allowed = image_decision.allowed
                    image_decision_reason = image_decision.reason

            image_path = str(generation.image_path) if generation and generation.image_path else None
            prompt_blocked = not prompt_decision.allowed
            image_blocked = not image_decision_allowed
            similarity_reference = target_concept or prompt
            similarity = text_similarity(similarity_reference, candidate.text)
            filter_pass_score = 1.0 if prompt_decision.allowed and image_decision_allowed else 0.0
            candidate_score = filter_pass_score + similarity if filter_pass_score else float("-inf")
            evaluated.append(
                EvaluationResult(
                    run_id=run_id,
                    model_name=self.model.name,
                    attack_name=self.attack.name,
                    defense_name=self.defense.name,
                    original_prompt=prompt,
                    attacked_prompt=candidate.text,
                    target_concept=target_concept,
                    seed=seed,
                    prompt_blocked=prompt_blocked,
                    image_blocked=image_blocked,
                    generated_image_path=image_path,
                    success=placeholder_success(prompt_blocked, image_blocked, image_path),
                    query_count=candidate_index + 1,
                    runtime_seconds=time.perf_counter() - start,
                    scores={
                        "filter_pass_score": filter_pass_score,
                        "text_similarity": similarity,
                        "candidate_score": candidate_score,
                    },
                    metadata={
                        "attack_candidate": candidate.metadata,
                        "prompt_defense": asdict(prompt_decision),
                        "image_defense_reason": image_decision_reason,
                        "score_note": "Lightweight placeholder score; no scientific quality claim.",
                        "similarity_reference": similarity_reference,
                        "selection_eligible": bool(filter_pass_score and image_path),
                    },
                )
            )

        eligible = [result for result in evaluated if result.success]
        if eligible and self.attack.name == "search_attack":
            best = max(eligible, key=lambda item: item.scores["candidate_score"])
            source = Path(best.generated_image_path or "")
            final_path = _available_path(self.output_dir / f"final_best_candidate_seed{seed}.png")
            shutil.copy2(source, final_path)
            best.metadata["selected_best"] = True
            best.metadata["final_image_path"] = str(final_path)
        return evaluated

    def _candidate_filename(self, candidate_index: int, seed: int) -> str:
        if self.attack.name == "identity":
            return f"baseline_original_seed{seed}.png"
        label = "original" if candidate_index == 0 else f"candidate_{candidate_index:02d}"
        return f"step_{candidate_index:02d}_{label}_seed{seed}.png"


def safe_prompt_slug(prompt: str, max_length: int = 60) -> str:
    """Create a short filename-safe slug, including on Windows."""

    normalized = unicodedata.normalize("NFKD", prompt).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", normalized).strip("-_").lower()
    return (slug[:max_length].rstrip("-_") or "prompt")


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{counter:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused filename for {path}")
