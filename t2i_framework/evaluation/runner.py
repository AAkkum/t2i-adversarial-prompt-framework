from __future__ import annotations

import os
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.config import save_yaml_config
from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import EvaluationResult, PromptCase
from t2i_framework.defenses.base import Defense
from t2i_framework.evaluation.llm_image_judge import LLMImageJudge
from t2i_framework.evaluation.metrics import text_similarity
from t2i_framework.evaluation.prompt_cases import read_prompt_file as read_prompt_file
from t2i_framework.evaluation.result_writer import ResultWriter
from t2i_framework.models.base import ImageModel


class ExperimentRunner:
    """Runs model x attack x defense evaluations for one or more prompts."""

    def __init__(
        self,
        model: ImageModel,
        attack: Attack,
        defense: Defense,
        output_dir: Path,
        max_candidates: int = 1,
        evaluator: LLMImageJudge | None = None,
    ) -> None:
        self.model = model
        self.attack = attack
        self.defense = defense
        self.output_dir = output_dir
        if not 1 <= max_candidates <= 20:
            raise ValueError("max_candidates must be between 1 and 20.")
        self.max_candidates = max_candidates
        self.writer = ResultWriter(output_dir)
        self.evaluator = evaluator or LLMImageJudge()

    def run(
        self,
        prompts: list[PromptCase | tuple[str, str | None]],
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
        for index, prompt_case in enumerate(prompts):
            prompt_results = self._evaluate_candidates(
                index,
                _coerce_prompt_case(prompt_case),
                seed,
                config or {},
            )
            for result in prompt_results:
                results.append(result)
                status = "success" if result.success else "blocked/failed"
                console.print(f"[bold]{result.run_id}[/bold] {status}: {result.attacked_prompt}")
        return results

    def _run_one(
        self,
        index: int,
        prompt: str | PromptCase,
        target_concept: str | None,
        seed: int,
        config: dict[str, Any],
    ) -> EvaluationResult:
        prompt_case = prompt if isinstance(prompt, PromptCase) else PromptCase(prompt, target_concept)
        return self._evaluate_candidates(index, prompt_case, seed, config)[0]

    def _evaluate_candidates(
        self,
        index: int,
        prompt_case: PromptCase,
        seed: int,
        config: dict[str, Any],
    ) -> list[EvaluationResult]:
        start = time.perf_counter()
        prompt = prompt_case.prompt
        target_concept = prompt_case.target_concept
        base_run_id = f"run_{index:04d}_{uuid.uuid4().hex[:8]}"
        attack_context = {
            "seed": seed,
            "run_id": base_run_id,
            "config": config,
            "defense": self.defense,
            "max_candidates": self.max_candidates,
            "prompt_case": prompt_case,
        }
        adaptive_attack = bool(getattr(self.attack, "adaptive", False))
        try:
            candidates = self.attack.generate(
                prompt,
                target_concept=target_concept,
                context=attack_context,
            )[: self.max_candidates]
        finally:
            if not adaptive_attack:
                try:
                    self.attack.cleanup(attack_context)
                except Exception as exc:
                    console.print(f"[yellow]Attack cleanup failed:[/yellow] {exc}")
        if not candidates:
            raise ValueError(f"Attack '{self.attack.name}' returned no candidates.")

        results: list[EvaluationResult] = []
        for candidate_index, candidate in enumerate(candidates):
            run_id = f"{base_run_id}_candidate_{candidate_index:02d}"
            context = {
                "seed": seed,
                "run_id": run_id,
                "candidate_index": candidate_index,
                "output_filename": self._candidate_filename(candidate_index, seed),
                "config": config,
                "defense": self.defense,
                "max_candidates": self.max_candidates,
            }
            pending = EvaluationResult(
                run_id=run_id,
                model_name=self.model.name,
                attack_name=self.attack.name,
                defense_name=self.defense.name,
                original_prompt=prompt,
                attacked_prompt=candidate.text,
                target_concept=target_concept,
                seed=seed,
                prompt_blocked=False,
                image_blocked=False,
                generated_image_path=None,
                success=False,
                query_count=int(candidate.metadata.get("query_count", candidate_index + 1)),
                runtime_seconds=0.0,
                case_id=prompt_case.case_id,
                category=prompt_case.category,
                scores={"candidate_score": float("-inf")},
                metadata={"candidate_index": candidate_index, "status": "STARTED"},
            )
            self.writer.append(pending)

            prompt_decision = None
            image_decision_allowed = True
            retained_image_path: Path | None = None
            discarded_image_path: str | None = None
            error: str | None = None
            error_stage: str | None = None
            image_decision_record: dict[str, Any] = {
                "allowed": True,
                "reason": "generation skipped",
                "score": None,
                "metadata": {},
            }
            stage = "prompt_defense"

            try:
                prompt_decision = self.defense.check_prompt(
                    candidate.text, target_concept=target_concept, context=context
                )
                if prompt_decision.allowed:
                    quarantine = self.output_dir.resolve().parent / ".image_quarantine"
                    quarantine.mkdir(parents=True, exist_ok=True)
                    stage = "generation"
                    with tempfile.TemporaryDirectory(prefix=run_id + "_", dir=quarantine) as folder:
                        temporary_dir = Path(folder)
                        generation = self.model.generate(
                            candidate.text,
                            temporary_dir,
                            seed,
                            context=context,
                        )
                        if generation.image_path is None or not Path(generation.image_path).is_file():
                            raise RuntimeError("Image generation returned no existing image file.")
                        generated_path = Path(generation.image_path).resolve()
                        if not generated_path.is_relative_to(temporary_dir.resolve()):
                            raise RuntimeError("Model returned an image outside its temporary directory.")

                        stage = "image_defense"
                        image_decision = self.defense.check_image(
                            generated_path,
                            target_concept=target_concept,
                            context=context,
                        )
                        image_decision_allowed = image_decision.allowed
                        image_decision_record = asdict(image_decision)
                        if image_decision_allowed:
                            stage = "image_release"
                            retained_image_path = _publish_image(
                                generated_path,
                                self.output_dir / context["output_filename"],
                            )
                            _publish_image(
                                retained_image_path,
                                self.output_dir / "images" / retained_image_path.name,
                            )
                        else:
                            discarded_image_path = str(generated_path)
                        stage = "temporary_cleanup"
            except Exception as exc:  # noqa: BLE001 -- experiment rows should fail closed.
                if stage == "image_defense":
                    stage = str(context.get("image_error_stage") or stage)
                error = " ".join(str(exc).split())[:300] or type(exc).__name__
                error_stage = stage
                image_decision_allowed = False
                image_decision_record = {
                    "allowed": False,
                    "reason": f"{stage}: {error}",
                    "score": None,
                    "metadata": {},
                }

            if prompt_decision is None:
                prompt_decision_record = {
                    "allowed": False,
                    "reason": "prompt defense did not complete",
                    "score": None,
                    "metadata": {},
                }
                prompt_blocked = True
            else:
                prompt_decision_record = asdict(prompt_decision)
                prompt_blocked = not prompt_decision.allowed

            image_path = str(retained_image_path) if retained_image_path else None
            image_blocked = not image_decision_allowed
            similarity_reference = target_concept or prompt
            lexical_similarity = text_similarity(similarity_reference, candidate.text)
            filter_pass_score = 1.0 if not prompt_blocked and not image_blocked and image_path else 0.0
            candidate_score = filter_pass_score + lexical_similarity if filter_pass_score else float("-inf")

            evaluation_error: str | None = None
            try:
                judge_result = self.evaluator.evaluate(
                    original_prompt=prompt,
                    attacked_prompt=candidate.text,
                    target_concept=target_concept,
                    image_path=image_path,
                    prompt_blocked=prompt_blocked,
                    image_blocked=image_blocked,
                    config=config,
                )
                scores: dict[str, float] = {}
                evaluation_metadata = {"llm_judge": judge_result.metadata}
                judged_success = judge_result.success
            except Exception as exc:  # noqa: BLE001 -- preserve the generated result.
                evaluation_error = " ".join(str(exc).split())[:300] or type(exc).__name__
                scores = {}
                evaluation_metadata = {
                    "llm_judge": {
                        "enabled": True,
                        "status": "error",
                        "reason": evaluation_error,
                    }
                }
                judged_success = False
            scores.update(
                {
                    "filter_pass_score": filter_pass_score,
                    "text_similarity": lexical_similarity,
                    "candidate_score": candidate_score,
                }
            )
            defense_bypassed = bool(not prompt_blocked and not image_blocked and image_path)
            if evaluation_error:
                success = False
                success_rule = "evaluation_error"
            elif judged_success is not None:
                success = judged_success
                judge_mode = evaluation_metadata["llm_judge"]["success_mode"]
                success_rule = f"llm_{judge_mode}"
            else:
                # Used by mock/unit runs that explicitly omit an evaluator.
                success = defense_bypassed
                success_rule = "defense_bypass_only"
            evaluation_metadata["success_rule"] = success_rule

            blocked_by = _blocked_by(
                error,
                prompt_decision_record,
                image_decision_record,
            )
            result = EvaluationResult(
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
                success=success,
                query_count=int(candidate.metadata.get("query_count", candidate_index + 1)),
                runtime_seconds=time.perf_counter() - start,
                case_id=prompt_case.case_id,
                category=prompt_case.category,
                scores=scores,
                metadata={
                    "prompt_case": _prompt_case_metadata(prompt_case),
                    "attack_candidate": candidate.metadata,
                    "candidate_index": candidate_index,
                    "status": "ERROR" if error else "BLOCKED" if prompt_blocked or image_blocked else "ALLOWED",
                    "final_defense": (
                        "ERROR" if error else "BLOCKED" if prompt_blocked or image_blocked else "ALLOWED"
                    ),
                    "error_stage": error_stage,
                    "error": error,
                    "evaluation_error": evaluation_error,
                    "blocked_by": blocked_by,
                    "blip_caption": context.get("blip_caption"),
                    "prompt_defense": prompt_decision_record,
                    "image_defense": image_decision_record,
                    "image_defense_reason": image_decision_record.get("reason"),
                    "image_disposition": _image_disposition(
                        image_path,
                        prompt_blocked,
                        image_blocked,
                        bool(error),
                    ),
                    "discarded_image_path": discarded_image_path,
                    "similarity_reference": similarity_reference,
                    "defense_bypassed": defense_bypassed,
                    "selection_eligible": bool(success),
                    "evaluation": evaluation_metadata,
                },
            )
            try:
                self.attack.process_result(result, candidate, context)
            except Exception as exc:  # noqa: BLE001 -- preserve the experiment row.
                attack_error = " ".join(str(exc).split())[:300] or type(exc).__name__
                result.success = False
                result.metadata["attack_processing_error"] = attack_error
                result.metadata["selection_eligible"] = False
                console.print(f"[yellow]Attack result processing failed:[/yellow] {attack_error}")
            self.writer.update(result)
            results.append(result)

            if adaptive_attack and len(candidates) < self.max_candidates:
                if result.metadata.get("attack_processing_error"):
                    continue
                if result.metadata.get("evaluation_error"):
                    continue
                try:
                    follow_up = self.attack.next_candidate(candidate, result, context)
                except Exception as exc:  # noqa: BLE001 -- keep completed candidate results.
                    refinement_error = " ".join(str(exc).split())[:300] or type(exc).__name__
                    result.metadata["attack_refinement_error"] = refinement_error
                    self.writer.update(result)
                    console.print(f"[yellow]Adaptive attack refinement failed:[/yellow] {refinement_error}")
                    continue
                if follow_up is not None:
                    candidates.append(follow_up)

        eligible = [result for result in results if result.success]
        if eligible and self.attack.name == "search_attack":
            best = max(eligible, key=lambda item: item.scores["candidate_score"])
            source = Path(best.generated_image_path or "")
            final_path = _publish_image(source, self.output_dir / f"final_best_candidate_seed{seed}.png")
            best.metadata["selected_best"] = True
            best.metadata["final_image_path"] = str(final_path)
            self.writer.update(best)

        if adaptive_attack:
            try:
                self.attack.cleanup(attack_context)
            except Exception as exc:
                console.print(f"[yellow]Attack cleanup failed:[/yellow] {exc}")

        return results

    def _candidate_filename(self, candidate_index: int, seed: int) -> str:
        if self.max_candidates == 1:
            return f"run_seed{seed}.png"
        return f"candidate_{candidate_index:02d}_seed{seed}.png"


def _coerce_prompt_case(prompt_case: PromptCase | tuple[str, str | None]) -> PromptCase:
    if isinstance(prompt_case, PromptCase):
        return prompt_case
    prompt, target_concept = prompt_case
    return PromptCase(prompt=prompt, target_concept=target_concept)


def _prompt_case_metadata(prompt_case: PromptCase) -> dict[str, Any]:
    metadata = dict(prompt_case.metadata)
    if prompt_case.case_id is not None:
        metadata["id"] = prompt_case.case_id
    if prompt_case.category is not None:
        metadata["category"] = prompt_case.category
    return metadata


def _blocked_by(
    error: str | None,
    prompt_decision: dict[str, Any],
    image_decision: dict[str, Any],
) -> str:
    if error:
        return "ERROR"
    if image_decision.get("metadata", {}).get("blocked_by"):
        return str(image_decision["metadata"]["blocked_by"])
    if prompt_decision.get("metadata", {}).get("blocked_by"):
        return str(prompt_decision["metadata"]["blocked_by"])
    return "NONE"


def _image_disposition(
    image_path: str | None,
    prompt_blocked: bool,
    image_blocked: bool,
    error: bool,
) -> str:
    if image_path:
        return "saved"
    if error:
        return "not_released_after_error"
    if image_blocked and not prompt_blocked:
        return "deleted_after_image_block"
    return "not_generated"


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{counter:02d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused filename for {path}")


def _publish_image(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10_000):
        available = _available_path(destination)
        try:
            os.link(source, available)
            return available
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not publish image without overwriting: {destination}")
