from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from t2i_framework.attacks.base import Attack
from t2i_framework.core.logging_utils import console
from t2i_framework.core.types import AttackCandidate, DefenseDecision
from t2i_framework.core.word_lists import load_replacement_map, load_term_set
from t2i_framework.judges.ollama_similarity import OllamaSimilarityJudge
from t2i_framework.paraphrasers.qwen_ollama import QwenOllamaParaphraser
from t2i_framework.similarity.clip_text import CLIPTextSimilarityScorer


Paraphraser = Callable[[str, str, int], list[str]]
SimilarityScorer = Callable[[str, str], float]
SimilarityJudge = Callable[[str, str, str], float]


class TextFoolerStyleAttack(Attack):
    """Adaptive TextFooler-style attack using deletion tests and paraphrase candidates."""

    name = "textfooler_style"

    DEFAULT_REPLACEMENTS = load_replacement_map("textfooler_replacements")
    DEFAULT_STOP_WORDS = load_term_set("stop_words")
    CONTEXT_LEAK_TERMS = load_term_set("context_leak_terms")
    WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

    def __init__(
        self,
        paraphraser: Paraphraser | None = None,
        candidate_count: int = 5,
        max_rounds: int = 2,
        stop_words: set[str] | None = None,
        use_qwen_fallback: bool = True,
        min_similarity: float = 0.7,
        similarity_scorer: SimilarityScorer | None = None,
        use_clip_similarity: bool = True,
        max_candidate_batches: int = 2,
        log_similarity: bool = True,
        use_llm_judge_fallback: bool = False,
        llm_judge_threshold: float = 0.75,
        llm_judge_model: str | None = None,
        llm_judge: SimilarityJudge | None = None,
        log_llm_judge: bool = True,
        filter_context_leaks: bool = False,
        log_candidate_filtering: bool = False,
        log_raw_paraphrases: bool = False,
        paraphraser_model: str | None = None,
        unload_ollama_after_attack: bool = True,
        log_ollama_unload: bool = True,
    ) -> None:
        self.paraphraser = paraphraser
        self.candidate_count = candidate_count
        self.max_rounds = max_rounds
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS
        self.use_qwen_fallback = use_qwen_fallback
        self.min_similarity = min_similarity
        self.similarity_scorer = similarity_scorer
        self.use_clip_similarity = use_clip_similarity
        self.max_candidate_batches = max_candidate_batches
        self.log_similarity = log_similarity
        self.use_llm_judge_fallback = use_llm_judge_fallback
        self.llm_judge_threshold = llm_judge_threshold
        self.llm_judge_model = llm_judge_model
        self.llm_judge = llm_judge
        self.log_llm_judge = log_llm_judge
        self.filter_context_leaks = filter_context_leaks
        self.log_candidate_filtering = log_candidate_filtering
        self.log_raw_paraphrases = log_raw_paraphrases
        self.paraphraser_model = paraphraser_model
        self.unload_ollama_after_attack = unload_ollama_after_attack
        self.log_ollama_unload = log_ollama_unload
        self._clip_similarity_scorer = None
        self._similarity_warning_printed = False
        self._ollama_paraphraser = None
        self._ollama_judge = None
        self._judge_warning_printed = False

    def generate(
        self,
        prompt: str,
        target_concept: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[AttackCandidate]:
        context = context or {}
        self._apply_context_config(context)
        defense = context.get("defense")
        if not hasattr(defense, "check_prompt"):
            return [
                AttackCandidate(
                    text=prompt,
                    metadata={"method": self.name, "status": "no_defense_in_context"},
                )
            ]

        trace: list[dict[str, Any]] = []
        original_decision = self._check(defense, prompt, target_concept, context)
        if original_decision.allowed:
            return [
                AttackCandidate(
                    text=prompt,
                    metadata={
                        "method": self.name,
                        "status": "original_allowed",
                        "original_defense": asdict(original_decision),
                    },
                )
            ]

        current_prompt = prompt
        current_decision = original_decision
        query_count = 1
        judge_query_count = 0

        for _round_index in range(self.max_rounds):
            changed = False
            ranked_units = self._rank_units(
                current_prompt,
                target_concept,
                defense,
                current_decision,
                context,
            )
            query_count += len(ranked_units)

            for unit, _importance in ranked_units:
                if not self._contains_unit(current_prompt, unit):
                    continue

                best_blocked_prompt = None
                best_blocked_decision = None
                best_allowed: dict[str, Any] | None = None

                for batch_index in range(self.max_candidate_batches):
                    replacements = self._replacement_candidates(
                        unit,
                        current_prompt,
                        target_concept,
                    )
                    if not replacements:
                        break

                    candidate_records = []
                    for replacement in replacements:
                        candidate_prompt = self._replace_unit(current_prompt, unit, replacement)
                        if candidate_prompt == current_prompt:
                            continue

                        similarity = self._score_similarity(unit, replacement)
                        similarity_allowed = similarity >= self.min_similarity
                        if self.log_similarity:
                            console.print(
                                "[textfooler_style] "
                                f'unit="{unit}" candidate="{replacement}" '
                                f"clip_similarity={similarity:.3f} "
                                f"threshold={self.min_similarity:.3f}",
                                markup=False,
                            )

                        trace_item: dict[str, Any] = {
                            "unit": unit,
                            "replacement": replacement,
                            "candidate_prompt": candidate_prompt,
                            "candidate_batch": batch_index,
                            "similarity": similarity,
                            "clip_similarity": similarity,
                            "similarity_method": "clip",
                            "similarity_allowed": similarity_allowed,
                            "min_similarity": self.min_similarity,
                        }

                        candidate_records.append(
                            {
                                "replacement": replacement,
                                "candidate_prompt": candidate_prompt,
                                "trace_item": trace_item,
                                "similarity": similarity,
                                "similarity_allowed": similarity_allowed,
                                "similarity_method": "clip",
                            }
                        )

                    if (
                        candidate_records
                        and self.use_llm_judge_fallback
                        and not any(record["similarity_allowed"] for record in candidate_records)
                    ):
                        if self.log_llm_judge:
                            console.print(
                                "[textfooler_style] "
                                f'CLIP rejected all candidates for "{unit}"; '
                                "using LLM judge fallback",
                                markup=False,
                            )
                        for record in candidate_records:
                            judge_score = self._score_llm_judge(
                                unit,
                                record["replacement"],
                                current_prompt,
                            )
                            judge_query_count += 1
                            judge_allowed = judge_score >= self.llm_judge_threshold
                            record["trace_item"].update(
                                {
                                    "judge_similarity": judge_score,
                                    "judge_allowed": judge_allowed,
                                    "judge_threshold": self.llm_judge_threshold,
                                }
                            )
                            if self.log_llm_judge:
                                console.print(
                                    "[textfooler_style] "
                                    f'llm_judge unit="{unit}" '
                                    f'candidate="{record["replacement"]}" '
                                    f"score={judge_score:.3f} "
                                    f"threshold={self.llm_judge_threshold:.3f}",
                                    markup=False,
                                )
                            if judge_allowed:
                                record["similarity"] = judge_score
                                record["similarity_allowed"] = True
                                record["similarity_method"] = "llm_judge"
                                record["trace_item"].update(
                                    {
                                        "similarity": judge_score,
                                        "similarity_allowed": True,
                                        "similarity_method": "llm_judge",
                                    }
                                )

                    for record in candidate_records:
                        replacement = record["replacement"]
                        candidate_prompt = record["candidate_prompt"]
                        trace_item = record["trace_item"]

                        if not record["similarity_allowed"]:
                            trace_item["status"] = (
                                "discarded_low_similarity_and_judge"
                                if "judge_similarity" in trace_item
                                else "discarded_low_similarity"
                            )
                            trace.append(trace_item)
                            continue

                        candidate_decision = self._check(
                            defense,
                            candidate_prompt,
                            target_concept,
                            context,
                        )
                        query_count += 1
                        trace_item.update(
                            {
                                "status": "defense_checked",
                                "allowed": candidate_decision.allowed,
                                "score": candidate_decision.score,
                                "reason": candidate_decision.reason,
                            }
                        )
                        trace.append(trace_item)

                        if candidate_decision.allowed:
                            allowed_candidate = {
                                "prompt": candidate_prompt,
                                "decision": candidate_decision,
                                "replacement": replacement,
                                "similarity": record["similarity"],
                                "similarity_method": record["similarity_method"],
                                "clip_similarity": trace_item.get("clip_similarity"),
                                "judge_similarity": trace_item.get("judge_similarity"),
                                "unit": unit,
                            }
                            if best_allowed is None or similarity > best_allowed["similarity"]:
                                best_allowed = allowed_candidate
                            continue

                        if best_blocked_decision is None or self._is_better(
                            candidate_decision,
                            best_blocked_decision,
                        ):
                            best_blocked_prompt = candidate_prompt
                            best_blocked_decision = candidate_decision

                    if best_allowed is not None:
                        break

                if best_allowed is not None:
                    final_decision = best_allowed["decision"]
                    return [
                        AttackCandidate(
                            text=best_allowed["prompt"],
                            metadata={
                                "method": self.name,
                                "status": "bypassed_prompt_defense",
                                "query_count": query_count,
                                "selected_unit": best_allowed["unit"],
                                "selected_replacement": best_allowed["replacement"],
                                "selected_similarity": best_allowed["similarity"],
                                "selected_similarity_method": best_allowed["similarity_method"],
                                "selected_clip_similarity": best_allowed["clip_similarity"],
                                "selected_judge_similarity": best_allowed["judge_similarity"],
                                "min_similarity": self.min_similarity,
                                "llm_judge_threshold": self.llm_judge_threshold,
                                "judge_query_count": judge_query_count,
                                "original_defense": asdict(original_decision),
                                "final_defense": asdict(final_decision),
                                "trace": trace,
                            },
                        )
                    ]

                if best_blocked_prompt is not None and best_blocked_decision is not None:
                    current_prompt = best_blocked_prompt
                    current_decision = best_blocked_decision
                    changed = True

            if not changed:
                break
            if current_decision.allowed:
                break

        return [
            AttackCandidate(
                text=current_prompt,
                metadata={
                    "method": self.name,
                    "status": "no_passing_candidate",
                    "query_count": query_count,
                    "min_similarity": self.min_similarity,
                    "llm_judge_threshold": self.llm_judge_threshold,
                    "judge_query_count": judge_query_count,
                    "original_defense": asdict(original_decision),
                    "final_defense": asdict(current_decision),
                    "trace": trace,
                },
            )
        ]

    def cleanup(self, context: dict[str, Any] | None = None) -> None:
        if not self.unload_ollama_after_attack:
            return

        providers = [self._ollama_paraphraser, self._ollama_judge]
        unloaded = set()
        for provider in providers:
            if provider is None:
                continue
            model = getattr(provider, "model", None)
            if not model or model in unloaded:
                continue
            try:
                provider.unload()
            except Exception as exc:
                console.print(
                    "[textfooler_style] "
                    f'could not unload Ollama model "{model}": {exc}',
                    markup=False,
                )
                continue
            unloaded.add(model)
            if self.log_ollama_unload:
                console.print(
                    "[textfooler_style] "
                    f'unloaded Ollama model "{model}"',
                    markup=False,
                )

    def _rank_units(
        self,
        prompt: str,
        target_concept: str | None,
        defense: Any,
        original_decision: DefenseDecision,
        context: dict[str, Any],
    ) -> list[tuple[str, float]]:
        units = self._candidate_units(prompt, target_concept, original_decision)
        ranked = []

        for unit in units:
            deleted_prompt = self._replace_unit(prompt, unit, "")
            deleted_decision = self._check(defense, deleted_prompt, target_concept, context)
            importance = self._importance_score(original_decision, deleted_decision, unit, target_concept)
            ranked.append((unit, importance))

        ranked.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
        return ranked

    def _candidate_units(
        self,
        prompt: str,
        target_concept: str | None,
        original_decision: DefenseDecision,
    ) -> list[str]:
        units = []
        matched_term = original_decision.metadata.get("matched_term")

        for unit in [target_concept, matched_term]:
            if unit and self._contains_unit(prompt, unit):
                units.append(unit)

        for source in self.DEFAULT_REPLACEMENTS:
            if self._contains_unit(prompt, source):
                units.append(source)

        for match in self.WORD_PATTERN.finditer(prompt):
            word = match.group(0)
            if len(word) <= 2 or word.lower() in self.stop_words or word.isdigit():
                continue
            units.append(word)

        unique_units = []
        seen = set()
        for unit in units:
            key = unit.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_units.append(unit)
        return unique_units

    def _importance_score(
        self,
        original_decision: DefenseDecision,
        deleted_decision: DefenseDecision,
        unit: str,
        target_concept: str | None,
    ) -> float:
        score = 0.0
        if not original_decision.allowed and deleted_decision.allowed:
            score += 100.0
        if original_decision.score is not None and deleted_decision.score is not None:
            score += max(0.0, original_decision.score - deleted_decision.score)
        if target_concept and unit.lower() == target_concept.lower():
            score += 10.0
        return score

    def _replacement_candidates(
        self,
        unit: str,
        context_prompt: str,
        target_concept: str | None,
    ) -> list[str]:
        candidates = []
        candidates.extend(self._generate_candidates(unit, context_prompt))

        fallback = self.DEFAULT_REPLACEMENTS.get(unit.lower())
        if fallback:
            candidates.append(fallback)

        unique_candidates = []
        seen = {unit.lower()}
        for candidate in candidates:
            candidate = self._normalize_prompt(str(candidate))
            if not candidate:
                self._log_candidate_filter(unit, candidate, "empty")
                continue
            if self._contains_unit(candidate, unit):
                self._log_candidate_filter(unit, candidate, "contains original unit")
                continue
            if self.filter_context_leaks and self._leaks_prompt_context(
                candidate,
                unit,
                context_prompt,
                target_concept,
            ):
                self._log_candidate_filter(unit, candidate, "leaks prompt context")
                continue
            key = candidate.lower()
            if key in seen:
                self._log_candidate_filter(unit, candidate, "duplicate")
                continue
            seen.add(key)
            unique_candidates.append(candidate)
            if len(unique_candidates) >= self.candidate_count:
                break
        return unique_candidates

    def _generate_candidates(self, unit: str, context_prompt: str) -> list[str]:
        if self.paraphraser is not None:
            candidates = self.paraphraser(unit, context_prompt, self.candidate_count)
            self._log_raw_paraphrases(unit, None, candidates)
            return candidates

        if not self.use_qwen_fallback:
            return []

        try:
            if self._ollama_paraphraser is None:
                kwargs = {"model": self.paraphraser_model} if self.paraphraser_model else {}
                self._ollama_paraphraser = QwenOllamaParaphraser(**kwargs)
            candidates = self._ollama_paraphraser.generate_candidates(
                unit,
                context=context_prompt,
                count=self.candidate_count,
            )
            self._log_raw_paraphrases(
                unit,
                self._ollama_paraphraser.last_raw_response,
                candidates,
            )
            return candidates
        except Exception:
            return []

    def _score_similarity(self, source: str, candidate: str) -> float:
        if self.similarity_scorer is not None:
            return float(self.similarity_scorer(source, candidate))

        if not self.use_clip_similarity:
            return 1.0

        try:
            if self._clip_similarity_scorer is None:
                self._clip_similarity_scorer = CLIPTextSimilarityScorer()
            return self._clip_similarity_scorer.score(source, candidate)
        except RuntimeError as exc:
            if not self._similarity_warning_printed:
                console.print(f"[textfooler_style] CLIP similarity disabled: {exc}")
                self._similarity_warning_printed = True
            return 1.0

    def _score_llm_judge(self, source: str, candidate: str, context_prompt: str) -> float:
        if self.llm_judge is not None:
            return float(self.llm_judge(source, candidate, context_prompt))

        try:
            if self._ollama_judge is None:
                kwargs = {"model": self.llm_judge_model} if self.llm_judge_model else {}
                self._ollama_judge = OllamaSimilarityJudge(**kwargs)
            return self._ollama_judge.score(source, candidate, context_prompt)
        except Exception as exc:
            if not self._judge_warning_printed:
                console.print(f"[textfooler_style] LLM judge fallback disabled: {exc}")
                self._judge_warning_printed = True
            return 0.0

    def _check(
        self,
        defense: Any,
        prompt: str,
        target_concept: str | None,
        context: dict[str, Any],
    ) -> DefenseDecision:
        return defense.check_prompt(prompt, target_concept=target_concept, context=context)

    def _is_better(self, candidate_decision: DefenseDecision, best_decision: DefenseDecision) -> bool:
        if candidate_decision.score is not None and best_decision.score is not None:
            return candidate_decision.score < best_decision.score
        if candidate_decision.score is not None:
            return True
        if best_decision.score is not None:
            return False
        return candidate_decision.allowed and not best_decision.allowed

    def _leaks_prompt_context(
        self,
        candidate: str,
        unit: str,
        context_prompt: str,
        target_concept: str | None,
    ) -> bool:
        if target_concept and target_concept.lower() != unit.lower():
            if self._contains_unit(candidate, target_concept):
                return True

        context_without_unit = self._replace_unit(context_prompt, unit, "")
        context_words = {
            match.group(0).lower()
            for match in self.WORD_PATTERN.finditer(context_without_unit)
        }
        leaked_action_terms = context_words & self.CONTEXT_LEAK_TERMS
        return any(self._contains_unit(candidate, term) for term in leaked_action_terms)

    def _log_candidate_filter(self, unit: str, candidate: str, reason: str) -> None:
        if not self.log_candidate_filtering:
            return
        console.print(
            "[textfooler_style] "
            f'filtered unit="{unit}" candidate="{candidate}" reason="{reason}"',
            markup=False,
        )

    def _log_raw_paraphrases(
        self,
        unit: str,
        raw_response: str | None,
        candidates: list[str],
    ) -> None:
        if not self.log_raw_paraphrases:
            return

        console.print(
            "[textfooler_style] "
            f'raw paraphrases for unit="{unit}":',
            markup=False,
        )
        if raw_response is not None:
            console.print(raw_response, markup=False)
        console.print(
            "[textfooler_style] "
            f'parsed candidates for unit="{unit}": {candidates}',
            markup=False,
        )

    def _contains_unit(self, prompt: str, unit: str) -> bool:
        return self._unit_pattern(unit).search(prompt) is not None

    def _replace_unit(self, prompt: str, unit: str, replacement: str) -> str:
        rewritten = self._unit_pattern(unit).sub(replacement, prompt, count=1)
        return self._normalize_prompt(rewritten)

    def _unit_pattern(self, unit: str) -> re.Pattern[str]:
        return re.compile(rf"(?<!\w){re.escape(unit)}(?!\w)", re.IGNORECASE)

    def _normalize_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"\s+([,.;:!?])", r"\1", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        return prompt.strip()

    def _apply_context_config(self, context: dict[str, Any]) -> None:
        attack_config = dict((context.get("config") or {}).get("attack", {}))
        attack_config.pop("name", None)

        for key in [
            "candidate_count",
            "max_rounds",
            "min_similarity",
            "use_clip_similarity",
            "max_candidate_batches",
            "log_similarity",
            "use_qwen_fallback",
            "use_llm_judge_fallback",
            "llm_judge_threshold",
            "llm_judge_model",
            "log_llm_judge",
            "filter_context_leaks",
            "log_candidate_filtering",
            "log_raw_paraphrases",
            "paraphraser_model",
            "unload_ollama_after_attack",
            "log_ollama_unload",
        ]:
            if key in attack_config:
                setattr(self, key, attack_config[key])
