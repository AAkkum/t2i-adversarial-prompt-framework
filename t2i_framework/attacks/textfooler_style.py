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
    """Adaptive TextFooler-style prompt rewriting attack.

    The attack is inspired by TextFooler's word-importance ranking and
    replacement loop, but adapts it to text-to-image prompt defenses. It first
    checks whether the original prompt is blocked, ranks important prompt units
    through deletion tests, asks a paraphraser for replacements, keeps only
    similarity-preserving candidates, and queries the defense until a candidate
    passes or the configured search budget is exhausted.
    """

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
        """Create a configurable TextFooler-style attack instance.

        Args:
            paraphraser: Optional injected paraphraser for tests or alternate
                backends. It receives the selected unit, full prompt context,
                and requested candidate count.
            candidate_count: Maximum number of replacement candidates to keep
                per paraphrase batch.
            max_rounds: Maximum number of times the attack may re-rank and
                continue from the best blocked rewrite.
            stop_words: Words ignored when building deletion-test units.
            use_qwen_fallback: Whether to use the Ollama Qwen paraphraser when
                no injected paraphraser is provided.
            min_similarity: Minimum candidate similarity required before a
                candidate is checked against the defense.
            similarity_scorer: Optional injected text similarity scorer.
            use_clip_similarity: Whether to use CLIP text similarity when no
                injected scorer is provided.
            max_candidate_batches: Number of paraphrase batches to request for
                each selected unit.
            log_similarity: Whether to print candidate similarity scores.
            use_llm_judge_fallback: Whether to ask an LLM judge when CLIP
                rejects every candidate in a batch.
            llm_judge_threshold: Minimum LLM judge score required to recover a
                CLIP-rejected candidate.
            llm_judge_model: Optional Ollama model name for the judge.
            llm_judge: Optional injected judge for tests or alternate backends.
            log_llm_judge: Whether to print judge scores.
            filter_context_leaks: Whether to discard replacements that leak
                target/action words from the prompt context.
            log_candidate_filtering: Whether to print why raw candidates are
                filtered out.
            log_raw_paraphrases: Whether to print raw and parsed paraphraser
                outputs.
            paraphraser_model: Optional Ollama model name for paraphrasing.
            unload_ollama_after_attack: Whether cleanup should ask Ollama to
                unload paraphraser/judge models after candidate generation.
            log_ollama_unload: Whether to print successful unload messages.
        """
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
        """Generate one adversarial prompt candidate for the runner.

        The method returns early if there is no defense object or if the
        original prompt already passes. Otherwise it performs the adaptive
        attack loop: rank units, try paraphrase replacements, accept a passing
        candidate, or continue from the best blocked candidate.

        Args:
            prompt: Original user prompt.
            target_concept: Optional restricted concept that should be attacked
                first when it appears in the prompt.
            context: Runner context. The attack expects `defense`, `config`,
                and optional bookkeeping values such as `seed`.

        Returns:
            A single-item list containing either the successful attacked prompt
            or the best prompt reached before the search budget ended.
        """
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

                unit_result = self._try_unit(
                    unit,
                    current_prompt,
                    target_concept,
                    defense,
                    context,
                    trace,
                )
                query_count += unit_result["query_count"]
                judge_query_count += unit_result["judge_query_count"]
                best_allowed = unit_result["best_allowed"]

                if best_allowed is not None:
                    final_decision = best_allowed["decision"]
                    return [
                        self._make_success_candidate(
                            best_allowed,
                            query_count,
                            judge_query_count,
                            original_decision,
                            final_decision,
                            trace,
                        )
                    ]

                if (
                    unit_result["best_blocked_prompt"] is not None
                    and unit_result["best_blocked_decision"] is not None
                ):
                    current_prompt = unit_result["best_blocked_prompt"]
                    current_decision = unit_result["best_blocked_decision"]
                    changed = True

            if not changed:
                break
            if current_decision.allowed:
                break

        return [
            self._make_failure_candidate(
                current_prompt,
                query_count,
                judge_query_count,
                original_decision,
                current_decision,
                trace,
            )
        ]

    def _try_unit(
        self,
        unit: str,
        current_prompt: str,
        target_concept: str | None,
        defense: Any,
        context: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Try to replace one ranked prompt unit.

        The method may request multiple paraphrase batches for the same unit.
        It returns the best passing candidate if one exists; otherwise it
        returns the best blocked candidate so the outer attack loop can continue
        from the strongest partial rewrite.

        Args:
            unit: Word or phrase selected by deletion-based ranking.
            current_prompt: Prompt state before replacing this unit.
            target_concept: Restricted concept passed by the runner.
            defense: Defense object used for prompt checks.
            context: Runner context passed through to the defense.
            trace: Shared trace list updated with every candidate decision.

        Returns:
            A dictionary containing `best_allowed`, `best_blocked_prompt`,
            `best_blocked_decision`, and query counters.
        """
        best_blocked_prompt = None
        best_blocked_decision = None
        best_allowed: dict[str, Any] | None = None
        query_count = 0
        judge_query_count = 0

        for batch_index in range(self.max_candidate_batches):
            candidate_records = self._build_candidate_records(
                unit,
                current_prompt,
                target_concept,
                batch_index,
            )
            if candidate_records is None:
                break

            judge_query_count += self._apply_llm_judge_fallback(
                unit,
                current_prompt,
                candidate_records,
            )
            result = self._evaluate_candidate_records(
                candidate_records,
                unit,
                target_concept,
                defense,
                context,
                trace,
            )
            query_count += result["query_count"]

            if result["best_allowed"] is not None:
                if (
                    best_allowed is None
                    or result["best_allowed"]["similarity"] > best_allowed["similarity"]
                ):
                    best_allowed = result["best_allowed"]

            if result["best_blocked_decision"] is not None and (
                best_blocked_decision is None
                or self._is_better(result["best_blocked_decision"], best_blocked_decision)
            ):
                best_blocked_prompt = result["best_blocked_prompt"]
                best_blocked_decision = result["best_blocked_decision"]

            if best_allowed is not None:
                break

        return {
            "best_allowed": best_allowed,
            "best_blocked_prompt": best_blocked_prompt,
            "best_blocked_decision": best_blocked_decision,
            "query_count": query_count,
            "judge_query_count": judge_query_count,
        }

    def _build_candidate_records(
        self,
        unit: str,
        current_prompt: str,
        target_concept: str | None,
        batch_index: int,
    ) -> list[dict[str, Any]] | None:
        """Create scored candidate records for one paraphrase batch.

        Raw paraphrases are converted into full prompt rewrites and scored with
        the configured similarity scorer. Records that are too dissimilar are
        kept in the trace later, but they are not sent to the defense unless a
        judge fallback accepts them.

        Args:
            unit: Word or phrase being replaced.
            current_prompt: Prompt state before replacement.
            target_concept: Restricted concept used by optional leak filtering.
            batch_index: Zero-based paraphrase batch number for trace metadata.

        Returns:
            A list of candidate records, or `None` when no replacements were
            available for this unit.
        """
        replacements = self._replacement_candidates(unit, current_prompt, target_concept)
        if not replacements:
            return None

        records = []
        for replacement in replacements:
            candidate_prompt = self._replace_unit(current_prompt, unit, replacement)
            if candidate_prompt == current_prompt:
                continue

            similarity = self._score_similarity(unit, replacement)
            similarity_allowed = similarity >= self.min_similarity
            self._log_similarity(unit, replacement, similarity)

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
            records.append(
                {
                    "replacement": replacement,
                    "candidate_prompt": candidate_prompt,
                    "trace_item": trace_item,
                    "similarity": similarity,
                    "similarity_allowed": similarity_allowed,
                    "similarity_method": "clip",
                }
            )
        return records

    def _apply_llm_judge_fallback(
        self,
        unit: str,
        current_prompt: str,
        candidate_records: list[dict[str, Any]],
    ) -> int:
        """Recover CLIP-rejected candidates with an optional LLM judge.

        The judge runs only when enabled and every candidate in the batch failed
        the normal similarity threshold. Accepted judge scores replace the
        candidate similarity value used for final selection.

        Args:
            unit: Word or phrase being replaced.
            current_prompt: Prompt context used by the judge.
            candidate_records: Mutable candidate records for the current batch.

        Returns:
            Number of judge queries performed.
        """
        if (
            not candidate_records
            or not self.use_llm_judge_fallback
            or any(record["similarity_allowed"] for record in candidate_records)
        ):
            return 0

        if self.log_llm_judge:
            console.print(
                "[textfooler_style] "
                f'CLIP rejected all candidates for "{unit}"; '
                "using LLM judge fallback",
                markup=False,
            )

        judge_query_count = 0
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
            self._log_llm_judge(unit, record["replacement"], judge_score)
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
        return judge_query_count

    def _evaluate_candidate_records(
        self,
        candidate_records: list[dict[str, Any]],
        unit: str,
        target_concept: str | None,
        defense: Any,
        context: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check similarity-approved candidates against the defense.

        Passing candidates are ranked by their active similarity score. Blocked
        candidates are ranked by `_is_better`, which uses defense scores when
        available and falls back to binary behavior for black-box defenses.

        Args:
            candidate_records: Candidate rewrites and trace metadata.
            unit: Word or phrase being replaced.
            target_concept: Restricted concept passed to the defense.
            defense: Defense object used for prompt checks.
            context: Runner context passed through to the defense.
            trace: Shared trace list updated with candidate outcomes.

        Returns:
            A dictionary containing the best allowed candidate, best blocked
            candidate, and the number of defense queries performed.
        """
        best_allowed: dict[str, Any] | None = None
        best_blocked_prompt = None
        best_blocked_decision = None
        query_count = 0

        for record in candidate_records:
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
                record["candidate_prompt"],
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
                    "prompt": record["candidate_prompt"],
                    "decision": candidate_decision,
                    "replacement": record["replacement"],
                    "similarity": record["similarity"],
                    "similarity_method": record["similarity_method"],
                    "clip_similarity": trace_item.get("clip_similarity"),
                    "judge_similarity": trace_item.get("judge_similarity"),
                    "unit": unit,
                }
                if (
                    best_allowed is None
                    or record["similarity"] > best_allowed["similarity"]
                ):
                    best_allowed = allowed_candidate
                continue

            if best_blocked_decision is None or self._is_better(
                candidate_decision,
                best_blocked_decision,
            ):
                best_blocked_prompt = record["candidate_prompt"]
                best_blocked_decision = candidate_decision

        return {
            "best_allowed": best_allowed,
            "best_blocked_prompt": best_blocked_prompt,
            "best_blocked_decision": best_blocked_decision,
            "query_count": query_count,
        }

    def _make_success_candidate(
        self,
        best_allowed: dict[str, Any],
        query_count: int,
        judge_query_count: int,
        original_decision: DefenseDecision,
        final_decision: DefenseDecision,
        trace: list[dict[str, Any]],
    ) -> AttackCandidate:
        """Build the successful attack candidate returned to the runner.

        Args:
            best_allowed: Best passing candidate record selected by the attack.
            query_count: Number of prompt-defense queries performed.
            judge_query_count: Number of LLM judge queries performed.
            original_decision: Defense decision for the original prompt.
            final_decision: Defense decision for the selected prompt.
            trace: Full attack trace for debugging and result analysis.

        Returns:
            AttackCandidate with the selected prompt and detailed metadata.
        """
        return AttackCandidate(
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

    def _make_failure_candidate(
        self,
        current_prompt: str,
        query_count: int,
        judge_query_count: int,
        original_decision: DefenseDecision,
        current_decision: DefenseDecision,
        trace: list[dict[str, Any]],
    ) -> AttackCandidate:
        """Build the fallback candidate when no prompt bypasses the defense.

        Args:
            current_prompt: Last prompt state reached by the search.
            query_count: Number of prompt-defense queries performed.
            judge_query_count: Number of LLM judge queries performed.
            original_decision: Defense decision for the original prompt.
            current_decision: Defense decision for the last prompt state.
            trace: Full attack trace for debugging and result analysis.

        Returns:
            AttackCandidate marked with `status="no_passing_candidate"`.
        """
        return AttackCandidate(
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

    def cleanup(self, context: dict[str, Any] | None = None) -> None:
        """Release optional Ollama paraphraser and judge models after the run.

        Args:
            context: Unused runner context, accepted to match the attack
                cleanup interface.
        """
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
        """Rank prompt units by deletion-test importance.

        Args:
            prompt: Current prompt state.
            target_concept: Optional restricted concept to prioritize.
            defense: Defense queried after deleting each candidate unit.
            original_decision: Defense decision for the current prompt.
            context: Runner context passed through to the defense.

        Returns:
            `(unit, importance)` pairs sorted from most to least important.
        """
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
        """Collect target, matched terms, known phrases, and content words.

        Args:
            prompt: Current prompt state.
            target_concept: Optional restricted concept from the runner.
            original_decision: Defense decision that may expose a matched term.

        Returns:
            Ordered unique units that can be deletion-tested and replaced.
        """
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
        """Compute how important a unit looked during deletion testing.

        The score strongly rewards deletions that make a blocked prompt pass.
        When the defense exposes numeric scores, lower deleted-prompt scores
        also increase importance. The explicit target receives a small bonus.

        Args:
            original_decision: Defense decision before deleting the unit.
            deleted_decision: Defense decision after deleting the unit.
            unit: Deleted word or phrase.
            target_concept: Optional restricted concept from the runner.

        Returns:
            Higher score means the unit should be tried earlier.
        """
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
        """Generate, normalize, deduplicate, and filter replacements.

        Args:
            unit: Word or phrase being replaced.
            context_prompt: Prompt context passed to the paraphraser.
            target_concept: Restricted concept used by optional leak filtering.

        Returns:
            Up to `candidate_count` candidate replacement strings.
        """
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
        """Request raw replacement candidates from an injected or Ollama paraphraser.

        Args:
            unit: Word or phrase being replaced.
            context_prompt: Full prompt context sent to the paraphraser.

        Returns:
            Raw candidate strings parsed by the paraphraser backend. Returns an
            empty list if no backend is enabled or the backend fails.
        """
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
        """Score whether a replacement preserves the selected unit's meaning.

        Args:
            source: Original word or phrase.
            candidate: Candidate replacement phrase.

        Returns:
            Similarity score in the range expected by the configured scorer. If
            CLIP is disabled or unavailable, returns `1.0` so candidates are not
            filtered by this stage.
        """
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
        """Ask the optional LLM judge for a replacement-preservation score.

        Args:
            source: Original word or phrase.
            candidate: Candidate replacement phrase.
            context_prompt: Full prompt context for judging.

        Returns:
            Judge score as a float, or `0.0` if the judge backend fails.
        """
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
        """Run the prompt-stage defense with the framework's standard signature."""
        return defense.check_prompt(prompt, target_concept=target_concept, context=context)

    def _is_better(self, candidate_decision: DefenseDecision, best_decision: DefenseDecision) -> bool:
        """Return whether a blocked candidate is a better partial rewrite.

        Score-returning defenses use lower scores as better because they are
        closer to passing. If scores are hidden, only an allowed candidate can
        beat a blocked one, so black-box blocked candidates keep their order.
        """
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
        """Detect optional candidate leakage from the surrounding prompt.

        Args:
            candidate: Replacement phrase returned by the paraphraser.
            unit: Word or phrase being replaced.
            context_prompt: Prompt before replacing the unit.
            target_concept: Restricted concept from the runner.

        Returns:
            True when the replacement reintroduces the target or known action
            context terms that should stay outside the replacement phrase.
        """
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

    def _log_similarity(self, unit: str, replacement: str, similarity: float) -> None:
        """Print the configured similarity score for a replacement candidate."""
        if not self.log_similarity:
            return
        console.print(
            "[textfooler_style] "
            f'unit="{unit}" candidate="{replacement}" '
            f"clip_similarity={similarity:.3f} "
            f"threshold={self.min_similarity:.3f}",
            markup=False,
        )

    def _log_llm_judge(self, unit: str, replacement: str, judge_score: float) -> None:
        """Print the LLM judge score for a replacement candidate."""
        if not self.log_llm_judge:
            return
        console.print(
            "[textfooler_style] "
            f'llm_judge unit="{unit}" '
            f'candidate="{replacement}" '
            f"score={judge_score:.3f} "
            f"threshold={self.llm_judge_threshold:.3f}",
            markup=False,
        )

    def _log_candidate_filter(self, unit: str, candidate: str, reason: str) -> None:
        """Print why a raw replacement candidate was discarded."""
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
        """Print raw paraphraser output and the parsed candidate list."""
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
        """Return whether a prompt contains a whole-word/phrase unit."""
        return self._unit_pattern(unit).search(prompt) is not None

    def _replace_unit(self, prompt: str, unit: str, replacement: str) -> str:
        """Replace the first whole-word/phrase occurrence of a unit."""
        rewritten = self._unit_pattern(unit).sub(replacement, prompt, count=1)
        return self._normalize_prompt(rewritten)

    def _unit_pattern(self, unit: str) -> re.Pattern[str]:
        """Build a case-insensitive regex for whole-unit matching."""
        return re.compile(rf"(?<!\w){re.escape(unit)}(?!\w)", re.IGNORECASE)

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize spacing around punctuation and repeated whitespace."""
        prompt = re.sub(r"\s+([,.;:!?])", r"\1", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        return prompt.strip()

    def _apply_context_config(self, context: dict[str, Any]) -> None:
        """Apply attack options from the merged runner config."""
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
