import threading
import time
from pathlib import Path

from t2i_framework.attacks.daca import (
    DACAAttack,
    OFFICIAL_RELEASE,
    ONTOLOGY_EDGES,
    ONTOLOGY_NODES,
)


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str, image_path=None, **kwargs) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_path": image_path,
                "options": kwargs,
            }
        )
        return self.responses.pop(0)


def _stage_responses() -> list[str]:
    # Six decomposers, four released specialist polishers, six edge assemblers,
    # one isolated-background assembler, and one final concatenation pass.
    return [
        "characters",
        "polished characters",
        "background",
        "belongings",
        "polished belongings",
        "clothing",
        "actions",
        "polished actions",
        "details",
        "polished details",
        "edge sentence 1",
        "edge sentence 2",
        "edge sentence 3",
        "edge sentence 4",
        "edge sentence 5",
        "edge sentence 6",
        "background sentence",
        "Final prompt: assembled adversarial image prompt",
    ]


def _official_release_responses() -> list[str]:
    return _stage_responses()[:16] + ["official final prompt"]


def test_daca_runs_paper_ontology_workflow() -> None:
    client = _FakeClient(_stage_responses())
    attack = DACAAttack(client=client)

    candidate = attack.generate(
        "an original scene",
        target_concept="protected concept",
        context={"config": {"attack": {"name": "daca"}}},
    )[0]

    assert candidate.text == "assembled adversarial image prompt"
    assert candidate.metadata["llm_call_count"] == 18
    assert candidate.metadata["ontology_nodes"] == list(ONTOLOGY_NODES)
    assert candidate.metadata["ontology_edges"] == [list(edge) for edge in ONTOLOGY_EDGES]
    assert candidate.metadata["target_concept_used_by_attack"] is False
    assert len(candidate.metadata["stage_timings_seconds"]) == 18
    assert candidate.metadata["attack_llm_runtime_seconds"] >= 0
    assert len(client.calls) == 18
    assert all(call["image_path"] is None for call in client.calls)


def test_daca_assembler_receives_decomposition_and_polish_outputs() -> None:
    client = _FakeClient(_stage_responses())
    attack = DACAAttack(client=client)

    attack.generate("an original scene", context={})

    first_assembler_prompt = str(client.calls[10]["user_prompt"])
    assert "belongings" in first_assembler_prompt
    assert "polished belongings" in first_assembler_prompt
    assert "characters" in first_assembler_prompt
    assert "polished characters" in first_assembler_prompt
    finalizer_prompt = str(client.calls[-1]["user_prompt"])
    assert "edge sentence 1" in finalizer_prompt
    assert "edge sentence 6" in finalizer_prompt
    assert "background sentence" in finalizer_prompt


def test_daca_uses_shared_client_options_and_configured_token_limits() -> None:
    client = _FakeClient(_stage_responses())
    attack = DACAAttack(client=client)

    attack.generate(
        "an original scene",
        context={
            "config": {
                "attack": {
                    "name": "daca",
                    "reasoning_effort": "none",
                    "max_tokens": {
                        "decomposer": 101,
                        "polisher": 202,
                        "assembler": 303,
                        "finalizer": 404,
                    },
                    "logging": {"progress": True, "outputs": True},
                }
            }
        },
    )

    assert client.calls[0]["options"] == {"max_tokens": 101, "reasoning_effort": "none"}
    assert client.calls[1]["options"] == {"max_tokens": 202, "reasoning_effort": "none"}
    assert client.calls[10]["options"] == {"max_tokens": 303, "reasoning_effort": "none"}
    assert client.calls[-1]["options"] == {"max_tokens": 404, "reasoning_effort": "none"}
    assert attack.log_progress is True
    assert attack.log_outputs is True


def test_daca_uses_its_dedicated_llm_server_config() -> None:
    attack = DACAAttack()

    attack._apply_context_config(
        {
            "config": {
                "attack": {"name": "daca"},
                "local_llm": {
                    "model": "evaluator-model",
                    "alias": "evaluator",
                    "port": 8083,
                },
                "daca_llm": {
                    "model": "attack-model",
                    "alias": "daca-qwen",
                    "port": 8084,
                },
            }
        }
    )

    assert attack.backend_model == "attack-model"
    assert attack.client.model == "daca-qwen"
    assert attack.client.base_url == "http://127.0.0.1:8084/v1"


def test_daca_cache_key_does_not_depend_on_worker_port() -> None:
    base_config = {
        "attack": {"name": "daca"},
        "daca_llm": {"model": "attack-model", "alias": "daca-qwen"},
    }
    first = DACAAttack()
    first._apply_context_config(
        {"config": {**base_config, "daca_llm": {**base_config["daca_llm"], "port": 8084}}}
    )
    second = DACAAttack()
    second._apply_context_config(
        {"config": {**base_config, "daca_llm": {**base_config["daca_llm"], "port": 8087}}}
    )

    assert first._cache_path("prompt", "target") == second._cache_path(
        "prompt", "target"
    )


def test_daca_uses_the_authors_released_prompt_assets() -> None:
    client = _FakeClient(_stage_responses())
    attack = DACAAttack(client=client)

    attack.generate("an original scene", context={})

    decomposer_prompt = str(client.calls[0]["user_prompt"])
    assert "Floyd was killed by the violent law enforcement" in decomposer_prompt
    assert decomposer_prompt.endswith("an original scene")


def test_official_release_reproduces_overwrite_and_raw_background_behavior() -> None:
    client = _FakeClient(_official_release_responses())
    attack = DACAAttack(client=client)

    candidate = attack.generate(
        "an original scene",
        context={
            "config": {
                "attack": {
                    "name": "daca",
                    "implementation_mode": OFFICIAL_RELEASE,
                    "temperature": 1.0,
                    "reasoning_effort": None,
                    "use_system_prompt": False,
                    "clean_outputs": False,
                }
            }
        },
    )[0]

    assert candidate.text == "official final prompt"
    assert candidate.metadata["implementation_mode"] == OFFICIAL_RELEASE
    assert candidate.metadata["llm_call_count"] == 17
    assert candidate.metadata["assembler_outputs"] == {
        "character": "edge sentence 5",
        "belongings": "edge sentence 6",
        "background": "background",
    }
    assert len(client.calls) == 17
    assert client.calls[0]["system_prompt"] == ""
    assert client.calls[0]["options"] == {
        "max_tokens": 512,
        "temperature": 1.0,
    }
    finalizer_prompt = str(client.calls[-1]["user_prompt"])
    assert "edge sentence 5" in finalizer_prompt
    assert "edge sentence 6" in finalizer_prompt
    assert "edge sentence 1" not in finalizer_prompt


def test_candidate_count_is_capped_by_runner_max_candidates() -> None:
    client = _FakeClient(_stage_responses() * 2)
    attack = DACAAttack(client=client)

    candidates = attack.generate(
        "an original scene",
        context={
            "max_candidates": 2,
            "config": {"attack": {"name": "daca", "candidate_count": 10}},
        },
    )

    assert len(candidates) == 2
    assert [candidate.metadata["candidate_index"] for candidate in candidates] == [0, 1]
    assert len(client.calls) == 36


def test_daca_reuses_persistent_candidates_across_attack_instances(tmp_path: Path) -> None:
    config = {
        "attack": {
            "name": "daca",
            "candidate_count": 2,
            "candidate_cache": {
                "enabled": True,
                "directory": str(tmp_path / "daca-cache"),
            },
        }
    }
    first_client = _FakeClient(_stage_responses() * 2)
    first = DACAAttack(client=first_client).generate(
        "an original scene", context={"max_candidates": 2, "config": config}
    )

    second_client = _FakeClient([])
    second = DACAAttack(client=second_client).generate(
        "an original scene", context={"max_candidates": 2, "config": config}
    )

    assert [candidate.text for candidate in second] == [candidate.text for candidate in first]
    assert len(first_client.calls) == 36
    assert second_client.calls == []
    assert all(candidate.metadata["candidate_cache_hit"] for candidate in second)


class _ConcurrentClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str, image_path=None, **kwargs) -> str:
        with self._lock:
            self.active += 1
            self.call_count += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.005)
        with self._lock:
            self.active -= 1
        return "candidate component"


def test_daca_parallelizes_independent_candidate_pipelines() -> None:
    client = _ConcurrentClient()
    candidates = DACAAttack(client=client).generate(
        "an original scene",
        context={
            "max_candidates": 2,
            "config": {
                "attack": {
                    "name": "daca",
                    "candidate_count": 2,
                    "parallel": {"workers": 2},
                }
            },
        },
    )

    assert len(candidates) == 2
    assert client.call_count == 36
    assert client.max_active == 2
    assert all(
        len(candidate.metadata["stage_timings_seconds"]) == 18
        for candidate in candidates
    )
