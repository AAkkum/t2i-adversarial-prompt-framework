import json

from t2i_framework.attacks.pgj import _LLMBackend


def test_pgj_cache_hit_does_not_load_transformers_model(tmp_path, monkeypatch) -> None:
    system_prompt = "system"
    user_prompt = "user"
    cache_path = tmp_path / "pgj.json"
    cache_path.write_text(
        json.dumps({f"{system_prompt}|||{user_prompt}": "cached candidate"}),
        encoding="utf-8",
    )
    backend = _LLMBackend("unused/model", device="cuda:0", cache_path=str(cache_path))

    def fail_if_loaded() -> None:
        raise AssertionError("A cache hit must not load the PGJ model.")

    monkeypatch.setattr(backend, "_load_model", fail_if_loaded)

    assert backend.generate(system_prompt, user_prompt) == "cached candidate"
    assert backend.model is None
