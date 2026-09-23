import json
from pathlib import Path

from t2i_framework.core import local_model_client
from t2i_framework.core.local_model_client import (
    LocalMultimodalClient,
    client_options,
    parse_json_object,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_parse_json_object_accepts_fenced_model_output() -> None:
    assert parse_json_object('```json\n{"value": 1}\n```') == {"value": 1}


def test_client_options_use_shared_server_config() -> None:
    options = client_options(
        {"local_llm": {"alias": "test-model", "host": "localhost", "port": 9000}}
    )

    assert options["model"] == "test-model"
    assert options["base_url"] == "http://localhost:9000/v1"


def test_openai_compatible_client_sends_multimodal_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(local_model_client.request, "urlopen", fake_urlopen)
    client = LocalMultimodalClient(
        provider="openai_compatible",
        model="local-qwen",
        base_url="http://127.0.0.1:8080/v1",
    )

    response = client.complete(
        "system",
        "review",
        image_path=image_path,
        max_tokens=128,
        reasoning_effort="none",
    )

    assert response == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    user_content = captured["payload"]["messages"][1]["content"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["max_tokens"] == 128
    assert captured["payload"]["reasoning_effort"] == "none"
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_ollama_client_sends_image_bytes(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response({"message": {"content": '{"ok": true}'}})

    monkeypatch.setattr(local_model_client.request, "urlopen", fake_urlopen)
    client = LocalMultimodalClient(
        provider="ollama",
        model="local-qwen",
        base_url="http://127.0.0.1:11434",
    )

    response = client.complete(
        "system",
        "review",
        image_path=image_path,
        max_tokens=128,
        reasoning_effort="none",
    )

    assert response == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["messages"][1]["images"]
    assert captured["payload"]["options"]["num_predict"] == 128
    assert captured["payload"]["think"] is False
