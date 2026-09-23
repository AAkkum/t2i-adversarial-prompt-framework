import json
from pathlib import Path

from t2i_framework.attacks import groot_client
from t2i_framework.attacks.groot_client import LocalMultimodalClient, parse_json_object


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

    monkeypatch.setattr(groot_client.request, "urlopen", fake_urlopen)
    client = LocalMultimodalClient(
        provider="openai_compatible",
        model="local-qwen",
        base_url="http://127.0.0.1:8080/v1",
    )

    response = client.complete("system", "review", image_path=image_path)

    assert response == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    user_content = captured["payload"]["messages"][1]["content"]
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["payload"]["temperature"] == 0.0


def test_ollama_client_sends_image_bytes(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Response({"message": {"content": '{"ok": true}'}})

    monkeypatch.setattr(groot_client.request, "urlopen", fake_urlopen)
    client = LocalMultimodalClient(
        provider="ollama",
        model="local-qwen",
        base_url="http://127.0.0.1:11434",
    )

    response = client.complete("system", "review", image_path=image_path)

    assert response == '{"ok": true}'
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["payload"]["messages"][1]["images"]
