from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import error, request


class LocalMultimodalClient:
    """HTTP client for local Ollama or OpenAI-compatible model servers."""

    def __init__(
        self,
        provider: str = "openai_compatible",
        model: str = "local-llm",
        base_url: str = "http://127.0.0.1:8082/v1",
        api_key: str | None = None,
        timeout_seconds: int = 300,
        temperature: float = 0.0,
        unload_after_request: bool = False,
    ) -> None:
        normalized_provider = provider.strip().lower().replace("-", "_")
        if normalized_provider not in {"openai_compatible", "ollama"}:
            raise ValueError("Local model provider must be 'openai_compatible' or 'ollama'.")
        self.provider = normalized_provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.unload_after_request = unload_after_request

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None = None,
        *,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> str:
        if self.provider == "ollama":
            result = self._complete_ollama(
                system_prompt,
                user_prompt,
                image_path,
                max_tokens,
                reasoning_effort,
                temperature,
            )
            if self.unload_after_request:
                self.unload()
            return result
        return self._complete_openai_compatible(
            system_prompt,
            user_prompt,
            image_path,
            max_tokens,
            reasoning_effort,
            temperature,
        )

    def unload(self) -> None:
        if self.provider != "ollama":
            return
        payload = {
            "model": self.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        generate_url = self.base_url
        if generate_url.endswith("/api/chat"):
            generate_url = generate_url[: -len("/api/chat")] + "/api/generate"
        elif not generate_url.endswith("/api/generate"):
            generate_url += "/api/generate"
        self._post_json(generate_url, payload)

    def _complete_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        temperature: float | None,
    ) -> str:
        user_content: str | list[dict[str, Any]] = user_prompt
        if image_path is not None:
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                },
            ]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
            if reasoning_effort == "none":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        response = self._post_json(endpoint, payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Local chat server returned an unexpected response.") from exc
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content)

    def _complete_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        temperature: float | None,
    ) -> str:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if image_path is not None:
            user_message["images"] = [base64.b64encode(image_path.read_bytes()).decode("ascii")]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(user_message)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature if temperature is None else float(temperature)
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = int(max_tokens)
        if reasoning_effort == "none":
            payload["think"] = False
        endpoint = self.base_url
        if not endpoint.endswith("/api/chat"):
            endpoint += "/api/chat"
        response = self._post_json(endpoint, payload)
        try:
            return str(response["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama returned an unexpected response.") from exc

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = " ".join(body.split())[:1000]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"Could not call local model server at {url}: "
                f"HTTP {exc.code} {exc.reason}{suffix}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - preserve local-server context.
            raise RuntimeError(f"Could not call local model server at {url}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Local model server returned a non-object JSON response.")
        return parsed


def client_options(
    config: dict[str, Any], section: str = "local_llm"
) -> dict[str, Any]:
    """Build client options from a named local model server configuration."""

    settings = dict(config.get(section, {}))
    host = str(settings.get("host", "127.0.0.1"))
    port = int(settings.get("port", 8082))
    return {
        "provider": str(settings.get("provider", "openai_compatible")),
        "model": str(settings.get("alias", "local-llm")),
        "base_url": str(settings.get("base_url", f"http://{host}:{port}/v1")),
        "api_key": settings.get("api_key"),
        "timeout_seconds": int(settings.get("timeout_seconds", 600)),
        "temperature": float(settings.get("temperature", 0.0)),
        "unload_after_request": bool(settings.get("unload_after_request", False)),
    }


def parse_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip().strip("`")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("Local model did not return a JSON object.")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Local model returned malformed JSON.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Local model response must be a JSON object.")
    return parsed


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
