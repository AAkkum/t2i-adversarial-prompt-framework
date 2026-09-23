from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from t2i_framework.core.config import load_yaml_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "local_llm.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the configured local llama.cpp server.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    settings = dict(config.get("local_llm", {}))
    model = str(settings.get("model", "")).strip()
    if not model:
        raise SystemExit(f"Set local_llm.model in {args.config}.")

    executable, command = _server_command()
    command.extend(
        [
            "--host",
            str(settings.get("host", "127.0.0.1")),
            "--port",
            str(settings.get("port", 8082)),
            "--alias",
            str(settings.get("alias", "local-llm")),
            "--ctx-size",
            str(settings.get("context_size", 8192)),
            "--n-gpu-layers",
            str(settings.get("gpu_layers", "auto")),
            "--jinja",
        ]
    )
    command.extend(_model_argument(model))

    mmproj = str(settings.get("mmproj", "")).strip()
    if mmproj:
        command.extend(["--mmproj", str(_existing_path(mmproj, "Multimodal projector"))])

    host = settings.get("host", "127.0.0.1")
    port = settings.get("port", 8082)
    print(f"Starting local LLM at http://{host}:{port}/v1")
    os.execv(executable, command)


def _server_command() -> tuple[str, list[str]]:
    llama_server = shutil.which("llama-server")
    if llama_server:
        return llama_server, [llama_server]

    llama = shutil.which("llama")
    if llama:
        return llama, [llama, "serve"]

    raise SystemExit("llama-server (or llama) is not installed or is not on PATH.")


def _model_argument(model: str) -> list[str]:
    local_path = _resolve_path(model)
    if local_path.is_file():
        return ["-m", str(local_path)]
    if model.startswith(("/", "./", "../", "~")) or model.lower().endswith(".gguf"):
        raise SystemExit(f"Local model file not found: {local_path}")
    return ["-hf", model]


def _existing_path(value: str, label: str) -> Path:
    path = _resolve_path(value)
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
