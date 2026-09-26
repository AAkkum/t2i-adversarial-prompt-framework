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
    parser.add_argument("--section", default="local_llm")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    settings = dict(config.get(args.section, {}))
    model = str(settings.get("model", "")).strip()
    if not model:
        raise SystemExit(f"Set {args.section}.model in {args.config}.")
    host = args.host or str(settings.get("host", "127.0.0.1"))
    port = args.port if args.port is not None else int(settings.get("port", 8082))

    executable, command = _server_command()
    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            str(settings.get("alias", "local-llm")),
            "--ctx-size",
            str(settings.get("context_size", 8192)),
            "--n-gpu-layers",
            str(settings.get("gpu_layers", "auto")),
            "--parallel",
            str(settings.get("parallel_slots", 4)),
            "--jinja",
        ]
    )
    command.extend(_model_argument(model))

    mmproj = str(settings.get("mmproj", "")).strip()
    if mmproj:
        command.extend(["--mmproj", str(_existing_path(mmproj, "Multimodal projector"))])

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
