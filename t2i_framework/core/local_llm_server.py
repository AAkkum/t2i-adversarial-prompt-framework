from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
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
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit llama.cpp offload device, for example CUDA1.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Physical GPU index; resolves CUDA or Vulkan via --list-devices.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("none", "layer", "row"),
        default=None,
    )
    parser.add_argument("--main-gpu", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    settings = dict(config.get(args.section, {}))
    model = str(settings.get("model", "")).strip()
    if not model:
        raise SystemExit(f"Set {args.section}.model in {args.config}.")
    host = args.host or str(settings.get("host", "127.0.0.1"))
    port = args.port if args.port is not None else int(settings.get("port", 8082))

    executable, command = _server_command()
    device_probe_command = command.copy()
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
    if args.device and args.device_index is not None:
        raise SystemExit("Use either --device or --device-index, not both.")
    device = args.device or str(settings.get("device", "")).strip()
    if args.device_index is not None:
        device = _resolve_device_name(device_probe_command, args.device_index)
    split_mode = args.split_mode or str(settings.get("split_mode", "")).strip()
    main_gpu = args.main_gpu
    if main_gpu is None and "main_gpu" in settings:
        main_gpu = int(settings["main_gpu"])
    _append_device_options(command, device, split_mode, main_gpu)
    command.extend(_model_argument(model))

    mmproj = str(settings.get("mmproj", "")).strip()
    if mmproj:
        command.extend(["--mmproj", str(_existing_path(mmproj, "Multimodal projector"))])

    device_note = f" on {device}" if device else ""
    print(f"Starting local LLM at http://{host}:{port}/v1{device_note}", flush=True)
    os.execv(executable, command)


def _append_device_options(
    command: list[str],
    device: str,
    split_mode: str,
    main_gpu: int | None,
) -> None:
    if device:
        command.extend(["--device", device])
    if split_mode:
        command.extend(["--split-mode", split_mode])
    if main_gpu is not None:
        command.extend(["--main-gpu", str(main_gpu)])


def _resolve_device_name(server_command: list[str], gpu_index: int) -> str:
    completed = subprocess.run(
        [*server_command, "--list-devices"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    names = _parse_device_names(output)
    if completed.returncode != 0 or not names:
        raise SystemExit(
            "Could not discover llama.cpp devices with --list-devices. "
            f"Output: {' '.join(output.split())[:500]}"
        )

    return _select_device_name(names, gpu_index)


def _select_device_name(names: list[str], gpu_index: int) -> str:
    cuda_name = f"CUDA{gpu_index}"
    if cuda_name in names:
        return cuda_name
    cuda_devices = [name for name in names if name.startswith("CUDA")]
    if len(cuda_devices) == 1:
        # CUDA_VISIBLE_DEVICES may remap one physical GPU to local CUDA0.
        return cuda_devices[0]

    vulkan_name = f"Vulkan{gpu_index}"
    if vulkan_name in names:
        return vulkan_name
    raise SystemExit(
        f"No llama.cpp device found for physical GPU {gpu_index}. "
        f"Available devices: {', '.join(names)}"
    )


def _parse_device_names(output: str) -> list[str]:
    return re.findall(r"^\s*((?:CUDA|Vulkan)\d+):", output, flags=re.MULTILINE)


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
