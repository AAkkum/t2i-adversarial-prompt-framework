from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import request

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "data/datasets/safety_nonsexual/safety_nonsexual_100.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute DACA candidates with one llama.cpp server per GPU."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--base-port", type=int, default=8084)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    dataset = _resolve(args.dataset)
    output = _resolve(
        args.output
        or Path("results/daca_cache_precompute") / time.strftime("%Y%m%d_%H%M%S")
    )
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("Provide at least one GPU in --gpus.")
    if not dataset.is_file():
        raise SystemExit(f"Dataset not found: {dataset}")

    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    config_dir = output / "configs"
    log_dir = output / "logs"
    for directory in (shard_dir, config_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shards = _write_shards(dataset, shard_dir, len(gpu_ids))
    servers: list[tuple[subprocess.Popen[bytes], object]] = []
    workers: list[tuple[int, subprocess.Popen[bytes], object]] = []
    try:
        for index, gpu_id in enumerate(gpu_ids):
            port = args.base_port + index
            if _port_is_open("127.0.0.1", port):
                raise RuntimeError(
                    f"Port {port} is already in use. Stop the existing server or "
                    "choose another --base-port."
                )
            log_handle = (log_dir / f"server_gpu{gpu_id}.log").open("wb")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_id,
                    "LLAMA_ARG_SPLIT_MODE": "none",
                    "LLAMA_ARG_MAIN_GPU": "0",
                }
            )
            command = [
                sys.executable,
                "-m",
                "t2i_framework.core.local_llm_server",
                "--config",
                "configs/attacks/daca.yaml",
                "--section",
                "daca_llm",
                "--port",
                str(port),
            ]
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            servers.append((process, log_handle))
            print(f"[server {index + 1}/{len(gpu_ids)}] GPU {gpu_id}, port {port}")
            _wait_for_server(
                process,
                "127.0.0.1",
                port,
                args.startup_timeout,
                log_dir / f"server_gpu{gpu_ids[index]}.log",
            )

        for index, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            port = args.base_port + index
            override_path = config_dir / f"worker_{index + 1:02d}.yaml"
            override_path.write_text(
                yaml.safe_dump({"daca_llm": {"port": port}}, sort_keys=True),
                encoding="utf-8",
            )
            worker_output = output / f"worker_{index + 1:02d}_gpu{gpu_id}"
            log_handle = (log_dir / f"worker_gpu{gpu_id}.log").open("wb")
            command = [
                sys.executable,
                "main.py",
                "--model",
                "mock",
                "--attack",
                "daca",
                "--defense",
                "none",
                "--prompt-file",
                str(shard),
                "--max-candidates",
                str(args.max_candidates),
                "--config",
                str(override_path),
                "--out",
                str(worker_output),
            ]
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            workers.append((index, process, log_handle))
            print(
                f"[worker {index + 1}/{len(gpu_ids)}] GPU {gpu_id}, "
                f"{_row_count(shard)} prompts"
            )

        failures: list[str] = []
        for index, process, log_handle in workers:
            return_code = process.wait()
            log_handle.close()
            status = "complete" if return_code == 0 else f"failed ({return_code})"
            print(f"[worker {index + 1}/{len(gpu_ids)}] {status}")
            if return_code != 0:
                failures.append(str(log_dir / f"worker_gpu{gpu_ids[index]}.log"))
        if failures:
            raise RuntimeError("DACA workers failed; inspect: " + ", ".join(failures))
    finally:
        for _, process, log_handle in workers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not log_handle.closed:
                log_handle.close()
        for process, log_handle in servers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log_handle.close()

    print("DACA cache precomputation complete.")
    print(f"Worker outputs and logs: {output}")


def _write_shards(dataset: Path, output: Path, count: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no CSV header: {dataset}")
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows:
        raise ValueError(f"Dataset contains no prompts: {dataset}")

    buckets: list[list[dict[str, str]]] = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        buckets[index % count].append(row)

    paths: list[Path] = []
    for index, bucket in enumerate(buckets):
        path = output / f"shard_{index + 1:02d}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bucket)
        paths.append(path)
    print(
        f"Split {len(rows)} prompts into {count} shards: "
        + ", ".join(str(len(bucket)) for bucket in buckets)
    )
    return paths


def _wait_for_server(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"DACA server on port {port} exited early; inspect {log_path}"
            )
        try:
            with request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f"[server ready] {url}")
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for DACA server on port {port}: {log_path}")


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) == 0


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
