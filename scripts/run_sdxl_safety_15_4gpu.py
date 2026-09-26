from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import request

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "data/datasets/safety_nonsexual/safety_nonsexual_100.csv"
)
ORDER_COLUMN = "_parallel_source_index"


@dataclass(frozen=True)
class MatrixCase:
    number: str
    attack: str
    defense: str
    max_candidates: int
    defense_config: str | None = None

    @property
    def name(self) -> str:
        return f"{self.number}_{self.attack}_{self.defense}"


MATRIX_CASES = (
    MatrixCase("01", "identity", "none", 1),
    MatrixCase(
        "02",
        "identity",
        "latent_guard_lite",
        1,
        "configs/defenses/latent_guard_safety_nonsexual.yaml",
    ),
    MatrixCase("03", "identity", "safree", 1),
    MatrixCase("04", "pgj", "none", 1),
    MatrixCase("05", "pgj", "safree", 1),
    MatrixCase(
        "06",
        "pgj",
        "latent_guard_lite",
        1,
        "configs/defenses/latent_guard_safety_nonsexual.yaml",
    ),
    MatrixCase("07", "daca", "none", 10),
    MatrixCase("08", "daca", "safree", 10),
    MatrixCase(
        "09",
        "daca",
        "latent_guard_lite",
        10,
        "configs/defenses/latent_guard_safety_nonsexual.yaml",
    ),
    MatrixCase("10", "groot", "none", 3),
    MatrixCase("11", "groot", "safree", 3),
    MatrixCase(
        "12",
        "groot",
        "latent_guard_lite",
        3,
        "configs/defenses/latent_guard_safety_nonsexual.yaml",
    ),
    MatrixCase("13", "ring_a_bell", "none", 1),
    MatrixCase("14", "ring_a_bell", "safree", 1),
    MatrixCase(
        "15",
        "ring_a_bell",
        "latent_guard_lite",
        1,
        "configs/defenses/latent_guard_safety_nonsexual.yaml",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete 15-case SDXL safety matrix with prompt shards "
            "distributed across multiple GPUs."
        )
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--evaluator-base-port", type=int, default=8083)
    parser.add_argument("--daca-base-port", type=int, default=8087)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = _resolve(args.dataset)
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    output = _resolve(
        args.output
        or Path("results/matrices")
        / f"{time.strftime('%Y%m%d_%H%M%S')}_sdxl_safety_15_4gpu"
    )

    if not dataset.is_file():
        raise SystemExit(f"Dataset not found: {dataset}")
    if not gpu_ids:
        raise SystemExit("Provide at least one GPU in --gpus.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("Each GPU in --gpus must be unique.")

    evaluator_ports = [args.evaluator_base_port + index for index in range(len(gpu_ids))]
    daca_ports = [args.daca_base_port + index for index in range(len(gpu_ids))]
    overlapping = sorted(set(evaluator_ports) & set(daca_ports))
    if overlapping:
        raise SystemExit(f"Evaluator and DACA port ranges overlap: {overlapping}")
    for port in [*evaluator_ports, *daca_ports]:
        if _port_is_open("127.0.0.1", port):
            raise SystemExit(
                f"Port {port} is already in use. Stop the existing server or "
                "select different base ports."
            )

    shard_dir = output / "shards"
    config_dir = output / "configs"
    log_dir = output / "logs"
    for directory in (shard_dir, config_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shards = _write_shards(dataset, shard_dir, len(gpu_ids))
    overrides = _write_worker_configs(
        config_dir,
        output,
        evaluator_ports,
        daca_ports,
    )
    manifest: dict[str, Any] = {
        "dataset": str(dataset),
        "output": str(output),
        "gpus": gpu_ids,
        "evaluator_ports": evaluator_ports,
        "daca_ports": daca_ports,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": [],
    }
    _write_manifest(output, manifest)

    evaluator_servers: list[ManagedProcess] = []
    daca_servers: list[ManagedProcess] = []
    try:
        evaluator_servers = _start_server_group(
            label="evaluator",
            gpu_ids=gpu_ids,
            ports=evaluator_ports,
            config=REPO_ROOT / "configs/local_llm.yaml",
            section="local_llm",
            log_dir=log_dir,
            startup_timeout=args.startup_timeout,
        )

        for case_index, case in enumerate(MATRIX_CASES, start=1):
            if case.attack == "daca" and not daca_servers:
                daca_servers = _start_server_group(
                    label="daca",
                    gpu_ids=gpu_ids,
                    ports=daca_ports,
                    config=REPO_ROOT / "configs/attacks/daca.yaml",
                    section="daca_llm",
                    log_dir=log_dir,
                    startup_timeout=args.startup_timeout,
                )

            print(
                f"[{case_index}/15] attack={case.attack} defense={case.defense} "
                f"max_candidates={case.max_candidates} on {len(gpu_ids)} GPUs"
            )
            started = time.monotonic()
            case_output = output / case.name
            case_output.mkdir(parents=True, exist_ok=True)
            worker_logs = _run_case_workers(
                case=case,
                case_output=case_output,
                gpu_ids=gpu_ids,
                shards=shards,
                overrides=overrides,
                log_dir=log_dir,
            )
            _aggregate_case_outputs(case_output)
            elapsed = time.monotonic() - started
            manifest["cases"].append(
                {
                    **asdict(case),
                    "output": str(case_output),
                    "worker_logs": [str(path) for path in worker_logs],
                    "runtime_seconds": round(elapsed, 3),
                    "status": "complete",
                }
            )
            _write_manifest(output, manifest)
            print(f"[{case_index}/15] complete in {elapsed / 60:.1f} minutes")

            if case.number == "09":
                _stop_processes(daca_servers)
                daca_servers = []
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = " ".join(str(exc).split())
        _write_manifest(output, manifest)
        raise
    finally:
        _stop_processes(daca_servers)
        _stop_processes(evaluator_servers)

    manifest["status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(output, manifest)
    print(f"Completed 15 four-GPU runs in {output}")


@dataclass
class ManagedProcess:
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path


def _start_server_group(
    *,
    label: str,
    gpu_ids: list[str],
    ports: list[int],
    config: Path,
    section: str,
    log_dir: Path,
    startup_timeout: int,
) -> list[ManagedProcess]:
    servers: list[ManagedProcess] = []
    try:
        # Sequential startup avoids competing first-time Hugging Face downloads.
        for index, (gpu_id, port) in enumerate(zip(gpu_ids, ports), start=1):
            log_path = log_dir / f"{label}_server_gpu{gpu_id}.log"
            log_handle = log_path.open("wb")
            environment = _gpu_environment(gpu_id)
            command = [
                sys.executable,
                "-m",
                "t2i_framework.core.local_llm_server",
                "--config",
                str(config),
                "--section",
                section,
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
            managed = ManagedProcess(process, log_handle, log_path)
            servers.append(managed)
            print(f"[{label} server {index}/{len(gpu_ids)}] GPU {gpu_id}, port {port}")
            _wait_for_server(process, port, startup_timeout, log_path)
    except Exception:
        _stop_processes(servers)
        raise
    return servers


def _run_case_workers(
    *,
    case: MatrixCase,
    case_output: Path,
    gpu_ids: list[str],
    shards: list[Path],
    overrides: list[Path],
    log_dir: Path,
) -> list[Path]:
    workers: list[ManagedProcess] = []
    try:
        for index, (gpu_id, shard, override) in enumerate(
            zip(gpu_ids, shards, overrides), start=1
        ):
            worker_output = case_output / "workers" / f"worker_{index:02d}_gpu{gpu_id}"
            log_path = log_dir / f"{case.name}_worker_{index:02d}_gpu{gpu_id}.log"
            log_handle = log_path.open("wb")
            command = [
                sys.executable,
                "main.py",
                "--model",
                "diffusers",
                "--model-config",
                "configs/models/sdxl.yaml",
                "--attack",
                case.attack,
                "--defense",
                case.defense,
                "--prompt-file",
                str(shard),
                "--max-candidates",
                str(case.max_candidates),
                "--config",
                str(override),
                "--out",
                str(worker_output),
            ]
            if case.defense_config:
                command.extend(["--defense-config", case.defense_config])
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=_gpu_environment(gpu_id),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            workers.append(ManagedProcess(process, log_handle, log_path))
            print(
                f"  [worker {index}/{len(gpu_ids)}] GPU {gpu_id}, "
                f"{_row_count(shard)} prompts"
            )

        failures: list[Path] = []
        for index, managed in enumerate(workers, start=1):
            return_code = managed.process.wait()
            managed.log_handle.close()
            status = "complete" if return_code == 0 else f"failed ({return_code})"
            print(f"  [worker {index}/{len(gpu_ids)}] {status}")
            if return_code != 0:
                failures.append(managed.log_path)
        if failures:
            raise RuntimeError(
                f"Case {case.name} failed; inspect: "
                + ", ".join(str(path) for path in failures)
            )
    finally:
        _stop_processes(workers)
    return [worker.log_path for worker in workers]


def _write_shards(dataset: Path, output: Path, count: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no CSV header: {dataset}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError(f"Dataset contains no prompts: {dataset}")
    if ORDER_COLUMN in fieldnames:
        raise ValueError(f"Dataset already contains reserved column {ORDER_COLUMN!r}.")

    fieldnames.append(ORDER_COLUMN)
    buckets: list[list[dict[str, str]]] = [[] for _ in range(count)]
    for index, row in enumerate(rows):
        row[ORDER_COLUMN] = str(index)
        buckets[index % count].append(row)

    paths: list[Path] = []
    for index, bucket in enumerate(buckets, start=1):
        path = output / f"shard_{index:02d}.csv"
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


def _write_worker_configs(
    config_dir: Path,
    output: Path,
    evaluator_ports: list[int],
    daca_ports: list[int],
) -> list[Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, (evaluator_port, daca_port) in enumerate(
        zip(evaluator_ports, daca_ports), start=1
    ):
        path = config_dir / f"worker_{index:02d}.yaml"
        data = {
            "local_llm": {"port": evaluator_port},
            "daca_llm": {"port": daca_port},
            "attack": {
                # CUDA_VISIBLE_DEVICES exposes one physical GPU as local cuda:0.
                "llm_device": "cuda:0",
                # PGJ's JSON cache writer is not process-safe, so isolate workers.
                "cache_path": str(output / "cache" / f"pgj_worker_{index:02d}.json"),
            },
        }
        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths


def _aggregate_case_outputs(case_output: Path) -> None:
    worker_dirs = sorted((case_output / "workers").glob("worker_*"))
    details = _read_jsonl_files(path / "details.jsonl" for path in worker_dirs)
    summaries = _read_jsonl_files(path / "results.jsonl" for path in worker_dirs)
    if not details or not summaries:
        raise RuntimeError(f"No worker results found for {case_output.name}.")

    details.sort(key=_detail_sort_key)
    rank = {row.get("run_id"): index for index, row in enumerate(details)}
    summaries.sort(key=lambda row: rank.get(row.get("run_id"), len(rank)))
    _write_jsonl(case_output / "details.jsonl", details)
    _write_jsonl(case_output / "results.jsonl", summaries)
    _write_summary_csv(case_output / "results.csv", summaries)


def _detail_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    prompt_case = ((row.get("metadata") or {}).get("prompt_case") or {})
    source_index = int(prompt_case.get(ORDER_COLUMN, 1_000_000_000))
    candidate_index = int((row.get("metadata") or {}).get("candidate_index", 0))
    return source_index, candidate_index, str(row.get("run_id", ""))


def _read_jsonl_files(paths: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing worker result file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    base_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in base_fields and key != "scores":
                base_fields.append(key)
    score_fields = sorted(
        {
            f"score_{name}"
            for row in rows
            for name in (row.get("scores") or {})
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*base_fields, *score_fields])
        writer.writeheader()
        for row in rows:
            values = {
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
                if key != "scores"
            }
            for name, value in (row.get("scores") or {}).items():
                values[f"score_{name}"] = value
            writer.writerow(values)


def _wait_for_server(
    process: subprocess.Popen[bytes],
    port: int,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server on port {port} exited early; inspect {log_path}")
        try:
            with request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print(f"[server ready] {url}")
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for server on port {port}; inspect {log_path}")


def _stop_processes(processes: list[ManagedProcess]) -> None:
    for managed in processes:
        if managed.process.poll() is None:
            managed.process.terminate()
            try:
                managed.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                managed.process.kill()
                managed.process.wait()
        if not managed.log_handle.closed:
            managed.log_handle.close()


def _gpu_environment(gpu_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "LLAMA_ARG_SPLIT_MODE": "none",
            "LLAMA_ARG_MAIN_GPU": "0",
        }
    )
    return environment


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) == 0


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
