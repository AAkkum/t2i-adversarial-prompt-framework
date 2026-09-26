import csv
import json
from pathlib import Path

import yaml

from scripts.run_sdxl_safety_15_4gpu import (
    MATRIX_CASES,
    ORDER_COLUMN,
    _aggregate_case_outputs,
    _write_shards,
    _write_worker_configs,
)
from t2i_framework.core.local_llm_server import (
    _append_device_options,
    _parse_device_names,
    _select_device_name,
)


def test_matrix_matches_the_fifteen_case_runner() -> None:
    assert len(MATRIX_CASES) == 15
    assert [(case.attack, case.defense) for case in MATRIX_CASES] == [
        ("identity", "none"),
        ("identity", "latent_guard_lite"),
        ("identity", "safree"),
        ("pgj", "none"),
        ("pgj", "safree"),
        ("pgj", "latent_guard_lite"),
        ("daca", "none"),
        ("daca", "safree"),
        ("daca", "latent_guard_lite"),
        ("groot", "none"),
        ("groot", "safree"),
        ("groot", "latent_guard_lite"),
        ("ring_a_bell", "none"),
        ("ring_a_bell", "safree"),
        ("ring_a_bell", "latent_guard_lite"),
    ]
    assert [case.max_candidates for case in MATRIX_CASES] == [
        1,
        1,
        1,
        1,
        1,
        1,
        10,
        10,
        10,
        3,
        3,
        3,
        1,
        1,
        1,
    ]


def test_write_shards_balances_rows_and_records_original_order(tmp_path: Path) -> None:
    dataset = tmp_path / "prompts.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "target_concept"])
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "id": str(index),
                    "prompt": f'prompt {index}, with "quotes"\nand a newline',
                    "target_concept": f"target {index}",
                }
            )

    paths = _write_shards(dataset, tmp_path / "shards", 4)
    rows = []
    sizes = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            shard_rows = list(csv.DictReader(handle))
        sizes.append(len(shard_rows))
        rows.extend(shard_rows)

    assert sizes == [3, 3, 2, 2]
    assert sorted(int(row[ORDER_COLUMN]) for row in rows) == list(range(10))
    assert all('with "quotes"\nand a newline' in row["prompt"] for row in rows)


def test_aggregate_case_outputs_restores_dataset_order(tmp_path: Path) -> None:
    case_output = tmp_path / "01_identity_none"
    for worker, source_indexes in ((1, (0, 2)), (2, (1, 3))):
        worker_dir = case_output / "workers" / f"worker_{worker:02d}_gpu{worker - 1}"
        worker_dir.mkdir(parents=True)
        details = []
        summaries = []
        for source_index in source_indexes:
            run_id = f"run-{source_index}"
            details.append(
                {
                    "run_id": run_id,
                    "metadata": {
                        "candidate_index": 0,
                        "prompt_case": {ORDER_COLUMN: str(source_index)},
                    },
                }
            )
            summaries.append(
                {
                    "run_id": run_id,
                    "original_prompt": f"prompt {source_index}",
                    "scores": {"quality": source_index / 10},
                }
            )
        _write_jsonl(worker_dir / "details.jsonl", details)
        _write_jsonl(worker_dir / "results.jsonl", summaries)

    _aggregate_case_outputs(case_output)

    aggregated = [
        json.loads(line)
        for line in (case_output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["run_id"] for row in aggregated] == ["run-0", "run-1", "run-2", "run-3"]
    assert "score_quality" in (case_output / "results.csv").read_text(encoding="utf-8")


def test_worker_configs_route_each_process_to_its_local_servers(tmp_path: Path) -> None:
    paths = _write_worker_configs(
        tmp_path / "configs",
        tmp_path / "output",
        [8083, 8084, 8085, 8086],
        [8087, 8088, 8089, 8090],
    )

    fourth = yaml.safe_load(paths[3].read_text(encoding="utf-8"))
    assert fourth["local_llm"]["port"] == 8086
    assert fourth["daca_llm"]["port"] == 8090
    assert fourth["attack"]["llm_device"] == "cuda:0"
    assert fourth["attack"]["cache_path"].endswith("pgj_worker_04.json")


def test_llama_server_accepts_explicit_physical_device_options() -> None:
    command = ["llama-server"]
    _append_device_options(command, "CUDA3", "none", 0)

    assert command == [
        "llama-server",
        "--device",
        "CUDA3",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
    ]


def test_llama_device_parser_supports_cuda_and_vulkan_backends() -> None:
    output = """
Available devices:
  CUDA0: NVIDIA RTX A6000
  Vulkan0: NVIDIA RTX A6000
  Vulkan1: NVIDIA RTX A6000
"""
    assert _parse_device_names(output) == ["CUDA0", "Vulkan0", "Vulkan1"]
    assert _select_device_name(["Vulkan0", "Vulkan1", "Vulkan2"], 2) == "Vulkan2"
    assert _select_device_name(["CUDA0"], 3) == "CUDA0"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
