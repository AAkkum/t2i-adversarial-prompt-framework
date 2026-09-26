import csv
from pathlib import Path

from scripts.precompute_daca_cache import _write_shards


def test_write_shards_balances_rows_and_preserves_csv_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "prompts.csv"
    fieldnames = ["id", "prompt", "target_concept", "category"]
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "id": str(index),
                    "prompt": f'prompt {index}, with "quotes"\nand a newline',
                    "target_concept": f"target {index}",
                    "category": "test",
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
    assert sorted(row["id"] for row in rows) == [str(index) for index in range(10)]
    assert all('with "quotes"\nand a newline' in row["prompt"] for row in rows)
