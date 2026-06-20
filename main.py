from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from t2i_framework.core.config import load_yaml_config
from t2i_framework.core.logging_utils import console
from t2i_framework.core.registry import (
    available_components,
    build_attack,
    build_defense,
    build_model,
)
from t2i_framework.evaluation.runner import ExperimentRunner, read_prompt_file

app = typer.Typer(help="Safe T2I adversarial prompt evaluation framework.")


@app.callback(invoke_without_command=True)
def main(
    model: str = typer.Option("mock", "--model", help="Model adapter name."),
    attack: str = typer.Option("identity", "--attack", help="Attack module name."),
    defense: str = typer.Option("none", "--defense", help="Defense module name."),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Single input prompt."),
    prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", help="CSV with prompt,target_concept."),
    target: Optional[str] = typer.Option(None, "--target", help="Target concept for a single prompt."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    out: Path = typer.Option(Path("results/run"), "--out", help="Output directory."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional YAML config path."),
    max_candidates: int = typer.Option(1, "--max-candidates", min=1, help="Candidates to consider."),
    list_components: bool = typer.Option(False, "--list-components", help="List registered components."),
) -> None:
    if list_components:
        _print_components()
        return

    if (prompt is None and prompt_file is None) or (prompt is not None and prompt_file is not None):
        raise typer.BadParameter("Provide exactly one of --prompt or --prompt-file.")

    try:
        image_model = build_model(model)
        attack_module = build_attack(attack)
        defense_module = build_defense(defense)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    prompts = [(prompt, target)] if prompt is not None else read_prompt_file(prompt_file)  # type: ignore[arg-type]
    config_data = load_yaml_config(config)
    runner = ExperimentRunner(
        model=image_model,
        attack=attack_module,
        defense=defense_module,
        output_dir=out,
        max_candidates=max_candidates,
    )
    results = runner.run(prompts=prompts, seed=seed, config=config_data)
    console.print(f"Wrote {len(results)} result row(s) to {out}")


def _print_components() -> None:
    table = Table(title="Available Components")
    table.add_column("Kind")
    table.add_column("Names")
    for kind, names in available_components().items():
        table.add_row(kind, ", ".join(names))
    console.print(table)


if __name__ == "__main__":
    app()
