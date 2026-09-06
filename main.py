from __future__ import annotations

from pathlib import Path

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
    prompt: str | None = typer.Option(None, "--prompt", help="Single input prompt."),
    prompt_file: Path | None = typer.Option(None, "--prompt-file", help="CSV with prompt,target_concept."),  # noqa: B008 -- Typer CLI declaration
    target: str | None = typer.Option(None, "--target", help="Target concept for a single prompt."),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    out: Path = typer.Option(Path("results/run"), "--out", help="Output directory."),  # noqa: B008 -- Typer CLI declaration
    config: Path | None = typer.Option(None, "--config", help="Optional YAML config path."),  # noqa: B008 -- Typer CLI declaration
    max_candidates: int = typer.Option(
        1, "--max-candidates", min=1, max=20, help="Total candidates to consider."
    ),
    list_components: bool = typer.Option(False, "--list-components", help="List registered components."),
) -> None:
    if list_components:
        _print_components()
        return

    if (prompt is None and prompt_file is None) or (prompt is not None and prompt_file is not None):
        raise typer.BadParameter("Provide exactly one of --prompt or --prompt-file.")

    config_data = load_yaml_config(config)
    model_config = dict(config_data.get("model", {}))
    configured_model_name = model_config.pop("name", None)
    if configured_model_name is not None and configured_model_name != model:
        raise typer.BadParameter(
            f"Config selects model '{configured_model_name}', but CLI selected '{model}'. "
            "Pass the same --model value or use a matching config."
        )

    try:
        image_model = build_model(model, **model_config)
        attack_module = build_attack(attack)
        defense_module = build_defense(defense)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except TypeError as exc:
        raise typer.BadParameter(f"Invalid configuration for model '{model}': {exc}") from exc

    prompts = [(prompt, target)] if prompt is not None else read_prompt_file(prompt_file)  # type: ignore[arg-type]
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
