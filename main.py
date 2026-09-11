from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from t2i_framework.core.config import default_component_config_path, load_merged_yaml_configs
from t2i_framework.core.logging_utils import console
from t2i_framework.core.registry import (
    available_components,
    build_attack,
    build_defense,
    build_model,
)
from t2i_framework.core.types import PromptCase
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
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Output directory. If omitted, a timestamped folder is created under results/runs/.",
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional YAML config path."),
    model_config: Optional[Path] = typer.Option(None, "--model-config", help="Optional model YAML config path."),
    attack_config: Optional[Path] = typer.Option(None, "--attack-config", help="Optional attack YAML config path."),
    defense_config: Optional[Path] = typer.Option(None, "--defense-config", help="Optional defense YAML config path."),
    max_candidates: int = typer.Option(
        1, "--max-candidates", min=1, max=20, help="Candidates to consider."
    ),
    list_components: bool = typer.Option(False, "--list-components", help="List registered components."),
) -> None:
    if list_components:
        _print_components()
        return

    if (prompt is None and prompt_file is None) or (prompt is not None and prompt_file is not None):
        raise typer.BadParameter("Provide exactly one of --prompt or --prompt-file.")

    auto_model_config = None if model_config is not None else default_component_config_path("models", model)
    auto_attack_config = None if attack_config is not None else default_component_config_path("attacks", attack)
    auto_defense_config = (
        None if defense_config is not None else default_component_config_path("defenses", defense)
    )
    config_data = load_merged_yaml_configs(
        auto_model_config,
        auto_attack_config,
        auto_defense_config,
        config,
        model_config,
        attack_config,
        defense_config,
    )
    model_config = dict(config_data.get("model", {}))
    configured_model_name = model_config.pop("name", None)
    model_builder_name = configured_model_name or model
    if configured_model_name is not None and configured_model_name != model and _is_registered_model_name(model):
        raise typer.BadParameter(
            f"Config selects model '{configured_model_name}', but CLI selected '{model}'. "
            "Pass the same --model value or use a matching config."
        )
    _validate_configured_component(config_data, "attack", attack)
    _validate_configured_component(config_data, "defense", defense)

    try:
        image_model = build_model(model_builder_name, **model_config)
        attack_module = build_attack(attack)
        defense_module = build_defense(defense)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except TypeError as exc:
        raise typer.BadParameter(f"Invalid configuration for model '{model}': {exc}") from exc

    prompts = (
        [PromptCase(prompt=prompt, target_concept=target)]
        if prompt is not None
        else read_prompt_file(prompt_file)  # type: ignore[arg-type]
    )
    output_dir = out or _timestamped_output_dir(model_builder_name, attack, defense)
    runner = ExperimentRunner(
        model=image_model,
        attack=attack_module,
        defense=defense_module,
        output_dir=output_dir,
        max_candidates=max_candidates,
    )
    results = runner.run(prompts=prompts, seed=seed, config=config_data)
    console.print(f"Wrote {len(results)} result row(s) to {output_dir}")


def _print_components() -> None:
    table = Table(title="Available Components")
    table.add_column("Kind")
    table.add_column("Names")
    for kind, names in available_components().items():
        table.add_row(kind, ", ".join(names))
    for kind in ("models", "attacks", "defenses", "evaluation"):
        presets = _available_config_presets(kind)
        if presets:
            table.add_row(f"{kind} presets", ", ".join(presets))
    console.print(table)


def _validate_configured_component(config_data: dict[str, object], section: str, selected_name: str) -> None:
    section_config = config_data.get(section)
    if not isinstance(section_config, dict):
        return
    configured_name = section_config.get("name")
    if configured_name is not None and configured_name != selected_name:
        raise typer.BadParameter(
            f"Config selects {section} '{configured_name}', but CLI selected '{selected_name}'. "
            f"Pass the same --{section} value or use a matching config."
        )


def _is_registered_model_name(name: str) -> bool:
    return name in available_components()["models"]


def _available_config_presets(kind: str) -> list[str]:
    path = Path("configs") / kind
    if not path.exists():
        return []
    return sorted(config_path.stem for config_path in path.glob("*.yaml"))


def _timestamped_output_dir(model: str, attack: str, defense: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "runs" / f"{stamp}_{model}_{attack}_{defense}"


if __name__ == "__main__":
    app()
