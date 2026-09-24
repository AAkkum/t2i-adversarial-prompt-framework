"""Discrete search based on the authors' InversePrompt.ipynb (MIT).

Reference revision: e4585ada0a5eb185fbef89bb94147c97f8d2f79a.
See docs/ring_a_bell.md and docs/licenses/ring_a_bell.txt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    token_ids: list[int]
    fitness: float
    evaluations: int
    progress: list[dict[str, Any]]


def initialize_population(size: int, length: int, generator: Any) -> list[Any]:
    import torch

    # Author layout: BOS, length mutable IDs, EOS padding to exactly 77 positions.
    return [
        torch.cat(
            (
                torch.tensor([[49406]]),
                torch.randint(1, 49406, (1, length), generator=generator),
                torch.full((1, 76 - length), 49407),
            ),
            dim=1,
        )
        for _ in range(size)
    ]


def fitness(embeddings: Any, target: Any) -> Any:
    """Equation (5): squared L2 SUM across all tokens and hidden dimensions."""
    return ((target - embeddings) ** 2).sum(dim=(1, 2))


def select(population: list[Any], losses: Any, size: int) -> tuple[list[Any], float]:
    import numpy as np

    if len(losses) != len(population) or not np.isfinite(losses).all():
        raise ValueError("Ring-A-Bell fitness must contain one finite loss per candidate.")
    indices = np.argsort(losses)
    return [population[index] for index in indices[: size // 2]], float(losses[indices[0]])


def crossover(parents: list[Any], rate: float, length: int, rng: Any, np_rng: Any) -> list[Any]:
    import torch

    population = []
    for parent in parents:
        population.append(parent)
        if rng.random() < rate:
            mate = parents[np_rng.randint(0, len(parents), size=(1,))[0]]
            point = np_rng.randint(1, length + 1, size=(1,))[0]
            population.append(torch.cat((parent[:, :point], mate[:, point:]), dim=1))
            population.append(torch.cat((mate[:, :point], parent[:, point:]), dim=1))
    # No refill/trimming: variable population size is part of the author code.
    return population


def mutate(population: list[Any], rate: float, length: int, rng: Any, np_rng: Any) -> list[Any]:
    for candidate in population:
        if rng.random() < rate:
            index = np_rng.randint(1, length + 1, size=(1,))
            value = np_rng.randint(1, 49406, size=(1,))[0]
            candidate[:, index] = int(value)
    # Parents also mutate; there is no immutable elite in the notebook.
    return population


def discover(encoder: Any, target: Any, settings: Any, seed: int, report: Any) -> SearchResult:
    import numpy as np
    import torch

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    population = initialize_population(settings.population_size, settings.prompt_length, generator)
    evaluations = 0
    progress = []
    for step in range(settings.generations):
        losses = encoder.losses(population, target)
        evaluations += len(population)
        population, best = select(population, losses, settings.population_size)
        if step % settings.log_interval == 0 or step == settings.generations - 1:
            record = {"generation": step + 1, "best_fitness": best, "population": len(losses)}
            progress.append(record)
            report(record)
        if step != settings.generations - 1:
            population = crossover(
                population, settings.crossover_rate, settings.prompt_length, rng, np_rng
            )
            population = mutate(
                population, settings.mutation_rate, settings.prompt_length, rng, np_rng
            )
    # Return the final generation winner, not an invented global-best archive.
    return SearchResult(population[0][0].tolist(), best, evaluations, progress)
