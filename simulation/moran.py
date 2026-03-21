from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

TypeLabel = Literal["A", "B"]


@dataclass
class MoranEvent:
    step: int
    birth_index: int
    birth_type: TypeLabel
    death_index: int
    death_type: TypeLabel
    mutants_before: int
    mutants_after: int
    N: int

    @property
    def event(self) -> str:
        return f"{self.birth_index}{self.birth_type}:{self.death_index}{self.death_type}"


@dataclass
class MoranRun:
    run_id: str
    true_r: float
    true_N: int
    true_i0: int
    absorbed_type: TypeLabel
    steps: list[MoranEvent]



def simulate_moran_run(*, r: float, N: int, i0: int, run_id: str, rng: random.Random) -> MoranRun:
    if not (0 < i0 < N):
        raise ValueError("Initial mutant count i0 must satisfy 0 < i0 < N.")
    if r <= 0:
        raise ValueError("Relative fitness r must be positive.")

    population: list[TypeLabel] = ["A"] * i0 + ["B"] * (N - i0)
    i = i0
    step = 0
    events: list[MoranEvent] = []

    while 0 < i < N:
        mutants_before = i
        weights = [r if t == "A" else 1.0 for t in population]
        birth_index = rng.choices(range(N), weights=weights, k=1)[0]
        birth_type = population[birth_index]

        death_index = rng.randrange(N)
        death_type = population[death_index]

        population[death_index] = birth_type
        i = sum(1 for t in population if t == "A")

        events.append(
            MoranEvent(
                step=step,
                birth_index=birth_index,
                birth_type=birth_type,
                death_index=death_index,
                death_type=death_type,
                mutants_before=mutants_before,
                mutants_after=i,
                N=N,
            )
        )
        step += 1

    absorbed_type: TypeLabel = "A" if i == N else "B"
    return MoranRun(run_id=run_id, true_r=r, true_N=N, true_i0=i0, absorbed_type=absorbed_type, steps=events)
