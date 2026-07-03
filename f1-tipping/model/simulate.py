"""Monte Carlo race engine: Plackett-Luce + independent DNF layer.

Plackett-Luce sampling uses the Gumbel-max trick: adding i.i.d. Gumbel
noise to each strength θᵢ and sorting descending draws a full finishing
order with exactly the PL sequential-softmax distribution — one argsort
per race instead of 20 sequential draws, fully vectorised.

DNF layer: each driver retires independently with probability pᵢ. DNF'd
drivers are demoted behind every survivor (relative order among DNFs
kept PL-consistent), which produces the correlated attrition scenarios
that promote several underdogs into the top 10 at once.
"""

from dataclasses import dataclass

import numpy as np

# Demotion applied to a DNF'd driver's Gumbel score. Gumbel noise spans
# a few units, so this puts every DNF behind every survivor.
DNF_DEMOTION = 1e6


@dataclass(frozen=True)
class SimSet:
    """A fixed set of simulated races (the common-random-numbers set)."""

    drivers: list[str]        # column order for the arrays below
    finish_pos: np.ndarray    # (n_sims, n) int8 — 0-based finishing position
    dnf: np.ndarray           # (n_sims, n) bool

    @property
    def n_sims(self) -> int:
        return self.finish_pos.shape[0]

    def index_of(self, code: str) -> int:
        return self.drivers.index(code)


def sample_finish_positions(
    theta: np.ndarray,
    dnf_probs: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample finishing positions for every driver in every race.

    Returns ``(finish_pos, dnf)`` where ``finish_pos[s, d]`` is driver
    d's 0-based classified position in race s.
    """
    n = theta.shape[0]
    scores = theta[None, :] + rng.gumbel(size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < dnf_probs[None, :]
    scores = np.where(dnf, scores - DNF_DEMOTION, scores)

    order = np.argsort(-scores, axis=1)              # order[s, pos] = driver
    finish_pos = np.empty((n_sims, n), dtype=np.int64)
    finish_pos[np.arange(n_sims)[:, None], order] = np.arange(n)[None, :]
    return finish_pos.astype(np.int8), dnf


def simulate_races(
    theta_by_code: dict[str, float],
    dnf_probs_by_code: dict[str, float],
    n_sims: int,
    seed: int,
) -> SimSet:
    """Simulate ``n_sims`` races for the fitted driver set."""
    drivers = list(theta_by_code)
    theta = np.array([theta_by_code[c] for c in drivers], dtype=np.float64)
    dnf_probs = np.array([dnf_probs_by_code[c] for c in drivers], dtype=np.float64)

    rng = np.random.default_rng(seed)
    finish_pos, dnf = sample_finish_positions(theta, dnf_probs, n_sims, rng)
    return SimSet(drivers=drivers, finish_pos=finish_pos, dnf=dnf)


def top_k_probs(finish_pos: np.ndarray, k: int) -> np.ndarray:
    """P(driver finishes in the top k), per driver column."""
    return (finish_pos < k).mean(axis=0)


def h2h_prob(finish_pos: np.ndarray, i: int, j: int) -> float:
    """P(driver column i finishes ahead of driver column j)."""
    return float((finish_pos[:, i] < finish_pos[:, j]).mean())


def mean_finish(finish_pos: np.ndarray) -> np.ndarray:
    """Expected 0-based finishing position, per driver column."""
    return finish_pos.mean(axis=0, dtype=np.float64)
