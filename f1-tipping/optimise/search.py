"""Ticket optimisation.

Ticket space is 20P10 ≈ 6.7×10¹¹, so: start from the probability-greedy
ticket (drivers sorted by expected finishing position), then steepest-
ascent local search with swap-within-ticket and substitute-from-outside
moves, with random restarts. All candidates are evaluated against the
same fixed simulation set (common random numbers), so comparisons are
low-variance.

Because the current comp scoring is additive per pick, a ticket's EV is
a sum of per-(driver, slot) expected points, which makes move deltas
O(1) and lets the Hungarian algorithm (`linear_sum_assignment`) provide
a provably EV-optimal seed — local search then serves as verification
and as the runner-up explorer. If ``scoring/rules.py`` is rewritten
with cross-pick interactions, set ``optimise.assume_additive: false``
to evaluate candidates on the raw per-sim scores instead.
"""

import _paths  # noqa: F401

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from model.simulate import SimSet, mean_finish
from scoring.rules import expected_points_matrix, ticket_scores


@dataclass(frozen=True)
class OptimiseReport:
    best_ticket: list[str]           # driver codes, P1..P10
    best_ev: float
    score_percentiles: dict[str, float]
    runner_ups: list[tuple[list[str], float]]
    n_sims: int


def greedy_ticket(sims: SimSet) -> np.ndarray:
    """Drivers sorted by expected finishing position, best 10 in order."""
    return np.argsort(mean_finish(sims.finish_pos))[:10]


def hungarian_ticket(exp_points: np.ndarray) -> np.ndarray:
    """EV-optimal assignment of drivers to slots (exact under additivity)."""
    rows, cols = linear_sum_assignment(-exp_points)
    ticket = np.empty(10, dtype=np.intp)
    ticket[cols] = rows
    return ticket


def _ev_additive(exp_points: np.ndarray, ticket: np.ndarray) -> float:
    return float(exp_points[ticket, np.arange(10)].sum())


def local_search(
    ticket: np.ndarray,
    exp_points: np.ndarray,
    sims: SimSet,
    multipliers: np.ndarray,
    max_iters: int,
    assume_additive: bool = True,
) -> tuple[np.ndarray, float]:
    """Steepest-ascent hill climb over swap and substitution moves."""
    n = exp_points.shape[0]

    def ev(t: np.ndarray) -> float:
        if assume_additive:
            return _ev_additive(exp_points, t)
        return float(ticket_scores(sims.finish_pos, t, multipliers).mean())

    current = ticket.copy()
    current_ev = ev(current)

    for _ in range(max_iters):
        best_delta, best_candidate = 0.0, None
        outside = np.setdiff1d(np.arange(n), current, assume_unique=False)

        for i in range(10):
            for j in range(i + 1, 10):  # swap slots i and j
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]
                delta = ev(cand) - current_ev
                if delta > best_delta + 1e-12:
                    best_delta, best_candidate = delta, cand
            for d in outside:           # substitute outside driver at slot i
                cand = current.copy()
                cand[i] = d
                delta = ev(cand) - current_ev
                if delta > best_delta + 1e-12:
                    best_delta, best_candidate = delta, cand

        if best_candidate is None:
            break
        current, current_ev = best_candidate, current_ev + best_delta

    return current, ev(current)


def optimise(
    sims: SimSet,
    multipliers: np.ndarray,
    n_restarts: int = 20,
    max_iters: int = 200,
    n_runner_ups: int = 5,
    assume_additive: bool = True,
    seed: int = 0,
) -> OptimiseReport:
    """Search ticket space for the highest expected points."""
    n = len(sims.drivers)
    exp_points = expected_points_matrix(sims.finish_pos, multipliers)
    rng = np.random.default_rng(seed)

    starts = [greedy_ticket(sims), hungarian_ticket(exp_points)]
    while len(starts) < max(n_restarts, 2):
        starts.append(rng.choice(n, size=10, replace=False))

    seen: dict[tuple[int, ...], float] = {}
    for k, start in enumerate(starts):
        ticket, ticket_ev = local_search(
            start, exp_points, sims, multipliers, max_iters, assume_additive
        )
        seen[tuple(ticket)] = ticket_ev
        if k in (0, 1):
            label = ("greedy start", "assignment start")[k]
            print(f"  {label:<17} -> local optimum EV {ticket_ev:.2f}")

    best = max(seen, key=seen.get)
    best_ticket = np.array(best, dtype=np.intp)
    best_ev = seen[best]

    # Runner-ups: local optima from other restarts plus the full
    # neighbourhood of the winner, ranked by EV on the same sim set.
    for i in range(10):
        for j in range(i + 1, 10):
            cand = best_ticket.copy()
            cand[i], cand[j] = cand[j], cand[i]
            seen.setdefault(tuple(cand), _ev_additive(exp_points, cand))
        for d in np.setdiff1d(np.arange(n), best_ticket):
            cand = best_ticket.copy()
            cand[i] = d
            seen.setdefault(tuple(cand), _ev_additive(exp_points, cand))
    ranked = sorted(
        (t for t in seen if t != best), key=seen.get, reverse=True
    )[:n_runner_ups]

    scores = ticket_scores(sims.finish_pos, best_ticket, multipliers)
    p10, p50, p90 = np.percentile(scores, [10, 50, 90])

    to_codes = lambda t: [sims.drivers[i] for i in t]  # noqa: E731
    return OptimiseReport(
        best_ticket=to_codes(best_ticket),
        best_ev=best_ev,
        score_percentiles={
            "mean": float(scores.mean()),
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
        },
        runner_ups=[(to_codes(np.array(t)), seen[t]) for t in ranked],
        n_sims=sims.n_sims,
    )
