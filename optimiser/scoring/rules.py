"""Comp scoring rules — a thin wrapper around the comp's own engine.

The canonical rules live in the parent repo's ``src/scorer.py`` and are
**imported, never copied**: ``score_ticket`` delegates to
``Scorer._score_main_race``, and the vectorised paths read every point
value from the ``Scorer`` class constants. If the comp changes its rules
there, this module follows automatically.

Current rules (from ``Scorer``), per pick in a top-10 ticket:

    exact position          POINTS_EXACT  (5)
    one position off        POINTS_CLOSE  (3)
    in top 10, wrong spot   POINTS_TOP10  (1)
    outside top 10          POINTS_MISS   (0)
    underdog multiplier     MULTIPLIER_UNDERDOG (2×) — from round 2, for
                            drivers outside the championship top 10
                            before the race

The vectorised paths exist because the optimiser evaluates
(tickets × simulations) scores; they are verified equal to the wrapped
``Scorer`` in ``tests/test_rules.py``.
"""

import _paths  # noqa: F401

from dataclasses import dataclass, field

import numpy as np

from scorer import Scorer  # the comp's scoring engine — never edited


@dataclass(frozen=True)
class ScoringContext:
    """Everything the underdog rule needs to know about the round."""

    round_num: int
    championship_top10: frozenset[str] = field(default_factory=frozenset)

    @property
    def underdog_active(self) -> bool:
        return self.round_num > 1 and bool(self.championship_top10)

    def multiplier(self, code: str) -> int:
        if self.underdog_active and code not in self.championship_top10:
            return Scorer.MULTIPLIER_UNDERDOG
        return 1


def score_ticket(
    ticket: list[str],
    result: list[str],
    context: ScoringContext | None = None,
) -> float:
    """Score one top-10 ticket against one finishing order.

    ``result`` may be the full classified order; only its first 10 count
    (matching the comp's results files). Delegates to the comp's
    ``Scorer._score_main_race``. Without a context — or with an empty
    championship set, where Scorer would double *every* pick — the
    underdog bonus is disabled (round-1 semantics).
    """
    context = context or ScoringContext(round_num=1)
    scorer = Scorer(round_num=context.round_num if context.underdog_active else 1)
    position_map = Scorer._position_map(list(result)[:10])
    picks = scorer._score_main_race(
        {"main_race": list(ticket)}, position_map, set(context.championship_top10)
    )
    return float(sum(p["points"] for p in picks))


def underdog_bonus(driver: str, predicted_pos: int, actual_pos: int) -> float:
    """Extension point for a position-scaled underdog bonus.

    TODO: the comp may move to a bonus scaled by driver position; today
    it is a flat multiplier (``Scorer.MULTIPLIER_UNDERDOG``) applied by
    ``Scorer._score_main_race`` regardless of ``predicted_pos`` /
    ``actual_pos``, so this returns that multiplier and the arguments
    are unused. Rewrite this (and ``points_tensor`` below) when the
    scaled rule is finalised — nothing in model/ needs touching.
    """
    return float(Scorer.MULTIPLIER_UNDERDOG)


# ─────────────────────────────────────────────────────────────
# Vectorised paths (numpy) for the optimiser
# ─────────────────────────────────────────────────────────────

def underdog_multipliers(drivers: list[str], context: ScoringContext) -> np.ndarray:
    """Per-driver score multiplier vector aligned with ``drivers``."""
    return np.array([context.multiplier(c) for c in drivers], dtype=np.int64)


def _base_points(actual: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Pre-multiplier points for picks with 0-based ``actual`` positions
    against predicted ``slots``. Mirrors ``Scorer._score_main_race``: a
    driver outside the top 10 has no position-map entry, so exact/close/
    top10 all require ``actual < 10``. Bool arithmetic keeps temporaries
    int16 even at 100k sims.
    """
    in_top10 = actual < 10
    exact = in_top10 & (actual == slots)
    close = in_top10 & (np.abs(actual - slots) == 1)
    rest = in_top10 & ~exact & ~close
    return (
        exact.astype(np.int16) * Scorer.POINTS_EXACT
        + close.astype(np.int16) * Scorer.POINTS_CLOSE
        + rest.astype(np.int16) * Scorer.POINTS_TOP10
        + (~in_top10).astype(np.int16) * Scorer.POINTS_MISS
    )


def points_tensor(finish_pos: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Points earned by (driver, slot) in every sim.

    ``finish_pos`` is (n_sims, n_drivers) 0-based positions; returns an
    (n_drivers, 10, n_sims) int16 tensor.
    """
    actual = finish_pos.T[:, None, :].astype(np.int16)      # (n, 1, sims)
    slots = np.arange(10, dtype=np.int16)[None, :, None]    # (1, 10, 1)
    return _base_points(actual, slots) * multipliers[:, None, None].astype(np.int16)


def expected_points_matrix(finish_pos: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Mean points per (driver, slot) over the sim set — (n_drivers, 10).

    The comp's scoring is additive across picks, so any ticket's EV is
    the sum of its 10 entries in this matrix.
    """
    return points_tensor(finish_pos, multipliers).mean(axis=2, dtype=np.float64)


def ticket_scores(
    finish_pos: np.ndarray,
    ticket_idx: np.ndarray,
    multipliers: np.ndarray,
) -> np.ndarray:
    """Per-sim total scores for one ticket of 10 driver column indices."""
    slots = np.arange(10, dtype=np.int16)[None, :]
    actual = finish_pos[:, ticket_idx].astype(np.int16)     # (sims, 10)
    pts = _base_points(actual, slots) * multipliers[ticket_idx][None, :].astype(np.int16)
    return pts.sum(axis=1)
