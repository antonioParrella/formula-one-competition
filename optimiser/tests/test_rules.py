"""Scoring: hand-computed examples, and equivalence of the vectorised
paths with the comp's own ``Scorer`` (which score_ticket wraps)."""

import numpy as np
import pytest

from scorer import Scorer
from scoring.rules import (
    ScoringContext,
    expected_points_matrix,
    score_ticket,
    ticket_scores,
    underdog_multipliers,
)

GRID = ["VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "HAD", "SAI", "ALO",
        "ALB", "GAS", "OCO", "STR", "HUL", "TSU", "LAW", "BEA", "BOR", "LIN"]

RESULT = ["VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "HAD", "SAI", "ALO"]
TICKET = ["VER", "PIA", "NOR", "HAM", "LEC", "RUS", "GAS", "HAD", "ALO", "TSU"]
# exact(5) close(3) close(3) close(3) close(3) exact(5) miss(0) exact(5)
# close(3) miss(0) = 30 before any underdog doubling.


def test_hand_computed_no_underdog_round1():
    assert score_ticket(TICKET, RESULT, ScoringContext(round_num=1)) == 30.0


def test_hand_computed_underdog_doubles_close_pick():
    # ALO outside the championship top 10 -> his close pick doubles: 30 + 3.
    champ = frozenset(["VER", "NOR", "PIA", "LEC", "HAM", "RUS", "ANT", "HAD",
                       "SAI", "STR"])
    ctx = ScoringContext(round_num=5, championship_top10=champ)
    assert score_ticket(TICKET, RESULT, ctx) == 33.0


def test_underdog_disabled_with_empty_champ_set():
    # Guard: an empty set must not turn the whole field into underdogs.
    ctx = ScoringContext(round_num=5)
    assert score_ticket(TICKET, RESULT, ctx) == 30.0


def _finish_pos_row(order: list[str]) -> np.ndarray:
    """(1, n) finish-position row from a full finishing order."""
    row = np.empty((1, len(GRID)), dtype=np.int8)
    for pos, code in enumerate(order):
        row[0, GRID.index(code)] = pos
    return row


@pytest.mark.parametrize("round_num", [1, 5])
def test_vectorised_scoring_equals_comp_scorer(round_num):
    """The numpy path must agree with Scorer._score_main_race exactly."""
    rng = np.random.default_rng(99)
    for _ in range(200):
        order = [GRID[i] for i in rng.permutation(len(GRID))]
        ticket_idx = rng.choice(len(GRID), size=10, replace=False)
        ticket = [GRID[i] for i in ticket_idx]
        champ = frozenset(GRID[i] for i in rng.choice(len(GRID), 10, replace=False))
        ctx = ScoringContext(round_num=round_num, championship_top10=champ)

        # Ground truth straight from the comp's engine.
        scorer = Scorer(round_num=round_num)
        picks = scorer._score_main_race(
            {"main_race": ticket}, Scorer._position_map(order[:10]), set(champ)
        )
        expected = float(sum(p["points"] for p in picks))

        assert score_ticket(ticket, order, ctx) == expected

        mult = underdog_multipliers(GRID, ctx)
        vectorised = ticket_scores(_finish_pos_row(order), ticket_idx, mult)
        assert float(vectorised[0]) == expected


def test_expected_points_matrix_matches_per_sim_scores():
    """Additivity: summing the E-matrix equals averaging per-sim scores."""
    rng = np.random.default_rng(3)
    finish_pos = np.stack(
        [rng.permutation(len(GRID)).astype(np.int8) for _ in range(500)]
    )
    ctx = ScoringContext(round_num=3, championship_top10=frozenset(GRID[:10]))
    mult = underdog_multipliers(GRID, ctx)

    exp_points = expected_points_matrix(finish_pos, mult)
    assert exp_points.shape == (len(GRID), 10)

    ticket_idx = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=np.intp)
    ev_matrix = exp_points[ticket_idx, np.arange(10)].sum()
    ev_sims = ticket_scores(finish_pos, ticket_idx, mult).mean()
    assert ev_matrix == pytest.approx(ev_sims)
