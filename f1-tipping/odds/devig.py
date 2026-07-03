"""Overround removal.

Turns raw exchange odds into implied probabilities:

- win market: proportional normalisation (default) or the power method
- top-N markets: probabilities must sum to N, proportional with a cap at 1
- H2H markets: two outcomes, normalise to 1

Also converts a whole snapshot (see ``odds/snapshot.py``) into the
market-probability structure consumed by ``model/fit.py``.
"""

import math

from scipy.optimize import brentq

PROB_CAP = 0.999  # no market outcome is treated as certain

# Structural market key -> top-k it informs.
TOPK_BY_MARKET = {"win": 1, "top3": 3, "top6": 6, "top10": 10}


def select_price(runner: dict) -> float | None:
    """Best available price for a runner: last traded where liquidity
    exists, else back/lay midpoint, else whichever side is quoted."""
    last = runner.get("last_traded")
    if last and last > 1.0:
        return float(last)
    back, lay = runner.get("back"), runner.get("lay")
    if back and lay:
        return (float(back) + float(lay)) / 2
    for side in (back, lay):
        if side and side > 1.0:
            return float(side)
    return None


def implied(odds: dict[str, float]) -> dict[str, float]:
    return {code: 1.0 / o for code, o in odds.items() if o and o > 1.0}


def devig_win(odds: dict[str, float], method: str = "proportional") -> dict[str, float]:
    """De-vig a winner market so probabilities sum to 1."""
    q = implied(odds)
    if not q:
        raise ValueError("Win market has no usable prices")
    if method == "proportional":
        total = sum(q.values())
        return {c: p / total for c, p in q.items()}
    if method == "power":
        return _devig_power(q)
    raise ValueError(f"Unknown de-vig method: {method!r}")


def _devig_power(q: dict[str, float]) -> dict[str, float]:
    """Power method: p_i = q_i^k with k solving sum(q_i^k) = 1.

    Penalises longshots more than proportional scaling does, matching the
    favourite-longshot bias in bookmaker (less so exchange) prices.
    """
    probs = [min(p, PROB_CAP) for p in q.values()]

    def excess(k: float) -> float:
        return sum(p**k for p in probs) - 1.0

    if excess(1.0) <= 0:  # no overround left — nothing to remove
        total = sum(probs)
        return {c: p / total for c, p in zip(q, probs)}
    k = brentq(excess, 1.0, 20.0)
    return {c: p**k for c, p in zip(q, probs)}


def devig_topn(odds: dict[str, float], n: int, method: str = "power") -> dict[str, float]:
    """De-vig a top-N market so probabilities sum to N.

    Default is the power method (p_i = q_i^k, k solving the sum): top-N
    overround sits almost entirely in the longshots, and proportional
    scaling instead drags near-certain favourites visibly below their
    win/top-6 prices, which no consistent race model can reproduce.
    """
    q = implied(odds)
    if len(q) < n:
        raise ValueError(f"Top-{n} market has only {len(q)} priced runners")

    if method == "power":
        probs = [min(p, PROB_CAP) for p in q.values()]

        def excess(k: float) -> float:
            return sum(p**k for p in probs) - n

        # excess is monotone decreasing in k; k=1 only if vig-free already.
        k = brentq(excess, 0.05, 50.0)
        return {c: p**k for c, p in zip(q, probs)}
    if method != "proportional":
        raise ValueError(f"Unknown de-vig method: {method!r}")

    # Proportional scaling can push a heavy favourite above 1 (e.g. a
    # 1.01 top-10 price); capped runners are pinned at PROB_CAP and the
    # rest rescaled to absorb the remainder.
    capped: dict[str, float] = {}
    free = dict(q)
    for _ in range(len(q)):
        remaining = n - sum(capped.values())
        scale = remaining / sum(free.values())
        scaled = {c: p * scale for c, p in free.items()}
        over = {c for c, p in scaled.items() if p > PROB_CAP}
        if not over:
            return {**capped, **scaled}
        for c in over:
            capped[c] = PROB_CAP
            del free[c]
    return capped  # unreachable in practice: n < field size


def devig_h2h(odds_a: float, odds_b: float) -> float:
    """P(A beats B) from a two-runner head-to-head market."""
    qa, qb = 1.0 / odds_a, 1.0 / odds_b
    return qa / (qa + qb)


def _market_weight(total_matched: float | None) -> float:
    """Liquidity weight for the fitting objective.

    Unknown liquidity gets weight 1; known liquidity scales gently with
    matched volume (a £100k market is trusted more than a £500 one).
    """
    if not total_matched or total_matched <= 0:
        return 1.0
    return min(1.5, max(0.2, math.log10(1.0 + total_matched) / 4.0))


def devig_snapshot(
    snapshot: dict,
    win_method: str = "proportional",
    topn_method: str = "power",
) -> dict:
    """Convert a snapshot into de-vigged market probabilities.

    Returns::

        {
          "drivers":      [codes, ...],           # union over all markets
          "topk":         {1: {code: p}, 3: ..., 6: ..., 10: ...},
          "weights":      {1: w, 3: w, ...},
          "h2h":          [((a, b), p_a, weight), ...],
          "markets_used": ["win: Race Winner (n=20)", ...],
        }
    """
    markets = snapshot.get("markets", {})
    topk: dict[int, dict[str, float]] = {}
    weights: dict[int, float] = {}
    used: list[str] = []
    drivers: set[str] = set()

    for key, k in TOPK_BY_MARKET.items():
        market = markets.get(key)
        if not market:
            continue
        odds = {
            code: price
            for code, runner in market["runners"].items()
            if (price := select_price(runner)) is not None
        }
        if not odds:
            continue
        topk[k] = (devig_win(odds, win_method) if k == 1
                   else devig_topn(odds, k, topn_method))
        weights[k] = _market_weight(market.get("total_matched"))
        drivers |= set(topk[k])
        used.append(f"{key}: {market.get('market_name', '?')} (n={len(odds)})")

    h2h: list[tuple[tuple[str, str], float, float]] = []
    for market in markets.get("h2h", []):
        runners = market.get("runners", {})
        if len(runners) != 2:
            continue
        (a, ra), (b, rb) = sorted(runners.items())
        pa, pb = select_price(ra), select_price(rb)
        if pa is None or pb is None:
            continue
        h2h.append(((a, b), devig_h2h(pa, pb), _market_weight(market.get("total_matched"))))
        drivers |= {a, b}
    if h2h:
        used.append(f"h2h: {len(h2h)} matchup market(s)")

    if 1 not in topk:
        raise RuntimeError(
            "Snapshot has no usable win market — refusing to fit on partial "
            f"data. Markets present: {sorted(markets)}"
        )

    return {
        "drivers": sorted(drivers),
        "topk": topk,
        "weights": weights,
        "h2h": h2h,
        "markets_used": used,
    }
