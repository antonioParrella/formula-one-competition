"""Overround removal.

Turns raw exchange odds into implied probabilities:

- win market: proportional normalisation (default) or the power method
- top-N markets: remove overround only; leave missing probability mass in
  incomplete books instead of inflating every priced runner
- H2H markets: two outcomes, normalise to 1
- "To Be Classified" Yes/No markets: two outcomes, normalise to 1;
  1 - P(classified) is the driver's DNF probability

Also converts a whole snapshot (see ``odds/snapshot.py``) into the
market-probability structure consumed by ``model/fit.py``.
"""

import math

from scipy.optimize import brentq

PROB_CAP = 0.999  # no market outcome is treated as certain
MAX_UNTRADED_SPREAD = 0.15  # probability points, matching prediction-market books
MIN_H2H_MATCHED = 100.0  # a handful of matched contracts cannot anchor a fit

# Structural market key -> top-k it informs. top5 is Kalshi's ladder
# step (KXF1TOP5); Betfair prices top6 instead — the fit uses whichever
# is present.
TOPK_BY_MARKET = {"win": 1, "top3": 3, "top5": 5, "top6": 6, "top10": 10}


def select_price(runner: dict) -> float | None:
    """Use a trade, or a reasonably tight two-sided book without one.

    A lone resting order is not evidence of a fair price. In particular,
    untraded Betfair longshots can have a 1.01 back offer with no lay, which
    must not be read as a near-certain winner.
    """
    last = runner.get("last_traded")
    back, lay = runner.get("back"), runner.get("lay")
    two_sided = bool(back and lay and back > 1.0 and lay > 1.0)
    tight = (two_sided and
             abs(1.0 / back - 1.0 / lay) <= MAX_UNTRADED_SPREAD)
    if last and last > 1.0:
        # A trade outside the current bid/ask range is stale relative to
        # the available book. Use that book if tight, else omit the runner.
        if two_sided and not back <= last <= lay:
            return (float(back) + float(lay)) / 2 if tight else None
        return float(last)
    if tight:
        return (float(back) + float(lay)) / 2
    return None


def has_traded_volume(market: dict) -> bool:
    """Unknown volume is allowed for manual inputs; explicit zero is not."""
    matched = market.get("total_matched")
    return matched is None or matched > 0


def has_usable_h2h_volume(market: dict) -> bool:
    """Keep manual H2H odds; reject known micro-volume market trades."""
    matched = market.get("total_matched")
    return matched is None or matched >= MIN_H2H_MATCHED


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
    """Remove top-N overround without inventing missing probability mass.

    Default is the power method (p_i = q_i^k, k solving the sum): top-N
    overround sits almost entirely in the longshots, and proportional
    scaling instead drags near-certain favourites visibly below their
    win/top-6 prices, which no consistent race model can reproduce.
    """
    q = implied(odds)
    if len(q) <= n:
        # With <= n priced runners the constraint sum(p) = n forces every
        # runner to certainty: degenerate (and rootless for the power
        # method, since q < 1 implies q^k < 1 for every k > 0).
        raise ValueError(
            f"Top-{n} market has only {len(q)} priced runners (needs > {n})"
        )
    # A short book can be missing several runners. Scaling its observed
    # probabilities up to N falsely treats every missing runner as a zero.
    if sum(q.values()) <= n:
        return q

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


def devig_classified(yes_price: float | None, no_price: float | None) -> float | None:
    """P(classified) for one driver from "To be Classified" prices.

    Both sides priced: standard two-way normalisation. One side only
    (the "No" market is often not listed or thin): its implied
    probability is used as-is, capped — the residual vig stays in, but
    that still beats a flat prior.
    """
    q_yes = 1.0 / yes_price if yes_price and yes_price > 1.0 else None
    q_no = 1.0 / no_price if no_price and no_price > 1.0 else None
    if q_yes is not None and q_no is not None:
        return q_yes / (q_yes + q_no)
    if q_yes is not None:
        return min(q_yes, PROB_CAP)
    if q_no is not None:
        return 1.0 - min(q_no, PROB_CAP)
    return None


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
          "drivers":      [codes, ...],           # union over fit markets
          "topk":         {1: {code: p}, 3: ..., 6: ..., 10: ...},
          "weights":      {1: w, 3: w, ...},
          "h2h":          [((a, b), p_a, weight), ...],
          "dnf":          {code: 1 - P(classified), ...},
          "markets_used": ["win: Race Winner (n=20)", ...],
        }

    ``dnf`` comes from the per-driver "To Be Classified" markets and is
    an input to the DNF layer, not a fit target; drivers priced only
    there are not added to the fitted field.
    """
    markets = snapshot.get("markets", {})
    topk: dict[int, dict[str, float]] = {}
    weights: dict[int, float] = {}
    used: list[str] = []
    drivers: set[str] = set()

    for key, k in TOPK_BY_MARKET.items():
        market = markets.get(key)
        if not market or not has_traded_volume(market):
            continue
        odds = {
            code: price
            for code, runner in market["runners"].items()
            if (price := select_price(runner)) is not None
        }
        if not odds:
            continue
        if k > 1 and len(odds) <= k:
            # Thin sources (wide spreads dropped) can leave a top-N market
            # with too few priced runners to de-vig — with <= N runners the
            # sum-to-N constraint forces all to certainty. Skip, don't die.
            print(f"WARNING: {key} market has only {len(odds)} priced "
                  f"runners (need > {k}), skipped")
            continue
        topk[k] = (devig_win(odds, win_method) if k == 1
                   else devig_topn(odds, k, topn_method))
        weights[k] = _market_weight(market.get("total_matched"))
        drivers |= set(topk[k])
        used.append(f"{key}: {market.get('market_name', '?')} (n={len(odds)})")

    h2h: list[tuple[tuple[str, str], float, float]] = []
    for market in markets.get("h2h", []):
        if not has_usable_h2h_volume(market):
            continue
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

    # "To be Classified": driver-runner markets for the Yes and (where
    # listed) No side. Per driver this is a binary event — two-way de-vig
    # when both sides are priced, raw implied probability otherwise.
    classified = markets.get("classified") or {}
    yes_market = classified.get("yes") or {}
    no_market = classified.get("no") or {}
    yes_runners = yes_market.get("runners", {}) if has_traded_volume(yes_market) else {}
    no_runners = no_market.get("runners", {}) if has_traded_volume(no_market) else {}
    cls_weight = _market_weight(yes_market.get("total_matched"))
    dnf: dict[str, float] = {}
    dnf_weights: dict[str, float] = {}
    for code in set(yes_runners) | set(no_runners):
        p_classified = devig_classified(
            select_price(yes_runners.get(code, {})),
            select_price(no_runners.get(code, {})),
        )
        if p_classified is not None:
            dnf[code] = 1.0 - p_classified
            dnf_weights[code] = cls_weight
    if dnf:
        sides = "+".join(s for s, runners in (("yes", yes_runners),
                                               ("no", no_runners)) if runners)
        used.append(f"classified({sides}): {len(dnf)} drivers")

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
        "dnf": dnf,
        "dnf_weights": dnf_weights,
        "markets_used": used,
    }
