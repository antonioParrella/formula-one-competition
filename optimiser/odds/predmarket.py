"""Shared helpers for probability-priced prediction markets.

Kalshi and Polymarket both quote binary markets as probabilities in
(0, 1) over fully public, unauthenticated JSON APIs. These helpers hold
what the two clients share: a plain stdlib HTTP GET (no SDK, no
credentials) and the conversion of probability quotes into the
decimal-odds runner entries of the snapshot schema (see
``odds/snapshot.py``), so everything downstream treats these sources
exactly like Betfair.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

_HEADERS = {"User-Agent": "f1-tipping-optimiser", "Accept": "application/json"}


def get_json(url: str, params: dict | None = None, timeout: float = 20.0):
    """GET a public JSON endpoint; RuntimeError with context on failure."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"GET {url} failed ({exc}). The endpoint is public — check "
            "connectivity, or fall back to another source / --manual."
        ) from None


def fnum(value) -> float | None:
    """Lenient float parse — these APIs mix numbers, strings and null."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_odds(prob) -> float | None:
    """Probability quote -> decimal odds; None outside (0, 1)."""
    p = fnum(prob)
    if p is None or not 0.0 < p < 1.0:
        return None
    return 1.0 / p


def runner_from_quotes(bid, ask, last, max_spread: float) -> dict | None:
    """Snapshot runner entry from probability-space quotes.

    ``back`` is the odds bought at (1/ask), ``lay`` the odds sold at
    (1/bid) — same ordering as Betfair (back <= lay). A two-sided book
    wider than ``max_spread`` is an unseeded placeholder (e.g. 0.02/0.98
    before the weekend) and both quotes are dropped rather than letting
    the garbage midpoint through. A one-sided book keeps its quote only
    when a last trade anchors it: a lone resting order that has never
    traded is an empty book wearing a price (a solitary 0.99 ask read as
    "99%" once fed a fit a 1% head-to-head that contradicted the win
    market by 5x). Returns None if no usable price remains.
    """
    bid, ask, last = fnum(bid), fnum(ask), fnum(last)
    if bid and ask and ask - bid > max_spread:
        bid = ask = None
    if to_odds(last) is None and (to_odds(bid) is None) != (to_odds(ask) is None):
        bid = ask = None
    runner = {"last_traded": to_odds(last), "back": to_odds(ask), "lay": to_odds(bid)}
    if all(v is None for v in runner.values()):
        return None
    return runner
