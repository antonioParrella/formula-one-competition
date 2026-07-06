"""Kalshi/Polymarket parsers — pure functions over recorded API shapes."""

import pytest

from odds.kalshi_client import parse_event_markets
from odds.polymarket_client import parse_binary_event, parse_h2h_market
from odds.predmarket import runner_from_quotes, to_odds


def test_to_odds_bounds():
    assert to_odds(0.25) == pytest.approx(4.0)
    assert to_odds("0.2500") == pytest.approx(4.0)  # Kalshi dollar strings
    assert to_odds(0.0) is None
    assert to_odds(1.0) is None
    assert to_odds(None) is None


def test_runner_from_quotes_tight_book():
    runner = runner_from_quotes("0.28", "0.30", "0.29", max_spread=0.15)
    assert runner["back"] == pytest.approx(1 / 0.30)
    assert runner["lay"] == pytest.approx(1 / 0.28)
    assert runner["last_traded"] == pytest.approx(1 / 0.29)


def test_runner_from_quotes_placeholder_book_dropped():
    # Unseeded books quote 0.02/0.98 — both sides must go, not the midpoint.
    assert runner_from_quotes(0.02, 0.98, None, max_spread=0.15) is None
    runner = runner_from_quotes(0.02, 0.98, 0.5, max_spread=0.15)
    assert runner == {"last_traded": 2.0, "back": None, "lay": None}


def test_runner_from_quotes_one_sided_book_kept():
    runner = runner_from_quotes(0, 0.01, 0, max_spread=0.15)
    assert runner["back"] == pytest.approx(100.0)
    assert runner["lay"] is None
    assert runner["last_traded"] is None


def _kalshi_market(name, bid, ask, last, vol, status="active"):
    return {"yes_sub_title": name, "yes_bid_dollars": bid,
            "yes_ask_dollars": ask, "last_price_dollars": last,
            "volume_fp": vol, "status": status}


def test_kalshi_parse_event_markets():
    raw = [
        _kalshi_market("Max Verstappen", "0.28", "0.30", "0.29", "1000"),
        _kalshi_market("Lando Norris", "0.10", "0.12", "0.11", "500"),
        _kalshi_market("Old Runner", "0.50", "0.50", "0.50", "9",
                       status="finalized"),
        _kalshi_market("Somebody New", "0.10", "0.20", "0.15", "50"),
    ]
    entry = parse_event_markets(raw, "KXF1RACE-TEST", {})
    assert set(entry["runners"]) == {"VER", "NOR"}
    assert entry["runners"]["VER"]["last_traded"] == pytest.approx(1 / 0.29)
    assert entry["runners"]["NOR"]["back"] == pytest.approx(1 / 0.12)
    # volume counts active mapped runners only (not finalized/unmatched)
    assert entry["total_matched"] == pytest.approx(1500)


def test_kalshi_parse_all_finalized_is_none():
    raw = [_kalshi_market("Max Verstappen", "0.5", "0.5", "0.5", "9",
                          status="finalized")]
    assert parse_event_markets(raw, "KXF1RACE-TEST", {}) is None


def _pm_market(**overrides):
    return {"closed": False, "active": True, **overrides}


def test_polymarket_parse_binary_event():
    event = {
        "title": "Test GP: Driver Winner", "slug": "test-gp-winner",
        "volume": "12345.6",
        "markets": [
            _pm_market(groupItemTitle="Max Verstappen", bestBid=0.047,
                       bestAsk=0.049, lastTradePrice=0.048),
            _pm_market(groupItemTitle="Driver A", bestBid=0.001,
                       bestAsk=0.002),                    # placeholder runner
            _pm_market(groupItemTitle="Pierre Gasly", bestBid=0.02,
                       bestAsk=0.98),                     # placeholder book
            _pm_market(groupItemTitle="Lando Norris", bestBid=0.30,
                       bestAsk=0.32, lastTradePrice=0.31, closed=True),
        ],
    }
    entry = parse_binary_event(event, {})
    assert set(entry["runners"]) == {"VER"}
    assert entry["runners"]["VER"]["back"] == pytest.approx(1 / 0.049)
    assert entry["market_id"] == "test-gp-winner"
    assert entry["total_matched"] == pytest.approx(12345.6)


def test_polymarket_parse_h2h_complement_prices():
    market = _pm_market(
        question="Who will finish higher: Alonso or Stroll?",
        outcomes='["Alonso", "Stroll"]',
        bestBid=0.55, bestAsk=0.60, lastTradePrice=0.58, volume="42",
    )
    entry = parse_h2h_market(market, {})
    alonso, stroll = entry["runners"]["ALO"], entry["runners"]["STR"]
    assert alonso["back"] == pytest.approx(1 / 0.60)
    assert alonso["lay"] == pytest.approx(1 / 0.55)
    assert alonso["last_traded"] == pytest.approx(1 / 0.58)
    assert stroll["back"] == pytest.approx(1 / 0.45)  # 1 - bid_a
    assert stroll["lay"] == pytest.approx(1 / 0.40)   # 1 - ask_a
    assert stroll["last_traded"] == pytest.approx(1 / 0.42)


def test_polymarket_h2h_unmapped_outcome_skipped():
    # Polymarket abbreviates Sainz to "Jr." — must be skipped, not guessed.
    market = _pm_market(question="Who will finish higher: Albon or Jr.?",
                        outcomes='["Albon", "Jr."]', bestBid=0.40,
                        bestAsk=0.45)
    assert parse_h2h_market(market, {}) is None


def test_polymarket_h2h_placeholder_book_skipped():
    market = _pm_market(question="Who will finish higher: Alonso or Stroll?",
                        outcomes='["Alonso", "Stroll"]', bestBid=0.01,
                        bestAsk=0.99)
    assert parse_h2h_market(market, {}) is None
