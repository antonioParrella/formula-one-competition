import pytest

from odds.combine import combine_snapshots
from odds.devig import _market_weight, devig_snapshot, devig_win


def _snap(source, markets, race="Test GP"):
    return {"race_name": race, "source": source,
            "fetched_at": "2026-07-06T00:00:00+00:00", "markets": markets}


def _market(odds, total_matched=None):
    return {"market_name": "m", "market_id": None,
            "total_matched": total_matched,
            "runners": {code: {"last_traded": o, "back": None, "lay": None}
                        for code, o in odds.items()}}


def test_combine_win_is_liquidity_weighted_average_of_devigged():
    a = _snap("betfair", {"win": _market({"VER": 2.0, "NOR": 2.2}, 100_000)})
    b = _snap("kalshi", {"win": _market({"VER": 1.8, "NOR": 2.6}, 50_000)})
    combined = combine_snapshots([a, b])
    assert combined["source"] == "combined"
    assert [c["source"] for c in combined["combined_from"]] == ["betfair", "kalshi"]

    p_a = devig_win({"VER": 2.0, "NOR": 2.2})
    p_b = devig_win({"VER": 1.8, "NOR": 2.6})
    w_a, w_b = _market_weight(100_000), _market_weight(50_000)
    expected = (w_a * p_a["VER"] + w_b * p_b["VER"]) / (w_a + w_b)

    odds_out = combined["markets"]["win"]["runners"]["VER"]["last_traded"]
    assert 1.0 / odds_out == pytest.approx(expected)
    # de-vigging the combined snapshot is a no-op on already vig-free probs
    out = devig_snapshot(combined)
    assert out["topk"][1]["VER"] == pytest.approx(expected, rel=1e-9)


def test_combine_keeps_drivers_priced_by_one_source_only():
    a = _snap("betfair", {"win": _market({"VER": 2.0, "NOR": 2.0}, 1000)})
    b = _snap("kalshi",
              {"win": _market({"VER": 2.0, "NOR": 2.5, "PIA": 6.0}, 1000)})
    runners = combine_snapshots([a, b])["markets"]["win"]["runners"]
    assert set(runners) == {"VER", "NOR", "PIA"}


def test_combine_concatenates_h2h_with_source_tags():
    h2h = [{"market_name": "VER v NOR", "market_id": None, "total_matched": 10,
            "runners": {"VER": {"last_traded": 1.8, "back": None, "lay": None},
                        "NOR": {"last_traded": 2.1, "back": None, "lay": None}}}]
    a = _snap("betfair",
              {"win": _market({"VER": 2.0, "NOR": 2.0}, 1000), "h2h": h2h})
    b = _snap("polymarket",
              {"win": _market({"VER": 2.0, "NOR": 2.0}, 1000), "h2h": h2h})
    combined = combine_snapshots([a, b])
    assert len(combined["markets"]["h2h"]) == 2
    assert combined["markets"]["h2h"][0]["market_name"] == "[betfair] VER v NOR"


def test_combine_merges_classified_most_liquid_first():
    yes_thick = {"market_name": "y", "market_id": None, "total_matched": 900,
                 "runners": {"VER": {"last_traded": 1.05, "back": None,
                                     "lay": None}}}
    yes_thin = {"market_name": "y", "market_id": None, "total_matched": 10,
                "runners": {"VER": {"last_traded": 1.20, "back": None,
                                    "lay": None},
                            "STR": {"last_traded": 1.10, "back": None,
                                    "lay": None}}}
    a = _snap("betfair", {"win": _market({"VER": 2.0, "NOR": 2.0}, 1),
                          "classified": {"yes": yes_thick}})
    b = _snap("manual", {"win": _market({"VER": 2.0, "NOR": 2.0}, 1),
                         "classified": {"yes": yes_thin}})
    combined = combine_snapshots([a, b])
    runners = combined["markets"]["classified"]["yes"]["runners"]
    assert runners["VER"]["last_traded"] == 1.05  # thick source wins VER
    assert runners["STR"]["last_traded"] == 1.10  # thin fills the gap


def test_combine_skips_degenerate_topn_from_one_source(capsys):
    # Polymarket shipped a top-3 with exactly 3 priced runners (every other
    # book a dropped placeholder) — skip that source's market, keep the rest.
    a = _snap("kalshi", {
        "win": _market({"VER": 2.0, "NOR": 2.0}, 1000),
        "top3": _market({"VER": 1.2, "NOR": 1.3, "PIA": 2.0, "LEC": 3.0,
                         "HAM": 5.0, "RUS": 8.0}, 500),
    })
    b = _snap("polymarket", {
        "win": _market({"VER": 2.0, "NOR": 2.0}, 1000),
        "top3": _market({"VER": 1.10, "NOR": 1.15, "PIA": 1.30}),
    })
    combined = combine_snapshots([a, b])
    assert combined["markets"]["top3"]["market_name"] == "combined top3 (kalshi)"
    assert "only 3 priced runners" in capsys.readouterr().out
    a = _snap("betfair", {"win": _market({"VER": 2.0, "NOR": 2.0}, 1)})
    with pytest.raises(ValueError, match="at least two"):
        combine_snapshots([a])
    b = _snap("kalshi", {"win": _market({"VER": 2.0, "NOR": 2.0}, 1)},
              race="Other GP")
    with pytest.raises(ValueError, match="different races"):
        combine_snapshots([a, b])


def test_devig_snapshot_accepts_top5_and_skips_short_topn(capsys):
    snapshot = _snap("kalshi", {
        "win": _market({"VER": 2.0, "NOR": 3.0, "PIA": 6.0, "LEC": 8.0,
                        "HAM": 10.0, "RUS": 12.0}, 1000),
        "top5": _market({"VER": 1.1, "NOR": 1.2, "PIA": 1.5, "LEC": 2.0,
                         "HAM": 2.5, "RUS": 3.0}, 500),
        "top10": _market({"VER": 1.01, "NOR": 1.02}, 100),  # too few priced
    })
    out = devig_snapshot(snapshot)
    assert 5 in out["topk"]
    assert sum(out["topk"][5].values()) == pytest.approx(5.0)
    assert 10 not in out["topk"]
    assert "top10 market has only 2 priced runners" in capsys.readouterr().out
