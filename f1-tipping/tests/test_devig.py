import pytest

from odds.devig import (
    PROB_CAP,
    devig_h2h,
    devig_snapshot,
    devig_topn,
    devig_win,
    select_price,
)


def test_proportional_win_sums_to_one_and_keeps_ratios():
    odds = {"VER": 2.0, "NOR": 4.0, "PIA": 8.0}  # implied sum 0.875 -> scale up
    probs = devig_win(odds, "proportional")
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["VER"] / probs["NOR"] == pytest.approx(2.0)
    assert probs["NOR"] / probs["PIA"] == pytest.approx(2.0)


def test_power_win_sums_to_one_and_penalises_longshots_more():
    odds = {"VER": 1.8, "NOR": 3.0, "PIA": 12.0, "HAD": 30.0}  # overround
    prop = devig_win(odds, "proportional")
    power = devig_win(odds, "power")
    assert sum(power.values()) == pytest.approx(1.0, abs=1e-9)
    # Power shifts probability from longshots to favourites.
    assert power["VER"] > prop["VER"]
    assert power["HAD"] < prop["HAD"]


def test_power_equal_odds_is_uniform():
    probs = devig_win({c: 3.5 for c in "ABCD"}, "power")
    for p in probs.values():
        assert p == pytest.approx(0.25)


@pytest.mark.parametrize("method", ["power", "proportional"])
def test_topn_sums_to_n(method):
    odds = {c: o for c, o in zip("ABCDEF", [1.3, 1.5, 2.0, 3.0, 5.0, 9.0])}
    probs = devig_topn(odds, 3, method)
    assert sum(probs.values()) == pytest.approx(3.0)
    assert all(0 < p <= PROB_CAP for p in probs.values())


def test_topn_power_takes_vig_from_longshots():
    # 14%+ overround: power leaves the 1.04 favourite nearly untouched
    # and squeezes the longshots instead of dragging everyone down.
    odds = {"A": 1.04, "B": 1.05, "C": 1.3, "D": 2.0, "E": 3.5, "F": 8.0,
            "G": 12.0, "H": 20.0}
    power = devig_topn(odds, 3, "power")
    prop = devig_topn(odds, 3, "proportional")
    assert sum(power.values()) == pytest.approx(3.0)
    assert power["A"] > prop["A"]
    assert power["H"] < prop["H"]
    assert power["A"] > 0.9  # favourite stays consistent with a 1.04 price


def test_topn_proportional_caps_favourites_at_one():
    # Scaling up would push the 1.01 shots above certainty.
    odds = {"A": 1.01, "B": 1.01, "C": 1.2, "D": 15.0, "E": 40.0, "F": 60.0}
    probs = devig_topn(odds, 3, "proportional")
    assert sum(probs.values()) == pytest.approx(3.0)
    assert probs["A"] == pytest.approx(PROB_CAP)
    assert max(probs.values()) <= PROB_CAP


def test_h2h():
    assert devig_h2h(2.0, 2.0) == pytest.approx(0.5)
    p = devig_h2h(1.8, 2.1)  # overround pair still normalises to 1
    assert p == pytest.approx((1 / 1.8) / (1 / 1.8 + 1 / 2.1))


def test_select_price_prefers_last_traded_then_midpoint():
    assert select_price({"last_traded": 3.0, "back": 2.8, "lay": 3.4}) == 3.0
    assert select_price({"last_traded": None, "back": 2.8, "lay": 3.4}) == pytest.approx(3.1)
    assert select_price({"last_traded": None, "back": 2.8, "lay": None}) == 2.8
    assert select_price({"last_traded": None, "back": None, "lay": None}) is None


def test_devig_snapshot_requires_win_market():
    snapshot = {
        "race_name": "Test",
        "markets": {"top10": {"market_name": "x", "runners": {
            c: {"last_traded": 1.5} for c in "ABCDEFGHIJKL"}}},
    }
    with pytest.raises(RuntimeError, match="no usable win market"):
        devig_snapshot(snapshot)
