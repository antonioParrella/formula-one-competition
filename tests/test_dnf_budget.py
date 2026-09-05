import json
import sys

sys.path.insert(0, "src")
from scorer import Scorer


def _write_tips(tips_dir, round_num, submissions):
    tips_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": round_num,
        "race_name": f"Round {round_num}",
        "submissions": submissions,
    }
    path = tips_dir / f"r{round_num:02d}_test_tips.json"
    path.write_text(json.dumps(payload))


def test_dnf_budget_accepts_only_remaining_slots(tmp_path):
    tips_dir = tmp_path / "tips"
    _write_tips(tips_dir, 1, [
        {"player": "Player", "dnf_picks": ["ALO", "STR", "PER", "BOT"]},
    ])
    _write_tips(tips_dir, 2, [
        {"player": "Player", "dnf_picks": ["LAW", "GAS", "OCO"]},
    ])

    scorer = Scorer(round_num=2, tips_dir=str(tips_dir))
    used_before = scorer._dnf_picks_used_before_round()
    result = scorer._score_dnfs(
        {"dnf_picks": ["LAW", "GAS", "OCO"]},
        {"LAW", "GAS", "OCO"},
        used_before=used_before["Player"],
    )

    assert result["score"] == Scorer.POINTS_DNF
    assert [pick["driver"] for pick in result["picks"]] == ["Lawson"]
    assert result["budget"] == {
        "limit": 5,
        "used_before": 4,
        "submitted": 3,
        "accepted": 1,
        "rejected_picks": ["Gasly", "Ocon"],
        "remaining_after": 0,
    }


def test_dnf_budget_is_independent_per_player(tmp_path):
    tips_dir = tmp_path / "tips"
    _write_tips(tips_dir, 1, [
        {"player": "Spent", "dnf_picks": ["ALO", "STR", "PER", "BOT", "LAW"]},
        {"player": "Fresh", "dnf_picks": []},
    ])

    scorer = Scorer(round_num=2, tips_dir=str(tips_dir))
    used_before = scorer._dnf_picks_used_before_round()
    spent = scorer._score_dnfs(
        {"dnf_picks": ["GAS"]}, {"GAS"}, used_before["Spent"]
    )
    fresh = scorer._score_dnfs(
        {"dnf_picks": ["GAS"]}, {"GAS"}, used_before.get("Fresh", 0)
    )

    assert spent["score"] == 0
    assert spent["picks"] == []
    assert spent["budget"]["rejected_picks"] == ["Gasly"]
    assert fresh["score"] == Scorer.POINTS_DNF
    assert fresh["budget"]["remaining_after"] == 4


def test_first_round_also_cannot_submit_more_than_five(tmp_path):
    scorer = Scorer(round_num=1, tips_dir=str(tmp_path / "tips"))
    result = scorer._score_dnfs(
        {"dnf_picks": ["ALO", "STR", "PER", "BOT", "LAW", "GAS"]},
        {"ALO", "STR", "PER", "BOT", "LAW", "GAS"},
    )

    assert result["score"] == 5 * Scorer.POINTS_DNF
    assert len(result["picks"]) == 5
    assert result["budget"]["rejected_picks"] == ["Gasly"]
    assert result["budget"]["remaining_after"] == 0


def test_round_scoring_applies_budget_and_records_rejection(tmp_path):
    tips_dir = tmp_path / "tips"
    _write_tips(tips_dir, 1, [
        {"player": "Spent", "dnf_picks": ["ALO", "STR", "PER", "BOT", "LAW"]},
    ])
    current_tips = {
        "round": 2,
        "race_name": "Round 2",
        "submissions": [{
            "player": "Spent",
            "submitted_at": "2026-03-05T12:00:00",
            "main_race": [],
            "dnf_picks": ["GAS"],
        }],
    }
    _write_tips(tips_dir, 2, current_tips["submissions"])
    results = {
        "top10": [],
        "dnfs": ["GAS"],
        "championship_top10_before_race": [],
        "is_sprint_weekend": False,
    }
    schedule = {
        "sessions": [
            {"session_name": "Practice 1", "date_start": "2026-03-06T01:30:00+00:00"},
            {"session_name": "Practice 2", "date_start": "2026-03-06T05:00:00+00:00"},
            {"session_name": "Practice 3", "date_start": "2026-03-07T01:30:00+00:00"},
            {"session_name": "Qualifying", "date_start": "2026-03-07T05:00:00+00:00"},
        ],
    }

    scorer = Scorer(round_num=2, tips_dir=str(tips_dir))
    scored = scorer._build_scored(current_tips, results, {}, schedule)
    player = scored["player_tips"][0]

    assert player["score"] == 0
    assert player["dnf_picks"] == []
    assert player["dnf_budget"]["rejected_picks"] == ["Gasly"]
