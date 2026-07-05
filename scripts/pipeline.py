"""
pipeline.py
────────────────────────────────────────────────────────────
Runs the full F1 tipping competition pipeline end-to-end.

Usage:
    python pipeline.py                        # all rounds
    python pipeline.py --round 4              # single round
    python pipeline.py --round 4 --tip-only  # skip results/scoring
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from survey_mars import SurveyMarsClient
from survey_index import SurveyIndex
from tips_parser import TipsParser
from fetch_schedule import ScheduleFetcher
from fetch_results import ResultsFetcher
from scorer import Scorer
from aggregator import Aggregator
from build_site import SiteBuilder


def run_pipeline(round_filter=None, tip_only=False, force_rebuild=False):
    """Run the full pipeline.

    Parameters
    ----------
    round_filter : int | None
        If set, only process this round. Otherwise processes all rounds.
    tip_only : bool
        If True, only fetch/save tips — skip results, scoring, aggregate, site.
    force_rebuild : bool
        If True, re-fetch and overwrite existing tips files.
    """
    print("=" * 60)
    print("  F1 TIPPING PIPELINE")
    print("=" * 60)

    # ── 1. Fetch tips ─────────────────────────────────────
    print("\n[1/6] Fetching tips from SurveyMars...")
    client = SurveyMarsClient()
    client.authenticate()

    index = SurveyIndex(client).fetch()
    index.print_summary()

    parser = TipsParser(client)
    surveys = index.all()

    if round_filter is not None:
        surveys = [s for s in surveys if s["round_num"] == round_filter]
        if not surveys:
            print(f"No survey found for round {round_filter}")
            return

    for survey in surveys:
        parser.fetch_and_save(
            survey_id=survey["survey_id"],
            round_num=survey["round_num"],
            race_name=survey["title"],
            force=force_rebuild,
        )

    if tip_only:
        print("\nTip-only mode — stopping after tips fetch.")
        return

    year = 2026

    # ── 2. Fetch session schedules ─────────────────────────
    print("\n[2/6] Fetching session schedules from OpenF1...")
    for survey in surveys:
        try:
            ScheduleFetcher(round_num=survey["round_num"], year=year).fetch_and_save(
                force=force_rebuild
            )
        except Exception as e:
            print(f"  WARNING: Failed to fetch schedule for round {survey['round_num']}: {e}")

    # ── 3. Fetch results ───────────────────────────────────
    print("\n[3/6] Fetching results from OpenF1...")
    for survey in surveys:
        try:
            fetcher = ResultsFetcher(round_num=survey["round_num"], year=year)
            fetcher.fetch_and_save()
        except Exception as e:
            print(f"  WARNING: Failed to fetch results for round {survey['round_num']}: {e}")

    # ── 4. Score rounds ────────────────────────────────────
    print("\n[4/6] Scoring rounds...")
    for survey in surveys:
        try:
            scorer = Scorer(round_num=survey["round_num"])
            scorer.score_and_save()
        except Exception as e:
            print(f"  WARNING: Failed to score round {survey['round_num']}: {e}")

    # ── 5. Aggregate standings ─────────────────────────────
    print("\n[5/6] Aggregating standings...")
    agg = Aggregator()
    agg.build_and_save()

    # ── 6. Build site ──────────────────────────────────────
    print("\n[6/6] Building website...")
    builder = SiteBuilder()
    builder.build_and_save()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run F1 tipping pipeline")
    ap.add_argument("--round", type=int, default=None, help="Process single round only")
    ap.add_argument("--tip-only", action="store_true", help="Fetch tips and stop")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if tips exist")
    args = ap.parse_args()

    run_pipeline(
        round_filter=args.round,
        tip_only=args.tip_only,
        force_rebuild=args.force,
    )
