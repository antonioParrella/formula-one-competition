---
name: run-race-pipeline
description: Run the F1 tipping competition’s race-weekend pipeline to collect tips, fetch OpenF1 data, score rounds and rebuild the leaderboard. Use when collecting submissions before a race, publishing scores after a race, or re-running one round after a correction. Covers the tip-only, single-round and force-fetch paths.
---

# Running the race pipeline

Run the pipeline from the repository root. Raw input is write-once and processed
files are regenerated from it, so use `--force` only when a source response has
changed or needs correcting.

```bash
python scripts/pipeline.py                    # all rounds
python scripts/pipeline.py --round 4          # one round
python scripts/pipeline.py --round 4 --tip-only  # tips only; no results, scoring or site build
python scripts/pipeline.py --round 4 --force  # re-fetch existing tips
```

The full run performs six steps: fetch SurveyMars tips, fetch OpenF1 schedules,
fetch results, score rounds, aggregate standings, and inject the rebuilt data
into `docs/index.html`.

Use the normal race-weekend sequence:

```bash
# Before the race: archive submissions without trying to score them.
python scripts/pipeline.py --round 5 --tip-only

# After the race: fetch the result, score it, rebuild the standings and site.
python scripts/pipeline.py --round 5
```

To rebuild only the site after existing rounds are scored:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from build_site import SiteBuilder; SiteBuilder().build_and_save()"
```
