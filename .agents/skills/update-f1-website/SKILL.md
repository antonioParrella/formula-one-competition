---
name: update-f1-website
description: Update and publish this repository's F1 tipping website after a completed race. Use when asked to refresh the leaderboard, process or publish the latest race, or perform the scheduled post-race check. Do not use for pre-race tips-only collection or optimiser work.
---

# Update the F1 website

Process one newly completed race through the existing pipeline, verify the
published data, and push only the resulting competition files. Treat a Monday
run as a conditional check: no completed race means a successful no-op.

## Establish a safe starting point

1. Work from the repository root and read `AGENTS.md` for the current data and
   scoring contracts.
2. Inspect the current branch, `git status --short`, and `git log -1`. Preserve
   all unrelated worktree changes. Do not clean, stash, reset, or stage them.
3. For a publishing run, fetch `origin/main` and require all of the following:
   the current branch is `main`, local `HEAD` equals `origin/main`, and the
   files this run may overwrite (`data/processed/standings.json`,
   `docs/index.html`, and the prospective round's files) have no pre-existing
   changes. Stop and report if any condition fails.
4. Never print SurveyMars credentials or the contents of `config/.env`.

## Decide whether a race needs processing

1. Read `total_rounds` and the latest recorded race from
   `data/processed/standings.json` and the latest `*_scored.json` file.
2. Check the internet for the current Formula 1 calendar and official race
   result. Prefer the official Formula 1 calendar/results; corroborate with the
   OpenF1 Race session and its non-empty final `session_result` when useful.
3. A race is ready only when a Grand Prix newer than the latest processed round
   has completed and a classified race result is available. Practice,
   qualifying, sprint-only results, a scheduled future race, or an abandoned
   meeting do not qualify.
4. Account for `config/cancelled_rounds.json` and the pipeline's
   cancellation-aware round numbering. Do not infer a new round from filenames
   alone.
5. If no newer completed race exists, do not run the pipeline, edit files,
   commit, or push. Report the latest processed round and the calendar evidence
   used for the no-op.
6. The pipeline currently declares its season year in `scripts/pipeline.py`.
   If the calendar year being checked differs, stop and report that the season
   configuration needs an intentional update rather than silently processing
   the wrong year.

## Process the new round

Run only the identified round and do not force existing raw data:

```powershell
& .\venv\Scripts\python.exe scripts\pipeline.py --round <ROUND>
```

Do not pass `--force`. Existing files under `data/raw/` are write-once. A
correction or deliberate re-fetch requires fresh user authorization.

The pipeline can print a warning and still exit successfully. Read its output
and require successful completion for the new round at every relevant stage:

- SurveyMars tips were found or an existing write-once tips file was used.
- A schedule and final race result exist for the round.
- The round was scored.
- Standings and `docs/index.html` were rebuilt.

Stop without publishing if authentication, network access, rate limits,
missing survey data, incomplete OpenF1 results, or any latest-round warning
leaves those outcomes uncertain.

## Verify the result

1. Inspect the new raw tips, schedule, result, and scored JSON. Confirm their
   round numbers agree, the race result has a full top ten, the scored players
   match the usable submissions, and penalties/DNF budget decisions are
   represented in the audit fields.
2. Confirm `standings.json` includes the new round, every player's total equals
   the sum of their round scores, ranks are ordered correctly, and the latest
   round winner and overall leader match `docs/index.html`.
3. Run `git diff --check`.
4. Run the test suite with a unique writable `--basetemp` and
   `-p no:cacheprovider`, then remove only that verified temporary directory.
   If the only failures are published-season assertions explicitly pinned to
   the previous round, update those assertions from the verified generated
   JSON and rerun. Do not weaken general scoring tests.
5. Review `git status --short`. Expected changes are limited to the new round's
   raw tips/schedule/result, its processed score, derived standings,
   `docs/index.html`, and any narrowly updated published-season assertion.
   An older raw file appearing as modified is a stop condition.

## Publish

Publishing is authorized when the user asks to publish/push or when this skill
is running from the scheduled post-race automation.

1. Stage the exact expected paths; never use `git add -A` or `git add .` in a
   dirty worktree.
2. Inspect the staged name/status list and staged diff checks before committing.
3. Commit with `Update standings after <race> GP` and push `main` to
   `origin/main`.
4. Confirm the pushed commit and report the round winner, overall leader and
   gap, test result, commit hash, and any unrelated changes left untouched.

Do not commit or push partial results. Remote divergence, merge conflicts,
unexpected files, or failed verification are stop conditions that must be
reported for review.
