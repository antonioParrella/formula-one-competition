---
name: run-f1-optimiser
description: Run this repository's pre-race F1 tipping optimiser and deliver its self-contained HTML analysis. Use when asked to generate optimal picks, refresh the odds/model/report for an upcoming Grand Prix, or perform the scheduled Friday optimiser run. Do not use to score completed races or update the competition leaderboard.
---

# Run the F1 optimiser

Produce a fresh, phone-readable HTML analysis for the next competition race.
Scheduled Friday runs are conditional: no race weekend means a successful
no-op, while missing or stale inputs mean a reported failure rather than an
old recommendation.

## Establish the target race

1. Work from the repository root and read `AGENTS.md` and
   `optimiser/CLAUDE.md` before running the model.
2. Inspect `git status --short`. Preserve all existing worktree changes and
   never stage, clean, reset, stash, commit, or push optimiser inputs or
   outputs as part of this workflow.
3. Check the official Formula 1 calendar and OpenF1 session schedule. For a
   scheduled Friday run, continue only when a Grand Prix race session is due
   during the current weekend. A sprint without a Grand Prix, a cancelled
   meeting, or a future weekend is not sufficient.
4. Resolve the competition round using the completed-round data, the fresh
   SurveyMars index, and `config/cancelled_rounds.json`. Query the index without
   fetching or saving tips. Require an active survey for the same race so the
   optimiser is not run for an event outside the competition.
5. Determine the market-facing Grand Prix name. Start with the official event
   title, but verify it against the live odds-source event names; circuit names
   and market names can differ.
6. Obtain the championship top ten entering the race from OpenF1. If OpenF1 is
   unavailable or incomplete, use official Formula 1 standings and record that
   fallback in the run summary. Require ten unique, mapped three-letter driver
   codes before continuing.

If these checks do not identify exactly one current race, do not run or email
the optimiser. Report the evidence and the stopping reason.

## Prepare an isolated run configuration

Do not edit `optimiser/config.yaml`; it may contain deliberate local tuning.
Create a uniquely named temporary YAML under `optimiser/data/` by running:

```powershell
& .\venv\Scripts\python.exe `
  .agents\skills\run-f1-optimiser\scripts\prepare_config.py `
  --base optimiser\config.yaml `
  --out optimiser\data\.run_config_r<ROUND>.yaml `
  --race-name "<MARKET GRAND PRIX NAME>" `
  --year <YEAR> `
  --round <ROUND> `
  --top10 <CODE1> <CODE2> <CODE3> <CODE4> <CODE5> `
          <CODE6> <CODE7> <CODE8> <CODE9> <CODE10>
```

The helper copies every modelling choice from the base configuration, changes
only the race identity and manual championship top ten, and keeps sprint
markets disabled because the competition optimiser targets the Grand Prix.
Refuse to overwrite an existing temporary config; use a new exact path.

## Run and verify

From the repository root, run all configured odds sources through the existing
launcher:

```powershell
& .\run.ps1 all --source all --config <ABSOLUTE_TEMP_CONFIG>
```

An individual source warning is acceptable only when at least one fresh source
still supplies enough structural markets for fitting and the command completes.
Do not use `--allow-stale`. Stop if all sources fail, the selected snapshot is
older than 24 hours, the fetched event is not the target Grand Prix, validation
does not complete, or the optimiser/report stage fails.

Require and inspect the matching fresh artifacts under `optimiser/data/`:

- one or more `odds_<race>_<timestamp>.json` snapshots;
- `model_fit_<race>.json`;
- `optimise_report_<race>.json`;
- `analysis_<race>.html`.

Confirm the JSON race identity, snapshot timestamp/source, ten unique drivers
in the best ticket, expected score and percentiles, ten underdog-context codes,
and a non-empty self-contained HTML document with a viewport declaration. The
HTML must not depend on local filesystem paths or external assets.

Run the optimiser tests with a unique writable `--basetemp` and
`-p no:cacheprovider`. A failing test is a stop condition. Remove only the
verified test temp directory and the exact temporary run config; retain the
odds snapshot and model/report artifacts as the reproducibility record.

## Email the report

Emailing is authorized when the user asks for it directly or when this skill is
running from the scheduled Friday automation.

Use the Gmail account named in the automation prompt and send to `me`. Send
only after every verification passes. Use a subject such as
`F1 optimiser — <Race> — <date>` and a short body containing the best ticket,
expected score, input timestamp, sources used, and any degraded-source warning.
Attach the generated HTML as `text/html` with an `.html` filename and attachment
disposition; do not paste the full report into the message body.

After Gmail confirms the send, report the race, attachment filename, recipient,
best ticket, and run timestamp. If Gmail is unavailable or sending fails, keep
the verified HTML locally and report its path; do not repeatedly resend without
confirming whether the first attempt succeeded.
