# Known Issues

Bugs found while scoring round 12 (Dutch GP, Zandvoort, 2026-08-23) on
2026-08-25. **All three are now fixed** — the diagnosis is kept here as the
record of what went wrong and why the fix is shaped the way it is.

| # | Issue | Status |
|---|-------|--------|
| 1 | `lastRace`/`bestRound` ignore sprint points | fixed |
| 2 | `NaN` in `championship_top10_before_race` | fixed |
| 3 | Pipeline crashes on Windows consoles (cp1252) | fixed |
| 4 | Pre-race ladder was the *after*-the-race ladder | fixed |

Round 12 was also the season's first mid-season driver change — Yuki Tsunoda
(#22) replaced Isack Hadjar (#6) for the whole weekend, running the sprint and
the race and finishing P11. That interacts with issue 2 in both directions, and
`tests/test_new_driver_underdog.py` pins the behaviour: the arrival has no
championship points so every main-race pick of him scores double, while the
departure keeps the points that hold him at P8 in the championship top 10, so a
pick of *him* must score single. Verified across all 12 rounds — every pick's
underdog flag matches the championship list for its round, and every pick's
points equal its base value times the multiplier.

---

## 1. `lastRace` and `bestRound` ignore sprint points

**Severity:** medium — wrong numbers on the public leaderboard, but season
totals and rankings were correct.

**Status:** fixed. Regression tests: `tests/test_aggregator_sprint.py`.

### What was wrong

[`Aggregator._build_standings`](src/aggregator.py) accumulated each round in two
passes. The first pass walked `player_tips` (the main race) and set four fields:

```python
p["points"]    += main_score
p["lastRace"]   = main_score
if p["bestRound"] is None or main_score > p["bestRound"]:
    p["bestRound"] = main_score
p["roundScores"][str(round_num)] = {"race": race_name, "points": main_score}
```

The second pass walked `sprint_tips` and folded the sprint score in — but only
into two of them:

```python
p["roundScores"][round_key]["points"] += sprint_score
p["points"] += sprint_score
```

`lastRace` and `bestRound` were never revisited, so on any sprint weekend they
kept the main-race-only value. [`SiteBuilder`](src/build_site.py) publishes both
to the leaderboard, so the live site showed them.

`points` and `roundScores` **were** correct, so the season standings and the
finishing order were right. Only the two display columns were affected.

### Blast radius

Every sprint round in the 2026 season: 2 (China, 50 pts), 4 (Miami, 35),
5 (Canada, 35), 9 (Silverstone, 30), 12 (Dutch, 15).

**`lastRace` after round 12** — was understated for the three players who scored
in the R12 sprint:

| Player | Was | Now |
|---------|-----|-----|
| Barry | 26 | 31 |
| Jake | 19 | 24 |
| Josh | 22 | 27 |

**`bestRound`** — was wrong for 6 of 11 players, because each of their best
rounds happened to be a sprint weekend:

| Player | Was | Now |
|---------|-----|-----|
| Rino | 49 | 54 |
| Jake | 41 | 46 |
| Dean | 53 | 58 |
| Veljko | 35 | 38 |
| Tara | 36 | 41 |
| Alex | 23 | 28 |

### The fix

`_build_standings` now collects `{player: sprint_score}` up front and makes a
single pass with the **round total** — main race plus sprint — through one
`_record_round` helper, so no field can see a partial score.

`lastRace` and `bestRound` are no longer tracked alongside `roundScores`; both
are derived from it at serialisation time (`max(...)` for `bestRound`, the
highest round key for `lastRace`), which removes the whole class of bug. Keying
`lastRace` off the round number also drops the old dependence on the order the
scored files happen to be read in.

`Aggregator().build_and_save()` rebuilt every round from the scored files — no
rescoring needed. Season totals, ranks, `games_played` and `roundScores` came
back byte-identical; only the 9 numbers above changed, in
`data/processed/standings.json` and `docs/index.html`.

---

## 2. `NaN` in `championship_top10_before_race`

**Severity:** low as it stood — round 12 was unaffected — but it could silently
corrupt underdog scoring, and it wrote invalid JSON.

**Status:** fixed. Regression tests: `tests/test_driver_acronyms.py`.

### What was wrong

[`ResultsFetcher._get_championship_top10`](src/fetch_results.py) left-merged the
championship standings onto the drivers table:

```python
merged = standings.merge(
    self.aggregator.drivers[["driver_number", "name_acronym"]],
    on="driver_number",
    how="left",
)
return merged["name_acronym"].tolist()
```

A `how="left"` merge yields `NaN` for any championship driver absent from
`drivers`, and nothing checked for it.

For round 12 that happened. OpenF1's `drivers?session_key=11348` (the Zandvoort
sprint) returns 22 drivers and omits car #6 — Isack Hadjar — even though
`drivers?driver_number=6&meeting_key=1292` *does* report him present in session
11348. Upstream data inconsistency, but the merge swallowed it.

Hadjar was **P8 in the championship**, so the saved top 10 read:

```json
["ANT", "HAM", "RUS", "LEC", "NOR", "VER", "PIA", nan, "GAS", "LAW"]
```

### Two consequences

**Scoring.** `championship_top10_before_race` drives the underdog multiplier —
picks outside the top 10 score double. A `NaN` in the list means that driver is
no longer recognised as a top-10 driver, so anyone picking him would have been
paid at the underdog rate.

Round 12 was **not** affected: nobody picked Hadjar, so the missing entry never
mattered, and re-scoring after the fix reproduced
`data/processed/r12_dutch_scored.json` byte for byte. Round 12 was also the only
round in the season with this problem — the other ten result files were clean.

**Invalid JSON.** Python's `json.dump` writes a bare `NaN` token, which is not
valid JSON, so `data/raw/results/r12_zandvoort_result.json` failed any strict
parser, including JavaScript's `JSON.parse`.

### The fix

Two layers, because the gap and the damage are separate problems.

*Repair the source.* `ResultAggregator._backfill_missing_drivers`
([src/leaderboard.py](src/leaderboard.py)) runs after the session fetches: any
`driver_number` referenced by the race results, sprint results or championship
standings but missing (or unnamed) in `drivers` is looked up via
`_lookup_driver`, which resolves Hadjar correctly. This sits on the shared
table, so it covers `top10` and `dnfs` too, not just the championship list.

`_lookup_driver` tries `drivers?driver_number=N&meeting_key=…` first, then walks
back through this season's earlier race sessions. The fallback matters because a
driver replaced mid-season keeps their championship points — and so their place
in the top 10 — while dropping out of the session driver tables, which is
exactly Hadjar's position from round 12 on; without it, `_require_acronyms`
below would raise and the round would not score at all. Note the bulk
`drivers?meeting_key=…` query is *not* usable here: it omits #6 at Zandvoort
even though the per-driver query finds him. An unscoped
`drivers?driver_number=N` is deliberately never tried — car numbers are reused
across seasons (#6 returns both `HAD` and `GOE`), so it can name the wrong
driver.

*Refuse to publish a hole.* `ResultsFetcher._require_acronyms` now guards all
four extractors and raises, naming the unresolved driver numbers, rather than
serialising a `NaN`. A driver OpenF1 cannot name at all stops the round instead
of quietly mispaying it.

`data/raw/results/r12_zandvoort_result.json` was regenerated and now holds
`"HAD"` at P8 and parses strictly.

---

## 3. Pipeline crashes on Windows consoles (cp1252)

**Severity:** cosmetic — a workaround existed.

**Status:** fixed. Regression tests: `tests/test_console_encoding.py`.

### What was wrong

`SurveyIndex.print_summary` prints a `─` box-drawing rule
(`src/survey_index.py:140`). On a default Windows console the stdout encoding is
cp1252 and the pipeline died before doing any work:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-59
```

`src/tips_parser.py:373,379` and `src/tips_reader.py:119` print non-ASCII too,
so fixing only the first rule would just have moved the crash.

The workaround was to force UTF-8 by hand:

```bash
PYTHONIOENCODING=utf-8 python scripts/pipeline.py --round 12
```

### The fix

`run_pipeline` calls `_use_utf8_console()` before its first `print`, which
reconfigures `sys.stdout`/`sys.stderr` to UTF-8 with `errors="replace"`. That
covers every print in the pipeline rather than one rule at a time, keeps the
box-drawing output intact wherever the terminal can render it, and degrades to
replacement characters instead of an exception where it can't. A stream that
cannot be reconfigured at all (redirected, wrapped) is left alone.

---

## 4. The pre-race ladder was really the after-the-race ladder

**Severity:** high — it mis-scored the underdog multiplier in live results.

**Status:** fixed. Regression tests: `tests/test_pre_race_ladder.py`.

### What was wrong

`championship_drivers` carries two orderings per driver:

| field | meaning |
|-------|---------|
| `position_start` | the ladder **before** the session |
| `position_current` | the ladder **after** it |

The API returns its rows in `position_current` order, and
[`ResultsFetcher._get_championship_top10`](src/fetch_results.py) took
`championship_standings.head(10)` — so `championship_top10_before_race` was
populated with the ladder as it stood *after* the weekend's points had been
added. Nobody was picking against that ladder: tips lock before Practice 1.

`ResultAggregator` already asks for the weekend's **first** competitive session
(the sprint, where there is one), so `position_start` on those standings is
exactly the ladder the players saw. Checked against every 2026 round: a round's
`position_start` matches the previous round's `position_current` driver for
driver, the one exception being a countback shuffle between two drivers on zero
points, well outside the top 10.

### Why it matters

The underdog rule is set membership, not order — a driver outside the top 10
scores double. So a reordering inside the ten is harmless, but a driver crossing
the P10/P11 line during the weekend is scored against a ladder that did not
exist when tips were submitted. That happened in three rounds:

| Round | Wrongly in the ladder | Wrongly out |
|-------|------|------|
| 2 Shanghai | PIA | GAS |
| 3 Suzuka | PIA | LIN |
| 6 Monaco | HAD | BEA |

Piastri at Suzuka and Hadjar at Monaco were both widely picked and should have
paid double, so every player was **under**-scored. Shanghai's swap changed two
underdog flags but no points — neither driver was picked into a scoring slot.

**Score corrections** (rounds 3 and 6; all other rounds unchanged):

| Player | Was | Now | Player | Was | Now |
|--------|-----|-----|--------|-----|-----|
| Rino | 288 | 290 | Barry | 225 | 227 |
| Jake | 280 | 282 | Veljko | 222 | 224 |
| Luca | 264 | 266 | Tara | 212 | 213 |
| Dean | 261 | 263 | Riki | 159 | 161 |
| Antonio | 257 | 259 | Alex | 133 | 139 |
| Josh | 251 | 253 | | | |

**No ranking changed** — the corrections are near-uniform.

### The fix

`ResultsFetcher._pre_race_ladder` sorts by `position_start` and takes the first
ten, and insists the result is *complete*: it raises if the column is absent, if
any driver has a null `position_start`, or if the top ten is not exactly
positions 1-10 (a gap, a tie, or standings too short). Ordering by an explicit
field rather than trusting row order is the actual fix; the completeness checks
stop a partial ladder from quietly becoming a short one.

The choice of session is now commented at the fetch site in
[src/leaderboard.py](src/leaderboard.py), since it is load-bearing: reading
`position_start` off the *race* session of a sprint weekend would give the
post-sprint ladder.

All twelve rounds were re-fetched, re-scored and republished. Only
`championship_top10_before_race` changed in the raw results; only rounds 2, 3
and 6 changed in the scored files.
