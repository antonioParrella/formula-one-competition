# Known Issues

Bugs found but not yet fixed. Discovered while scoring round 12 (Dutch GP,
Zandvoort, 2026-08-23) on 2026-08-25.

---

## 1. `lastRace` and `bestRound` ignore sprint points

**Severity:** medium — wrong numbers on the public leaderboard, but season
totals and rankings are correct.

**Status:** open.

### What's wrong

[`Aggregator._build_standings`](src/aggregator.py) accumulates each round in two
passes. The first pass walks `player_tips` (the main race) and sets four fields:

```python
p["points"]    += main_score      # aggregator.py:90
p["lastRace"]   = main_score      # aggregator.py:91
if p["bestRound"] is None or main_score > p["bestRound"]:
    p["bestRound"] = main_score   # aggregator.py:92-93
p["roundScores"][str(round_num)] = {"race": race_name, "points": main_score}
```

The second pass walks `sprint_tips` and folds the sprint score in — but only
into two of them:

```python
p["roundScores"][round_key]["points"] += sprint_score   # aggregator.py:109
p["points"] += sprint_score                             # aggregator.py:111
```

`lastRace` and `bestRound` are never revisited. On any sprint weekend they keep
the main-race-only value.

### Why it matters

[`SiteBuilder`](src/build_site.py) publishes both fields to the leaderboard
(`build_site.py:114-115`), so the live site shows them.

`points` and `roundScores` **are** correct, so the season standings and the
finishing order are right. Only the two display columns are affected.

### Blast radius

Every sprint round in the 2026 season:

| Round | Race | Sprint points awarded |
|-------|-------------|-----|
| 2 | China | 50 |
| 4 | Miami | 35 |
| 5 | Canada | 35 |
| 9 | Silverstone | 30 |
| 12 | Dutch | 15 |

**`lastRace` after round 12** — understated for the three players who scored in
the R12 sprint:

| Player | Published | Correct |
|---------|-----|-----|
| Barry | 26 | 31 |
| Jake | 19 | 24 |
| Josh | 22 | 27 |

**`bestRound`** — wrong for 6 of 11 players, because each player's best round so
far happened to be a sprint weekend:

| Player | Published | Correct |
|---------|-----|-----|
| Rino | 49 | 54 |
| Jake | 41 | 46 |
| Dean | 53 | 58 |
| Veljko | 35 | 38 |
| Tara | 36 | 41 |
| Alex | 23 | 28 |

### Suggested fix

Compute the round total before touching any of the four fields, rather than
patching `points`/`roundScores` afterwards. Build a `{player: sprint_score}` map
from `sprint_tips` up front, then run a single pass that adds
`main_score + sprint_score` everywhere.

Note that `bestRound` is derived state — recomputing it as
`max(roundScores.values())` at serialisation time would remove the whole class
of bug.

Re-running `Aggregator().build_and_save()` after the fix rebuilds every round
from the scored files, so no rescoring is needed — but it **will** change
`bestRound` for the six players above on the published site.

---

## 2. `NaN` in `championship_top10_before_race`

**Severity:** low as it stands — round 12 is unaffected — but it can silently
corrupt underdog scoring, and it writes invalid JSON.

**Status:** open.

### What's wrong

[`ResultsFetcher._get_championship_top10`](src/fetch_results.py) left-merges the
championship standings onto the drivers table:

```python
merged = standings.merge(
    self.aggregator.drivers[["driver_number", "name_acronym"]],
    on="driver_number",
    how="left",
)                                     # fetch_results.py:149-153
return merged["name_acronym"].tolist()
```

A `how="left"` merge yields `NaN` for any championship driver absent from
`drivers`, and nothing checks for it.

For round 12 that happened. OpenF1's `drivers?session_key=11348` (the Zandvoort
sprint) returns 22 drivers and omits car #6 — Isack Hadjar — even though
`drivers?driver_number=6&meeting_key=1292` *does* report him present in session
11348. Upstream data inconsistency, but the merge swallows it.

Hadjar was **P8 in the championship**, so the saved top 10 reads:

```json
["ANT", "HAM", "RUS", "LEC", "NOR", "VER", "PIA", nan, "GAS", "LAW"]
```

### Two consequences

**Scoring.** `championship_top10_before_race` drives the underdog multiplier —
picks outside the top 10 score double. A `NaN` in the list means that driver is
no longer recognised as a top-10 driver, so anyone picking him would be paid at
the underdog rate.

Round 12 is **not** affected: nobody picked Hadjar, so the missing entry never
mattered. The scores in `data/processed/r12_dutch_scored.json` are correct as
published. Round 12 is also the only round in the season with this problem — the
other ten result files are clean.

**Invalid JSON.** Python's `json.dump` writes a bare `NaN` token, which is not
valid JSON. `data/raw/results/r12_zandvoort_result.json` fails any strict parser,
including JavaScript's `JSON.parse`. Nothing reads it strictly today, and it does
not leak into `docs/index.html`, but it is a trap for anything added later.

### Suggested fix

Fail loudly instead of silently. After the merge, check for nulls and either
raise, or fall back to a second lookup (`drivers?driver_number=N&meeting_key=…`
resolves Hadjar correctly). At minimum, never serialise a `NaN` into the results
file.

---

## 3. Pipeline crashes on Windows consoles (cp1252)

**Severity:** cosmetic — a workaround exists.

**Status:** open.

`SurveyIndex.print_summary` prints a `─` box-drawing character
(`src/survey_index.py:140`). On a default Windows console the stdout encoding is
cp1252 and the pipeline dies with `UnicodeEncodeError` before doing any work:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-59
```

**Workaround:** run with UTF-8 forced.

```bash
PYTHONIOENCODING=utf-8 python scripts/pipeline.py --round 12
```

**Fix options:** set `PYTHONIOENCODING` in `run.ps1`/`run`, call
`sys.stdout.reconfigure(encoding="utf-8")` at pipeline start, or use ASCII `-`
for the separator rules.
