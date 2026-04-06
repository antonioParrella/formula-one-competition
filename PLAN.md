# Plan: "Other Picks" — Binary/Contrarian Picks

## Context

Add a new category of picks alongside Main Race, Sprint, and DNFs. These are binary choices (e.g. "who finishes higher: NOR or LEC?" or "over/under 2.5 safety cars") where points are weighted by how uncommon the correct answer was — getting the minority pick right scores more. Not every race will have these.

---

## 1. Backend Design

### 1a. SurveyMars Question Structure

**Option A: Dedicated question indices (Recommended)**

Reserve question index range `70001`–`70010` for binary picks:
- `70001`, `70002` = option columns (e.g. `NOR` for pick, `LEC` for alternative, or `OVER`/`UNDER`)
- `70001` = the player's selection

Each binary pick is one "question" in SurveyMars with the player selecting one of two options.

```json
// Raw tips format extension
{
  "round": 3,
  "race_name": "Japan",
  "other_picks": [
    {
      "id": "head_to_head_1",
      "type": "who_higher",
      "optionA": "NOR",
      "optionB": "LEC",
      "player_picks": [
        { "player": "Luca", "pick": "NOR" },
        { "player": "Barry", "pick": "LEC" }
      ]
    }
  ]
}
```

**Pros:** Each pick is a self-contained question + two options. Easy to parse.
**Cons:** Requires specific survey question setup each race.

**Option B: Single question with structured answer**

One SurveyMars question `70000` where answers are strings like `NOR:HAM|OVER_2.5|UNDER_1.5`. Parse on our side.

**Pros:** Fewer survey questions to manage.
**Cons:** Fragile parsing, manual string format, error-prone.

**Recommendation: Option A** — cleaner, more robust, better automation.

### 1b. Scoring Logic

**Crowd-weighted scoring formula:**

```python
def score_binary(pick, correct_option, all_picks_for_this_question):
    # What fraction of players picked the CORRECT option
    correct_fraction = sum(1 for p in all_picks if p["pick"] == correct_option) / len(all_picks)

    if pick != correct_option:
        return 0  # Wrong = 0

    # Contrarian multiplier: the rarer the correct pick, the more it's worth
    # If 90% got it right → each gets 1pt
    # If 10% got it right → each gets 9pts
    # If 50/50 → each gets 2pts
    # Formula: 1 / correct_fraction (capped at a max)
    base_points = 1 / correct_fraction
    return min(base_points, 10)  # Cap at 10 to prevent runaway scores
```

**Example:** 10 players, question is "who finishes higher: NOR or ALB?"
- 8 players pick NOR, 2 players pick ALB
- ALB actually finishes higher
- The 2 who picked ALB each get `1 / 0.2 = 5 points`
- The 8 who picked NOR get 0

**Total points for a round = sum of all `other_picks` scores for that player.**

### 1c. Data Flow

```
SurveyMars API (question 70001-70010)
    ↓
TipsParser: detect "other_picks" question indices
    ↓  save to data/raw/tips/r{round}_{slug}_tips.json (extend existing file)
    ↓
Score round: read correct answers from results (or override file)
    ↓  calculate crowd-weighted scores
    ↓
Extend scored file format:
    data/processed/r{round}_{slug}_scored.json
    → add "other_picks" section with per-player scores + breakdown

Extend standings:
    → "other_picks" becomes part of total race score

Extend build_site.py:
    → _build_other_tips() → OTHER_TIPS JS array
```

### 1d. Providing Correct Answers

Since these can't be auto-resolved from OpenF1 (they're arbitrary questions), correct answers come from **the override file**:

```json
// data/overrides/r03_japan_overrides.json
{
  "round": 3,
  "note": "Japan GP other picks results",
  "other_picks_results": [
    { "id": "head_to_head_1", "correct": "LEC" },
    { "id": "over_under_1", "correct": "OVER" }
  ],
  "overrides": [ ... ]
}
```

This keeps it manual but **very fast** — just one JSON edit after the race, and the rest of the pipeline auto-scores.

### 1e. Sprint/Weekend Variability

Not every weekend will have other picks. `TipsParser._has_other_picks()` checks for question index `70001` presence. If absent, the section is simply omitted from that round's data.

---

## 2. New Data Structures

### Raw Tips Extension (in existing tips JSON)
```json
"other_picks": [
  {
    "id": "h2h_1",
    "type": "who_higher",
    "optionA": "NOR",
    "optionB": "LEC",
    "player_picks": [
      { "player": "Luca", "pick": "NOR" },
      ...
    ]
  }
]
```

### Processed Scored Extension
```json
"other_picks": [
  {
    "id": "h2h_1",
    "optionA": "NOR",
    "optionB": "LEC",
    "correct": "LEC",
    "player_scores": [
      { "player": "Luca", "pick": "NOR", "correct": false, "points": 0 },
      { "player": "Barry", "pick": "LEC", "correct": true, "points": 5 }
    ]
  }
]
```

### JavaScript OTHER_TIPS Array
```js
{
  round: 3,
  raceName: "Japan",
  picks: [
    {
      id: "h2h_1",
      type: "who_higher",
      optionA: "NOR",
      optionB: "LEC",
      correct: "LEC",
      results: [
        { player: "Luca", pick: "NOR", correct: false, points: 0 },
        { player: "Barry", pick: "LEC", correct: true, points: 5 }
      ]
    }
  ]
}
```

---

## 3. Frontend / UI Design

### Challenge: Mobile tabs are already cramped

**Current tabs:** Main Race | Sprint | DNFs | ???

**Option A: Dropdown selector instead of tab bar** (Recommended)

Replace the sub-tab bar with a dropdown/segmented control that doesn't scale horizontally:

```
┌──────────────────────────┐
│ ▼ Main Race              │  ← single dropdown
│   Sprint                 │
│   DNFs                   │
│   Other Picks            │
└──────────────────────────┘
```

On desktop: keep tabs. On mobile: replace with dropdown. This removes horizontal scaling entirely.

**Option B: Overflow scroll on mobile**

Keep tabs but make them horizontally scrollable with snap:
```
← swipe → [Main Race] [Sprint] [DNFs] [Other]
```

**Pros:** Minimal HTML/CSS change.
**Cons:** Still feels cramped on small screens, hidden options require swipe.

**Option C: Merge "Other" into an existing tab**

Put Other Picks inside the DNF tab as a second section below the DNF table (since both are "special" picks that don't happen every week).

**Pros:** No new tab needed.
**Cons:** Merges conceptually different things.

**Recommendation: Option A with dropdown** — cleanest on mobile, tabs stay on desktop.

### Cool UI for Other Picks

**Design: "Split vote" visualization**

For each binary question, show:

```
┌─────────────────────────────────────────┐
│ Who finishes higher?  NOR vs LEC        │
│ Winner: LEC ✓                           │
│                                         │
│ ━━━━━━━━━━░░░░░░░░                     │
│ NOR 80%     LEC 20%                      │
│ (8 players) (2 players)                  │
│   → 0 pts      → 5 pts each              │
│                                         │
│ Players who got it right: Barry, Luca    │
└─────────────────────────────────────────┘
```

Visual elements:
- **Split bar**: A horizontal bar divided proportionally — shows the crowd distribution at a glance. Option A colored one way, Option B another. The correct side gets a checkmark.
- **Point multiplier badge**: Shows the contrarian multiplier (e.g. "×5")
- **Player list**: Who picked what, color-coded (green = correct, red = wrong)

**Chart.js integration (optional):**
If you want to go further, add an "Other Picks" chart tab showing cumulative other-picks points per player over the season — reuse the existing chart infrastructure.

### Desktop View

```
Question                    │ Option A │ Option B │ Players who got it right
────────────────────────────┼──────────┼──────────┼──────────────────────────
Who finishes higher?        │ 80% ❌    │ 20% ✓    │ Barry, Luca
  NOR vs LEC                │ 0 pts     │ 5 pts    │

Over/under 2.5 safety cars  │ 40% ✓     │ 60% ❌    │ Antonio, Sarah
  Over vs Under             │ 2.5 pts   │ 0 pts    │
```

---

## 4. Files to Modify

| File | Change |
|---|---|
| `tips_parser.py` | Add `_has_other_picks()`, parse question indices 70001–70010 |
| `score_round.py` | Add `_score_other_picks()` with crowd-weighted logic, merge into totals |
| `aggregate.py` | Include other_picks scores in standings |
| `build_site.py` | Add `_build_other_tips()`, inject `OTHER_TIPS` into HTML |
| `docs/index.html` | New sub-tab/dropdown, render function for Other Picks, split bar CSS |

---

## 5. Implementation Phases

**Phase 1: Data foundation**
- Extend raw tips format in `TipsParser` to detect and save other_picks
- Manual override file format for providing correct answers
- Test: parse a mock other_picks survey, verify raw JSON

**Phase 2: Scoring**
- Crowd-weighted scoring logic in scorer
- Merge into existing scored files and standings
- Test: verify scoring math with known inputs

**Phase 3: Website build**
- `_build_other_tips()` in `build_site.py`
- Inject `OTHER_TIPS` data

**Phase 4: Frontend UI**
- Mobile dropdown selector (replace cramped tab bar)
- Split bar visualization for crowd distribution
- Per-player result display on desktop and mobile

---

## 6. Open Questions

1. **Do you want the dropdown UI on mobile even on desktop?** (I recommend keeping tabs on desktop, dropdown only on mobile)
2. **Should there be a max cap on contrarian points?** (Proposed: 10x max to prevent one question from dominating)
3. **Do you want cumulative "Other Picks" in the points chart?** Or a separate chart?
4. **What binary pick types do you envision for the first race?** (head-to-head, over/under, or others?)
5. **Should incorrect picks lose points or just score 0?** (Proposed: 0, not negative)
