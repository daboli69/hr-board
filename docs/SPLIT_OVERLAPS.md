# Daily venue overlap boards

`etl.split_overlaps.build` runs inside the normal board build. It selects today's
game/player identities and opposing probable starters, including projected and
roster-only entries already supplied by Going Yard. Those statuses remain visible.
Doubleheaders retain separate game identities. No league-wide ranking is built.

The official All-Star game date determines the lower boundary; all statistics
exclude the slate date. MLB game logs provide exact counting stats and pitching
outs. Rates are recomputed from counts, never averaged from per-game rate fields.
Game logs are raw observations, not precomputed home/away splits. Only the
applicable venue is aggregated. No opposite-venue split is requested or built.
The existing daily Statcast pull supplies batted-ball and swing evidence; no
second season-wide request is added to the normal ETL. The targeted standalone
verification command does fetch its own frame.

## Fixed score version: venue-overlap-1

For each side, take the mean of two `rate/reference` ratios, each capped at 2:

| Family | Batter inputs | Pitcher allowed inputs |
|---|---|---|
| HR | ISO / .170; Barrel% / 8 | ISO / .170; Barrel% / 8 |
| Hit | AVG / .250; Contact% / 75 | AVG / .250; Contact% / 75 |
| HRR | OBP / .320; ISO / .170 | OBP / .320; ISO / .170 |

Strength = 100 × batter component × pitcher component (range 0–400).
The reference scales are fixed interpretive anchors, not fitted population
estimates. The score is an unvalidated research index, not an event probability.
Changing thresholds or sort never changes the score. Missing required evidence
produces a null score, excluded from qualifiers even at zero thresholds.

Sample confidence = 100 × min(PA/(PA+100), BF/(BF+100)). It measures sample depth,
not forecast accuracy or a statistical confidence interval. Raw PA, BF, IP,
batted-ball counts and Statcast PA remain available for coverage inspection.
Pitching IP is displayed as decimal innings, not baseball's outs notation.
Contact% uses contact swings / all swings; wOBA uses Savant's observed weighted
numerator and denominator. Barrel% uses classified batted balls. No missing
tracking measurements are imputed.

Pitch usage uses the last five official regular-season `gamesStarted == 1`
appearances before the slate date (including starts before the break if needed).
Each handedness denominator is its observed pitch count. Overall usage determines
pitch order. Fewer starts, missing coverage, and absent handedness samples are
shown explicitly. Pitch mix, lineup position and all other Going Yard signals
are excluded from scoring.

## Persistence and grading

`docs/snapshots/pregame/<date>/<game>.splits.json` freezes the first successful
pregame split observation, independently of the original `<game>.json` record.
It includes `split_overlaps`, all features, model version, three distinct signal
families, complete default-threshold ranks and threshold values, including null
ranks for nonqualifiers. Existing frozen records are never retrofitted. Tracking
begins with the first successful pregame split build after this release, even
when a baseline record already exists for that game.

`etl.grade_pregame` adds `split_signals` under `split_games` in `docs/pregame-results.json`, joining by
game and player and using official final boxscores: HR, H, and H+R+RBI. Missing
appearances remain null; they are not losses. The original features and thresholds
travel with each graded row. Scoring weights are never automatically retuned.

Client thresholds persist in localStorage. IndexedDB `going-split-research`
preserves full loaded snapshots and separate custom ranking observations keyed
by snapshot, thresholds and sort. No user data is uploaded. JSON export includes
the complete current feature and ranking snapshot. Server defaults and individual
device thresholds therefore remain distinguishable. Browser storage can be
cleared by the user/browser; export is the portable record.

Both Top Overlaps (cards) and All Qualifiers (table) expose the entire qualifying
field. Pitchers is a deduplicated per-game view, using the first matchup in the
active sort. Every displayed stat has a sort control; null values sort last.

## Targeted validation

```
python tests/test_split_overlaps.py
node --test tests/split-overlaps.test.cjs
python tests/test_pregame_records.py
python tests/test_frozen_grading.py
python -m etl.split_overlaps --output /path/to/validation.json
```
