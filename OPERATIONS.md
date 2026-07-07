# Going Yard — Operations Manual

The owner's manual for running, debugging, and eventually tuning this app.
Written at the end of the initial build; everything below was learned the hard way.

## The one rule

**The heat engine is frozen.** `compute.heat_score`, the four signals
(Pull% · EV · Barrel% · IdealAA%, trailing 14 days), their anchors, and
`PITCH_WEIGHTS` are the differentiator. Live tracking already shows the composite
beating every individual input (the signature of a well-built blend). Nothing tunes
until the tracker sample says so — see "When the data matures."

## Architecture in one breath

GitHub Actions cron → `etl/build_board.py` (Statcast pull → normalize → profiles →
heat → labels/badges/park/weather/robbed/milestones) → `docs/board.json` + daily
snapshot → static `docs/index.html` renders everything client-side. A second cron
(`grade-board`) backfills `docs/history.json` from snapshots (self-healing, 10-day
lookback). `backtest.yml` (manual) replays the season into `docs/backtest.json`.

## Debugging playbook (in order)

1. **`docs/board.json → build_health`** — every subsystem's skip reason + counts.
2. **`board.json → label_diag`** — label pipeline funnel, per-gate pass %, and the
   three nearest-miss hitters with the exact gates they failed.
3. **Actions log lines** — `[build] ...`, `[labels] ...`, `[track] ...` are all
   greppable statuses.
4. **The canary** (`tests/canary.py`, runs before every build): if it fails, the
   runner's pandas changed semantics again. That family of bug (nullable/Arrow
   dtypes) broke labels twice and the grader once; `normalize_frame()` in
   `statcast_data.py` is the choke point that closes it. Never bypass it.
5. **Tracker didn't grade?** Snapshots + lookback self-heal missed days once the
   cause is fixed; run the grade workflow manually to backfill immediately.
6. **A park shows neutral/blank?** Venue name drift. Geometry resolves via
   `park_geometry.canonical()` (substring match) and factors via `@ABBR` keys —
   a truly new stadium needs one row in `PARK_GEO` + `TEAM_VENUE`.

## The dials (current value → when to touch)

| Dial | Where | Value | Notes |
|---|---|---|---|
| SMASH bar | index.html `slateSmash` + build `_smash_score` | score ≥ 6.5, 3+ reasons, heat ≥ 55, top-3/day | Tune only from `by_smash` conversion after ~90 flags (~1 month). Keep JS/Python in sync — parity matters for grading. |
| Robbed floor | build park pass | 330 ft, LA 15–45 | If the Robbed Board is too chatty/quiet after a live week. |
| Near-HR proxy | `hitter_labels` | 325 ft, LA ≥ 15 | The one invented stat in the PF labels. Calibrate vs PF per-hitter using `label_diag.near_misses`. |
| Blast | `hitter_labels` | official Savant sliding scale (squared×100 + bat speed ≥ 164), per tracked BBE | Don't re-derive; this exact formula was the bug once. |
| Opener | `starter_lengths` consumers | median start ≤ 2.0 IP over 2+ starts, or 5+ apps with zero starts | Call-ups making a first start read as normal SPs — known edge. |
| Trend | `compute.trend` | 10-BBE floor, L15 confirmation, ±25% override | Display/conviction only — not heat. |
| Wind sensitivity | `wind_sens.py` | weekly, shrink 25 dates, clamp 0.15–1.8 | Learns itself; delete `docs/wind_sens.json` to force refresh. |
| Milestones | `career_hr_milestones` | within 5 of a 50-multiple | Cosmetic. |

## Honest approximations (documented, not bugs)

Near-HR and Blast are best-effort mirrors of PF's unpublished stats. The backtest
replays the **core** model only (actual starters, no park/weather layer) — treat its
edge as a floor. Roof calls are forecast heuristics. Model odds are the blend's fair
value, not simulated frequencies — the *disagreements* with book lines are the signal.

## When the data matures

- **~150 hitters graded in the 70+ tier** (~3 weeks): the tier's true rate is known
  well enough to size bets. Backtest can answer sooner — run it.
- **~150+ outcomes per tag**: the trust table earns the right to set parlay tag
  weights from data instead of priors.
- **~90 SMASH flags**: keep/raise/lower the 6.5 bar on evidence.
- Change **one dial at a time** and note the date — a mid-sample change muddies
  everything before it.

## Deploy hygiene

Push → run "Update HR Board" → hard-refresh with a `?v=` bump (UI-only changes skip
the workflow). Back up bets from the Bets tab occasionally — they live only in the
phone's localStorage. `history.json` and `docs/snapshots/` are the crown jewels;
they're in git, which is the backup.
