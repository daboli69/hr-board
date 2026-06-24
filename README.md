# Going Yard — a free, zero-touch MLB home run board

A self-updating scouting board that ranks every hitter in today's lineups on the
four signals that drive home runs, **all measured over the last 2 weeks of play**:

1. **Pull%** — pulled fly balls / line drives (40%+ good; ~66% of HR are pulled)
2. **EV** — average exit velocity (90+ floor, 94+ elite)
3. **Barrel%** — the purest power-contact signal (80-86% of HR are barreled)
4. **Ideal Attack Angle%** — share of competitive swings in the 5-20 deg band (58%+ good)

Color-shaded so clearing the threshold is a glance — not a 20-minute video crawl.
Season and career sit beside each recent number for context. No subscriptions,
no manual data grabbing.

## What it does

Every scheduled run:
1. Pulls today's slate, probable pitchers, and posted lineups from **MLB StatsAPI**
   (official, free, no key).
2. Pulls season Statcast from **Baseball Savant** via `pybaseball` (free, no key)
   and computes, with identical definitions, each hitter's **L5 / L15 / L30**
   recent windows and **season** line.
3. Pulls **career** rates from FanGraphs (Statcast era, 2015+).
4. Scores each hitter with a transparent **Heat** composite and writes `docs/board.json`.
5. The static board at `docs/index.html` reads that JSON.

Total cost: **$0/mo.** The only data that would cost money (live odds) is
deliberately left out — this is a data board, not a betting feed.

## Data sources (all free, all API/programmatic — nothing manual)

| Layer | Source | Key? |
|---|---|---|
| Schedule, probables, lineups, handedness | MLB StatsAPI | no |
| Recent + season contact quality | Baseball Savant (`pybaseball`) | no |
| Career rates | FanGraphs (`pybaseball`) | no |
| Park HR factors (by handedness) | static table in `etl/parks.py` | — |

## One-time setup (≈10 min)

1. Push this folder to a **public** GitHub repo (public = unlimited free Actions
   minutes; private gets 2,000/mo, also plenty).
2. **Settings → Pages →** Deploy from a branch → branch `main`, folder **`/docs`**.
   Your board will live at `https://<you>.github.io/<repo>/`.
3. **Settings → Actions → General →** Workflow permissions → **Read and write**.
4. **Actions tab → Update HR Board → Run workflow** to kick the first build.
   First run is slow (it downloads the season's Statcast once, then caches).

That's it. After that it refreshes itself at 11a / 2p / 5p / 6:30p ET so lineups
fill in automatically as teams post them. Share the Pages URL with friends —
a handful of readers won't dent any free tier.

## Run it locally to test first

```bash
pip install -r requirements.txt
python -m etl.build_board        # writes docs/board.json
cd docs && python -m http.server 8000   # open http://localhost:8000
```

The repo ships with a **sample `docs/board.json`** so the board renders
immediately, before any pull. The first real run overwrites it.

## Reading the board

- **Dense sortable grid**: one compact row per hitter showing all four signals
  (Pull% / EV / Brl% / IdealAA%) at a glance, threshold-shaded (orange clears the
  "good" mark, blue is below the floor). **Tap any column header** (HEAT / PULL /
  EV / BRL / IAA) to re-sort the whole slate by that metric — so you explore the
  data yourself instead of trusting the Heat order.
- **Filters**: Heat-floor chips (40+/55+/70+), per-game dropdown, search, hide-thin.


## Daily Tracker

A second view (top toggle) that grades the model against reality. The
**Grade HR Board** workflow runs each morning (6:30 AM ET) after games are
final: it reads that day's board, pulls actual HRs, classifies each as off the
**starting pitcher or the bullpen**, and joins results back to what we ranked.
It appends to `docs/history.json`, and the Tracker view shows:

- HR rate by **Heat tier** (does higher Heat actually homer more?)
- HR rate by **opposing-arm form** (do SHELLABLE / STEADY-BAD arms get taken deep?)
- **SP vs BP** split of the HRs
- a daily log

## Scoring notes

- **Hitter Heat** = six power signals on the last 2 weeks: Pull-air%, Barrel%,
  ISO, EV, Ideal AA%, SLG (ISO/SLG added as power-outcome confirmation), then
  nudged by the opposing arm's vulnerability. Weights/anchors at the top of
  `etl/compute.py`.
- **Park factor is display-only** — it does NOT move the Heat score. Environment
  is shown for context; the data and matchup drive the ranking.
- **Pitcher form** is a level × trend matrix: SHELLABLE (bad + worsening),
  STEADY-BAD (consistently hittable — still a target), SLIPPING (was fine, now
  cracking), BAD-IMPROVING / BOUNCING-BACK (downgrade — underlying numbers ticking up),
  DEALING (avoid).

- **Tap a row**: full detail — recent·season·career for every signal, 2wk/L5/L30
  trend, context metrics, and the opposing arm's full HR-vulnerability breakdown
  with red flags.

## Tuning (one place each)

- **Signal thresholds (poor/good/elite) and weights:** top of `etl/compute.py` (`ANCHORS`, `WEIGHTS`).
- **Park factors:** `etl/parks.py`.
- **Recent window length, season start:** env vars in the workflow / `build_board.py`.

## Honest notes

- `pybaseball` scrapes Savant/FanGraphs rather than hitting a keyed REST API.
  It's free and reliable but can occasionally rate-limit; the ETL fails soft
  (a bad column degrades to `–`, the board still publishes).
- Lineups post a few hours before first pitch, so the morning run shows
  probables and any early lineups; later runs fill the rest in.
- Career Statcast metrics only exist from 2015 on.

## Decision helpers (time-savers)

- **Top Plays** (panel on the Board): the strongest non-thin hitters not facing a
  DEALING arm, each with a **confirmation tier** (LOADED / STRONG / SOLID / LEAN —
  how many of the 6 signals cleared "good"), batting-order spot, and a one-line
  **why** (e.g. "52% air-pull · 14% brl · 91 EV · vs STEADY-BAD arm"). Scan this
  instead of 270 rows.
- **Stacks** (view toggle): every vulnerable arm with 2+ hot hitters (Heat 55+)
  facing him, sorted by Heat — built for grand-slam parlays / same-game pairing.
- **Batting-order spot** (#1-#9) shows on every row and confirms the lineup is posted.
- **Tier badge** on each hitter = confirmation strength at a glance, independent of Heat.

## Pitcher platoon splits & bullpen (added)

- **Pitcher vs RHB / LHB**: each arm's HR-vulnerability is now split by batter
  handedness (from Statcast `stand`). The hitter detail shows vs-RHB and vs-LHB
  allowed rates and highlights the side matching *this* hitter — so you can see if
  the arm gets mashed by the hand your guy bats from.
- **Opposing bullpen**: every team's relievers (everyone who isn't the game's
  starter) are aggregated into a bullpen HR-vulnerability score + form, with the
  same RHB/LHB platoon split. Each hitter shows the opposing pen's overall score,
  its read vs that hitter's hand, and pen red flags. *Display-only for now* — not
  baked into Heat, since the starter is the primary matchup. Easy to weight later.
- **Handedness** shows as a colored badge (R orange / L blue / S purple) at the
  front of every name so it's always visible, plus a **Bats filter** (R / L / S).

