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

- **Heat chip** (0–100): ranks the four signals only. Blue (below) → orange (elite).
- **Metric strip**: the four signals in order — Pull% / EV / Brl% / IdealAA%. Top
  number is the **last 2 weeks**, shaded orange when it clears the "good" mark and
  blue when it's below the "poor" floor. Small line below = season · career.
- **Tap a row**: 2-week-vs-game-window trend, extra context metrics (bat speed,
  HH%, ISO, LA), the opposing arm + park (context, not in Heat), warning flags
  (e.g. "empty IAA — no EV behind it"), and the full Heat math.

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
