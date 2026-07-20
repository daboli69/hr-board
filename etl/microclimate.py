"""
microclimate.py — the horse-genetics edge, transposed to baseball.

Reconstructs the ACTUAL temperature at the moment of each batted ball by joining:
  1. MLB Stats API GUMBO feed  -> real per-plate-appearance wall-clock timestamp
     (statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live -> liveData.plays.allPlays[].about.startTime)
  2. Open-Meteo historical hourly weather (no signup, no key) at the ballpark lat/lon
     -> temperature for that exact hour

Then buckets each hitter's batted-ball quality (EV, xwOBA) by the real conditions at contact,
so we can ask the question no aggregator can answer: does THIS hitter's power hold up as the
temperature drops late in a night game? Some hitters are temperature-fragile; that fragility
is nowhere in any results database because temperature-at-contact has never been joined in.

This is genuine per-pitch time — NOT the inning interpolation used as a fallback elsewhere.
Statcast has no exposed timestamp; GUMBO does. That join is the whole edge.

Runs as its own ETL over completed games, building a persistent season-long profile per hitter
that build_board reads and attaches. Fails soft: any game or weather miss degrades to skipping
that game, never crashes the aggregate.
"""

from __future__ import annotations
import json
import time
import datetime as _dt
import urllib.request
import urllib.parse
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

GUMBO_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
# Open-Meteo historical/forecast archive — free, no key, hourly temperature.
WX_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
          "&start_date={d}&end_date={d}&hourly=temperature_2m,wind_speed_10m"
          "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=UTC")

OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "microclimate.json"

# ballpark coordinates (lat, lon) keyed by team abbreviation. Used for the weather join.
PARK_COORDS = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8907, -84.4677), "BAL": (39.2839, -76.6218),
    "BOS": (42.3467, -71.0972), "CHC": (41.9484, -87.6553), "CWS": (41.8299, -87.6338),
    "CIN": (39.0975, -84.5069), "CLE": (41.4962, -81.6852), "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485), "HOU": (29.7570, -95.3555), "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400), "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712), "MIN": (44.9817, -93.2776), "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262), "OAK": (37.7516, -122.2005), "ATH": (38.5570, -121.4680),
    "PHI": (39.9061, -75.1665), "PIT": (40.4469, -80.0057), "SD": (32.7073, -117.1566),
    "SF": (37.7786, -122.3893), "SEA": (47.5914, -122.3325), "STL": (38.6226, -90.1928),
    "TB": (27.7683, -82.6534), "TEX": (32.7473, -97.0842), "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}

_HTTP_HEADERS = {"User-Agent": "going-yard/1.0"}


def _get_json(url: str, retries: int = 2):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HTTP_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
    if last:
        print(f"[micro] fetch failed: {url[:70]}... ({last})")
    return None


def game_pks_for_date(date: str) -> list:
    """Completed game_pks for a date via the schedule endpoint."""
    data = _get_json(SCHED_URL.format(date=date))
    if not data:
        return []
    out = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            state = (g.get("status") or {}).get("abstractGameState")
            if state == "Final":
                out.append(g.get("gamePk"))
    return [p for p in out if p]


def _hourly_temp_map(lat: float, lon: float, date: str) -> dict:
    """{utc_hour_int: temp_f} for the ballpark on that date. One weather call per game."""
    data = _get_json(WX_URL.format(lat=lat, lon=lon, d=date))
    if not data or "hourly" not in data:
        return {}
    h = data["hourly"]
    times = h.get("time", [])
    temps = h.get("temperature_2m", [])
    winds = h.get("wind_speed_10m", [])
    out = {}
    for i, t in enumerate(times):
        try:
            hour = int(t[11:13])   # "YYYY-MM-DDTHH:MM" -> HH
            out[hour] = {
                "temp_f": temps[i] if i < len(temps) else None,
                "wind_mph": winds[i] if i < len(winds) else None,
            }
        except Exception:
            continue
    return out


def _home_abbr(feed: dict) -> str:
    try:
        return feed["gameData"]["teams"]["home"]["abbreviation"]
    except Exception:
        return None


def extract_batted_balls(feed: dict) -> list:
    """Walk GUMBO allPlays -> one record per BATTED BALL with its real UTC timestamp.
    Returns [{batter_id, batter_name, ts_hour(int UTC), ev, xwoba, is_hr, inning}]."""
    out = []
    try:
        plays = feed["liveData"]["plays"]["allPlays"]
    except Exception:
        return out
    for pl in plays:
        try:
            about = pl.get("about", {})
            matchup = pl.get("matchup", {})
            batter = matchup.get("batter", {})
            bid = batter.get("id")
            # timestamp: prefer the play endTime (contact happens near the end of the PA);
            # fall back to the last pitch event's startTime, then the play startTime.
            ts = about.get("endTime") or about.get("startTime")
            events = pl.get("playEvents", [])
            for ev in reversed(events):
                if ev.get("isPitch") and ev.get("startTime"):
                    ts = ev.get("startTime")   # the actual contact pitch
                    break
            if not ts or bid is None:
                continue
            # only batted balls: hitData present, or event implies contact
            hit = pl.get("result", {})
            hitData = None
            for ev in events:
                if ev.get("hitData"):
                    hitData = ev["hitData"]
            if hitData is None:
                continue   # not a batted ball (walk, K, etc.)
            ev_speed = hitData.get("launchSpeed")
            # parse the UTC hour
            try:
                hour = int(ts[11:13])
            except Exception:
                continue
            out.append({
                "batter_id": int(bid),
                "batter_name": batter.get("fullName"),
                "hour": hour,
                "ev": ev_speed,
                "is_hr": (hit.get("eventType") == "home_run"),
                "inning": about.get("inning"),
            })
        except Exception:
            continue
    return out


def process_game(pk: int, date: str) -> list:
    """One completed game -> batted balls tagged with real temperature at contact."""
    feed = _get_json(GUMBO_URL.format(pk=pk))
    if not feed:
        return []
    home = _home_abbr(feed)
    coords = PARK_COORDS.get(home)
    if not coords:
        return []
    wx = _hourly_temp_map(coords[0], coords[1], date)
    if not wx:
        return []
    balls = extract_batted_balls(feed)
    for b in balls:
        cell = wx.get(b["hour"])
        b["temp_f"] = cell.get("temp_f") if cell else None
        b["wind_mph"] = cell.get("wind_mph") if cell else None
    return [b for b in balls if b.get("temp_f") is not None]


def build_profiles(all_balls: list, min_n: int = 25) -> dict:
    """Aggregate batted balls into per-hitter temperature-sensitivity profiles.
    {batter_id: {name, warm:{avg_ev,hr_rate,n}, cool:{...}, temp_sensitivity_ev,
                 late:{...}, early:{...}, n_total}}
    warm/cool split at the hitter's own median contact temperature, so it's relative to the
    conditions HE plays in (a Marlins hitter's 'cool' differs from a Rockies hitter's).
    """
    if pd is None or not all_balls:
        return {}
    df = pd.DataFrame(all_balls)
    df = df[df["temp_f"].notna() & df["ev"].notna()]
    if df.empty:
        return {}
    out = {}
    for bid, g in df.groupby("batter_id"):
        if len(g) < min_n:
            continue
        med = g["temp_f"].median()

        def _dmg(sub):
            if not len(sub):
                return None
            return {
                "avg_ev": round(float(sub["ev"].mean()), 1),
                "hr_rate": round(float(sub["is_hr"].mean()), 4),
                "n": int(len(sub)),
            }

        warm = _dmg(g[g["temp_f"] >= med])
        cool = _dmg(g[g["temp_f"] < med])
        late = _dmg(g[g["inning"] >= 7]) if "inning" in g.columns else None
        early = _dmg(g[g["inning"] < 7]) if "inning" in g.columns else None
        sens = None
        if warm and cool and warm["avg_ev"] is not None and cool["avg_ev"] is not None:
            sens = round(warm["avg_ev"] - cool["avg_ev"], 1)   # +ve = worse when cool
        out[str(int(bid))] = {
            "name": g["batter_name"].iloc[0],
            "warm": warm, "cool": cool, "late": late, "early": early,
            "temp_sensitivity_ev": sens,
            "median_temp": round(float(med), 1),
            "n_total": int(len(g)),
        }
    return out


def run(start_date: str, end_date: str = None, existing: dict = None) -> dict:
    """Build (or extend) microclimate profiles over a date range. Idempotent per game via
    a processed-games set so re-runs only add new completed games."""
    end_date = end_date or start_date
    existing = existing or {}
    processed = set(existing.get("_processed_games", []))
    all_balls = existing.get("_balls", [])

    d0 = _dt.date.fromisoformat(start_date)
    d1 = _dt.date.fromisoformat(end_date)
    day = d0
    new_games = 0
    while day <= d1:
        ds = day.isoformat()
        for pk in game_pks_for_date(ds):
            if pk in processed:
                continue
            balls = process_game(pk, ds)
            all_balls.extend(balls)
            processed.add(pk)
            new_games += 1
            time.sleep(0.3)   # be polite to the free endpoints
        day += _dt.timedelta(days=1)

    profiles = build_profiles(all_balls)
    print(f"[micro] processed {new_games} new games, {len(all_balls)} batted balls, "
          f"{len(profiles)} hitter profiles")
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profiles": profiles,
        # keep raw balls + processed set so future runs extend rather than rebuild
        "_balls": all_balls,
        "_processed_games": sorted(processed),
    }


def _load_existing() -> dict:
    try:
        with open(OUT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _write(payload: dict):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    import os
    os.replace(tmp, OUT_PATH)


def main():
    import sys
    import os
    # default: extend from the last processed date through yesterday
    existing = _load_existing()
    start = os.environ.get("MICRO_START")
    if not start:
        # start the day after the latest processed, or 30 days back if fresh
        if existing.get("_processed_games"):
            start = (_dt.date.today() - _dt.timedelta(days=3)).isoformat()
        else:
            start = os.environ.get("SEASON_START",
                                   (_dt.date.today() - _dt.timedelta(days=30)).isoformat())
    end = os.environ.get("MICRO_END", (_dt.date.today() - _dt.timedelta(days=1)).isoformat())
    payload = run(start, end, existing)
    # slim the public copy: drop the raw balls from what the frontend loads (keep in a
    # separate state file), but write everything to OUT_PATH for idempotent re-runs.
    _write(payload)
    print(f"[micro] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
