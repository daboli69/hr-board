"""
Per-park wind sensitivity, learned from this season's own data.

The same 15 mph out-wind is a big deal at an open park and near-irrelevant at one
whose stands block the flow (or where the forecast never matches the field, e.g.
Oracle). Instead of hand-coding opinions, each park's daily HR-per-batted-ball is
regressed on that day's wind-out component (historical hourly weather via the free
Open-Meteo archive, one call per park per refresh). The park's slope relative to
the league slope becomes a multiplier applied to the wind vector in the physics
model — heavily shrunk toward 1.0 on thin samples, clamped, cached for a week.

Fails soft everywhere: any network / data problem -> {} -> every park uses 1.0
(exactly today's behavior).
"""
from __future__ import annotations
import json
import math
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from etl import park_geometry as PG
from etl.park_factors import TEAM_VENUE

TTL = 7 * 86400          # recompute weekly
SHRINK_DATES = 25.0      # pseudo-dates pulling a park toward sens 1.0
CLAMP = (0.15, 1.8)
GAME_HOUR_LOCAL = 19     # approx first pitch used for historical wind (statcast has no time)


def _archive_wind(lat, lon, start, end, timeout=15):
    """{date: (speed_mph, from_deg)} at ~GAME_HOUR local, via Open-Meteo archive."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "mph", "timezone": "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    h = data.get("hourly", {})
    out = {}
    for i, t in enumerate(h.get("time", [])):
        if t.endswith(f"T{GAME_HOUR_LOCAL:02d}:00"):
            d = t.split("T")[0]
            ws = (h.get("wind_speed_10m") or [None])[i]
            wd = (h.get("wind_direction_10m") or [None])[i]
            if ws is not None and wd is not None:
                out[d] = (float(ws), float(wd))
    return out


def compute_wind_sensitivity(df: pd.DataFrame) -> dict:
    """{venue_or_@abbr: sens} from the season frame + archived weather."""
    need = {"home_team", "game_date", "events", "launch_speed"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[df["launch_speed"].notna()].copy()
    d["hr"] = d["events"].eq("home_run")
    d["game_date"] = d["game_date"].astype(str).str[:10]

    rows = []                      # (abbr, x=wind_out, y=hr_rate, n_bb) per park-date
    per_park_pts = {}
    for abbr, grp in d.groupby("home_team"):
        venue = TEAM_VENUE.get(str(abbr))
        if not venue:
            continue
        lat, lon, _ = PG.park_coords(venue)
        cf = PG.cf_bearing(venue)
        if not lat or cf is None:
            continue
        daily = grp.groupby("game_date").agg(bb=("hr", "size"), hr=("hr", "sum"))
        daily = daily[daily["bb"] >= 20]
        if len(daily) < 8:
            continue
        try:
            wxs = _archive_wind(lat, lon, daily.index.min(), daily.index.max())
        except Exception:
            continue
        pts = []
        for date, r in daily.iterrows():
            w = wxs.get(date)
            if not w:
                continue
            spd, frm = w
            blow_to = (frm + 180.0) % 360.0
            x = spd * math.cos(math.radians(blow_to - cf))    # +out to CF, -in
            pts.append((x, r["hr"] / r["bb"], r["bb"]))
        if len(pts) >= 8:
            per_park_pts[str(abbr)] = pts
            rows.extend((str(abbr), *p) for p in pts)
    if not rows:
        return {}

    league_beta_num = league_beta_den = 0.0
    betas = {}
    for abbr, pts in per_park_pts.items():
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        w = np.array([p[2] for p in pts], float)
        xm = np.average(x, weights=w); ym = np.average(y, weights=w)
        den = float(np.sum(w * (x - xm) ** 2))
        if den <= 0:
            continue
        num = float(np.sum(w * (x - xm) * (y - ym)))
        betas[abbr] = (num / den, len(pts))
        league_beta_num += num
        league_beta_den += den
    if league_beta_den <= 0:
        return {}
    beta_l = league_beta_num / league_beta_den
    if beta_l <= 0:                 # league-wide wind-out must help HRs or signal is broken
        return {}

    out = {}
    for abbr, (beta, n) in betas.items():
        raw = beta / beta_l
        sens = (n * raw + SHRINK_DATES * 1.0) / (n + SHRINK_DATES)
        sens = float(np.clip(sens, *CLAMP))
        out["@" + abbr] = round(sens, 2)
        v = TEAM_VENUE.get(abbr)
        if v:
            out[v] = round(sens, 2)
    print(f"[wind_sens] learned sensitivity for {len(betas)} parks (league beta {beta_l:.2e})")
    return out


def load_wind_sensitivity(cache_path, df=None):
    """Weekly-cached load; always leaves a file so the commit can't fail; {} on any issue."""
    try:
        cached = json.load(open(cache_path))
        if time.time() - cached.get("ts", 0) < TTL and cached.get("sens"):
            return cached["sens"]
    except Exception:
        cached = None
    sens = {}
    if df is not None:
        try:
            sens = compute_wind_sensitivity(df)
        except Exception as e:
            print(f"[wind_sens] compute failed: {e}")
    if not sens and cached and cached.get("sens"):
        sens = cached["sens"]                    # stale beats nothing
    try:
        json.dump({"ts": time.time(), "sens": sens}, open(cache_path, "w"))
    except Exception:
        pass
    return sens


def sens_for(sens, venue, team=None):
    if not sens:
        return 1.0
    v = sens.get(venue)
    if v is None and team:
        v = sens.get("@" + str(team))
    return float(v) if v is not None else 1.0
