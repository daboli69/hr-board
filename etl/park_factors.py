"""
In-house empirical HR park factors, computed from the season's Statcast batted balls — the
same raw data Baseball Savant builds its park factors on. Zero manual entry, self-updating,
and it reflects each park's CURRENT configuration because it's this season's actual outcomes
(walls that moved this year show up immediately).

Method (controls for hitter quality so a good offense can't inflate its own park):
for every batted ball league-wide we know exit velocity, launch angle, and whether it left
the yard. We build an expected-HR baseline = the league HR rate for balls of that EV/LA.
A park's factor = actual HR / expected HR there — "balls of this quality became homers N x
the league rate in this park" — which isolates the park (short porch, altitude, marine
layer) from who was hitting. Split by batter hand, shrunk toward 1.0 on small samples.

Output: {venue_name: {"all": mult, "R": mult, "L": mult}}, 1.00 = league average.
"""
import json
import os
import time
import numpy as np
import pandas as pd

# Statcast home_team abbreviation -> venue name (matches the board's statsapi names).
# This is fixed plumbing (only changes when a team relocates), not hand-entered factors.
TEAM_VENUE = {
    "AZ": "Chase Field", "ARI": "Chase Field", "ATL": "Truist Park", "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park", "CHC": "Wrigley Field", "CWS": "Rate Field", "CHW": "Rate Field",
    "CIN": "Great American Ball Park", "CLE": "Progressive Field", "COL": "Coors Field",
    "DET": "Comerica Park", "HOU": "Daikin Park", "KC": "Kauffman Stadium", "KCR": "Kauffman Stadium",
    "LAA": "Angel Stadium", "LAD": "Dodger Stadium", "MIA": "loanDepot park",
    "MIL": "American Family Field", "MIN": "Target Field", "NYM": "Citi Field",
    "NYY": "Yankee Stadium", "OAK": "Sutter Health Park", "ATH": "Sutter Health Park",
    "PHI": "Citizens Bank Park", "PIT": "PNC Park", "SD": "Petco Park", "SDP": "Petco Park",
    "SEA": "T-Mobile Park", "SF": "Oracle Park", "SFG": "Oracle Park", "STL": "Busch Stadium",
    "TB": "George M. Steinbrenner Field", "TBR": "George M. Steinbrenner Field",
    "TEX": "Globe Life Field", "TOR": "Rogers Centre", "WSH": "Nationals Park", "WSN": "Nationals Park",
}

EV_BINS = np.arange(20.0, 126.0, 4.0)     # exit velocity bins (mph)
LA_BINS = np.arange(-40.0, 76.0, 4.0)     # launch angle bins (deg)
PARK_SHRINK = 12.0    # HR-equivalent pseudo-count pulling a park toward 1.0 on small samples
BIN_SHRINK = 60.0     # smooths the EV/LA baseline cells toward the global HR rate


def _xhr(ev, la, hr):
    """Expected-HR per batted ball = league HR rate in its EV/LA cell (smoothed)."""
    glob = float(hr.mean()) if len(hr) else 0.0
    tot, _, _ = np.histogram2d(ev, la, bins=[EV_BINS, LA_BINS])
    hrh, _, _ = np.histogram2d(ev[hr], la[hr], bins=[EV_BINS, LA_BINS])
    rate = (hrh + glob * BIN_SHRINK) / (tot + BIN_SHRINK)        # per-cell smoothed HR rate
    ix = np.clip(np.digitize(ev, EV_BINS) - 1, 0, rate.shape[0] - 1)
    iy = np.clip(np.digitize(la, LA_BINS) - 1, 0, rate.shape[1] - 1)
    return rate[ix, iy]


def compute_park_factors(df):
    """Build {venue: {all,R,L}} from a season Statcast frame. {} if data is too thin."""
    if df is None or len(df) == 0:
        return {}
    need = {"launch_speed", "launch_angle", "events", "home_team"}
    if not need.issubset(df.columns):
        print(f"[park_factors] missing columns {need - set(df.columns)}; skipping.")
        return {}
    d = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()  # batted balls only
    if len(d) < 5000:
        print(f"[park_factors] only {len(d)} batted balls; too thin, skipping.")
        return {}
    ev = d["launch_speed"].to_numpy(float)
    la = d["launch_angle"].to_numpy(float)
    hr = (d["events"] == "home_run").to_numpy()
    stand = d["stand"].to_numpy(str) if "stand" in d.columns else np.array(["R"] * len(d))
    park = d["home_team"].astype(str).to_numpy()
    xhr = _xhr(ev, la, hr)

    def _factors(mask):
        raw, xs = {}, {}
        for abbr in np.unique(park[mask]):
            sel = mask & (park == abbr)
            a = float(hr[sel].sum())
            x = float(xhr[sel].sum())
            if x <= 0:
                continue
            raw[abbr] = (a + PARK_SHRINK) / (x + PARK_SHRINK)   # shrunk toward 1.0
            xs[abbr] = x
        if not raw:
            return {}
        wsum = sum(xs.values())
        norm = sum(raw[k] * xs[k] for k in raw) / wsum if wsum else 1.0   # exposure-weighted mean
        if norm <= 0:
            norm = 1.0
        return {k: round(raw[k] / norm, 3) for k in raw}                  # league mean -> 1.00

    allm = _factors(np.ones(len(d), bool))
    rm = _factors(stand == "R")
    lm = _factors(stand == "L")

    venues = {}
    for abbr, m in allm.items():
        rec = {"all": m, "R": rm.get(abbr, m), "L": lm.get(abbr, m)}
        venue = TEAM_VENUE.get(abbr)
        if venue:
            venues[venue] = rec
        # durable key: the team abbreviation. Venue names drift (renames, temporary
        # parks, relocations) and a missed name silently zeroes the anchor; "@ABBR"
        # always resolves. Unknown abbrs are kept here instead of dropped.
        venues["@" + str(abbr)] = rec
    named = sum(1 for k in venues if not k.startswith("@"))
    print(f"[park_factors] computed {len(allm)} parks from {len(d)} batted balls "
          f"({named} name-keyed, all abbr-keyed, hand-split).")
    return venues


def load_park_factors(cache_path, df=None, year=None):
    """Compute from the season frame when given one (always, each build) and cache it.
    With no frame, read the last cache. Always leaves a file behind so the commit can't
    fail on a missing path. Returns {venue: {...}} ({} -> physics-only fallback)."""
    venues = {}
    if df is not None:
        try:
            venues = compute_park_factors(df)
        except Exception as e:
            print(f"[park_factors] compute failed: {e}")
            venues = {}
    if not venues and os.path.exists(cache_path):
        try:
            venues = json.load(open(cache_path)).get("venues", {})  # fall back to last good
            if venues:
                print(f"[park_factors] using {len(venues)} cached parks.")
        except Exception:
            venues = {}
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump({"year": year, "computed": time.time(), "method": "statcast_xhr",
                   "venues": venues}, open(cache_path, "w"))
    except Exception as e:
        print(f"[park_factors] cache write failed: {e}")
    return venues


def _normalize(name):
    return "".join(c for c in (name or "").lower() if c.isalnum())


def factor_for(venues, venue_name, hand="all", team=None):
    """HR multiplier for a venue + batter hand. Resolves by venue name, then normalized
    name, then the home team's abbreviation ("@ABBR" — durable across venue renames and
    temporary parks). 1.0 if unknown (neutral / physics-only)."""
    if not venues:
        return 1.0
    rec = venues.get(venue_name)
    if rec is None and venue_name:
        target = _normalize(venue_name)
        for k, v in venues.items():
            if not k.startswith("@") and _normalize(k) == target:
                rec = v
                break
    if rec is None and team:
        rec = venues.get("@" + str(team))
    if rec is None:
        return 1.0
    h = hand if hand in ("R", "L") else "all"
    return rec.get(h, rec.get("all", 1.0))
