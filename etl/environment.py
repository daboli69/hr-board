"""Environment & bullpen mechanics — geometry-aware near misses, air density, fatigue flags.

The flat 350-ft near-miss threshold this replaces was wrong in a way that mattered: 350 feet is
a routine fly to centre at almost every park and a home run down the line at several. It
counted the wrong balls as near misses at both ends. What actually defines "nearly out" is the
distance to the wall AT THE ANGLE THE BALL WAS HIT, which the app already has geometry for.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from . import park_geometry as PG          # package import (normal ETL run)
except ImportError:                            # standalone / test import
    import park_geometry as PG

# ---------------------------------------------------------------------------
# 1. Distance-to-Fence Delta
# ---------------------------------------------------------------------------

NEAR_MISS_FT = 5.0        # within 5 ft of the wall counts as a near miss

# The Statcast frame carries `home_team` (a 3-letter abbreviation), not a park name — but the
# fence geometry is keyed by park name. Without this bridge `near_miss_log` finds no venue on
# any row and silently returns nothing, which is exactly what produced "near-miss 0" on a
# build where every other geometry call worked.
TEAM_PARK = {
    "ARI": "Chase Field", "AZ": "Chase Field", "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards", "BOS": "Fenway Park", "CHC": "Wrigley Field",
    "CWS": "Rate Field", "CHW": "Rate Field", "CIN": "Great American Ball Park",
    "CLE": "Progressive Field", "COL": "Coors Field", "DET": "Comerica Park",
    "HOU": "Daikin Park", "KC": "Kauffman Stadium", "KCR": "Kauffman Stadium",
    "LAA": "Angel Stadium", "LAD": "Dodger Stadium", "MIA": "loanDepot park",
    "MIL": "American Family Field", "MIN": "Target Field", "NYM": "Citi Field",
    "NYY": "Yankee Stadium", "OAK": "Sutter Health Park", "ATH": "Sutter Health Park",
    "PHI": "Citizens Bank Park", "PIT": "PNC Park", "SD": "Petco Park", "SDP": "Petco Park",
    "SF": "Oracle Park", "SFG": "Oracle Park", "SEA": "T-Mobile Park",
    "STL": "Busch Stadium", "TB": "George M. Steinbrenner Field",
    "TBR": "George M. Steinbrenner Field", "TEX": "Globe Life Field",
    "TOR": "Rogers Centre", "WSH": "Nationals Park", "WSN": "Nationals Park",
}


def park_for_row(row, venue_col="venue"):
    """Resolve a park name from a Statcast row: explicit venue first, else the home team."""
    v = row.get(venue_col) if hasattr(row, "get") else None
    if v:
        return v
    ht = row.get("home_team") if hasattr(row, "get") else None
    return TEAM_PARK.get(str(ht).upper()) if ht else None


def spray_angle_deg(hc_x, hc_y, stand=None):
    """Statcast hit coordinates -> spray angle in degrees.

    Negative = left field, positive = right field, 0 = straightaway centre. This is the
    STADIUM frame, which is what the fence geometry is indexed by — deliberately not flipped
    to the batter's pull/oppo frame, because the wall does not care who is hitting.
    """
    if hc_x is None or hc_y is None:
        return None
    try:
        return float(np.degrees(np.arctan2(float(hc_x) - 125.42, 198.27 - float(hc_y))))
    except Exception:
        return None


def fence_delta(venue, hc_x, hc_y, hit_distance):
    """How far short of (or past) the wall this ball landed, at its own spray angle.

    Returns delta in feet: positive = cleared the wall's distance, negative = fell short.
    A delta of -3 at Fenway's 379-ft power alley is a near miss; the same ball is 30 ft short
    in Detroit's left-centre. A flat threshold cannot express that.
    """
    ang = spray_angle_deg(hc_x, hc_y)
    if ang is None or hit_distance is None:
        return None
    ang = max(-45.0, min(45.0, ang))
    try:
        wall, _ = PG.wall_at(venue, np.array([ang], dtype=float))
        wall_ft = float(np.asarray(wall, float)[0])
    except Exception:
        return None
    if not wall_ft or wall_ft != wall_ft:
        return None
    return round(float(hit_distance) - wall_ft, 1)


def is_near_miss(venue, hc_x, hc_y, hit_distance, was_hr=False, tol=NEAR_MISS_FT):
    """A batted ball that reached within `tol` feet of the wall but did not leave.

    Near misses are worth tracking because they are contact that WOULD have cleared in a
    different park, on a warmer night, or with the wind blowing the other way. That is the
    part of a hitter's profile most likely to convert next time.
    """
    if was_hr:
        return False
    d = fence_delta(venue, hc_x, hc_y, hit_distance)
    return d is not None and d >= -abs(tol)


def near_miss_log(df, batter_ids, venue_col="venue", days=14, asof=None):
    """{bid: {"near": n, "cleared": n, "avg_delta": x, "best_delta": x}} over a trailing window."""
    need = {"batter", "hc_x", "hc_y", "hit_distance_sc"}   # venue resolved via home_team
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[df["hit_distance_sc"].notna()]
    if days and asof and "game_date" in d.columns:
        cut = pd.to_datetime(asof) - pd.Timedelta(days=days)
        d = d[pd.to_datetime(d["game_date"], errors="coerce") >= cut]
    if d.empty:
        return {}
    wanted = {int(b) for b in batter_ids if b is not None}
    out = {}
    for bid, g in d.groupby("batter"):
        if int(bid) not in wanted:
            continue
        deltas, near, cleared = [], 0, 0
        for _, r in g.iterrows():
            venue = park_for_row(r, venue_col)
            if not venue:
                continue
            dd = fence_delta(venue, r.get("hc_x"), r.get("hc_y"), r.get("hit_distance_sc"))
            if dd is None:
                continue
            deltas.append(dd)
            is_hr = str(r.get("events")) == "home_run"
            if is_hr:
                cleared += 1
            elif dd >= -NEAR_MISS_FT:
                near += 1
        if deltas:
            out[int(bid)] = {
                "near": near, "cleared": cleared, "n": len(deltas),
                "avg_delta": round(float(np.mean(deltas)), 1),
                "best_delta": round(float(max(deltas)), 1),
            }
    return out


# ---------------------------------------------------------------------------
# 2. Atmospheric physics (Alan Nathan)
# ---------------------------------------------------------------------------

# Park elevations in feet. Altitude dominates air density far more than weather does — Coors
# sits about 20% thinner than sea level, which is worth roughly 25 feet of carry on a fly ball
# and is why it is the only park that meaningfully distorts a run environment on geometry alone.
PARK_ELEVATION_FT = {
    "Coors Field": 5200, "Chase Field": 1086, "Truist Park": 1050,
    "American Family Field": 635, "Great American Ball Park": 550, "PNC Park": 730,
    "Kauffman Stadium": 750, "Busch Stadium": 465, "Globe Life Field": 545,
    "Comerica Park": 585, "Guaranteed Rate Field": 595, "Rate Field": 595,
    "Wrigley Field": 595, "Progressive Field": 660, "Target Field": 815,
    "Angel Stadium": 160, "Dodger Stadium": 340, "Oracle Park": 15,
    "Petco Park": 20, "T-Mobile Park": 15, "Yankee Stadium": 55, "Citi Field": 20,
    "Fenway Park": 20, "Oriole Park at Camden Yards": 30, "Nationals Park": 25,
    "Citizens Bank Park": 20, "loanDepot park": 10, "Tropicana Field": 15,
    "Rogers Centre": 250, "Sutter Health Park": 30, "Daikin Park": 45,
    "Minute Maid Park": 45, "George M. Steinbrenner Field": 15,
}


def air_density(temp_f=70.0, elevation_ft=0.0, humidity_pct=50.0, pressure_inhg=None):
    """Air density in kg/m^3, following Nathan's treatment.

    Three inputs, in order of how much they matter: altitude (large), temperature (moderate),
    humidity (small, and counter-intuitively humid air is LIGHTER than dry air because water
    vapour is lighter than the nitrogen and oxygen it displaces).
    """
    T_c = (float(temp_f) - 32.0) * 5.0 / 9.0
    T_k = T_c + 273.15
    if pressure_inhg is None:
        # barometric formula from elevation
        p_sea = 29.92
        pressure_inhg = p_sea * math.exp(-float(elevation_ft) / 27000.0)
    p_pa = float(pressure_inhg) * 3386.39
    # saturation vapour pressure (Tetens), then partial pressure of water
    svp = 610.78 * math.exp(17.27 * T_c / (T_c + 237.3))
    pv = (float(humidity_pct) / 100.0) * svp
    pd_ = p_pa - pv
    R_d, R_v = 287.058, 461.495
    return (pd_ / (R_d * T_k)) + (pv / (R_v * T_k))


REF_DENSITY = air_density(70.0, 0.0, 50.0)      # sea level, 70F, 50% RH


def carry_multiplier(temp_f=70.0, elevation_ft=0.0, humidity_pct=50.0, pressure_inhg=None):
    """Fly-ball carry relative to a sea-level 70F reference.

    Drag scales with air density, so thinner air means a ball that decelerates less and lands
    farther. The exponent below is empirical: a 1% drop in density buys roughly 0.32% more
    distance on a well-struck fly ball, which is why Coors plays about 5-6% longer.
    """
    rho = air_density(temp_f, elevation_ft, humidity_pct, pressure_inhg)
    return round((REF_DENSITY / rho) ** 0.32, 4)


def carry_feet(distance_ft, temp_f=70.0, elevation_ft=0.0, humidity_pct=50.0):
    """Distance a ball would travel under given conditions, given its neutral distance."""
    return round(float(distance_ft) * carry_multiplier(temp_f, elevation_ft, humidity_pct), 1)


# ---------------------------------------------------------------------------
# 3. Bullpen fatigue
# ---------------------------------------------------------------------------

UNAVAILABLE, FATIGUED, AVAILABLE = "UNAVAILABLE", "FATIGUED", "AVAILABLE"

FATIGUE_K_PENALTY = 0.92        # -8% K%
FATIGUE_BB_PENALTY = 1.05       # +5% BB%
REPLACEMENT_WOBA = 0.360


def reliever_status(pitch_log):
    """Classify one reliever from his trailing appearances.

    pitch_log: [{"days_ago": 1, "pitches": 28}, ...]

    The thresholds encode what a manager will actually do, not what is physically possible:
    an arm at 35+ pitches yesterday is not coming in tonight regardless of how he feels, so
    modelling him as available is modelling a pitcher who will not appear.
    """
    if not pitch_log:
        return AVAILABLE, {}
    by_day = {}
    for e in pitch_log:
        d = int(e.get("days_ago", 99))
        by_day[d] = by_day.get(d, 0) + int(e.get("pitches") or 0)
    y = by_day.get(1, 0)
    last3 = sum(v for k, v in by_day.items() if 1 <= k <= 3)
    consec = 0
    for d in (1, 2, 3):
        if by_day.get(d, 0) > 0:
            consec += 1
        else:
            break
    detail = {"yesterday": y, "last3": last3, "consecutive_days": consec}
    if y >= 35 or consec >= 3 or last3 >= 55:
        return UNAVAILABLE, detail
    if consec >= 2 or y >= 25:
        return FATIGUED, detail
    return AVAILABLE, detail


def bullpen_state(relievers):
    """Aggregate an opponent bullpen into usable rates.

    relievers: [{"id":, "k_pct":, "bb_pct":, "xwoba":, "leverage": 1..3, "pitch_log": [...]}]

    UNAVAILABLE arms are dropped entirely — that is the mechanically correct move, because
    those innings genuinely fall to someone else. FATIGUED arms stay but pitch worse. The
    result is that losing two leverage arms does not just dilute the average slightly, it
    changes who is on the mound.
    """
    avail, out = [], []
    for r in (relievers or []):
        status, detail = reliever_status(r.get("pitch_log"))
        rec = dict(r)
        rec["status"], rec["fatigue_detail"] = status, detail
        if status == UNAVAILABLE:
            out.append(rec)
            continue
        if status == FATIGUED:
            if rec.get("k_pct") is not None:
                rec["k_pct"] = float(rec["k_pct"]) * FATIGUE_K_PENALTY
            if rec.get("bb_pct") is not None:
                rec["bb_pct"] = float(rec["bb_pct"]) * FATIGUE_BB_PENALTY
            if rec.get("xwoba") is not None:
                rec["xwoba"] = float(rec["xwoba"]) + 0.012
        avail.append(rec)

    if not avail:
        return {"k_pct": 0.205, "bb_pct": 0.095, "xwoba": REPLACEMENT_WOBA,
                "n_available": 0, "n_out": len(out), "fatigue": 1.0,
                "detail": [{"id": r.get("id"), "status": r["status"]} for r in out]}

    # weight by leverage: the arms a manager saves for the 8th matter more than the mop-up guy
    def wavg(key, default):
        num = den = 0.0
        for r in avail:
            v = r.get(key)
            if v is None:
                continue
            w = float(r.get("leverage") or 1.0)
            num += w * float(v)
            den += w
        return (num / den) if den else default

    n_fat = sum(1 for r in avail if r["status"] == FATIGUED)
    fatigue = min(1.0, (len(out) * 1.0 + n_fat * 0.5) / max(1.0, len(relievers or [1])))
    xw = wavg("xwoba", 0.315)
    # a depleted pen slides toward replacement — the innings are covered by worse arms
    xw = xw + fatigue * (REPLACEMENT_WOBA - xw) * 0.65
    return {
        "k_pct": round(wavg("k_pct", 0.225), 4),
        "bb_pct": round(wavg("bb_pct", 0.085), 4),
        "xwoba": round(xw, 4),
        "n_available": len(avail), "n_out": len(out),
        "n_fatigued": n_fat, "fatigue": round(fatigue, 3),
        "detail": [{"id": r.get("id"), "status": r["status"], **r.get("fatigue_detail", {})}
                   for r in (avail + out)],
    }


# ---------------------------------------------------------------------------
# 4. Directional defense proxy — no external OAA needed
# ---------------------------------------------------------------------------

SPRAY_BUCKETS = (("LF", -45.0, -15.0), ("CF", -15.0, 15.0), ("RF", 15.0, 45.0))


def directional_defense_proxy(df, days=30, asof=None, min_balls=60):
    """{TEAM: {"LF": delta, "CF": delta, "RF": delta}} — expected hits minus actual hits allowed.

    Built entirely from the Statcast frame already in memory: for every ball in play, Statcast's
    own xBA says how often that batted ball becomes a hit given its speed and angle. Summing xBA
    over a spray bucket gives the hits a league-average defense would allow; comparing that to
    the hits actually allowed isolates the fielders.

    POSITIVE delta = defense allowed FEWER hits than expected = good defense in that direction.
    Units are hits saved per 100 balls in play, which keeps parks with different batted-ball
    volumes comparable.
    """
    need = {"estimated_ba_using_speedangle", "events", "hc_x", "hc_y",
            "inning_topbot", "home_team", "away_team"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[df["estimated_ba_using_speedangle"].notna() & df["hc_x"].notna() & df["hc_y"].notna()]
    if days and asof and "game_date" in d.columns:
        cut = pd.to_datetime(asof) - pd.Timedelta(days=days)
        d = d[pd.to_datetime(d["game_date"], errors="coerce") >= cut]
    if d.empty:
        return {}
    d = d.copy()
    # the FIELDING team is the home team in the top half, away team in the bottom
    topbot = d["inning_topbot"].astype(str)
    d["_def"] = np.where(topbot.str.startswith("Top"), d["home_team"], d["away_team"])
    d["_ang"] = np.degrees(np.arctan2(d["hc_x"] - 125.42, 198.27 - d["hc_y"]))
    HITS = {"single", "double", "triple"}          # HRs excluded: no defender can field them
    ev = d["events"].astype(str)
    d = d[~ev.eq("home_run")]
    d["_hit"] = d["events"].astype(str).isin(HITS).astype(float)
    d["_xba"] = pd.to_numeric(d["estimated_ba_using_speedangle"], errors="coerce")

    out = {}
    for team, g in d.groupby("_def"):
        rec = {}
        for name, lo, hi in SPRAY_BUCKETS:
            b = g[(g["_ang"] >= lo) & (g["_ang"] < hi)]
            if len(b) < min_balls:
                continue
            exp_hits = float(b["_xba"].sum())
            act_hits = float(b["_hit"].sum())
            rec[name] = round(100.0 * (exp_hits - act_hits) / len(b), 2)
        if rec:
            out[str(team)] = rec
    return out


def defense_for_hitter(team_dir_def, spray_profile, bats="R"):
    """Map a team's LF/CF/RF defense onto a hitter's pull/centre/oppo tendencies.

    The buckets are in stadium coordinates; a hitter's spray profile is in his own frame. A
    right-handed batter pulls to LEFT field, a left-handed batter pulls to RIGHT — so the same
    strong left side is a problem for one and irrelevant to the other. Getting this mapping
    backwards would invert the adjustment, which is worse than not having it.
    """
    if not team_dir_def or not spray_profile:
        return None
    pull_field = "LF" if bats == "R" else "RF"
    oppo_field = "RF" if bats == "R" else "LF"
    return {
        "pull": team_dir_def.get(pull_field),
        "center": team_dir_def.get("CF"),
        "oppo": team_dir_def.get(oppo_field),
    }
