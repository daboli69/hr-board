"""
features.py — parallel feature-extraction edges, computed alongside (never touching) the
frozen HR heat model. Each is a reconstructed feature the market undersamples or can't buy
pre-built. Outputs attach to board rows under a `features` key; the frontend surfaces them
as context, and the tracker/backtest can later validate each in isolation via CLV.

Four edges:
  1. pitch_matchup      — hitter's pitch-family power AND swing-miss profile vs the pitch mix
                          the opposing starter actually throws (location-aware). Not BvP.
  2. microclimate       — hitter batted-ball quality bucketed by RECONSTRUCTED conditions at
                          contact (inning-interpolated game time -> hourly weather), so we can
                          ask "does his power hold as the temp drops late?" Approximate by
                          design — see the honesty note in reconstruct_pitch_conditions().
  3. reliever_fatigue   — cumulative, leverage-weighted workload index for each reliever over
                          the last 5 days, not a binary rest flag.
  4. late_hr_context    — pregame flag: elevated late-inning HR expectancy when a team's pen is
                          gassed and its starter is short. Builds on #3.

Design rules (same as props/runs):
  - Pure functions, no global state, degrade to None/{} cleanly on missing data.
  - Never import or modify compute.heat_score. These are parallel signals.
  - Every reconstructed feature is explicitly labeled approximate where it is.
"""

from __future__ import annotations
import math
import datetime as _dt

try:
    import pandas as pd
    import numpy as np
except Exception:      # keep import-safe in thin environments
    pd = None
    np = None

# reuse the family map from statcast_data so buckets stay consistent
try:
    from statcast_data import PITCH_BUCKET
except Exception:
    PITCH_BUCKET = {
        "FF": "FB", "FA": "FB", "SI": "FB", "FT": "FB", "FC": "FB",
        "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR", "CS": "BR", "SV": "BR", "SC": "BR", "KN": "BR",
        "CH": "OFF", "FS": "OFF", "FO": "OFF",
    }

FAMILIES = ("FB", "BR", "OFF")
_SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
_WHIFFS = {"swinging_strike", "swinging_strike_blocked"}


# ---------------------------------------------------------------------------
# 1. PITCH-TYPE MATCHUP
# ---------------------------------------------------------------------------

def hitter_pitch_profile(rows) -> dict:
    """Per pitch-family: the hitter's power (xwOBAcon) AND swing-miss (whiff rate).
    Returns {fam: {xwobacon, whiff, n_bb, n_sw}}. Location split (up/down) added when
    plate_z is present, since a hitter who mashes low fastballs but whiffs high is a
    different animal than the family average.
    """
    if pd is None or rows is None or rows.empty or "pitch_type" not in rows.columns:
        return {}
    w = rows.copy()
    w["fam"] = w["pitch_type"].map(PITCH_BUCKET)
    w = w[w["fam"].notna()]
    if w.empty:
        return {}
    desc = w["description"].astype(str) if "description" in w.columns else None
    out = {}
    for fam in FAMILIES:
        sub = w[w["fam"] == fam]
        if sub.empty:
            continue
        bb = sub[sub["launch_speed"].notna()] if "launch_speed" in sub.columns else sub.iloc[0:0]
        # power: mean xwOBA on contact
        xw = None
        if "estimated_woba_using_speedangle" in bb.columns and len(bb):
            vals = pd.to_numeric(bb["estimated_woba_using_speedangle"], errors="coerce").dropna()
            if len(vals):
                xw = round(float(vals.mean()), 3)
        # swing-miss: whiffs / swings
        whiff = None
        n_sw = 0
        if desc is not None:
            d = desc[sub.index]
            swings = d.isin(_SWINGS).sum()
            whiffs = d.isin(_WHIFFS).sum()
            n_sw = int(swings)
            if swings > 0:
                whiff = round(float(whiffs) / float(swings), 3)
        entry = {"xwobacon": xw, "whiff": whiff, "n_bb": int(len(bb)), "n_sw": n_sw}
        # location split on the vertical plane (up vs down) when available
        if "plate_z" in sub.columns:
            pz = pd.to_numeric(sub["plate_z"], errors="coerce")
            hi = sub[pz >= 2.8]      # upper third-ish (sz roughly 1.5-3.5 ft)
            lo = sub[pz < 2.2]
            def _xw(s):
                if not len(s) or "estimated_woba_using_speedangle" not in s.columns:
                    return None
                v = pd.to_numeric(s[s["launch_speed"].notna()]["estimated_woba_using_speedangle"], errors="coerce").dropna()
                return round(float(v.mean()), 3) if len(v) else None
            entry["xw_up"] = _xw(hi)
            entry["xw_dn"] = _xw(lo)
        out[fam] = entry
    return out


def pitcher_zone_grid(rows) -> dict:
    """3x3 zone grid of where a pitcher lives, using the Statcast `zone` field (1-9 in the
    strike zone, 11-14 chase). Returns {zone: usage_pct} over zones 1-9 plus a 'chase' bucket.
    Feeds the heatmap on the Edges tab. Zones map to a 3x3 grid:
        1 2 3   (up:    left/mid/right from catcher view)
        4 5 6   (mid)
        7 8 9   (down)
    """
    if pd is None or rows is None or rows.empty or "zone" not in rows.columns:
        return {}
    z = pd.to_numeric(rows["zone"], errors="coerce").dropna()
    if not len(z):
        return {}
    total = len(z)
    grid = {}
    for zone in range(1, 10):
        grid[str(zone)] = round(float((z == zone).sum()) / total, 3)
    grid["chase"] = round(float(z.isin([11, 12, 13, 14]).sum()) / total, 3)
    return grid


def pitcher_zone_damage(rows, min_n: int = 6) -> dict:
    """Where a pitcher gets HURT, by strike-zone cell (1-9) — the 'meatball zones'. Mirrors
    batter_zone_damage but for balls put in play against THIS pitcher: xwOBAcon allowed + barrel%
    allowed per cell. The darkest (highest-damage) cells are the meatballs a matching batter
    exploits. {zone: {xwobacon, barrel_pct, n}}.
    """
    if pd is None or rows is None or rows.empty or "zone" not in rows.columns:
        return {}
    if "launch_speed" not in rows.columns:
        return {}
    bb = rows[rows["launch_speed"].notna()].copy()
    if bb.empty:
        return {}
    bb["_z"] = pd.to_numeric(bb["zone"], errors="coerce")
    out = {}
    for zone in range(1, 10):
        sub = bb[bb["_z"] == zone]
        if len(sub) < min_n:
            continue
        xw = pd.to_numeric(sub.get("estimated_woba_using_speedangle"), errors="coerce").dropna()
        if not len(xw):
            continue
        entry = {"xwobacon": round(float(xw.mean()), 3), "n": int(len(sub))}
        if "launch_speed_angle" in sub.columns:
            lsa = pd.to_numeric(sub["launch_speed_angle"], errors="coerce")
            entry["barrel_pct"] = round(float(100.0 * (lsa == 6).sum() / len(sub)), 1)
        out[str(zone)] = entry
    return out


def zone_overlap(batter_hr_by_zone: dict, pitcher_dmg: dict,
                 meatball_thresh: float = 0.370, top_n_meatballs: int = 4) -> dict:
    """THE ZONE SIGNAL — the app's #1 read.

    Counts the HOME RUNS this batter has hit in the exact zones this pitcher is most hittable in.
    ZONE 5 means: he has hit 5 HRs in this pitcher's meatball zones.

      * meatball zones = the pitcher's most-damaged cells (xwOBA allowed >= meatball_thresh,
        capped at his top_n_meatballs worst cells so a bad pitcher doesn't flag all nine)
      * count = total HRs the batter hit in those cells
      * cells = the overlapping zones (each renders as one amber dot on the map)

    Thresholds: 3+ = minimum viable, 5+ = premium. Below 3 = no badge, no matchup.
    center_count weights the middle third (2,4,5,6,8) — HRs on meatballs over the heart of the
    plate are higher-confidence than corner overlaps.

    Returns {count, cells, hr_by_cell, center_count, badge, meatballs}.
    """
    if not batter_hr_by_zone or not pitcher_dmg:
        return {"count": 0, "cells": [], "hr_by_cell": {}, "center_count": 0,
                "badge": None, "meatballs": []}
    CENTER = {"2", "4", "5", "6", "8"}
    # rank the pitcher's cells by damage allowed; keep those over the bar, worst first
    ranked = []
    for zk, zv in (pitcher_dmg or {}).items():
        xw = zv.get("xwobacon") if isinstance(zv, dict) else zv
        if xw is not None and float(xw) >= meatball_thresh:
            ranked.append((str(zk), float(xw)))
    ranked.sort(key=lambda kv: -kv[1])
    meatballs = [zk for zk, _ in ranked[:top_n_meatballs]]

    hr_by_cell = {}
    total_hr = 0
    center = 0
    for zk in meatballs:
        n = batter_hr_by_zone.get(zk) or 0
        if n:
            hr_by_cell[zk] = int(n)
            total_hr += int(n)
            if zk in CENTER:
                center += int(n)
    cells = sorted(int(z) for z in hr_by_cell)
    badge = None
    if total_hr >= 5:
        badge = {"n": total_hr, "tier": "premium"}
    elif total_hr >= 3:
        badge = {"n": total_hr, "tier": "viable"}
    return {"count": total_hr, "cells": cells, "hr_by_cell": hr_by_cell,
            "center_count": center, "badge": badge,
            "meatballs": sorted(int(z) for z in meatballs)}


def batter_hr_zones(rows) -> dict:
    """Where a hitter's HOME RUNS actually came from, by strike-zone cell (1-9).
    This is the literal HR-location map that drives the ZONE signal: {zone: hr_count}.
    Unlike xwOBAcon (contact quality), this counts real home runs hit in each cell.
    """
    if pd is None or rows is None or getattr(rows, "empty", True):
        return {}
    if "zone" not in rows.columns or "events" not in rows.columns:
        return {}
    hr = rows[rows["events"].astype(str) == "home_run"]
    if hr.empty:
        return {}
    z = pd.to_numeric(hr["zone"], errors="coerce")
    out = {}
    for zone in range(1, 10):
        n = int((z == zone).sum())
        if n:
            out[str(zone)] = n
    return out


def batter_zone_damage(rows, min_n: int = 6) -> dict:
    """A hitter's contact quality by strike-zone cell (1-9). For a HR app we return not just
    xwOBAcon but the two most HR-predictive contact facts: barrel rate (launch_speed_angle==6)
    and average distance on air balls — so the UI can show whether damage in a zone is
    over-the-fence power or gap doubles. {zone: {xwobacon, barrel_pct, avg_dist, n}}.
    Requires min_n batted balls in a cell (default 6) so a 2-3 ball fluke doesn't read as
    'crushes this zone'. Joined against a pitcher's zone_grid.
    """
    if pd is None or rows is None or rows.empty or "zone" not in rows.columns:
        return {}
    if "launch_speed" not in rows.columns:
        return {}
    bb = rows[rows["launch_speed"].notna()].copy()
    if bb.empty:
        return {}
    bb["_z"] = pd.to_numeric(bb["zone"], errors="coerce")
    out = {}
    for zone in range(1, 10):
        sub = bb[bb["_z"] == zone]
        if len(sub) < min_n:
            continue
        xw = pd.to_numeric(sub.get("estimated_woba_using_speedangle"), errors="coerce").dropna()
        if not len(xw):
            continue
        entry = {"xwobacon": round(float(xw.mean()), 3), "n": int(len(sub))}
        # barrel rate — the single most HR-predictive contact event
        if "launch_speed_angle" in sub.columns:
            lsa = pd.to_numeric(sub["launch_speed_angle"], errors="coerce")
            entry["barrel_pct"] = round(float(100.0 * (lsa == 6).sum() / len(sub)), 1)
        # average distance on balls hit in the air (LA >= 10) — is the damage leaving the yard?
        if "hit_distance_sc" in sub.columns and "launch_angle" in sub.columns:
            la = pd.to_numeric(sub["launch_angle"], errors="coerce")
            dist = pd.to_numeric(sub["hit_distance_sc"], errors="coerce")
            air = dist[(la >= 10) & dist.notna()]
            if len(air):
                entry["avg_dist"] = int(round(float(air.mean())))
        out[str(zone)] = entry
    return out


def batter_vs_pitcher_zones(batter_zone: dict, pitcher_grid: dict) -> dict:
    """The join: how much does THIS batter punish the zones THIS pitcher lives in?
    Usage-weighted sum of (batter zone xwOBAcon) over the pitcher's zone distribution.
    Returns {score, hot_zone, hot_xw, pitcher_zone_usage}. This is the spatial version of
    the pitch-matchup edge — the number that ranks batters on the pitcher's drill-down.
    """
    if not batter_zone or not pitcher_grid:
        return {}
    num, wsum = 0.0, 0.0
    best_zone, best_val = None, -1
    for zone in [str(z) for z in range(1, 10)]:
        usage = pitcher_grid.get(zone, 0)
        bz = batter_zone.get(zone)
        if usage <= 0 or not bz or bz.get("xwobacon") is None:
            continue
        num += usage * bz["xwobacon"]
        wsum += usage
        # find the batter's best-punished zone that the pitcher actually throws to
        if bz["xwobacon"] * usage > best_val:
            best_val = bz["xwobacon"] * usage
            best_zone = zone
    if wsum <= 0:
        return {}
    weighted = num / wsum
    return {
        "score": round(weighted, 3),                 # weighted xwOBAcon in his zones
        "hot_zone": int(best_zone) if best_zone else None,
        "hot_xw": batter_zone.get(best_zone, {}).get("xwobacon") if best_zone else None,
        "pitcher_hot_usage": pitcher_grid.get(best_zone) if best_zone else None,
    }


def pitcher_arsenal(rows) -> dict:
    """The pitch mix a PITCHER actually throws: {fam: usage_pct}, plus vertical tendency
    (does he live up or down). This is what the hitter will see — the other half of the join.
    """
    if pd is None or rows is None or rows.empty or "pitch_type" not in rows.columns:
        return {}
    w = rows.copy()
    w["fam"] = w["pitch_type"].map(PITCH_BUCKET)
    w = w[w["fam"].notna()]
    if w.empty:
        return {}
    total = len(w)
    usage = {fam: round(float((w["fam"] == fam).sum()) / total, 3) for fam in FAMILIES}
    # average velocity per family — shown on the pitch-mix cards
    velo = {}
    if "release_speed" in w.columns:
        rs = pd.to_numeric(w["release_speed"], errors="coerce")
        for fam in FAMILIES:
            v = rs[(w["fam"] == fam)].dropna()
            if len(v) >= 10:
                velo[fam] = round(float(v.mean()), 1)
    up_pct = None
    if "plate_z" in w.columns:
        pz = pd.to_numeric(w["plate_z"], errors="coerce").dropna()
        if len(pz):
            up_pct = round(float((pz >= 2.8).sum()) / len(pz), 3)
    return {"usage": usage, "velo": velo or None, "n": int(total), "up_pct": up_pct}


def pitch_matchup(hitter_profile: dict, arsenal: dict, league_xwobacon: float = 0.360) -> dict:
    """Join a hitter's pitch-family profile to the pitcher's actual arsenal. Produces an
    edge score: usage-weighted expected damage vs the family the pitcher THROWS, contrasted
    with league-average contact. Positive = the pitcher's mix plays into this hitter's
    strengths; negative = the mix is where the hitter is weak.

    This is the market-invisible feature: not 'is he a good hitter' (priced) but 'does THIS
    pitcher's specific mix feed THIS hitter's specific strength' (not in any box score).
    """
    if not hitter_profile or not arsenal or not arsenal.get("usage"):
        return {}
    usage = arsenal["usage"]
    # power edge: usage-weighted (hitter xwobacon - league) over families the pitcher throws
    num_pow, num_whiff, wsum, wsum_wh = 0.0, 0.0, 0.0, 0.0
    covered = 0
    for fam, u in usage.items():
        hp = hitter_profile.get(fam)
        if not hp or u <= 0:
            continue
        if hp.get("xwobacon") is not None and hp.get("n_bb", 0) >= 15:
            num_pow += u * (hp["xwobacon"] - league_xwobacon)
            wsum += u
            covered += 1
        if hp.get("whiff") is not None and hp.get("n_sw", 0) >= 25:
            num_whiff += u * hp["whiff"]
            wsum_wh += u
    if wsum <= 0:
        return {}
    power_edge = num_pow / wsum                       # xwOBA points above/below league
    whiff_rate = (num_whiff / wsum_wh) if wsum_wh > 0 else None
    # location amplifier: if pitcher lives up and hitter mashes up (or vice versa)
    loc_amp = 0.0
    up_pct = arsenal.get("up_pct")
    fb = hitter_profile.get("FB") or {}
    if up_pct is not None and fb.get("xw_up") is not None and fb.get("xw_dn") is not None:
        # hitter's up-vs-down power skew, weighted by how much the pitcher lives up
        skew = fb["xw_up"] - fb["xw_dn"]              # +ve = better up
        loc_amp = skew * (up_pct - 0.5) * 2.0         # aligns when both up or both down
    # normalize to a 0-100 "matchup score" centered at 50 (neutral). ~0.08 xwOBA = big.
    score = 50 + (power_edge + loc_amp) * 300
    score = max(0, min(100, round(score, 1)))
    return {
        "score": score,
        "power_edge": round(power_edge + loc_amp, 3),    # xwOBA pts vs league on his mix
        "whiff_vs_mix": round(whiff_rate, 3) if whiff_rate is not None else None,
        "families_covered": covered,
        "pitcher_top_fam": max(usage, key=usage.get),
        "pitcher_top_usage": usage[max(usage, key=usage.get)],
    }


# ---------------------------------------------------------------------------
# 2. MICROCLIMATE (reconstructed, approximate)
# ---------------------------------------------------------------------------

def reconstruct_pitch_conditions(rows, game_start_map: dict, weather_hourly: dict) -> dict:
    """Bucket a hitter's batted balls by RECONSTRUCTED conditions at contact.

    HONESTY NOTE: Statcast per-pitch data has no reliable wall-clock timestamp exposed by
    pybaseball (sv_id is inconsistent). So we APPROXIMATE the time of each plate appearance:
      game_start + inning * ~18min. That places each ball within roughly the right hour, which
      is all the hourly-weather join needs. This is 'conditions during the inning', not
      'conditions at the exact pitch'. It is deliberately coarse and labeled as such.

    game_start_map: {game_pk: iso_start_time}
    weather_hourly: {game_pk: {hour_offset: {temp_f, wind_mph, wind_out}}}
    Returns {'warm': {...}, 'cool': {...}, 'late': {...}} damage splits + a temp-sensitivity read.
    """
    if pd is None or rows is None or rows.empty or "inning" not in rows.columns:
        return {}
    if not game_start_map or not weather_hourly:
        return {}
    w = rows.copy()
    bb = w[w["launch_speed"].notna()] if "launch_speed" in w.columns else w.iloc[0:0]
    if bb.empty:
        return {}
    # assign each batted ball an approximate temperature via inning offset
    temps, is_late = [], []
    for _, r in bb.iterrows():
        gp = r.get("game_pk")
        inn = r.get("inning")
        wx = weather_hourly.get(gp) or weather_hourly.get(str(gp))
        if wx is None or inn is None or (isinstance(inn, float) and math.isnan(inn)):
            temps.append(None); is_late.append(False); continue
        hour_off = min(int((int(inn) - 1) * 18 / 60), 4)   # ~18 min/inning, cap at +4h
        cell = wx.get(hour_off) or wx.get(str(hour_off)) or wx.get(0) or wx.get("0")
        temps.append(cell.get("temp_f") if cell else None)
        is_late.append(int(inn) >= 7)
    bb = bb.assign(_temp=temps, _late=is_late)
    valid = bb[bb["_temp"].notna()]
    if len(valid) < 20:
        return {}   # not enough reconstructed points to say anything

    def _dmg(sub):
        if not len(sub):
            return None
        ev = pd.to_numeric(sub["launch_speed"], errors="coerce").dropna()
        xw = pd.to_numeric(sub.get("estimated_woba_using_speedangle"), errors="coerce").dropna() \
            if "estimated_woba_using_speedangle" in sub.columns else pd.Series([], dtype=float)
        return {"avg_ev": round(float(ev.mean()), 1) if len(ev) else None,
                "xwobacon": round(float(xw.mean()), 3) if len(xw) else None,
                "n": int(len(sub))}

    med = valid["_temp"].median()
    warm = _dmg(valid[valid["_temp"] >= med])
    cool = _dmg(valid[valid["_temp"] < med])
    late = _dmg(valid[valid["_late"]])
    early = _dmg(valid[~valid["_late"]])
    # temperature sensitivity: how much does EV move warm->cool
    temp_sens = None
    if warm and cool and warm["avg_ev"] is not None and cool["avg_ev"] is not None:
        temp_sens = round(warm["avg_ev"] - cool["avg_ev"], 1)   # +ve = worse when cool
    return {
        "warm": warm, "cool": cool, "late": late, "early": early,
        "temp_sensitivity_ev": temp_sens,
        "approx": True,   # ALWAYS flag: this is inning-interpolated, not per-pitch
        "median_temp": round(float(med), 1),
    }


# ---------------------------------------------------------------------------
# 3. RELIEVER CUMULATIVE FATIGUE
# ---------------------------------------------------------------------------

def reliever_fatigue(appearances: list, as_of_date: str) -> dict:
    """Leverage-weighted cumulative workload over the trailing 5 days.

    appearances: list of {date, pitches, high_leverage(bool), back_to_back(bool)} for ONE
    reliever, most recent first. Each recent outing contributes fatigue that decays by day,
    amplified by pitch count and leverage. Returns a 0-100 fatigue index (higher = more
    gassed) plus the human-readable driver. This replaces the binary available/rested flag
    with a continuous stress measure the books don't model.
    """
    if not appearances:
        return {"index": 0, "state": "FRESH", "driver": "no recent work"}
    try:
        asof = _dt.date.fromisoformat(as_of_date[:10])
    except Exception:
        return {}
    load = 0.0
    drivers = []
    for a in appearances:
        try:
            d = _dt.date.fromisoformat(str(a.get("date"))[:10])
        except Exception:
            continue
        days_ago = (asof - d).days
        if days_ago < 0 or days_ago > 5:
            continue
        pitches = float(a.get("pitches") or 0)
        if pitches <= 0:
            continue
        # base fatigue from pitch count, normalized (25 pitches = a full inning-ish)
        base = pitches / 25.0
        # leverage amplifier: high-leverage innings cost more (adrenaline, max effort)
        lev = 1.35 if a.get("high_leverage") else 1.0
        # recency decay: today's work weighs full, 5 days ago barely
        decay = max(0.0, 1.0 - days_ago / 6.0)
        contribution = base * lev * decay
        load += contribution
        if days_ago <= 1 and pitches >= 20:
            drivers.append(f"{int(pitches)}p {days_ago}d ago")
        if a.get("back_to_back"):
            load += 0.4   # consecutive-day penalty
            drivers.append("back-to-back")
    # map cumulative load to 0-100 (load ~2.5+ = heavily worked)
    index = min(100, round(load / 3.0 * 100))
    if index >= 70:
        state = "GASSED"
    elif index >= 45:
        state = "WORN"
    elif index >= 20:
        state = "LIGHT"
    else:
        state = "FRESH"
    return {
        "index": index, "state": state,
        "driver": ", ".join(dict.fromkeys(drivers)) if drivers else "cumulative recent work",
    }


# ---------------------------------------------------------------------------
# 4. LATE-INNING HR CONTEXT (pregame flag, builds on #3)
# ---------------------------------------------------------------------------

def late_hr_context(starter_expected_ip: float, pen_fatigue_indices: list,
                    pen_hr_rate: float = None) -> dict:
    """Pregame flag for elevated LATE HR expectancy. When the starter is short (low expected
    IP) and the bullpen behind him is collectively gassed, innings 6-9 will feature tired,
    hittable arms the pregame HR line didn't fully price.

    pen_fatigue_indices: list of fatigue indexes (0-100) for the available pen arms.
    Returns a 0-100 'late HR boost' score + label. This is the reconstructed feature: the
    join of starter length x aggregate pen fatigue that no single stat exposes.
    """
    if starter_expected_ip is None or not pen_fatigue_indices:
        return {}
    # how much of the game the pen must cover
    pen_innings = max(0.0, 9.0 - float(starter_expected_ip))
    pen_share = min(1.0, pen_innings / 9.0)
    # aggregate pen fatigue (mean of available arms), 0-1
    avg_fatigue = sum(pen_fatigue_indices) / len(pen_fatigue_indices) / 100.0
    # the feature: pen must cover a lot AND the pen is tired
    boost = pen_share * avg_fatigue
    score = round(min(100, boost * 140), 1)   # scale so a bad combo lands ~70-90
    if score >= 65:
        label = "HIGH late-HR spot"
    elif score >= 40:
        label = "elevated late-HR"
    elif score >= 20:
        label = "mild late-HR lean"
    else:
        label = "neutral"
    return {
        "score": score, "label": label,
        "pen_innings_needed": round(pen_innings, 1),
        "pen_avg_fatigue": round(avg_fatigue * 100),
    }


# ---------------------------------------------------------------------------
# HR POWER PROFILE (parallel display lens — never touches the frozen heat model)
# ---------------------------------------------------------------------------

def hr_power_profile(rows) -> dict:
    """A HR-focused read on a hitter's raw batted-ball power, from the contact events most
    predictive of home runs. DISPLAY lens shown alongside heat — does NOT feed or modify the
    frozen 4-signal heat model.

    Returns {barrel_pct, hard_pct, avg_fb_dist, max_dist, hr_swing_pct, n}:
      barrel_pct   — % of batted balls barreled (launch_speed_angle==6). Strongest HR signal.
      hard_pct     — % hit 95+ mph (raw thump).
      avg_fb_dist  — average distance on fly balls (LA 20-40): the carry that clears fences.
      max_dist     — longest batted ball in the window (ceiling / jackpot relevance).
      hr_swing_pct — % of batted balls with a home-run-shaped swing (EV>=95 AND LA 20-35),
                     whether or not it left the yard.
    """
    if pd is None or rows is None or getattr(rows, "empty", True):
        return {}
    if "launch_speed" not in rows.columns:
        return {}
    bb = rows[rows["launch_speed"].notna()].copy()
    n = len(bb)
    if n < 15:
        return {}
    ev = pd.to_numeric(bb["launch_speed"], errors="coerce")
    la = pd.to_numeric(bb.get("launch_angle"), errors="coerce") if "launch_angle" in bb.columns else None
    dist = pd.to_numeric(bb.get("hit_distance_sc"), errors="coerce") if "hit_distance_sc" in bb.columns else None
    out = {"n": int(n)}
    if "launch_speed_angle" in bb.columns:
        lsa = pd.to_numeric(bb["launch_speed_angle"], errors="coerce")
        out["barrel_pct"] = round(float(100.0 * (lsa == 6).sum() / n), 1)
    out["hard_pct"] = round(float(100.0 * (ev >= 95).sum() / n), 1)
    if la is not None and dist is not None:
        fb = dist[(la >= 20) & (la <= 40) & dist.notna()]
        if len(fb):
            out["avg_fb_dist"] = int(round(float(fb.mean())))
    if dist is not None and dist.notna().any():
        out["max_dist"] = int(round(float(dist.max())))
    if la is not None:
        hr_swing = ((ev >= 95) & (la >= 20) & (la <= 35)).sum()
        out["hr_swing_pct"] = round(float(100.0 * hr_swing / n), 1)
    return out


# ---------------------------------------------------------------------------
# SQUARE UP RATING (parallel display lens — never touches the frozen heat model)
# ---------------------------------------------------------------------------

def square_up_rating(rows) -> dict:
    """A 0-100 quality-of-contact score: how consistently does this batter square the ball up?
    Built entirely from Statcast batted-ball data already pulled — Sweet Spot%, Barrel%, Hard
    Hit%, and average Exit Velo — blended so it reflects overall squaring-up, not any single
    metric. DISPLAY lens shown next to xwOBA/EV in the detail sheet; does NOT feed heat.

    Component scaling (each mapped 0-1, then weighted):
      Sweet Spot% (LA 8-32)   — 25%  (the launch window where damage happens)
      Barrel%                 — 30%  (the elite squared-up + right-angle events)
      Hard Hit% (95+ mph)     — 25%  (raw thump)
      Avg Exit Velo           — 20%  (consistency of contact quality)
    Returns {rating, sweet_spot_pct, barrel_pct, hard_pct, avg_ev, tier, n}.
    """
    if pd is None or rows is None or getattr(rows, "empty", True):
        return {}
    if "launch_speed" not in rows.columns:
        return {}
    bb = rows[rows["launch_speed"].notna()].copy()
    n = len(bb)
    if n < 15:
        return {}
    ev = pd.to_numeric(bb["launch_speed"], errors="coerce")
    la = pd.to_numeric(bb.get("launch_angle"), errors="coerce") if "launch_angle" in bb.columns else None
    sweet = float(((la >= 8) & (la <= 32)).sum()) / n if la is not None else None
    barrel = None
    if "launch_speed_angle" in bb.columns:
        lsa = pd.to_numeric(bb["launch_speed_angle"], errors="coerce")
        barrel = float((lsa == 6).sum()) / n
    hard = float((ev >= 95).sum()) / n
    avg_ev = float(ev.mean())

    # scale each to 0-1 against sensible MLB ranges
    def clamp(x): return max(0.0, min(1.0, x))
    s_sweet = clamp((sweet - 0.28) / 0.16) if sweet is not None else 0.5   # ~28% floor, ~44% elite
    s_barrel = clamp((barrel - 0.03) / 0.12) if barrel is not None else 0.5  # 3%..15%
    s_hard = clamp((hard - 0.28) / 0.22)                                   # 28%..50%
    s_ev = clamp((avg_ev - 86.0) / 8.0)                                    # 86..94 mph

    # reweight if sweet or barrel is missing so weights still sum to 1
    parts = [(s_sweet, 0.25, sweet is not None),
             (s_barrel, 0.30, barrel is not None),
             (s_hard, 0.25, True),
             (s_ev, 0.20, True)]
    wsum = sum(w for _, w, ok in parts if ok)
    rating = sum(s * w for s, w, ok in parts if ok) / wsum * 100 if wsum else 0
    rating = round(rating, 1)
    tier = ("Elite" if rating >= 75 else "Strong" if rating >= 60
            else "Average" if rating >= 45 else "Weak")
    out = {"rating": rating, "hard_pct": round(hard * 100, 1),
           "avg_ev": round(avg_ev, 1), "tier": tier, "n": int(n)}
    if sweet is not None: out["sweet_spot_pct"] = round(sweet * 100, 1)
    if barrel is not None: out["barrel_pct"] = round(barrel * 100, 1)
    return out


# ---------------------------------------------------------------------------
# PLATE DISCIPLINE (chase% + in-zone contact%) — parallel lens, never touches heat
# ---------------------------------------------------------------------------

_OUT_ZONES = {11, 12, 13, 14}     # Statcast zone codes for pitches OUTSIDE the strike zone
_IN_ZONES = set(range(1, 10))     # 1-9 = inside the strike zone

def plate_discipline_raw(rows) -> dict:
    """Per-batter chase% and in-zone contact% from pitch-level data.
      chase_pct    = swings at OUT-of-zone pitches / OUT-of-zone pitches seen (lower = better)
      zcontact_pct = contact on IN-zone swings / IN-zone swings (higher = better)
    Returns {chase_pct, zcontact_pct, oz_pitches, iz_swings} or {} if too little data.
    Percentiles are assigned separately (needs the league distribution).
    """
    if pd is None or rows is None or getattr(rows, "empty", True):
        return {}
    if "zone" not in rows.columns or "description" not in rows.columns:
        return {}
    z = pd.to_numeric(rows["zone"], errors="coerce")
    desc = rows["description"].astype(str)
    is_swing = desc.isin(list(_SWINGS))
    is_contact = desc.isin(["foul", "hit_into_play"])     # bat-on-ball (not swinging_strike)

    oz = z.isin(list(_OUT_ZONES))
    iz = z.isin(list(_IN_ZONES))
    oz_pitches = int(oz.sum())
    iz_swings = int((iz & is_swing).sum())
    out = {}
    if oz_pitches >= 20:
        out["chase_pct"] = round(float(100.0 * (oz & is_swing).sum() / oz_pitches), 1)
        out["oz_pitches"] = oz_pitches
    if iz_swings >= 20:
        out["zcontact_pct"] = round(float(100.0 * (iz & is_swing & is_contact).sum() / iz_swings), 1)
        out["iz_swings"] = iz_swings
    return out


def discipline_percentiles(all_raw: dict) -> dict:
    """Given {batter_id: {chase_pct, zcontact_pct,...}} for every batter, assign each a league
    percentile (0-100) for both metrics. chase is INVERTED (lower chase = higher percentile, so
    'higher is always better' like Savant). Adds chase_pctl / zcontact_pctl and the icon flags:
      eye        — chase_pctl >= 75  (elite discipline, rarely chases)
      crosshair  — zcontact_pctl >= 75 (elite in-zone contact)
      warning    — chase_pctl <= 25  (high chaser)
    Also returns the HR-probability multiplier the spec specifies (display/parlay only, never heat).
    """
    import numpy as _np
    chase_vals = sorted(v["chase_pct"] for v in all_raw.values() if v.get("chase_pct") is not None)
    zc_vals = sorted(v["zcontact_pct"] for v in all_raw.values() if v.get("zcontact_pct") is not None)

    def pctl(sorted_vals, x, invert=False):
        if not sorted_vals:
            return None
        # fraction of the league at or below x
        import bisect
        rank = bisect.bisect_right(sorted_vals, x) / len(sorted_vals)
        p = round(rank * 100)
        return round(100 - p) if invert else p

    out = {}
    for bid, v in all_raw.items():
        d = dict(v)
        if v.get("chase_pct") is not None:
            d["chase_pctl"] = pctl(chase_vals, v["chase_pct"], invert=True)   # low chase -> high pctl
        if v.get("zcontact_pct") is not None:
            d["zcontact_pctl"] = pctl(zc_vals, v["zcontact_pct"])
        eye = d.get("chase_pctl") is not None and d["chase_pctl"] >= 75
        crosshair = d.get("zcontact_pctl") is not None and d["zcontact_pctl"] >= 75
        warning = d.get("chase_pctl") is not None and d["chase_pctl"] <= 25
        d["eye"] = bool(eye)
        d["crosshair"] = bool(crosshair)
        d["warning"] = bool(warning)
        # HR-probability multiplier per the spec (parlay/display only): eye +6%, crosshair +5%,
        # warning -6%. Grade delta: +1 eye, +1 crosshair, -1 warning.
        mult = 1.0
        grade = 0
        if eye: mult *= 1.06; grade += 1
        if crosshair: mult *= 1.05; grade += 1
        if warning: mult *= 0.94; grade -= 1
        d["hr_mult"] = round(mult, 3)
        d["grade_delta"] = grade
        out[bid] = d
    return out


# ---------------------------------------------------------------------------
# BOMB SCORE — the composite batter-vs-pitcher matchup score (0-100)
# ---------------------------------------------------------------------------

def bomb_score(iso=None, slg=None, overlap_count=0, park_boost=None, platoon=None,
               pitcher_era=None, hot_streak=None, tto_score=None) -> dict:
    """Composite matchup score for ONE batter vs ONE pitcher, 0-100.

    Inputs (all optional — the score reweights over what's present):
      iso            batter recent ISO (power)
      slg            batter recent SLG
      overlap_count  ZONE overlap: amber dots vs this pitcher's meatballs (the headline signal)
      park_boost     today's park HR boost, % (e.g. +8)
      platoon        True if the batter has the handedness edge, False if not, None unknown
      pitcher_era    the opposing starter's season ERA (higher = more hittable)
      hot_streak     recent-form multiplier/trend (>1 = heating up)
      tto_score      pitcher's times-through-order vulnerability, 0-100

    Weighting favors ZONE overlap, which the spec calls the single most important signal.
    Returns {score, tier, parts:{...}} where tier is elite (>=65) / high (>=55) / mid / low.
    NOTE: parallel to the frozen heat model — this NEVER feeds heat.
    """
    def clamp(x):
        return max(0.0, min(1.0, x))

    parts = {}
    comps = []          # (value 0-1, weight)

    if iso is not None:
        v = clamp((iso - 0.120) / 0.180)          # .120 floor -> .300 elite
        parts["iso"] = round(v * 100)
        comps.append((v, 0.20))
    if slg is not None:
        v = clamp((slg - 0.350) / 0.250)          # .350 -> .600
        parts["slg"] = round(v * 100)
        comps.append((v, 0.10))
    # ZONE overlap — headline signal. 0 dots = 0, 3 = viable, 5+ = premium/max
    v_zone = clamp(overlap_count / 5.0)
    parts["zone"] = round(v_zone * 100)
    comps.append((v_zone, 0.25))
    if park_boost is not None:
        v = clamp((park_boost + 10.0) / 30.0)     # -10% -> +20%
        parts["park"] = round(v * 100)
        comps.append((v, 0.12))
    if platoon is not None:
        v = 1.0 if platoon else 0.35
        parts["platoon"] = round(v * 100)
        comps.append((v, 0.08))
    if pitcher_era is not None:
        v = clamp((pitcher_era - 2.80) / 2.40)    # 2.80 stingy -> 5.20 batting practice
        parts["era"] = round(v * 100)
        comps.append((v, 0.15))
    if tto_score is not None:
        v = clamp(tto_score / 100.0)
        parts["tto"] = round(v * 100)
        comps.append((v, 0.05))

    if not comps:
        return {}
    wsum = sum(w for _, w in comps)
    base = sum(val * w for val, w in comps) / wsum * 100

    # hot-streak bonus: up to +5 points when the bat is genuinely heating up
    bonus = 0.0
    if hot_streak is not None:
        try:
            bonus = max(0.0, min(5.0, (float(hot_streak) - 1.0) * 25.0))
        except (TypeError, ValueError):
            bonus = 0.0
    score = round(max(0.0, min(100.0, base + bonus)), 1)
    tier = ("elite" if score >= 65 else "high" if score >= 55
            else "mid" if score >= 40 else "low")
    return {"score": score, "tier": tier, "parts": parts,
            "hot_bonus": round(bonus, 1) if bonus else 0}


# ---------------------------------------------------------------------------
# HR VULNERABILITY SCORE — the pitcher-side 0-100 composite (ADDITIONAL score,
# parallel to the existing Statcast "get-shelled" model; neither feeds heat)
# ---------------------------------------------------------------------------

def vuln_score(era=None, whip=None, park_factor=None, hand_hr=None,
               zone_damage=None, danger_count=None) -> dict:
    """0-100 rating of how susceptible this pitcher is to giving up a home run TODAY.

    Point budget (per spec — each component contributes up to its max):
        ERA .................... 30 pts
        Park factor ............ 20 pts
        HR splits by hand ...... 15 pts
        WHIP ................... 15 pts
        Zone damage ............ 12 pts
        Dangerous batter count .. 8 pts
                                 ----
                                 100

    Args:
      era          season ERA (higher = more vulnerable)
      whip         season WHIP (higher = more traffic, more damage)
      park_factor  today's park HR factor (1.00 neutral; >1 boosts HR)
      hand_hr      {"R": {hr, pa}, "L": {hr, pa}} 2-yr HR allowed by batter hand —
                   we take the WORSE side, since the opponent can stack that hand
      zone_damage  {zone: {xwobacon, ...}} damage allowed by zone (the meatball map)
      danger_count how many genuinely dangerous bats he faces today

    Returns {score, tier, parts:{...}, missing:[...]}.
    tier: 'elite_target' (>=70) / 'strong' (>=50) / 'moderate' (>=30) / 'avoid'
    Components that are unavailable are reported in `missing` and their points are
    redistributed proportionally, so a partial score is still on a 0-100 scale.
    """
    def clamp(x):
        return max(0.0, min(1.0, x))

    parts = {}
    missing = []
    earned = 0.0        # points actually earned
    possible = 0.0      # points available given what we know

    # --- ERA: 30 pts. 2.80 (stingy) -> 5.50 (batting practice)
    if era is not None:
        v = clamp((float(era) - 2.80) / 2.70)
        pts = v * 30.0
        parts["era"] = {"value": round(float(era), 2), "pts": round(pts, 1), "max": 30}
        earned += pts; possible += 30.0
    else:
        missing.append("era")

    # --- Park factor: 20 pts. 0.90 (suppressing) -> 1.20 (launching pad)
    if park_factor is not None:
        v = clamp((float(park_factor) - 0.90) / 0.30)
        pts = v * 20.0
        parts["park"] = {"value": round(float(park_factor), 2), "pts": round(pts, 1), "max": 20}
        earned += pts; possible += 20.0
    else:
        missing.append("park")

    # --- HR splits by hand: 15 pts. Use the WORSE side's HR/PA rate (that's the exploitable one).
    if hand_hr:
        best_rate = None
        worst_side = None
        for side in ("R", "L"):
            s = (hand_hr or {}).get(side) or {}
            pa = s.get("pa") or 0
            if pa >= 60:                       # need real sample before trusting a split
                rate = (s.get("hr") or 0) / float(pa)
                if best_rate is None or rate > best_rate:
                    best_rate = rate; worst_side = side
        if best_rate is not None:
            v = clamp(best_rate / 0.045)       # 4.5% HR/PA allowed = maxed out
            pts = v * 15.0
            parts["hand_hr"] = {"value": round(best_rate * 100, 2), "side": worst_side,
                                "pts": round(pts, 1), "max": 15}
            earned += pts; possible += 15.0
        else:
            missing.append("hand_hr")
    else:
        missing.append("hand_hr")

    # --- WHIP: 15 pts. 1.00 (clean) -> 1.55 (constant traffic)
    if whip is not None:
        v = clamp((float(whip) - 1.00) / 0.55)
        pts = v * 15.0
        parts["whip"] = {"value": round(float(whip), 2), "pts": round(pts, 1), "max": 15}
        earned += pts; possible += 15.0
    else:
        missing.append("whip")

    # --- Zone damage: 12 pts. How many/how bad are his meatball cells.
    if zone_damage:
        xws = []
        for zk, zv in (zone_damage or {}).items():
            xw = zv.get("xwobacon") if isinstance(zv, dict) else zv
            if xw is not None:
                xws.append(float(xw))
        if xws:
            xws.sort(reverse=True)
            top = xws[:3]                       # his three worst cells
            avg_top = sum(top) / len(top)
            n_meat = sum(1 for x in xws if x >= 0.370)
            v_dmg = clamp((avg_top - 0.320) / 0.180)      # .320 -> .500
            v_cnt = clamp(n_meat / 4.0)                   # 4+ meatball cells = maxed
            v = 0.65 * v_dmg + 0.35 * v_cnt
            pts = v * 12.0
            parts["zone"] = {"worst_avg": round(avg_top, 3), "meatballs": n_meat,
                             "pts": round(pts, 1), "max": 12}
            earned += pts; possible += 12.0
        else:
            missing.append("zone")
    else:
        missing.append("zone")

    # --- Dangerous batter count: 8 pts. How many real threats in today's opposing lineup.
    if danger_count is not None:
        v = clamp(float(danger_count) / 5.0)    # 5+ dangerous bats = maxed
        pts = v * 8.0
        parts["danger"] = {"value": int(danger_count), "pts": round(pts, 1), "max": 8}
        earned += pts; possible += 8.0
    else:
        missing.append("danger")

    if possible <= 0:
        return {}
    # rescale to 0-100 over the components we actually have
    score = round(earned / possible * 100.0, 1)
    tier = ("elite_target" if score >= 70 else "strong" if score >= 50
            else "moderate" if score >= 30 else "avoid")
    return {"score": score, "tier": tier, "parts": parts,
            "missing": missing, "coverage": round(possible, 0)}


# ---------------------------------------------------------------------------
# MATCHUP GRADE — ELITE / STRONG / MOD / WEAK (the unifying per-batter grade)
# ---------------------------------------------------------------------------

def matchup_grade(iso=None, overlap_count=0, vuln_tier=None, park_factor=None,
                  hot_form=None, discipline_delta=0) -> dict:
    """Per-batter grade for TODAY'S specific matchup, by FACTOR CONVERGENCE (not a weighted
    average). Five factors either align or they don't; the grade reflects how many stack up:

        ELITE  — 5 factors aligned, with at least 2 at their premium level
        STRONG — 4+ factors aligned
        MOD    — 2-3 factors aligned
        WEAK   — 0-1

    Factors:
      1. ISO tier      HIGH (>=.200) aligns, ELITE (>=.250) is premium
      2. Zone overlap  3+ amber dots aligns, 5+ is premium
      3. Vuln tier     'strong' aligns, 'elite_target' is premium
      4. Park factor   >=1.05 aligns, >=1.10 is premium
      5. Hot bat       recent form trending up aligns, strongly up is premium

    discipline_delta (from the Eye/Crosshair/Warning icons) nudges the final grade one step
    up or down — an elite-eye hitter who forces pitchers into the zone gets the bump.
    Returns {grade, aligned, premium, factors:{...}, notes:[...]}.
    """
    factors = {}
    aligned = 0
    premium = 0
    notes = []

    # 1. ISO tier
    if iso is not None:
        if iso >= 0.250:
            factors["iso"] = "ELITE"; aligned += 1; premium += 1
            notes.append(f".{int(round(iso*1000)):03d} ISO (elite power)")
        elif iso >= 0.200:
            factors["iso"] = "HIGH"; aligned += 1
            notes.append(f".{int(round(iso*1000)):03d} ISO")
        elif iso >= 0.150:
            factors["iso"] = "MID"
        else:
            factors["iso"] = "LOW"
    else:
        factors["iso"] = None

    # 2. Zone overlap (the headline signal)
    if overlap_count >= 5:
        factors["zone"] = "PREMIUM"; aligned += 1; premium += 1
        notes.append(f"ZONE {overlap_count} (premium overlap)")
    elif overlap_count >= 3:
        factors["zone"] = "VIABLE"; aligned += 1
        notes.append(f"ZONE {overlap_count}")
    else:
        factors["zone"] = "NONE"

    # 3. Opposing pitcher vulnerability
    if vuln_tier == "elite_target":
        factors["vuln"] = "ELITE TARGET"; aligned += 1; premium += 1
        notes.append("elite-target arm")
    elif vuln_tier == "strong":
        factors["vuln"] = "STRONG"; aligned += 1
        notes.append("vulnerable arm")
    elif vuln_tier:
        factors["vuln"] = vuln_tier.upper()
    else:
        factors["vuln"] = None

    # 4. Park
    if park_factor is not None:
        if park_factor >= 1.10:
            factors["park"] = "BOOST+"; aligned += 1; premium += 1
            notes.append(f"{park_factor:.2f}x park")
        elif park_factor >= 1.05:
            factors["park"] = "BOOST"; aligned += 1
            notes.append(f"{park_factor:.2f}x park")
        elif park_factor <= 0.95:
            factors["park"] = "SUPPRESS"
        else:
            factors["park"] = "NEUTRAL"
    else:
        factors["park"] = None

    # 5. Hot bat form
    if hot_form is not None:
        try:
            hf = float(hot_form)
            if hf >= 1.15:
                factors["form"] = "HOT"; aligned += 1; premium += 1
                notes.append("bat is hot")
            elif hf > 1.02:
                factors["form"] = "WARM"; aligned += 1
                notes.append("trending up")
            elif hf < 0.95:
                factors["form"] = "COLD"
            else:
                factors["form"] = "STEADY"
        except (TypeError, ValueError):
            factors["form"] = None
    else:
        factors["form"] = None

    # base grade from convergence
    if aligned >= 5 and premium >= 2:
        grade = "ELITE"
    elif aligned >= 4:
        grade = "STRONG"
    elif aligned >= 2:
        grade = "MOD"
    else:
        grade = "WEAK"

    # plate-discipline nudge (Eye/Crosshair up, Warning down) — one step, never past the ends
    order = ["WEAK", "MOD", "STRONG", "ELITE"]
    if discipline_delta:
        i = order.index(grade)
        if discipline_delta > 0 and grade != "ELITE":
            # only promote when the hitter is already close (avoid inflating a WEAK spot)
            if aligned >= (3 if grade == "MOD" else 4):
                i = min(len(order) - 1, i + 1)
                notes.append("plate-discipline bump")
        elif discipline_delta < 0:
            i = max(0, i - 1)
            notes.append("chaser downgrade")
        grade = order[i]

    return {"grade": grade, "aligned": aligned, "premium": premium,
            "factors": factors, "notes": notes}


# ---------------------------------------------------------------------------
# LEAGUE-WIDE STATCAST PERCENTILES (true MLB percentiles, computed from the full
# season pull — no scraping). Feeds the Savant-style percentile card + discipline icons.
# ---------------------------------------------------------------------------

_PCTL_CATEGORIES = [
    # (key, label, higher_is_better)
    ("avg_ev",       "Exit Velocity",  True),
    ("barrel_pct",   "Barrel%",        True),
    ("hard_pct",     "Hard Hit%",      True),
    ("xwoba",        "xwOBA",          True),
    ("bb_pct",       "Walk Rate",      True),
    ("chase_pct",    "Chase Rate",     False),   # lower is better -> inverted
    ("whiff_pct",    "Whiff%",         False),   # lower is better -> inverted
    ("zcontact_pct", "Zone Contact%",  True),
    ("k_pct",        "K%",             False),   # lower is better -> inverted
]


def league_batter_stats(df, min_pa: int = 25, min_bbe: int = 15) -> dict:
    """Compute the nine Savant percentile categories for EVERY qualified batter in the pull.
    This is the whole league (400+ hitters), not just today's slate, so percentiles are true
    MLB percentiles rather than slate-relative.
    Returns {batter_id: {avg_ev, barrel_pct, hard_pct, xwoba, bb_pct, chase_pct, whiff_pct,
                         zcontact_pct, k_pct, pa, bbe}}
    """
    if pd is None or df is None or getattr(df, "empty", True):
        return {}
    need = {"batter", "events", "description"}
    if not need.issubset(df.columns):
        return {}

    PA_END = {"single", "double", "triple", "home_run", "field_out", "strikeout",
              "strikeout_double_play", "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
              "field_error", "grounded_into_double_play", "force_out", "double_play",
              "fielders_choice", "fielders_choice_out", "catcher_interf", "intent_walk",
              "triple_play", "sac_fly_double_play"}
    out = {}
    d = df
    desc = d["description"].astype(str)
    is_swing = desc.isin(list(_SWINGS))
    is_whiff = desc.isin(["swinging_strike", "swinging_strike_blocked"])
    is_contact = desc.isin(["foul", "hit_into_play"])
    z = pd.to_numeric(d["zone"], errors="coerce") if "zone" in d.columns else None
    ev = pd.to_numeric(d["launch_speed"], errors="coerce") if "launch_speed" in d.columns else None
    lsa = pd.to_numeric(d["launch_speed_angle"], errors="coerce") if "launch_speed_angle" in d.columns else None
    xw = pd.to_numeric(d.get("estimated_woba_using_speedangle"), errors="coerce") if "estimated_woba_using_speedangle" in d.columns else None

    work = pd.DataFrame({
        "batter": d["batter"],
        "_pa": d["events"].isin(list(PA_END)).astype(float),
        "_k": d["events"].isin(["strikeout", "strikeout_double_play"]).astype(float),
        "_bb": d["events"].isin(["walk", "intent_walk"]).astype(float),
        "_swing": is_swing.astype(float),
        "_whiff": is_whiff.astype(float),
    })
    if z is not None:
        work["_oz"] = z.isin([11, 12, 13, 14]).astype(float)
        work["_iz"] = z.isin(list(range(1, 10))).astype(float)
        work["_oz_swing"] = (work["_oz"] * work["_swing"])
        work["_iz_swing"] = (work["_iz"] * work["_swing"])
        work["_iz_contact"] = (work["_iz"] * is_swing.astype(float) * is_contact.astype(float))
    if ev is not None:
        work["_ev"] = ev
        work["_bbe"] = ev.notna().astype(float)
        work["_hard"] = (ev >= 95).astype(float) * ev.notna().astype(float)
    if lsa is not None:
        work["_brl"] = (lsa == 6).astype(float)
    if xw is not None:
        work["_xw"] = xw

    for bid, g in work.groupby("batter"):
        try:
            bid_i = int(bid)
        except (TypeError, ValueError):
            continue
        pa = float(g["_pa"].sum())
        if pa < min_pa:
            continue
        rec = {"pa": int(pa)}
        rec["k_pct"] = round(float(100.0 * g["_k"].sum() / pa), 1)
        rec["bb_pct"] = round(float(100.0 * g["_bb"].sum() / pa), 1)
        sw = float(g["_swing"].sum())
        if sw >= 30:
            rec["whiff_pct"] = round(float(100.0 * g["_whiff"].sum() / sw), 1)
        if "_oz" in g:
            ozp = float(g["_oz"].sum())
            if ozp >= 30:
                rec["chase_pct"] = round(float(100.0 * g["_oz_swing"].sum() / ozp), 1)
            izs = float(g["_iz_swing"].sum())
            if izs >= 30:
                rec["zcontact_pct"] = round(float(100.0 * g["_iz_contact"].sum() / izs), 1)
        if "_bbe" in g:
            bbe = float(g["_bbe"].sum())
            if bbe >= min_bbe:
                rec["bbe"] = int(bbe)
                rec["avg_ev"] = round(float(g["_ev"].mean()), 1)
                rec["hard_pct"] = round(float(100.0 * g["_hard"].sum() / bbe), 1)
                if "_brl" in g:
                    rec["barrel_pct"] = round(float(100.0 * g["_brl"].sum() / bbe), 1)
        if "_xw" in g:
            xv = g["_xw"].dropna()
            if len(xv) >= min_bbe:
                rec["xwoba"] = round(float(xv.mean()), 3)
        out[bid_i] = rec
    return out


def league_percentiles(league: dict) -> dict:
    """Turn raw league stats into 0-100 percentiles per category, Savant-style.
    Directional stats (chase, whiff, K%) are inverted so higher percentile is ALWAYS better.
    Returns {batter_id: {cat: {"value": v, "pctl": p, "label": name}}}
    """
    import bisect
    dists = {}
    for key, _label, _hib in _PCTL_CATEGORIES:
        vals = sorted(v[key] for v in league.values() if v.get(key) is not None)
        if vals:
            dists[key] = vals

    out = {}
    for bid, rec in league.items():
        cats = {}
        for key, label, higher_is_better in _PCTL_CATEGORIES:
            v = rec.get(key)
            if v is None or key not in dists:
                continue
            vals = dists[key]
            p = round(bisect.bisect_right(vals, v) / len(vals) * 100)
            if not higher_is_better:
                p = 100 - p
            cats[key] = {"value": v, "pctl": max(0, min(100, p)), "label": label}
        if cats:
            out[bid] = cats
    return out
