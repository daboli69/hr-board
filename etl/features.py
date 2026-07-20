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


def batter_zone_damage(rows) -> dict:
    """A hitter's xwOBAcon by strike-zone cell (1-9). Which zones does he punish?
    Returns {zone: {xwobacon, n}}. Joined against a pitcher's zone_grid to find the batters
    who most exploit exactly where that pitcher lives.
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
        if len(sub) < 3:
            continue
        xw = pd.to_numeric(sub.get("estimated_woba_using_speedangle"), errors="coerce").dropna()
        if len(xw):
            out[str(zone)] = {"xwobacon": round(float(xw.mean()), 3), "n": int(len(sub))}
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
    up_pct = None
    if "plate_z" in w.columns:
        pz = pd.to_numeric(w["plate_z"], errors="coerce").dropna()
        if len(pz):
            up_pct = round(float((pz >= 2.8).sum()) / len(pz), 3)
    return {"usage": usage, "n": int(total), "up_pct": up_pct}


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
