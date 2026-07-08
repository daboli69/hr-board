"""
props.py — scoring for non-HR props (hits, HRR, strikeouts).

Explicit separation from the HR heat model: this file does NOT call anything
in compute.py and its outputs are never fed back into heat_score. Everything
here reads batter/pitcher profile dicts (from statcast_data) and produces
independent 0-100 scores per prop type.

Trailing-14-day methodology (same as HR heat):
  * Batter side reads the trailing 14-day `recent` window from batter_profiles
    (identical window the HR model uses). Thin-sample confidence discount
    (`_confidence`) pulls hitters with <40 tracked batted balls toward the
    league median.
  * Pitcher side reads the trailing 14-day `recent` window from pitcher_profiles
    via `_pitcher_2wk`. Because pitchers only start every 5-6 days, 14 days
    typically means 2-3 starts (~30-60 PA); when that recent sample is thin
    (<60 PA) the blender weights it against the season line proportional to
    confidence — same "prefer 2 weeks, fall back to season when thin" logic
    the HR pitcher score uses.
  * Anchor thresholds (elite / good / floor) are population-derived from
    league-wide distributions, NOT tuned to any tracker sample.

Prop types:
  hit_heat   — expected likelihood of getting a base hit (1H prop)
               Higher = better for OVER 0.5 hits.
  hrr_heat   — hits+runs+RBIs composite. Uses hit_heat as base + lineup-spot
               PA multiplier + HR upside boost (HR = 1H+1R+1RBI in one swing).
  k_heat     — expected strikeout likelihood. Higher = more likely to K.
               Bet OVER on Ks with high k_heat, UNDER with low k_heat.

Not touched by these functions:
  compute.heat_score, PITCH_WEIGHTS, or anything in the HR pipeline.
"""
from __future__ import annotations


# Population anchors — set from ~2015-2024 league distributions, not tracker data.
# These are the "elite / good / average" cutoffs each signal maps against.
_HIT_ANCHORS = {
    # batter side (recent form): higher is better
    "xba":         {"floor": 0.220, "good": 0.290, "elite": 0.340},
    "hardhit":     {"floor": 30.0,  "good": 42.0,  "elite": 50.0},
    "ld_pct":      {"floor": 18.0,  "good": 24.0,  "elite": 28.0},
    "bb_minus_k":  {"floor": -18.0, "good": -6.0,  "elite": 4.0},   # net plate discipline
    "contact":     {"floor": 76.0,  "good": 82.0,  "elite": 88.0},
    # pitcher side (season): worse for pitcher = better for hitter
    "opp_swstr":   {"floor": 15.0,  "good": 10.5,  "elite": 8.0},   # inverted: lower opp_swstr = better
    "opp_xba":     {"floor": 0.230, "good": 0.280, "elite": 0.320},
}

_K_ANCHORS = {
    # batter K prone: higher = more likely to K
    "k_pct":       {"floor": 15.0,  "good": 24.0,  "elite": 32.0},
    "swstr":       {"floor": 8.0,   "good": 12.0,  "elite": 16.0},
    "contact":     {"floor": 84.0,  "good": 78.0,  "elite": 72.0},  # inverted
    # pitcher K stuff: higher = more likely to induce K
    "opp_k_pct":   {"floor": 18.0,  "good": 24.0,  "elite": 30.0},
    "opp_swstr":   {"floor": 9.0,   "good": 12.0,  "elite": 15.0},
}


def _norm(val, anc, invert=False):
    """Map a value to 0-1 against three-point (floor/good/elite) anchors. Piecewise
    linear: below floor -> 0, floor->good -> 0..0.6, good->elite -> 0.6..1, above elite -> 1.
    invert=True flips: below floor gives 1, above elite gives 0 (for stats where
    LOWER means BETTER, e.g. opposing pitcher SwStr for a hitter)."""
    if val is None:
        return None
    floor, good, elite = anc["floor"], anc["good"], anc["elite"]
    if invert:
        # low is good: floor is the WORST, elite is BEST
        if val >= floor: return 0.0
        if val <= elite: return 1.0
        if val >= good:
            # floor..good -> 0..0.6
            return 0.6 * (floor - val) / (floor - good)
        # good..elite -> 0.6..1
        return 0.6 + 0.4 * (good - val) / (good - elite)
    else:
        if val <= floor: return 0.0
        if val >= elite: return 1.0
        if val <= good:
            return 0.6 * (val - floor) / (good - floor)
        return 0.6 + 0.4 * (val - good) / (elite - good)


def _confidence(bb_count):
    """Same discount as heat_score uses — thin samples get pulled toward the median."""
    if bb_count is None or bb_count < 5:
        return 0.5
    if bb_count >= 40:
        return 1.0
    return 0.5 + 0.5 * (bb_count - 5) / 35


def _pitcher_2wk(pprof):
    """Extract pitcher signals honoring the 2-week methodology, with confidence
    weighting toward season when the trailing 14-day sample is thin.

    Mirrors what the HR pipeline does: pitchers only pitch every 5-6 days, so 14 days
    of data = 2-3 starts = often 30-60 PA. When the recent sample is stable
    (>=60 PA) we use it straight; when thin we blend toward season proportional
    to confidence. Empty pprof or missing signals return None cleanly."""
    if not pprof:
        return {}
    recent = pprof.get("recent") or {}
    season = pprof.get("season") or {}
    recent_pa = recent.get("pa") or 0
    # confidence in 14-day recent form: 0 at pa<10, 1 at pa>=60
    conf = 0.0 if recent_pa < 10 else 1.0 if recent_pa >= 60 else (recent_pa - 10) / 50.0

    def blend(key):
        r = recent.get(key)
        s = season.get(key)
        if r is None and s is None:
            return None
        if r is None:
            return s
        if s is None:
            return r
        return round(conf * r + (1 - conf) * s, 3)

    return {
        "swstr_pct_allowed": blend("swstr_pct_allowed"),
        "k_pct_allowed":     blend("k_pct_allowed"),
        "bb_pct_allowed":    blend("bb_pct_allowed"),
        "ba_allowed":        blend("ba_allowed"),
        "xba_allowed":       blend("xba_allowed"),
        "ld_pct_allowed":    blend("ld_pct_allowed"),
        "recent_pa":         recent_pa,
        "recent_weight":     round(conf, 2),
    }


def hit_heat(batter_recent, pitcher_prof):
    """0-100 score for likelihood of at least one hit today. Returns (score, breakdown).
    Higher = better OVER 0.5 hits bet.

    Both sides use 14-day trailing form: batter_recent is the trailing 14-day batter
    profile (same window used by HR heat), pitcher_prof is the full pitcher profile
    dict (from statcast_data.pitcher_profiles) — this function pulls its 14-day
    subwindow via _pitcher_2wk and blends toward season only when the arm has thin
    recent-start volume."""
    if not batter_recent:
        return None, {}
    b = batter_recent
    p = _pitcher_2wk(pitcher_prof)
    signals = {
        "xba":         _norm(b.get("xba"),          _HIT_ANCHORS["xba"]),
        "hardhit":     _norm(b.get("hardhit_pct"),  _HIT_ANCHORS["hardhit"]),
        "ld_pct":      _norm(b.get("ld_pct_hit"),   _HIT_ANCHORS["ld_pct"]),
        "bb_minus_k":  _norm((b.get("bb_pct") or 0) - (b.get("k_pct") or 0),
                             _HIT_ANCHORS["bb_minus_k"]),
        "contact":     _norm(b.get("contact_pct"),  _HIT_ANCHORS["contact"]),
        "opp_swstr":   _norm(p.get("swstr_pct_allowed"), _HIT_ANCHORS["opp_swstr"], invert=True),
        "opp_xba":     _norm(p.get("xba_allowed"),  _HIT_ANCHORS["opp_xba"]),
    }
    # Weights: xBA carries most weight (best proxy for hits), then contact/discipline,
    # then pitcher hittability. All numbers derived from population plausibility, not tuned.
    weights = {"xba": 2.2, "hardhit": 1.4, "ld_pct": 1.3, "bb_minus_k": 1.2,
               "contact": 1.1, "opp_swstr": 1.4, "opp_xba": 1.4}
    numer = 0.0; denom = 0.0
    for k, v in signals.items():
        if v is None: continue
        w = weights[k]; numer += w * v; denom += w
    if not denom:
        return None, {"signals": signals}
    raw = numer / denom
    conf = _confidence(b.get("bb_count"))
    # Center un-known part at 0.45 (roughly league-median hit probability), pull toward it
    score = 100.0 * (0.45 + conf * (raw - 0.45))
    return round(max(0.0, min(100.0, score)), 1), {
        "signals": signals, "conf": conf, "raw": round(raw, 3),
        "pitcher_recent_weight": p.get("recent_weight"),
    }


def hrr_heat(batter_recent, pitcher_prof, lineup_spot=None, hr_heat=None):
    """0-100 score for hits+runs+RBIs. Layered on top of hit_heat:
      * hit skill (from hit_heat, 14-day both sides) is the base
      * lineup spot multiplier (top of order = more PAs = more R/RBI opps)
      * HR upside boost (an HR guarantees 1H+1R+1RBI in one swing)"""
    hh, _ = hit_heat(batter_recent, pitcher_prof)
    if hh is None:
        return None, {}
    # lineup PA multiplier: leadoff/2 get ~4.6 PA, 8/9 get ~3.5
    spot_mult = 1.0
    if lineup_spot:
        pa_est = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2, 6: 4.0, 7: 3.9, 8: 3.7, 9: 3.5}
        pa_est_med = 4.0
        m = pa_est.get(int(lineup_spot))
        if m: spot_mult = m / pa_est_med
    # HR upside — a hitter with high HR heat contributes to HRR via HRs
    hr_boost = 0.0
    if hr_heat is not None and hr_heat > 40:
        hr_boost = min(15.0, (hr_heat - 40) * 0.35)   # up to +15 points for elite HR guys
    score = hh * spot_mult + hr_boost
    return round(max(0.0, min(100.0, score)), 1), {
        "base_hit": hh, "spot_mult": round(spot_mult, 3), "hr_boost": round(hr_boost, 1),
    }


_PK_ANCHORS = {
    # pitcher's own K stuff (14-day trailing, blends toward season)
    "k_pct":       {"floor": 15.0,  "good": 24.0,  "elite": 30.0},
    "swstr":       {"floor": 8.0,   "good": 12.0,  "elite": 15.0},
    # opposing lineup K vulnerability (season, averaged across projected lineup)
    "opp_lineup_k":{"floor": 18.0,  "good": 24.0,  "elite": 28.0},
}


def pitcher_k_heat(pitcher_prof, opp_lineup_k_pct=None, opener=False):
    """0-100 score for pitcher-K-total prop (OVER 5.5 Ks, OVER 6.5 Ks, etc).
    Higher = more likely to hit the OVER on Ks.

    Signals (2-week methodology, same 60-PA confidence-blender pattern):
      * pitcher K% (his own strikeout rate, trailing 14 days blended)
      * pitcher SwStr% (leading indicator — batters missing his stuff)
      * opposing lineup K vulnerability (avg K% across opposing hitters)
      * volume flag — an opener throwing 1 IP tops has almost no path to
        a 5+ K game; his score gets multiplied down accordingly.

    Returns (score, breakdown) with the same shape as heat_score / hit_heat."""
    if not pitcher_prof:
        return None, {}
    p = _pitcher_2wk(pitcher_prof)
    signals = {
        "k_pct":         _norm(p.get("k_pct_allowed"),   _PK_ANCHORS["k_pct"]),
        "swstr":         _norm(p.get("swstr_pct_allowed"), _PK_ANCHORS["swstr"]),
        "opp_lineup_k":  _norm(opp_lineup_k_pct,          _PK_ANCHORS["opp_lineup_k"])
                         if opp_lineup_k_pct is not None else None,
    }
    weights = {"k_pct": 2.5, "swstr": 1.6, "opp_lineup_k": 1.4}
    numer = 0.0; denom = 0.0
    for k, v in signals.items():
        if v is None: continue
        w = weights[k]; numer += w * v; denom += w
    if not denom:
        return None, {"signals": signals}
    raw = numer / denom
    # Confidence from the recent-vs-season blend weight — a pitcher with 60+
    # recent PA gets the full score, one with 10 PA gets pulled toward league median.
    conf = 0.5 + 0.5 * (p.get("recent_weight") or 0.0)
    score = 100.0 * (0.40 + conf * (raw - 0.40))
    # Opener downgrade: a listed opener throwing 1 IP caps out at ~2 Ks
    # regardless of stuff; the K prop line is almost always a full-appearance
    # number, so an opener's chance of hitting O5.5 is structurally near zero.
    if opener:
        score = score * 0.35
    return round(max(0.0, min(100.0, score)), 1), {
        "signals": signals, "conf": round(conf, 2), "raw": round(raw, 3),
        "pitcher_recent_weight": p.get("recent_weight"),
        "opener": bool(opener),
    }


# Legacy hitter K score kept as internal helper — not surfaced on the Props tab
# (bettors bet pitcher K props, not hitter K props). Retained in case a future
# feature wants it as a Bear-side signal ("this hitter is likely to K vs this arm").
def k_heat_hitter(batter_recent, pitcher_prof):
    """0-100 score for likelihood the hitter strikes out this game. Kept as
    internal helper; not exposed in the UI. See pitcher_k_heat for the actual
    prop-facing K score."""
    if not batter_recent:
        return None, {}
    b = batter_recent
    p = _pitcher_2wk(pitcher_prof)
    signals = {
        "batter_k":    _norm(b.get("k_pct"),        _K_ANCHORS["k_pct"]),
        "batter_swstr":_norm(b.get("swstr_pct"),    _K_ANCHORS["swstr"]),
        "batter_ct":   _norm(b.get("contact_pct"),  _K_ANCHORS["contact"], invert=True),
        "pitcher_k":   _norm(p.get("k_pct_allowed"), _K_ANCHORS["opp_k_pct"]),
        "pitcher_ss":  _norm(p.get("swstr_pct_allowed"), _K_ANCHORS["opp_swstr"]),
    }
    weights = {"batter_k": 1.8, "batter_swstr": 1.5, "batter_ct": 1.3,
               "pitcher_k": 1.6, "pitcher_ss": 1.4}
    numer = 0.0; denom = 0.0
    for k, v in signals.items():
        if v is None: continue
        w = weights[k]; numer += w * v; denom += w
    if not denom:
        return None, {"signals": signals}
    raw = numer / denom
    conf = _confidence(b.get("bb_count"))
    score = 100.0 * (0.35 + conf * (raw - 0.35))
    return round(max(0.0, min(100.0, score)), 1), {
        "signals": signals, "conf": conf, "raw": round(raw, 3),
        "pitcher_recent_weight": p.get("recent_weight"),
    }
