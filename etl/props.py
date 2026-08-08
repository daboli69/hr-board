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

try:
    from . import kengine, hitmodel
except ImportError:                       # standalone import (tests)
    import kengine, hitmodel


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
    # --- hit-specific additions ---
    # These target the HIT profile, which is close to the opposite of the HR profile: flatter
    # contact, whole-field usage, and an arm who lets the ball get put in play.
    "oppo":        {"floor": 18.0,  "good": 25.0,  "elite": 32.0},   # 24.6% is MLB average
    "spray":       {"floor": 55.0,  "good": 42.0,  "elite": 30.0},   # inverted: lower = more even
    "babip_la":    {"floor": 45.0,  "good": 58.0,  "elite": 68.0},   # share of contact -4..26 deg
    "opp_k":       {"floor": 27.0,  "good": 21.0,  "elite": 17.0},   # inverted: low-K arm = more BIP
    "opp_bb":      {"floor": 4.0,   "good": 7.5,   "elite": 10.0},   # inverted below
    "sprint":      {"floor": 26.0,  "good": 28.0,  "elite": 29.5},   # ft/s; 29+ beats out infield hits
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


def hit_prob_gated(batter_recent, pitcher_prof, shapes=None, spray_profile=None,
                   def_by_zone=None, sprint_speed=None, lineup_spot=None,
                   implied_team_total=None):
    """1+ hit probability via the contact gate, with the anchor model as fallback.

    Returns (probability, breakdown). The gate is sequential, not additive: a hitter cannot get
    a hit on a ball he never puts in play, so contact quality is only allowed to matter after
    the at-bat survives the whiff. The old weighted-sum model ranked a .380-xBAcon/35%-whiff
    hitter above a .320/16% hitter; gating correctly reverses that.
    """
    b = (batter_recent or {})
    p = (pitcher_prof or {}).get("season") or (pitcher_prof or {}).get("recent") or {}
    bat = {
        "whiff": (b.get("whiff_pct") or 0) / 100.0 or None,
        "k_pct": (b.get("k_pct") or 0) / 100.0 or None,
        "xba_con": b.get("xba"),
        "bb_pct": (b.get("bb_pct") or 0) / 100.0 or None,
    }
    pit = {
        "whiff": (p.get("swstr_pct_allowed") or 0) / 100.0 or None,
        "k_pct": (p.get("k_pct_allowed") or 0) / 100.0 or None,
        "xba_con": p.get("xba_allowed"),
        "bb_pct": (p.get("bb_pct_allowed") or 0) / 100.0 or None,
    }
    xpa = hitmodel.expected_pa(lineup_spot or 5, implied_team_total)
    try:
        res = hitmodel.contact_gated_hit_prob(
            bat, pit, shapes=shapes, spray_profile=spray_profile,
            def_by_zone=def_by_zone, sprint_speed=sprint_speed, xpa=xpa)
        return res["p_hit"], res
    except Exception as e:
        score, bd = hit_heat(batter_recent, pitcher_prof, sprint_speed=sprint_speed)
        return None, {"fallback": "anchor model", "hit_heat": score, "error": str(e), **bd}


def hit_heat(batter_recent, pitcher_prof, sprint_speed=None):
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
        # --- whole-field usage: beats defensive positioning, and the single strongest
        # batter-side feature in published hit models after xBA itself ---
        "oppo":        _norm(b.get("oppo_pct"),     _HIT_ANCHORS["oppo"]),
        "spray":       _norm(b.get("spray_score"),  _HIT_ANCHORS["spray"], invert=True),
        # --- the launch window that actually produces hits (-4 to 26 deg), which is flatter
        # than the HR power band. Without this the hit model quietly inherits the HR model's
        # preference for lift, which costs batting average. ---
        "babip_la":    _norm(b.get("babip_la_pct"), _HIT_ANCHORS["babip_la"]),
        "opp_swstr":   _norm(p.get("swstr_pct_allowed"), _HIT_ANCHORS["opp_swstr"], invert=True),
        "opp_xba":     _norm(p.get("xba_allowed"),  _HIT_ANCHORS["opp_xba"]),
        # --- opportunity: a hitter cannot single if the at-bat ends in a strikeout or a walk.
        # A low-K, low-walk arm puts the ball in play more often, which raises the ceiling on
        # hit chances before any contact-quality question is asked. ---
        "opp_k":       _norm(p.get("k_pct_allowed"),  _HIT_ANCHORS["opp_k"], invert=True),
        "opp_bb":      _norm(p.get("bb_pct_allowed"), _HIT_ANCHORS["opp_bb"], invert=True),
        # --- legs: turns topped and chopped ground balls into infield singles. Statcast folds
        # this into its own xBA for weak contact, and it is purely a HIT signal — it does
        # nothing for home runs. ---
        "sprint":      _norm(sprint_speed, _HIT_ANCHORS["sprint"]),
    }
    # xBA still carries the most weight — it is the closest single proxy for "did this become a
    # hit". Whole-field usage and the BABIP launch window are next: both are about beating the
    # defense rather than hitting the ball harder, which is what separates hit skill from power.
    weights = {"xba": 2.2, "hardhit": 1.0, "ld_pct": 1.2, "bb_minus_k": 1.1,
               "contact": 1.2, "oppo": 1.5, "spray": 1.0, "babip_la": 1.4,
               "opp_swstr": 1.2, "opp_xba": 1.3, "opp_k": 1.2, "opp_bb": 0.8,
               "sprint": 0.9}
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
def pitcher_k_projection(pitcher_csw, lineup_k_rates, arsenal=None, framing_runs=None,
                         pitch_limit=None, velo_drop=None, opener=False,
                         trailing_k_pct=None):
    """Expected strikeouts via CSW true-talent + dynamic xBF + arsenal-depth TTOP.

    Replaces the static 22-batters-faced projection. That constant was the largest single source
    of inflated K numbers: a 75-pitch arm at 4.3 pitches per PA faces 17 batters, not 22.
    Returns the kengine payload, including the exact Poisson-binomial distribution.
    """
    return kengine.predict_pitcher_k_count(
        pitcher_csw, lineup_k_rates, arsenal=arsenal, framing_runs=framing_runs,
        pitch_limit=pitch_limit, velo_drop=velo_drop, opener=opener,
        trailing_k_pct=trailing_k_pct)


def k_prob_at_line(dist, line):
    """P(Ks > line) from the exact distribution rather than a Poisson approximation."""
    return kengine.prob_over_ks(dist, line)


def hrr_projection(hit_result, lineup_spot, implied_team_total=None,
                   batters_ahead=None, batters_behind=None, hr_rate_per_pa=0.032):
    """Hits + Runs + RBIs: volume first (xPA scaled by the implied total), context second.

    The ordering is deliberate — no amount of lineup context rescues a hitter who only gets
    three trips to the plate, which is why bottom-of-the-order HRR tickets chronically
    underperform their rate stats.
    """
    return hitmodel.hrr_projection(
        hit_result, lineup_spot, implied_team_total=implied_team_total,
        batters_ahead=batters_ahead, batters_behind=batters_behind,
        hr_rate_per_pa=hr_rate_per_pa)


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
