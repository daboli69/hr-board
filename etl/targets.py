"""Team targets — which defenses are most likely to give up home runs tonight.

MULTIPLICATIVE aggregator, per spec:

    Target_Score_Raw = BASELINE_HR_RATE * SP_Factor**w_sp * Pen_Factor**w_pen
                        * Staff_Factor**w_staff * Park_Factor**w_park * Wx_Factor**w_wx

Exponents are the backtested macro weights (Starter 34 / Bullpen 26 / Staff HR/9 18 / Park 12 /
Weather 10) NORMALISED to exponent scale, not re-derived -- the shares are unchanged from the
weighted-mean version this replaces, only the combination rule changed from a weighted average to
a weighted geometric product. See W_* below for the exact numbers.

Target_Score_Raw is mapped onto 0-100 with a log curve anchored so all-neutral inputs (every
factor = 1.00x) land at 50 and the strongest lift the backtest has actually graded (the 4+
convergence-family band, ~1.94x) lands near 100 -- MAX_LIFT = 2.0.

This revision also enriches the STARTER factor (still one weighted term, still 34% of the
exponent budget) with three additional inputs that were sitting unused elsewhere in the ETL:

  1. xFIP-ERA gap                (a.fg.xfip vs the pitcher's ERA)      -- regression signal
  2. Fly-ball / ground-ball mix  (a.fb_pct / a.gb_pct)                 -- HR opportunity signal
  3. Platoon alignment           (pitcher_edges.hand_splits vs lineup handedness)

None of these add a new weighted component -- W_SP is still 0.85, the same 34-share as before.
They are multipliers INSIDE the starter factor, exactly the way `arm_form` already was.

All three are OPTIONAL keyword arguments with a neutral default, so this remains a safe drop-in:
a caller that only ever passes the original arguments (starter_vuln, arm_form, pen, team_hr9,
park_boost, temp_f, wind_out) gets the same starter factor as before, just now run through the
multiplicative combiner instead of the weighted mean. The new inputs are DARK until
etl/build_board.py's call site is updated to pass them -- that wiring is intentionally not
included here, since it was not part of this file's brief, and target_score() must not silently
assume fields the current caller does not yet supply.
"""
from __future__ import annotations

import math

BASELINE_HR_RATE = 0.109   # season base rate this app has graded, kept as the log-mapping anchor

# Backtested macro weights, UNCHANGED from the additive version, expressed as multiplicative
# exponents. Each is the original percentage share divided by 40 -- chosen so the exponents land
# in a well-behaved 0.25-0.85 range rather than compounding into extreme multipliers.
W_SP = 0.85     # 34 / 40
W_PEN = 0.65    # 26 / 40
W_STAFF = 0.45  # 18 / 40
W_PARK = 0.30   # 12 / 40
W_WX = 0.25     # 10 / 40

MAX_LIFT = 2.0

TIER_PRIME = 68
TIER_STRONG = 56
TIER_LEAN = 44


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Starter factor -- HR Vulnerability score, enriched with xFIP, FB/GB mix, and platoon alignment.
# ---------------------------------------------------------------------------

def _xfip_regression_bump(xfip, era):
    """Multiplier for a pitcher whose xFIP sits well above his ERA -- expected HR regression.

    xFIP strips out a pitcher's actual HR/FB rate and substitutes a league-average one, so a
    large POSITIVE gap (xFIP > ERA) means his current results are propped up by a HR/FB rate
    that has run better than his underlying batted-ball profile supports -- the textbook
    regression-candidate signature. A large NEGATIVE gap (he has been unlucky) works the other
    way and is credited down, since a pitcher who "should" have a lower ERA is not the matchup
    this scanner exists to surface.

    Note: this pipeline computes xFIP but not a separate FIP figure, so the comparison used
    everywhere in this module is xFIP vs ERA, not xFIP vs FIP.

    Every 1.0 point of (xFIP - ERA) moves the multiplier 6%, clamped to +/-18% total -- enough
    to matter, not enough for one FanGraphs pull to dominate the starter factor on its own.
    """
    if xfip is None or era is None:
        return 1.0, None
    gap = float(xfip) - float(era)
    mult = _clamp(1.0 + gap * 0.06, 0.82, 1.18)
    note = None
    if gap >= 0.75:
        note = f"xFIP {float(xfip):.2f} vs ERA {float(era):.2f} (regression candidate)"
    elif gap <= -0.75:
        note = f"xFIP {float(xfip):.2f} vs ERA {float(era):.2f} (has been unlucky)"
    return mult, note


def _flyball_groundball_bump(fb_pct, gb_pct):
    """Multiplier from a pitcher's own batted-ball mix -- more fly balls, more HR opportunity.

    fb_pct/gb_pct come from etl/statcast_data.py:pitcher_batted_profile() (launch-angle based:
    GB < 10deg, LD 10-24deg, FB >= 25deg), already computed in the ETL and attached to
    pitcher_props, but not previously read by this module.

    League sits near 35% FB / 43% GB. A staff-average extreme fly-ball arm (42%+ FB) is
    credited up; a heavy ground-ball arm (50%+ GB) is credited down. Both are real, graded
    sabermetric signals -- fly-ball rate correlates directly with HR/9 in a way GB rate does not
    -- but both are SEASON-LONG and slow-moving, so the multiplier is kept smaller than the
    nightly-changing xFIP-ERA gap above.
    """
    if fb_pct is None and gb_pct is None:
        return 1.0, None
    mult = 1.0
    bits = []
    if fb_pct is not None:
        f = float(fb_pct)
        mult *= _clamp(1.0 + (f - 35.0) / 100.0, 0.90, 1.12)
        if f >= 42:
            bits.append(f"{f:.0f}% FB")
    if gb_pct is not None:
        g = float(gb_pct)
        mult *= _clamp(1.0 - (g - 43.0) / 130.0, 0.90, 1.10)
        if g >= 50:
            bits.append(f"{g:.0f}% GB (suppresses)")
    return _clamp(mult, 0.88, 1.15), (" · ".join(bits) or None)


def _platoon_bump(hand_splits, lineup_hand_pct):
    """Multiplier when tonight's lineup's dominant hand lines up with the pitcher's weaker side.

    hand_splits is pitcher_edges' raw {"two_yr": {"R": {"hr","pa"}, "L": {"hr","pa"}}, ...} --
    counts, not a pre-computed rate, so the rate is derived here with a minimum-PA gate (60 PA)
    to keep a small-sample platoon split from driving the score.

    lineup_hand_pct is the SHARE of the batting lineup that bats right-handed (0-1); the mirror
    share is implicitly left-handed. Switch hitters should already be resolved to their
    effective side by the caller before this percentage is computed, the same way the rest of
    the app resolves switch-hitter handedness against a specific throwing arm.
    """
    if not hand_splits or lineup_hand_pct is None:
        return 1.0, None
    two_yr = (hand_splits or {}).get("two_yr") or {}
    rates = {}
    for side in ("R", "L"):
        d = two_yr.get(side) or {}
        pa = d.get("pa") or 0
        if pa >= 60:
            rates[side] = d.get("hr", 0) / pa
    if len(rates) < 2:
        return 1.0, None
    worse_side = "R" if rates["R"] >= rates["L"] else "L"
    exposure = lineup_hand_pct if worse_side == "R" else (1.0 - lineup_hand_pct)
    gap = rates[worse_side] - rates["L" if worse_side == "R" else "R"]
    if gap <= 0 or exposure < 0.55:
        return 1.0, None
    # Scale by both the size of the platoon split AND how much of the lineup exploits it -- a
    # real split against a lineup only 40% the exploiting hand should not spike the score.
    mult = _clamp(1.0 + gap * 3.0 * exposure, 1.0, 1.20)
    note = f"lineup leans {worse_side}HB into his weaker side ({100*exposure:.0f}% exposure)"
    return mult, note


def _sp_factor(starter_vuln, arm_form, xfip=None, era=None,
              fb_pct=None, gb_pct=None, hand_splits=None, lineup_hand_pct=None):
    """HR Vulnerability score (0-100, same source as the Arms tab) -> a multiplier around 1.0,
    enriched by xFIP regression, FB/GB mix, and platoon alignment -- all optional."""
    if starter_vuln is None:
        return 1.0, None, False
    v = _clamp(float(starter_vuln) / 100.0, 0.0, 1.0)
    factor = 0.60 + v * 0.90          # vuln 0 -> 0.60x, vuln 50 -> 1.05x, vuln 100 -> 1.50x

    bump = {"HITTABLE": 1.06, "SHELLABLE": 1.04, "SLIPPING": 1.03,
            "STEADY": 1.00, "DEALING": 0.94}.get(str(arm_form or "").upper(), 1.00)
    factor *= bump

    notes = [f"SP vuln {round(float(starter_vuln))}" + (f" · {arm_form}" if arm_form else "")]

    xg_mult, xg_note = _xfip_regression_bump(xfip, era)
    factor *= xg_mult
    if xg_note:
        notes.append(xg_note)

    fg_mult, fg_note = _flyball_groundball_bump(fb_pct, gb_pct)
    factor *= fg_mult
    if fg_note:
        notes.append(fg_note)

    pl_mult, pl_note = _platoon_bump(hand_splits, lineup_hand_pct)
    factor *= pl_mult
    if pl_note:
        notes.append(pl_note)

    return _clamp(factor, 0.45, 1.85), " · ".join(notes), True


# ---------------------------------------------------------------------------
# Remaining factors -- unchanged in substance, only now returning a MULTIPLIER instead of a
# 0-1 additive score.
# ---------------------------------------------------------------------------

def _pen_factor(pen):
    have = False
    factor = 1.0
    notes = []
    rv = (pen or {}).get("rank_val")
    if rv is not None:
        have = True
        r = _clamp(float(rv) / 100.0, 0.0, 1.0)
        factor *= 0.62 + r * 0.86
        notes.append(f"pen {round(float(rv))}")
        if str((pen or {}).get("label") or "").upper() in ("WORN", "GASSED"):
            factor *= 1.10
            notes.append(str(pen["label"]).lower())
        out = (pen or {}).get("n_unavailable") or 0
        live = (pen or {}).get("live_pen") or {}
        if live.get("n_out") is not None:
            out = max(out, live["n_out"])
        if out >= 2:
            factor *= 1.05
            notes.append(f"{out} arms down")
    return factor, (" · ".join(notes) or None), have


def _staff_factor(team_hr9):
    """Season staff HR/9, its OWN exponent term -- kept separate from the pen factor's own use
    of the SAME number (via rank_val) so this evidence is not weighted under two names."""
    if team_hr9 is None:
        return 1.0, None, False
    return _clamp(float(team_hr9) / 1.15, 0.75, 1.35), None, True


def _park_factor(park_boost):
    if park_boost is None:
        return 1.0, None, False
    b = float(park_boost)
    mult = _clamp(1.0 + b / 100.0 * 0.45, 0.85, 1.30)
    note = f"park {'+' if b > 0 else ''}{round(b)}%" if abs(b) >= 3 else None
    return mult, note, True


def _wx_factor(temp_f, wind_out):
    if temp_f is None and not wind_out:
        return 1.0, None, False
    factor = 1.0
    bits = []
    if temp_f is not None:
        tf = float(temp_f)
        factor *= _clamp(1.0 + (tf - 70.0) / 300.0, 0.92, 1.12)
        if tf >= 85:
            bits.append("hot")
        elif tf <= 55:
            bits.append("cold")
    if wind_out:
        factor *= 1.04
        bits.append("wind out")
    return factor, (" · ".join(bits) or None), True


def driver_pills(pen=None, park_boost=None, temp_f=None, wind_out=None, starter_vuln=None):
    """Short labelled pills for the scanner card: {k, v, i} with i = 0-1 intensity for colour."""
    pills = []
    if starter_vuln is not None:
        v = float(starter_vuln)
        pills.append({"k": "SP", "v": ("High Vuln" if v >= 70 else
                                       "Vulnerable" if v >= 55 else
                                       "Average" if v >= 40 else "Tough"),
                      "i": _clamp(v / 100.0, 0.0, 1.0)})
    if pen:
        rv = pen.get("rank_val")
        if rv is not None:
            worn = str(pen.get("label") or "").upper() in ("WORN", "GASSED")
            r = float(rv)
            lab = ("Gassed" if worn and r >= 55 else "Worn" if worn else
                   "High Vuln" if r >= 65 else "Exploitable" if r >= 50 else "Solid")
            pills.append({"k": "Pen", "v": lab, "i": _clamp(r / 100.0, 0.0, 1.0)})
    if park_boost is not None and abs(float(park_boost)) >= 3:
        b = float(park_boost)
        pills.append({"k": "Park", "v": f"{'+' if b > 0 else ''}{round(b)}%",
                      "i": _clamp((b + 15.0) / 30.0, 0.0, 1.0)})
    if temp_f is not None or wind_out:
        bits = []
        if temp_f is not None:
            tf = float(temp_f)
            bits.append("Hot" if tf >= 85 else "Warm" if tf >= 72 else
                        "Cool" if tf >= 58 else "Cold")
        if wind_out:
            bits.append("wind out")
        if bits:
            score = _clamp((float(temp_f) - 45.0) / 50.0, 0.0, 1.0) if temp_f is not None else 0.5
            if wind_out:
                score = _clamp(score + 0.18, 0.0, 1.0)
            pills.append({"k": "Wx", "v": " · ".join(bits), "i": score})
    return pills[:4]


def target_score(starter_vuln=None, arm_form=None, pen=None, team_hr9=None,
                 park_boost=None, temp_f=None, wind_out=None,
                 xfip=None, era=None, fb_pct=None, gb_pct=None,
                 hand_splits=None, lineup_hand_pct=None):
    """Score a defense 0-100 via the multiplicative SP x Pen x Staff x Park x Wx framework.

    xfip/era/fb_pct/gb_pct/hand_splits/lineup_hand_pct are all OPTIONAL enrichments to the
    starter factor; every one defaults to None and degrades to "no adjustment" rather than
    raising -- the same fallback pattern this module already uses for a missing starter
    entirely.

    A missing top-level factor (no starter posted, no pen data) still falls back to a NEUTRAL
    1.00x rather than being dropped and renormalised, per the original spec: the pipeline must
    never fail on partial data, and the tradeoff -- a TBD card reads closer to average than it
    may turn out to be -- is made visible via the `flags` list rather than silently absorbed.
    """
    sp_f, sp_note, sp_have = _sp_factor(
        starter_vuln, arm_form, xfip=xfip, era=era, fb_pct=fb_pct, gb_pct=gb_pct,
        hand_splits=hand_splits, lineup_hand_pct=lineup_hand_pct)
    pen_f, pen_note, pen_have = _pen_factor(pen)
    staff_f, _staff_note, staff_have = _staff_factor(team_hr9)
    park_f, park_note, park_have = _park_factor(park_boost)
    wx_f, wx_note, wx_have = _wx_factor(temp_f, wind_out)

    if not (sp_have or pen_have or staff_have or park_have or wx_have):
        return None

    raw = (BASELINE_HR_RATE
           * (sp_f ** W_SP) * (pen_f ** W_PEN) * (staff_f ** W_STAFF)
           * (park_f ** W_PARK) * (wx_f ** W_WX))

    ratio = max(raw / BASELINE_HR_RATE, 1e-6) if BASELINE_HR_RATE else 1.0
    score = _clamp(50.0 + 50.0 * (math.log(ratio) / math.log(MAX_LIFT)), 0.0, 100.0)

    components = {
        "starter": round(_clamp(sp_f / 1.5, 0.0, 1.0) * 100) if sp_have else None,
        "bullpen": round(_clamp(pen_f / 1.5, 0.0, 1.0) * 100) if pen_have else None,
        "staff": round(_clamp((staff_f - 0.75) / 0.60, 0.0, 1.0) * 100) if staff_have else None,
        "park": round(_clamp((park_f - 0.85) / 0.45, 0.0, 1.0) * 100) if park_have else None,
        "weather": round(_clamp((wx_f - 0.92) / 0.20, 0.0, 1.0) * 100) if wx_have else None,
    }
    components = {k: v for k, v in components.items() if v is not None}

    w_total = W_SP + W_PEN + W_STAFF + W_PARK + W_WX
    weights = {"starter": round(W_SP / w_total * 100), "bullpen": round(W_PEN / w_total * 100),
               "staff": round(W_STAFF / w_total * 100), "park": round(W_PARK / w_total * 100),
               "weather": round(W_WX / w_total * 100)}

    drivers = [n for n in (sp_note, pen_note, park_note, wx_note) if n]

    flags = []
    if not sp_have:
        flags.append("SP TBD")
    if not pen_have:
        flags.append("no pen data")

    have_count = sum((sp_have, pen_have, staff_have, park_have, wx_have))
    coverage = round(100 * have_count / 5)

    return {
        "score": round(score, 1),
        "components": components,
        "weights": weights,
        "drivers": drivers[:4],
        "pills": driver_pills(pen=pen, park_boost=park_boost, temp_f=temp_f,
                              wind_out=wind_out, starter_vuln=starter_vuln),
        "flags": flags,
        "coverage": coverage,
    }


def tier(score):
    if score is None:
        return None
    if score >= TIER_PRIME:
        return "PRIME"
    if score >= TIER_STRONG:
        return "STRONG"
    if score >= TIER_LEAN:
        return "LEAN"
    return "AVOID"
