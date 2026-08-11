"""Team targets — which defenses are most likely to give up home runs tonight.

REWRITTEN as a multiplicative aggregator over three modules that already exist elsewhere in the
app, rather than a fifth independent scoring system:

    SP_Factor   <- HR Vulnerability score, same source as the Arms tab
    Pen_Factor  <- bullpen ranking + staff HR/9, same source as the Bullpens tab
    BPP_Factor  <- BallparkPal park/weather multiplier, same keys the UI pills already read

    Target_Score_Raw = BASELINE_HR_RATE * SP_Factor**W_SP * Pen_Factor**W_PEN * BPP_Factor**W_BPP

Exponents are derived from the backtest-graded lift shares for each bucket (starter 34,
bullpen+staff 44, park+weather 22 -> normalised), not chosen to look right:

    W_SP  = 0.82   (34/100 share)
    W_PEN = 1.06   (44/100 share -- bullpen exploitability + staff HR/9 folded together)
    W_BPP = 0.53   (22/100 share -- park + weather folded together)

A missing factor (no probable starter posted) falls back to a NEUTRAL 1.00x multiplier rather
than renormalising the remaining weights, per spec -- this keeps the pipeline from ever failing
on partial data, at the cost of a TBD card reading closer to average than it might turn out to
be. A "SP TBD" flag is still emitted so that tradeoff is visible on the card.

Schema is unchanged: target_score() still returns score/components/weights/drivers/coverage/
pills/flags, and tier() still returns PRIME/STRONG/LEAN/AVOID, because docs/index.html is locked
and string-matches those exact tier names for badge colour.
"""
from __future__ import annotations

import math

BASELINE_HR_RATE = 0.109   # season base rate this app has graded, kept as the log-mapping anchor

# Backtest-calibrated exponents. See module docstring for the share derivation.
W_SP = 0.82
W_PEN = 1.06
W_BPP = 0.53

# The raw multiplicative score is mapped onto 0-100 with a log curve anchored so that all-neutral
# inputs (every factor = 1.00x) land at 50, and the strongest realistic combination the backtest
# has actually observed (~1.94x lift, the 4-family HR ceiling) lands near 100.
MAX_LIFT = 2.0

# Same tier cutoffs the UI has always used -- unchanged so PRIME/STRONG/LEAN colour correctly.
TIER_PRIME = 68
TIER_STRONG = 56
TIER_LEAN = 44


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _sp_factor(starter_vuln, arm_form):
    """HR Vulnerability score (0-100, same source as the Arms tab) -> a multiplier around 1.0."""
    if starter_vuln is None:
        return 1.0, None, False
    v = _clamp(float(starter_vuln) / 100.0, 0.0, 1.0)
    factor = 0.60 + v * 0.90          # vuln 0 -> 0.60x, vuln 50 -> 1.05x, vuln 100 -> 1.50x
    bump = {"HITTABLE": 1.06, "SHELLABLE": 1.04, "SLIPPING": 1.03,
            "STEADY": 1.00, "DEALING": 0.94}.get(str(arm_form or "").upper(), 1.00)
    factor *= bump
    note = f"SP vuln {round(float(starter_vuln))}"
    if arm_form:
        note += f" · {arm_form}"
    return factor, note, True


def _pen_factor(pen, team_hr9):
    """Bullpen ranking + staff HR/9 (same sources as the Bullpens tab) -> a multiplier."""
    have = False
    factor = 1.0
    notes = []

    rv = (pen or {}).get("rank_val")
    if rv is not None:
        have = True
        r = _clamp(float(rv) / 100.0, 0.0, 1.0)
        factor *= 0.62 + r * 0.86     # rank 0 -> 0.62x, rank 50 -> 1.05x, rank 100 -> 1.48x
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

    if team_hr9 is not None:
        have = True
        # League sits near 1.15 HR/9; this is a modifier on top of the pen ranking rather than
        # its own weighted term, since both describe the same pitching staff.
        factor *= _clamp(float(team_hr9) / 1.15, 0.75, 1.35)
        notes.append(f"staff {float(team_hr9):.2f} HR/9")

    return factor, (" · ".join(notes) or None), have


def _bpp_factor(park_boost, temp_f, wind_out):
    """BallparkPal park/weather multiplier -> a single environment factor.

    Uses the same park_boost -> multiplier relationship already used elsewhere in the ETL
    (hr_proxy = 1 + boost/100 * 0.45) so this factor agrees with the number the park pill on
    other screens already shows, rather than inventing a second scale.
    """
    have = False
    factor = 1.0
    notes = []

    if park_boost is not None:
        have = True
        b = float(park_boost)
        factor *= _clamp(1.0 + b / 100.0 * 0.45, 0.85, 1.30)
        if abs(b) >= 3:
            notes.append(f"park {'+' if b > 0 else ''}{round(b)}%")

    if temp_f is not None or wind_out:
        have = True
        if temp_f is not None:
            tf = float(temp_f)
            factor *= _clamp(1.0 + (tf - 70.0) / 300.0, 0.92, 1.12)
            if tf >= 85:
                notes.append("hot")
            elif tf <= 55:
                notes.append("cold")
        if wind_out:
            factor *= 1.04
            notes.append("wind out")

    return factor, (" · ".join(notes) or None), have


def driver_pills(components, drivers, pen=None, park_boost=None, temp_f=None,
                 wind_out=None, starter_vuln=None):
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
                 park_boost=None, temp_f=None, wind_out=None):
    """Score a defense 0-100 via the multiplicative SP x Pen x BPP framework.

    A missing factor falls back to a NEUTRAL 1.00x rather than being dropped and renormalised --
    this is the spec'd behaviour so the pipeline never fails on partial data (e.g. no probable
    starter yet). The tradeoff is explicit: a TBD card reads closer to average than it might
    turn out to be once the starter posts, which is why an "SP TBD" flag still ships alongside
    the score rather than letting the neutral default pass silently.
    """
    sp_f, sp_note, sp_have = _sp_factor(starter_vuln, arm_form)
    pen_f, pen_note, pen_have = _pen_factor(pen, team_hr9)
    bpp_f, bpp_note, bpp_have = _bpp_factor(park_boost, temp_f, wind_out)

    if not (sp_have or pen_have or bpp_have):
        return None

    raw = BASELINE_HR_RATE * (sp_f ** W_SP) * (pen_f ** W_PEN) * (bpp_f ** W_BPP)

    # Log-map onto 0-100: raw == baseline -> 50, raw == baseline * MAX_LIFT -> 100 (and
    # symmetrically down to 0 at raw == baseline / MAX_LIFT).
    ratio = raw / BASELINE_HR_RATE if BASELINE_HR_RATE else 1.0
    ratio = max(ratio, 1e-6)
    score = 50.0 + 50.0 * (math.log(ratio) / math.log(MAX_LIFT))
    score = _clamp(score, 0.0, 100.0)

    components = {
        "starter": round(_clamp(sp_f / 1.5, 0.0, 1.0) * 100) if sp_have else None,
        "bullpen": round(_clamp(pen_f / 1.5, 0.0, 1.0) * 100) if pen_have else None,
        "park": round(_clamp((bpp_f - 0.85) / 0.45, 0.0, 1.0) * 100) if bpp_have else None,
    }
    components = {k: v for k, v in components.items() if v is not None}

    weights = {"starter": round(W_SP / (W_SP + W_PEN + W_BPP) * 100),
               "bullpen": round(W_PEN / (W_SP + W_PEN + W_BPP) * 100),
               "park": round(W_BPP / (W_SP + W_PEN + W_BPP) * 100)}

    drivers = [n for n in (sp_note, pen_note, bpp_note) if n]

    flags = []
    if not sp_have:
        flags.append("SP TBD")
    if not pen_have:
        flags.append("no pen data")

    have_count = sum((sp_have, pen_have, bpp_have))
    coverage = round(100 * have_count / 3)

    return {
        "score": round(score, 1),
        "components": components,
        "weights": weights,
        "drivers": drivers[:4],
        "pills": driver_pills(components, drivers, pen=pen, park_boost=park_boost,
                              temp_f=temp_f, wind_out=wind_out, starter_vuln=starter_vuln),
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
