"""Team targets — which defenses are most likely to give up home runs tonight.

The board ranks HITTERS. This ranks the other side: for each game it scores the pitching team on
how likely they are to concede, then presents it as "batting team vs that defense", so you can
pick the matchup first and the hitters second. That ordering is what the parlay playbook asks
for and there was no screen that supported it.

WEIGHTS ARE FROM THE BACKTEST, NOT INTUITION. Graded over 113 days:

    zone damage (5+ premium meatball zones)   1.39x    <- strongest
    starter form HITTABLE                     1.21x
    park lean+                                1.18x
    park strong+                              1.07x    <- weaker than it looks

That park line is worth pausing on. "Strong" HR parks graded at only 1.07x while merely "lean"
parks graded 1.18x, which is almost certainly the market pricing obvious launching pads
correctly — everyone knows Coors. So park is weighted LOW here despite being the factor people
reach for first. The starter and the bullpen carry most of the signal because they are the parts
that change nightly and get priced less efficiently.
"""
from __future__ import annotations


# Component weights, summing to 100. Ordered by graded lift, not by how obvious they feel.
W_STARTER = 34.0      # the single biggest lever: he throws 60-70% of the innings
W_BULLPEN = 26.0      # decides the late innings, and worn pens are the market's blind spot
W_TEAM_HR9 = 18.0     # season-long evidence the staff gives up power
W_PARK = 12.0         # real but modest, and largely priced in
W_WEATHER = 10.0      # carry moves the margin, not the outcome


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _starter_component(vuln_score, arm_form):
    """0-1 from the starter's HR vulnerability, nudged by tonight's form label."""
    if vuln_score is None:
        return None, "no starter posted"
    base = _clamp(float(vuln_score) / 100.0)
    # Form labels graded: HITTABLE 1.21x, DEALING 1.02x. Small nudge, not a rewrite.
    bump = {"HITTABLE": 0.08, "SHELLABLE": 0.05, "SLIPPING": 0.04,
            "STEADY": 0.0, "DEALING": -0.06}.get(str(arm_form or "").upper(), 0.0)
    note = f"starter vuln {round(float(vuln_score))}"
    if bump:
        note += f" · {arm_form}"
    return _clamp(base + bump), note


def _bullpen_component(pen):
    """0-1 from the bullpen ranking: exploitability, HR/9, and how worn it is."""
    if not pen:
        return None, "no pen data"
    rv = pen.get("rank_val")
    if rv is None:
        return None, "no pen data"
    base = _clamp(float(rv) / 100.0)
    bits = [f"pen {round(float(rv))}"]
    hr9 = pen.get("hr9")
    if hr9 is not None and float(hr9) >= 1.30:
        base = _clamp(base + 0.06)
        bits.append(f"{float(hr9):.2f} HR/9")
    # A worn pen is the part of this the market is slowest to move on, which is exactly why the
    # bump is meaningful rather than cosmetic.
    if str(pen.get("label") or "").upper() in ("WORN", "GASSED"):
        base = _clamp(base + 0.10)
        bits.append(pen["label"].lower())
    out = pen.get("n_unavailable") or 0
    live = pen.get("live_pen") or {}
    if live.get("n_out") is not None:
        out = max(out, live["n_out"])
    if out >= 2:
        base = _clamp(base + 0.05)
        bits.append(f"{out} arms down")
    return base, " · ".join(bits)


def _team_hr9_component(hr9):
    """0-1 from the staff's season HR/9. League sits near 1.15."""
    if hr9 is None:
        return None, None
    # 0.80 -> 0, 1.60 -> 1
    return _clamp((float(hr9) - 0.80) / 0.80), f"staff {float(hr9):.2f} HR/9"


def _park_component(boost_pct):
    """0-1 from the park's HR boost. Deliberately shallow — see the module docstring."""
    if boost_pct is None:
        return None, None
    b = float(boost_pct)
    lab = "park +" + str(round(b)) + "%" if b >= 3 else ("park " + str(round(b)) + "%" if b <= -3 else None)
    return _clamp((b + 15.0) / 30.0), lab


def _weather_component(temp_f, wind_out):
    """0-1 from carry conditions. Warm air and a wind blowing out both help the ball."""
    if temp_f is None and wind_out is None:
        return None, None
    score, bits = 0.5, []
    if temp_f is not None:
        score = _clamp((float(temp_f) - 45.0) / 50.0)
        if float(temp_f) >= 82:
            bits.append(f"{round(float(temp_f))}°F")
        elif float(temp_f) <= 55:
            bits.append(f"cold {round(float(temp_f))}°F")
    if wind_out is not None:
        score = _clamp(score + (0.18 if wind_out else -0.12))
        if wind_out:
            bits.append("wind out")
    return score, (" · ".join(bits) or None)


def target_score(starter_vuln=None, arm_form=None, pen=None, team_hr9=None,
                 park_boost=None, temp_f=None, wind_out=None):
    """Score a defense 0-100 on how likely it is to give up home runs tonight.

    Missing components are DROPPED and the weights renormalised over what is present, rather than
    filled with a neutral 0.5. Substituting an average for a missing starter would quietly move
    every game toward the middle and make a game with no posted starter look like a considered
    50 instead of an unknown — the reason is reported so a thin score is visible as thin.
    """
    parts, drivers = [], []

    for val, note, w, key in (
        _starter_component(starter_vuln, arm_form) + (W_STARTER, "starter"),
        _bullpen_component(pen) + (W_BULLPEN, "bullpen"),
        _team_hr9_component(team_hr9) + (W_TEAM_HR9, "staff"),
        _park_component(park_boost) + (W_PARK, "park"),
        _weather_component(temp_f, wind_out) + (W_WEATHER, "weather"),
    ):
        if val is None:
            continue
        parts.append((key, val, w))
        if note:
            drivers.append(note)

    if not parts:
        return None

    total_w = sum(w for _, _, w in parts)
    score = sum(v * w for _, v, w in parts) / total_w * 100.0

    return {
        "score": round(score, 1),
        "components": {k: round(100 * v) for k, v, _ in parts},
        "weights": {k: round(w / total_w * 100) for k, _, w in parts},
        "drivers": drivers[:4],
        "coverage": round(100 * total_w / (W_STARTER + W_BULLPEN + W_TEAM_HR9 + W_PARK + W_WEATHER)),
    }


def tier(score):
    if score is None:
        return None
    if score >= 68:
        return "PRIME"
    if score >= 56:
        return "STRONG"
    if score >= 44:
        return "LEAN"
    return "AVOID"
