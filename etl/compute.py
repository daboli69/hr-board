"""
The Heat score (0-100), rebuilt around the FOUR signals that drive the ranking,
all measured on the last ~2 weeks of play:

  1. Pull-air%   — share of fly balls / line drives that are pulled.
                   ~66% of HR are pulled; 40%+ is the "good" mark.
                   (All-fields monsters like Wood / Ohtani score lower here but
                    make it up on EV + barrel — by design, not a penalty.)
  2. Avg EV      — harder = farther; 90 mph is the floor, 94+ is elite.
  3. Barrel%     — 80-86% of HR are barreled; the purest power-contact signal.
  4. Ideal AA%   — % of competitive swings with attack angle in the 5-20 deg band
                   (the new Statcast bat-tracking metric). 58%+ good, 70% elite.

Each signal is scored against three anchors (poor / good / elite) instead of a
single average, so "did he clear the threshold" is what moves the number — which
is how you actually read these. Everything is transparent: every term and any
warning flags are attached to each hitter's score_breakdown.

Park factor and the opposing arm are intentionally NOT folded into Heat — they're
situational and shown on the card so you apply them by eye. Easy to fold in later
if you want.
"""
from __future__ import annotations

# (poor, good, elite) anchor points per signal
ANCHORS = {
    "pull_air_pct": (28.0, 40.0, 55.0),
    "avg_ev":       (86.0, 88.5, 91.5),   # lowered: ~88.5 2wk avg now clears "good"
    "barrel_pct":   (6.0, 11.0, 17.0),
    "ideal_aa_pct": (45.0, 58.0, 70.0),
    "iso":          (0.140, 0.200, 0.290),   # isolated power — purest power outcome
    "slg":          (0.380, 0.450, 0.560),   # slugging (includes singles, lighter weight)
}

# points each signal can contribute (sums to 100)
WEIGHTS = {
    "barrel_pct": 22,
    "pull_air_pct": 20,
    "iso": 16,
    "avg_ev": 14,
    "ideal_aa_pct": 14,
    "slg": 14,
}

# kept for the UI / older callers
LEAGUE_AVG = {
    "barrel_pct": 8.0, "hardhit_pct": 40.0, "iso": 0.160, "avg_ev": 89.0,
    "launch_angle": 12.5, "fb_pct": 25.0, "swstr_pct": 11.0,
    "pull_air_pct": 38.0, "ideal_aa_pct": 52.0,
}


def anchor_scale(v, poor, good, elite):
    """Piecewise: below poor ramps 0->.25, poor->good .25->.65, good->elite .65->1."""
    if v is None:
        return 0.40
    if v <= poor:
        return max(0.0, 0.25 * v / poor) if poor > 0 else 0.0
    if v <= good:
        return 0.25 + 0.40 * (v - poor) / (good - poor)
    if v <= elite:
        return 0.65 + 0.35 * (v - good) / (elite - good)
    return 1.0


def heat_score(recent: dict, pitcher_score: int | None = None) -> tuple[int, dict]:
    """
    Score a hitter on his last-2-weeks form across the four signals, then nudge by
    the opposing pitcher's HR-vulnerability (50 = neutral; 100 boosts ~+22%, 0 cuts).
    """
    if not recent:
        return 0, {"note": "no recent data"}

    bd, total = {}, 0.0
    for key, w in WEIGHTS.items():
        s = anchor_scale(recent.get(key), *ANCHORS[key])
        pts = round(s * w, 1)
        bd[key] = pts
        total += pts

    # plain-language flags
    flags = []
    ev = recent.get("avg_ev")
    iaa = recent.get("ideal_aa_pct")
    pull = recent.get("pull_air_pct")
    brl = recent.get("barrel_pct")
    bbe = recent.get("bb_count") or 0
    if bbe < 12:
        flags.append(f"thin sample ({bbe} BBE)")
    if ev is not None and ev < ANCHORS["avg_ev"][0]:
        flags.append("EV below 87 floor")
    if iaa is not None and ev is not None and iaa >= ANCHORS["ideal_aa_pct"][1] and ev < 88:
        flags.append("empty IAA (no EV behind it)")
    if pull is not None and pull >= 40 and ev is not None and ev >= 90 and brl is not None and brl >= 11:
        flags.append("full HR profile")

    bd["flags"] = flags

    # opposing-pitcher matchup multiplier
    base = total
    if pitcher_score is not None:
        mult = 1.0 + (pitcher_score - 50) / 50.0 * 0.22
        mult = max(0.78, min(1.22, mult))
        bd["base_four"] = round(base, 1)
        bd["pitcher_mult"] = round(mult, 3)
        total = base * mult

    return int(round(min(100, total))), bd


# ----------------------------------------------------------------------------
# PITCHER HR-VULNERABILITY ("get-shelled") model
# ----------------------------------------------------------------------------
# Higher score = more likely to get taken deep. Anchors are (safe, mid, danger).
# For swstr the relationship is inverted (fewer whiffs = more vulnerable).

PITCH_ANCHORS = {
    "barrel_pct_allowed": (5.0, 8.5, 12.0),
    "hr_per_pa":          (2.3, 3.3, 5.0),
    "hardhit_pct_allowed":(35.0, 40.0, 46.0),
    "avg_ev_allowed":     (86.5, 89.0, 91.5),
    "ideal_aa_allowed":   (45.0, 53.0, 62.0),
    "pull_air_allowed":   (32.0, 40.0, 50.0),
}
PITCH_WEIGHTS = {
    "barrel_pct_allowed": 24,
    "hr_per_pa": 20,
    "hardhit_pct_allowed": 16,
    "avg_ev_allowed": 12,
    "ideal_aa_allowed": 12,
    "pull_air_allowed": 10,
    "swstr_inv": 6,           # low whiff rate = vulnerable
}
SWSTR_ANCHORS = (13.0, 10.0, 7.5)   # safe, mid, danger (inverted)


def _vuln(metrics: dict) -> float:
    """0-100 vulnerability from a metric dict (one window)."""
    if not metrics:
        return 0.0
    total = 0.0
    for key, w in PITCH_WEIGHTS.items():
        if key == "swstr_inv":
            v = metrics.get("swstr_pct_allowed")
            safe, mid, danger = SWSTR_ANCHORS
            # inverted: lower v -> higher scale
            if v is None:
                s = 0.40
            elif v >= safe:
                s = max(0.0, 0.25 * (2 - v / safe))
            elif v >= mid:
                s = 0.25 + 0.40 * (safe - v) / (safe - mid)
            elif v >= danger:
                s = 0.65 + 0.35 * (mid - v) / (mid - danger)
            else:
                s = 1.0
            total += max(0.0, min(1.0, s)) * w
        else:
            total += anchor_scale(metrics.get(key), *PITCH_ANCHORS[key]) * w
    return round(total, 1)


def pitcher_hr_score(recent: dict, season: dict) -> dict:
    """
    Returns the pitcher's HR-vulnerability picture:
      score        — blended (recent-weighted) 0-100, used to modulate hitters
      recent_score — vulnerability on the last 2 weeks (the recent-form read)
      season_score — vulnerability season-long
      form         — recent-form identifier {label, color}
      flags        — plain-language red flags that he may get shelled
    """
    rec_s = _vuln(recent)
    sea_s = _vuln(season)
    # blend leans recent but keeps season as ballast; if no recent, use season
    if recent and recent.get("bbe"):
        score = round(0.6 * rec_s + 0.4 * sea_s, 1)
    else:
        score = sea_s
    delta = round(rec_s - sea_s, 1)

    flags = []
    r = recent or {}
    if (r.get("bbe") or 0) < 10:
        flags.append(f"thin recent sample ({r.get('bbe', 0)} BBE)")
    if r.get("barrel_pct_allowed") is not None and r["barrel_pct_allowed"] >= 10:
        flags.append(f"barrels up ({r['barrel_pct_allowed']}%)")
    if r.get("hardhit_pct_allowed") is not None and r["hardhit_pct_allowed"] >= 43:
        flags.append(f"hard contact ({r['hardhit_pct_allowed']}%)")
    if r.get("avg_ev_allowed") is not None and r["avg_ev_allowed"] >= 90.5:
        flags.append(f"loud contact ({r['avg_ev_allowed']} EV)")
    if r.get("hr_per_pa") is not None and r["hr_per_pa"] >= 4.0:
        flags.append(f"HR-prone lately ({r['hr_per_pa']}% of PA)")
    if r.get("velo_trend") is not None and r["velo_trend"] <= -0.8:
        flags.append(f"velo down {r['velo_trend']} mph")
    if r.get("swstr_pct_allowed") is not None and r["swstr_pct_allowed"] <= 8.5:
        flags.append(f"not missing bats ({r['swstr_pct_allowed']}% swstr)")
    if r.get("ideal_aa_allowed") is not None and r["ideal_aa_allowed"] >= 58:
        flags.append(f"hitters timing him up ({r['ideal_aa_allowed']}% ideal AA)")
    if r.get("pull_air_allowed") is not None and r["pull_air_allowed"] >= 45:
        flags.append(f"pulled in the air ({r['pull_air_allowed']}%)")
    # homers despite Ks (his note): decent whiff but still leaking HR
    if (r.get("swstr_pct_allowed") is not None and r["swstr_pct_allowed"] >= 11
            and r.get("hr_per_pa") is not None and r["hr_per_pa"] >= 3.5):
        flags.append("HRs even with Ks")
    if delta >= 10:
        flags.append("trending worse vs season")
    # steady-bad: consistently vulnerable across both windows (a bad pitcher is a bad pitcher)
    if rec_s >= 58 and sea_s >= 55 and abs(delta) < 6:
        flags.append("consistently hittable (season + recent both poor)")
    # bad but improving: underlying numbers ticking up — downgrade the target
    if rec_s >= 50 and delta <= -8:
        flags.append("underlying stats improving — caution")

    # recent-form identifier = absolute vulnerability LEVEL x TREND direction.
    # A bad pitcher is a bad pitcher even when steady; a bad pitcher whose
    # underlying numbers are improving is a different (downgrade) story.
    bad = rec_s >= 60          # absolutely vulnerable right now
    midbad = 48 <= rec_s < 60
    worsening = delta >= 6     # recent notably worse than season
    improving = delta <= -6    # recent notably better than season

    if bad and worsening:
        form = {"label": "SHELLABLE", "color": "#E4572E"}      # bad and getting worse — prime
    elif bad and improving:
        form = {"label": "BAD-IMPROVING", "color": "#E0913A"}  # still bad but trending up — caution
    elif bad:
        form = {"label": "STEADY-BAD", "color": "#E4572E"}     # consistently hittable — still a target
    elif worsening:
        form = {"label": "SLIPPING", "color": "#E0913A"}       # was fine, now cracking — opportunity
    elif midbad:
        form = {"label": "HITTABLE", "color": "#E0913A"}       # middling, leaks some
    elif improving and sea_s >= 50:
        form = {"label": "BOUNCING-BACK", "color": "#5FB97A"}  # bad season but sharpening up — avoid
    elif rec_s <= 35:
        form = {"label": "DEALING", "color": "#5FB97A"}        # genuinely good — avoid
    else:
        form = {"label": "STEADY", "color": "#8A95A3"}

    return {
        "score": int(round(min(100, score))),
        "recent_score": int(round(min(100, rec_s))),
        "season_score": int(round(min(100, sea_s))),
        "delta": delta,
        "form": form,
        "flags": flags,
    }
