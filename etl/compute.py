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


def form_score(w: dict) -> float:
    """0-100 power-form score from a window's metrics (the 6 signals, no arm)."""
    if not w:
        return 0.0
    tot = 0.0
    for k, wt in WEIGHTS.items():
        tot += anchor_scale(w.get(k), *ANCHORS[k]) * wt
    return round(tot, 1)


# Trend uses CONTACT-QUALITY only (barrel / pull-air / EV / attack angle). Results
# (ISO/SLG) lag and run lucky, so leaving them out makes "trending up" a cleaner
# leading indicator of imminent power rather than an echo of recent luck.
_QUALITY_WEIGHTS = {"barrel_pct": 32, "pull_air_pct": 28, "avg_ev": 20, "ideal_aa_pct": 20}


def _quality_form(w: dict) -> float:
    if not w:
        return 0.0
    tot = 0.0
    for k, wt in _QUALITY_WEIGHTS.items():
        tot += anchor_scale(w.get(k), *ANCHORS[k]) * wt
    return round(tot, 1)


def trend(short_w: dict, long_w: dict) -> dict:
    """
    Direction + % change of recent power form: short window (L5) vs longer (L30).
    Sample-aware:
      - short < 6 BBE  -> flat (too little to read)
      - long  < 15 BBE -> NEW player; if the short window is hot, flag up-new
      - else           -> signed % change between the two form scores
    Returns {dir: up/down/flat, pct: int|None, new: bool}.
    """
    s_bbe = (short_w or {}).get("bb_count") or 0
    l_bbe = (long_w or {}).get("bb_count") or 0
    if s_bbe < 6:
        return {"dir": "flat", "pct": None, "new": l_bbe < 15}
    s = _quality_form(short_w)
    if l_bbe < 15:                         # call-up / off-IL: no real baseline
        if s >= 50:
            return {"dir": "up", "pct": None, "new": True}
        return {"dir": "flat", "pct": None, "new": True}
    l = _quality_form(long_w)
    if l <= 0:
        return {"dir": "flat", "pct": None, "new": False}
    base = max(l, 30)                       # floor the baseline so a cold hitter's
    pct = max(-99, min(99, round((s - l) / base * 100)))  # tiny gain isn't +60%; cap sanity
    if pct >= 8:
        return {"dir": "up", "pct": pct, "new": False}
    if pct <= -8:
        return {"dir": "down", "pct": pct, "new": False}
    return {"dir": "flat", "pct": pct, "new": False}


def heat_score(recent: dict, pitcher_score: int | None = None) -> tuple[int, dict]:
    """
    Score a hitter on his last-2-weeks form across the four signals, then nudge by
    the opposing pitcher's HR-vulnerability (50 = neutral; 100 boosts ~+22%, 0 cuts).
    """
    if not recent:
        return 0, {"note": "no recent data"}

    bd, total, cleared = {}, 0.0, 0
    for key, w in WEIGHTS.items():
        s = anchor_scale(recent.get(key), *ANCHORS[key])
        pts = round(s * w, 1)
        bd[key] = pts
        total += pts
        if recent.get(key) is not None and s >= 0.65:   # cleared its "good" anchor
            cleared += 1
    bd["cleared"] = cleared          # 0-6 signals at/above good
    bd["signals"] = {k: (recent.get(k) is not None and anchor_scale(recent.get(k), *ANCHORS[k]) >= 0.65)
                     for k in WEIGHTS}
    # confirmation tier (echoes a quad/triple/double read)
    tier = ("LOADED" if cleared >= 5 else "STRONG" if cleared == 4
            else "SOLID" if cleared == 3 else "LEAN" if cleared == 2 else "THIN")
    bd["tier"] = tier

    # plain-language flags
    flags = []
    ev = recent.get("avg_ev")
    iaa = recent.get("ideal_aa_pct")
    pull = recent.get("pull_air_pct")
    brl = recent.get("barrel_pct")
    bbe = recent.get("bb_count") or 0
    pa = recent.get("pa") or 0
    if pa < 25 or bbe < 18:          # ~5 games or fewer = unreliable 2-week read
        flags.append(f"small sample ({pa} PA, {bbe} BBE)")
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


PITCH_FAM_LABEL = {"FB": "Fastball", "BR": "Breaking", "OFF": "Offspeed"}


def pitch_mix_profile(recent_splits: dict | None, usage: dict | None) -> dict | None:
    """
    Weight a hitter's recent per-pitch-family avg EV / launch angle / whiff% by the
    opposing starter's usage of those families — i.e. what he does vs THIS arm's mix
    over the last 2 weeks. Returns mix-adjusted values, or None if not computable.
    """
    if not recent_splits or not usage:
        return None
    ev_n = ev_d = la_n = la_d = wh_n = wh_d = br_n = br_d = 0.0
    for f in ("FB", "BR", "OFF"):
        u = usage.get(f)
        hs = recent_splits.get(f)
        if u is None or not hs:
            continue
        if hs.get("avg_ev") is not None:
            ev_n += u * hs["avg_ev"]; ev_d += u
        if hs.get("la") is not None:
            la_n += u * hs["la"]; la_d += u
        if hs.get("whiff_pct") is not None:
            wh_n += u * hs["whiff_pct"]; wh_d += u
        if hs.get("barrel_pct") is not None:
            br_n += u * hs["barrel_pct"]; br_d += u
    if not ev_d:
        return None
    return {
        "avg_ev": round(ev_n / ev_d, 1),
        "barrel_pct": round(br_n / br_d, 1) if br_d else None,
        "la": round(la_n / la_d, 1) if la_d else None,
        "whiff_pct": round(wh_n / wh_d, 1) if wh_d else None,
    }


def pitch_matchup(hitter_splits: dict | None, usage: dict | None,
                  overall_barrel: float | None = None) -> dict | None:
    """
    Cross a hitter's barrel% by pitch family with the pitcher's usage of those
    families. Surfaces whether the pitcher's mix feeds the hitter's strengths.
    Display-only context — not part of Heat.
    """
    if not hitter_splits or not usage:
        return None
    fams, wsum, wbar = [], 0.0, 0.0
    for f in ("FB", "BR", "OFF"):
        u = usage.get(f)
        hs = hitter_splits.get(f)
        if u is None or not hs or hs.get("barrel_pct") is None:
            continue
        wsum += u
        wbar += u * hs["barrel_pct"]
        fams.append((f, u, hs["barrel_pct"]))
    if wsum <= 0:
        return None
    weighted = round(wbar / wsum, 1)
    best = max(fams, key=lambda x: x[1] * x[2])
    edge = round(weighted - overall_barrel, 1) if overall_barrel is not None else None
    return {
        "weighted_barrel": weighted,
        "edge": edge,                         # vs the hitter's own overall barrel%
        "best": {"fam": best[0], "label": PITCH_FAM_LABEL[best[0]], "usage": best[1], "barrel": best[2]},
        "fams": {f: {"usage": u, "barrel": b} for f, u, b in fams},
    }


def luck_read(gap: float | None, xwobacon: float | None = None) -> str | None:
    """
    Read the xwOBAcon-vs-actual gap. Two independent things matter:
      - HOW FAR results exceed contact (gap magnitude) = regression risk
      - HOW GOOD the underlying contact is (xwOBAcon level) = the floor he regresses to
    Elite contact (>=.420) is genuinely rare, so "locked in" stays meaningful.
    """
    if gap is None:
        return None
    elite = xwobacon is not None and xwobacon >= 0.420
    weak = xwobacon is not None and xwobacon < 0.360
    if gap >= 0.045:
        return "running cold — due to break out"
    if gap <= -0.090:                       # results FAR beyond the contact
        return ("red hot — but producing well beyond even elite contact; some cooling likely"
                if elite else
                "red hot but lucky — results far above his contact; regression likely")
    if gap <= -0.045:                       # moderately above contact
        if elite:
            return "hot, and elite contact backs most of it up"
        if weak:
            return "hot on light contact — likely to cool"
        return "hot — running a bit above his contact"
    # results roughly match contact
    if elite:
        return "locked in — elite contact, results match; genuinely hot"
    if weak:
        return "fair — but the underlying contact is light"
    return "fair — results in line with his contact"


def pitcher_badges(*, recent=None, score=None, recent_score=None, season_score=None,
                   two_yr=None) -> list:
    """Defined, actionable vulnerability badges for a starter — the pitcher analogue
    of the hitter trait badges. Priority order = most HR-relevant first."""
    r = recent or {}
    out = []
    if score is not None and score >= 70:
        out.append({"t": "HR PRONE", "k": "arm"})
    if r.get("barrel_pct_allowed") is not None and r["barrel_pct_allowed"] >= 10:
        out.append({"t": "BARRELED", "k": "mix"})
    if (r.get("avg_ev_allowed") is not None and r["avg_ev_allowed"] >= 90.5) or \
       (r.get("hardhit_pct_allowed") is not None and r["hardhit_pct_allowed"] >= 43):
        out.append({"t": "LOUD", "k": "pow"})
    if r.get("pull_air_allowed") is not None and r["pull_air_allowed"] >= 45:
        out.append({"t": "PULL-AIR", "k": "hot"})
    # platoon: which hand has he coughed up HRs to (2yr)? flag the worse one
    if two_yr:
        worst = None
        for hand, lbl in (("R", "WEAK vs R"), ("L", "WEAK vs L")):
            s = two_yr.get(hand)
            if s and s.get("pa") and s["pa"] >= 200:
                rate = s["hr"] / s["pa"]
                if rate >= 0.035 and (worst is None or rate > worst[0]):
                    worst = (rate, lbl)
        if worst:
            out.append({"t": worst[1], "k": "plat"})
    if r.get("velo_trend") is not None and r["velo_trend"] <= -0.8:
        out.append({"t": "VELO ↓", "k": "due"})
    if recent_score is not None and season_score is not None and (recent_score - season_score) >= 8:
        out.append({"t": "SLIPPING", "k": "arm"})
    if not out and score is not None and score <= 35:
        out.append({"t": "TOUGH", "k": "tough"})
    return out


def player_badges(*, opp_form=None, hand_hr=None, eff_hand=None, pitch_matchup=None,
                  luck_gap=None, trend=None, max_ev=None, pen_score=None, xwobacon=None) -> list:
    """Ordered, actionable trait badges for a hitter — each one a specific reason
    he's interesting today. Priority order = most HR-relevant first."""
    out = []
    if opp_form in ("SHELLABLE", "STEADY-BAD"):
        out.append({"t": "WEAK ARM", "k": "arm"})
    if hand_hr and eff_hand in ("R", "L"):
        side = hand_hr.get(eff_hand)
        if side and side.get("pa") and (side.get("hr", 0) >= 25 or
                                        (side["hr"] / side["pa"]) >= 0.035):
            out.append({"t": "PLATOON", "k": "plat"})
    if pitch_matchup and pitch_matchup.get("edge") is not None and pitch_matchup["edge"] >= 2:
        out.append({"t": "PITCH EDGE", "k": "mix"})
    # exactly ONE form badge — never contradictory pairs. Temperature spectrum:
    #   DUE (unlucky) · WARMING (contact rising) · HOT (elite & earned) · MAY COOL (over his contact)
    if luck_gap is not None and luck_gap >= 0.045:
        out.append({"t": "DUE", "k": "due"})
    elif luck_gap is not None and luck_gap <= -0.090:
        out.append({"t": "MAY COOL", "k": "cool"})
    elif xwobacon is not None and xwobacon >= 0.420 and (luck_gap is None or luck_gap > -0.060):
        out.append({"t": "HOT", "k": "lock"})
    elif trend and trend.get("dir") == "up":
        out.append({"t": "WARMING", "k": "hot"})
    if max_ev is not None and max_ev >= 112:
        out.append({"t": "POWER", "k": "pow"})
    if pen_score is not None and pen_score >= 78:
        out.append({"t": "WEAK PEN", "k": "pen"})
    return out


def read_angle(*, hand=None, trend=None, pitch_matchup=None, luck_gap=None,
               opp_form=None, hand_hr=None, eff_hand=None, xwobacon=None) -> str:
    """
    One synthesized sentence — the model's read on a hitter today, assembled from the
    strongest 1-2 matchup facts plus a trend/luck qualifier. Returns '' if nothing
    stands out (so the card stays quiet rather than forcing a weak narrative).
    """
    handlbl = {"R": "RHB", "L": "LHB", "S": "switch"}.get(hand, "")
    eff_lbl = {"R": "RHB", "L": "LHB"}.get(eff_hand, handlbl)
    bits = []
    # 1) pitch-mix power edge — the strongest "why" when present
    pm = pitch_matchup
    if pm and pm.get("best") and pm.get("edge") is not None and pm["edge"] >= 2:
        b = pm["best"]
        lab = b["label"].lower()
        bits.append(f"barrels {lab} ({b['barrel']}%) vs a {b['usage']}% {lab} arm")
    # 2) the arm has coughed up HRs to this hitter's hand
    if hand_hr and eff_hand in ("R", "L"):
        side = hand_hr.get(eff_hand)
        if side and side.get("pa") and (side.get("hr", 0) >= 20 or
                                        (side["hr"] / side["pa"]) >= 0.035):
            bits.append(f"vs an arm that's allowed {side['hr']} HR to {eff_lbl} (2yr)")
    # 3) shellable starter
    if not bits and opp_form in ("SHELLABLE", "STEADY-BAD"):
        bits.append(f"facing a {opp_form.lower().replace('-', ' ')} starter")
    # qualifier: due/hot + trend
    qual = None
    if luck_gap is not None and luck_gap >= 0.045:
        qual = "due — hitting it hard, results lagging"
    elif luck_gap is not None and luck_gap <= -0.090:
        qual = "may cool — results above what his contact supports"
    elif xwobacon is not None and xwobacon >= 0.420 and (luck_gap is None or luck_gap > -0.060):
        qual = "hot — elite contact, the heat is earned"
    elif trend and trend.get("dir") == "up" and trend.get("pct"):
        qual = f"warming up — contact rising (+{trend['pct']}%)"
    elif trend and trend.get("dir") == "down" and trend.get("pct"):
        qual = f"contact cooling ({trend['pct']}%)"
    if not bits and not qual:
        return ""
    lead = (handlbl + " " if handlbl else "")
    if bits:
        s = lead + ", ".join(bits[:2]) + (f" — {qual}" if qual else "")
    else:                                  # nothing notable but a trend/luck note
        s = lead.rstrip() + (f" — {qual}" if qual else "")
    return s[0].upper() + s[1:]


def hand_vuln(split: dict | None) -> dict | None:
    """HR-vulnerability score of a pitcher (or pen) vs ONE batter hand, from a
    split's {season, recent} dicts. Returns None if the sample is too small to read."""
    if not split:
        return None
    season = split.get("season") or {}
    recent = split.get("recent") or {}
    if (season.get("bbe") or 0) < 20:        # not enough vs that hand to trust
        return None
    res = pitcher_hr_score(recent, season)
    res["bbe"] = season.get("bbe")
    return res


def platoon_note(splits: dict | None) -> dict | None:
    """Compare a pitcher's vulnerability vs RHB vs LHB. Returns which hand mashes it."""
    if not splits:
        return None
    r = hand_vuln(splits.get("R"))
    l = hand_vuln(splits.get("L"))
    rs = r["score"] if r else None
    ls = l["score"] if l else None
    if rs is None and ls is None:
        return None
    worse, gap = None, None
    if rs is not None and ls is not None:
        worse = "R" if rs >= ls else "L"
        gap = abs(rs - ls)
    return {"R": rs, "L": ls, "worse": worse, "gap": gap}


def bullpen_vuln(pen: dict | None, bat_hand: str | None = None) -> dict | None:
    """
    Score a team bullpen's HR-vulnerability (overall) plus, if the batter's hand is
    given, the pen's vulnerability vs that specific hand (the platoon-vs-pen read).
    """
    if not pen:
        return None
    overall = pitcher_hr_score(pen.get("recent") or {}, pen.get("season") or {})
    out = {
        "score": overall["score"],
        "form": overall["form"],
        "flags": overall.get("flags", [])[:3],
        "arms": pen.get("arms"),
    }
    splits = pen.get("splits") or {}
    if bat_hand in ("R", "L"):
        vh = hand_vuln(splits.get(bat_hand))
        out["vs_hand"] = vh["score"] if vh else None
        out["hand"] = bat_hand
    out["platoon"] = platoon_note(splits)
    return out


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
