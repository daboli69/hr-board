"""
grandslam.py — scores each hitter's grand-slam likelihood for the DraftKings GS jackpot.

A grand slam needs two things to line up, and most models only look at the second:
  1. TRAFFIC — the bases must be loaded when this hitter comes up. Modeled from the on-base
     ability of the hitters BATTING AHEAD of him + the opposing starter's wildness (BB rate).
  2. PUNISH — when the bases are loaded, the pitcher is forced to throw strikes (can't risk a
     walk-in run), so he grooves fastballs in the zone. We want the hitter most able to punish
     an in-zone fastball: power (barrel/EV/pull) + the elite gate.
Plus two amplifiers:
  3. PITCHER SHIFT — pitchers whose effectiveness drops when forced into the zone (flagged only
     when we can measure it; bases-loaded sample is thin, so this is a bonus not a base).
  4. PEN FATIGUE — grand slams happen when innings spiral; a gassed pen / short starter raises
     the odds of the bases-loaded mistake.

All parallel to the frozen HR heat model — never modifies it. Pure functions.
"""

from __future__ import annotations


def traffic_score(hitter_spot: int, lineup: list, opp_bb_pct_allowed: float) -> dict:
    """How likely the bases are loaded (or busy) when this hitter bats.
    lineup: list of player dicts in batting order (each with on-base proxy). We look at the
    THREE hitters batting ahead of this spot — the ones who load the bases for him.
    opp_bb_pct_allowed: starter's walk rate — a wild pitcher creates the traffic jam.
    Returns {score 0-100, obp_ahead, wildness}.
    """
    if not lineup or hitter_spot is None:
        return {}
    # the three spots ahead (wrapping the order: spot 1's "ahead" is 8,9,7 etc.)
    ahead_spots = [((hitter_spot - 1 - k - 1) % 9) + 1 for k in range(3)]
    obps = []
    for sp in ahead_spots:
        pl = next((x for x in lineup if x.get("lineup_spot") == sp), None)
        if pl:
            ob = pl.get("_ob")
            if ob is not None:
                obps.append(ob)
    if not obps:
        return {}
    obp_ahead = sum(obps) / len(obps)
    # normalize: league OBP ~.320; .360+ is a strong on-base trio
    ob_component = max(0.0, min(1.0, (obp_ahead - 0.290) / 0.100))
    # wildness: league BB% ~8.5%; 10%+ is wild
    wild = 0.0
    if opp_bb_pct_allowed is not None:
        wild = max(0.0, min(1.0, (opp_bb_pct_allowed - 6.0) / 6.0))
    # traffic blends on-base ahead (primary) with pitcher wildness (amplifier)
    score = round((ob_component * 0.7 + wild * 0.3) * 100, 1)
    return {"score": score, "obp_ahead": round(obp_ahead, 3),
            "wildness": round(opp_bb_pct_allowed, 1) if opp_bb_pct_allowed is not None else None}


def punish_score(metrics_season: dict, elite: dict, in_zone_fb: dict = None) -> dict:
    """How well this hitter punishes an in-zone fastball — the pitch he'll see with the bases
    loaded. Built from power profile (barrel/EV/pull) + the elite gate, and — when available —
    his ACTUAL in-zone-fastball barrel/hardhit (the sharpest version).
    Returns {score 0-100, drivers[]}.
    """
    drivers = []
    comp = 0.0
    n = 0
    ev = (metrics_season or {}).get("avg_ev")
    br = (metrics_season or {}).get("barrel_pct")
    pa = (metrics_season or {}).get("pull_air_pct")
    if ev is not None:
        comp += max(0.0, min(1.0, (ev - 86.0) / 8.0)); n += 1
        if ev >= 91: drivers.append(f"{ev:.1f} EV")
    if br is not None:
        comp += max(0.0, min(1.0, (br - 6.0) / 10.0)); n += 1
        if br >= 12: drivers.append(f"{br:.0f}% barrel")
    if pa is not None:
        comp += max(0.0, min(1.0, (pa - 30.0) / 20.0)); n += 1
        if pa >= 45: drivers.append(f"{pa:.0f}% pull-air")
    # in-zone fastball punish (the real signal, when we have it)
    if in_zone_fb and in_zone_fb.get("barrel_pct") is not None:
        izb = in_zone_fb["barrel_pct"]
        comp += max(0.0, min(1.0, (izb - 6.0) / 12.0)) * 1.5; n += 1.5   # weight it heavier
        if izb >= 12: drivers.append(f"{izb:.0f}% brl on in-zone FB")
    if n == 0:
        return {}
    base = comp / n
    # elite gate bonus
    if elite and elite.get("elite"):
        base = min(1.0, base + 0.08)
        drivers.append("elite profile")
    return {"score": round(base * 100, 1), "drivers": drivers[:4]}


def grand_slam_score(traffic: dict, punish: dict, pen_boost: float = 0.0,
                     shift_boost: float = 0.0) -> dict:
    """Combine traffic (bases loaded?) x punish (can he golf a grooved FB?) + amplifiers.
    Traffic and punish are BOTH required — a masher who never bats with ducks on the pond
    won't slam, and a weak bat with loaded bases won't either. So we multiply them, then add
    the situational amplifiers.
    """
    if not traffic or not punish:
        return {}
    t = traffic.get("score", 0) / 100.0
    p = punish.get("score", 0) / 100.0
    # geometric core: both must be present
    core = (t * p) ** 0.5
    score = core * 100 + pen_boost + shift_boost
    score = max(0, min(100, round(score, 1)))
    return {
        "score": score,
        "traffic": traffic.get("score"),
        "punish": punish.get("score"),
        "pen_boost": round(pen_boost, 1) if pen_boost else 0,
        "shift_boost": round(shift_boost, 1) if shift_boost else 0,
        "drivers": punish.get("drivers", []),
    }
