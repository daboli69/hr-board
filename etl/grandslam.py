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


def punish_score(metrics_season: dict, elite: dict, in_zone_fb: dict = None,
                 badges: set = None, ideal_aa_clear: bool = False) -> dict:
    """How well this hitter punishes an in-zone fastball — the pitch he'll see with the bases
    loaded. Built from power profile (barrel/EV/pull) + the elite gate, and — when available —
    his ACTUAL in-zone-fastball barrel/hardhit (the sharpest version).

    badges/ideal_aa_clear use the SAME graded lifts as the Cross-Game HR Optimizer, checked
    directly against the tracker (lock 1.31x, pow 1.29x, mix 1.20x but thin at n=100, hrsp 1.07x
    barely above base rate; ideal launch angle +12% season/+31% this month). Kept consistent
    across features rather than re-deriving a second set of numbers for the same evidence.
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
        elif br >= 10: drivers.append(f"{br:.0f}% barrel (clears 10% floor)")
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
    if elite and elite.get("elite"):
        base = min(1.0, base + 0.08)
        drivers.append("elite profile")
    if badges:
        for bk, lift in (("lock", 1.31), ("pow", 1.29), ("mix", 1.20), ("hrsp", 1.07)):
            if bk in badges:
                base = min(1.0, base * (1.0 + (lift - 1.0) * 0.35))
                drivers.append(f"{bk} badge")
                break   # one badge credit, not a stack -- these badges correlate with each other
    if ideal_aa_clear:
        base = min(1.0, base * (1.0 + (1.12 - 1.0) * 0.45))
        drivers.append("ideal launch angle")
    return {"score": round(base * 100, 1), "drivers": drivers[:5]}


def bullpen_wildness(pen: dict) -> float:
    """0-1 bullpen-meltdown-risk contribution, from the SAME bullpen_rankings record the
    Bullpens tab already reads (rank_val 0-100, label). No separate bullpen walk-rate field
    exists in this app's data -- rank_val is the established proxy for "how likely is this pen
    to be the problem," already used for exactly this purpose elsewhere.
    """
    if not pen:
        return 0.0
    rv = pen.get("rank_val")
    w = max(0.0, min(1.0, (rv - 50.0) / 50.0)) if rv is not None else 0.0
    if str(pen.get("label") or "").upper() in ("WORN", "GASSED"):
        w = min(1.0, w + 0.20)
    return w


def loaded_bases_prob(ahead_p_ob, wildness=0.085):
    """P(bases loaded when this hitter comes up), from the three hitters ahead of him.

    The old traffic score was a 0-100 heuristic built from an OBP proxy. This computes the thing
    itself: a grand slam needs all three preceding hitters to reach and none of them to be erased
    on the bases, which is a product of real per-PA probabilities rather than a normalised score.

    `ahead_p_ob` are per-PA on-base probabilities for the three spots ahead. They now come from
    the contact-gated hit model, which is a genuine per-hitter number — the previous proxy tied
    everyone in a lineup slot to roughly the same value.

    ERASURE is the term the heuristic had no way to express. Three men reaching does not mean
    three men still standing: double plays, caught stealing and force-outs remove roughly a
    quarter of baserunners before the next batter. Without it this reads about 30% too high.
    """
    if not ahead_p_ob or len(ahead_p_ob) < 3:
        return None
    ERASE = 0.24

    # Treating "bases loaded" as three consecutive hitters reaching UNDERSTATES it, because that
    # is not the only route: walks, hit-by-pitch, errors and a runner already aboard all load
    # the bases without three straight men reaching in sequence. Left uncorrected the model
    # priced an average lineup spot at 0.58x the real rate.
    #
    # 1.734 is SOLVED against the observed league rate — roughly 150 grand slams across 4,860
    # team-games, which is 0.343% per hitter-game — not chosen to look right. If that rate
    # drifts, re-fit it rather than nudging it.
    ROUTES = 1.734

    p = 1.0
    for ob in ahead_p_ob[:3]:
        ob = max(0.10, min(0.60, float(ob) + wildness - 0.085))
        p *= ob * (1.0 - ERASE)
    return max(0.0, min(0.25, p * ROUTES))


def slam_probability(p_loaded, hr_per_pa, park_mult=1.0, near_miss_boost=0.0):
    """P(grand slam this game) = P(bases loaded in a PA) x P(he homers in it), across ~4.3 PAs.

    Reported as a real probability so it can be compared against a book price. The old score was
    ordinal only — useful for ranking, useless for deciding whether a number is worth taking.
    """
    if p_loaded is None or not hr_per_pa:
        return None
    hr = float(hr_per_pa) * float(park_mult or 1.0) * (1.0 + float(near_miss_boost or 0.0))
    per_pa = p_loaded * hr
    return max(0.0, min(0.08, 1.0 - (1.0 - per_pa) ** 4.3))


def grand_slam_score(traffic: dict, punish: dict, pen_boost: float = 0.0,
                     shift_boost: float = 0.0, p_slam: float = None,
                     pen_label: str = None) -> dict:
    """Combine traffic (bases loaded?) x punish (can he golf a grooved FB?) + amplifiers.
    Traffic and punish are BOTH required — a masher who never bats with ducks on the pond
    won't slam, and a weak bat with loaded bases won't either. So we multiply them, then add
    the situational amplifiers.
    """
    if not traffic or not punish:
        return {}
    t = traffic.get("score", 0) / 100.0
    p = punish.get("score", 0) / 100.0
    core = (t * p) ** 0.5
    score = core * 100 + pen_boost + shift_boost
    score = max(0, min(100, round(score, 1)))
    fair_odds = None
    if p_slam and p_slam > 0:
        dec = 1.0 / p_slam
        fair_odds = round((dec - 1.0) * 100) if dec >= 2.0 else round(-100.0 / (dec - 1.0))

    # Combined drivers -- traffic's own numbers (ahead OBP, SP wildness) were previously dropped
    # entirely; only punish's drivers ever reached the final output, so a slam that scored high
    # because three great on-base hitters bat ahead of him showed no evidence of that at all.
    drivers = []
    if traffic.get("obp_ahead") is not None:
        drivers.append(f"Ahead OBP: {traffic['obp_ahead']:.3f}")
    if traffic.get("wildness") is not None:
        wtxt = f"SP BB%: {traffic['wildness']:.1f}%"
        if pen_label:
            wtxt += f" + {pen_label} Pen"
        drivers.append(wtxt)
    drivers.extend(punish.get("drivers", []))

    return {
        "score": score,
        "p_slam": (round(p_slam, 5) if p_slam is not None else None),
        "fair_odds": fair_odds,
        "traffic": traffic.get("score"),
        "punish": punish.get("score"),
        "pen_boost": round(pen_boost, 1) if pen_boost else 0,
        "shift_boost": round(shift_boost, 1) if shift_boost else 0,
        "drivers": drivers[:5],
    }
