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


# Task 1: lineup slot modifier for P(bases loaded) -- standard historical baseball frequencies
# (NOT this app's own backtest -- checked directly: there is no bases-loaded-by-slot number
# anywhere in BACKTEST.by_edge, so this is a reasoned general heuristic, not a verified figure,
# and is labelled that way rather than dressed up as backtested).
#
# IMPORTANT CAVEAT, read before trusting this at full strength: loaded_bases_prob() already
# computes P(bases loaded) from the REAL on-base probability of the three hitters actually
# batting ahead of this specific batter tonight. The reason slots 3-5 see more bases-loaded
# PAs in real baseball IS that they bat behind good-OBP hitters -- which the _ahead OBP
# calculation already captures directly, specifically, for tonight's real lineup. Applying a
# second, generic slot-based multiplier on top double-counts that same underlying cause through
# a blunter instrument. Damped to 25% of the stated bump/reduction for exactly that reason --
# a light nudge for anything the specific-hitters calculation might still miss (e.g. real
# lineup-construction tendencies beyond just OBP), not a full second correction.
LINEUP_SLOT_MODIFIER_FULL = {1: 0.92, 2: 1.00, 3: 1.06, 4: 1.08, 5: 1.05,
                             6: 1.00, 7: 1.00, 8: 0.94, 9: 0.90}
_SLOT_DAMP = 0.25
lineup_slot_modifier = {s: 1.0 + (m - 1.0) * _SLOT_DAMP for s, m in LINEUP_SLOT_MODIFIER_FULL.items()}


def loaded_bases_prob(ahead_p_ob, wildness=0.085, hitter_spot=None):
    """P(bases loaded when this hitter comes up), from the three hitters ahead of him.

    hitter_spot: Task 1's lineup_slot_modifier, applied at 25% strength -- see the module-level
    comment above for why this is damped rather than applied at full stated strength.

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
    p = p * ROUTES
    if hitter_spot is not None:
        p *= lineup_slot_modifier.get(int(hitter_spot), 1.0)
    return max(0.0, min(0.25, p))


# Real, currently-graded badge/metric lifts for Grand Slam's Matchup Lift (Part B) -- officially
# greenlit after resolving a discrepancy against the live app's own Trends UI (docs/index.html's
# BKLABEL mapping): the raw key `mix` displays as "PITCH EDGE" (not `arm`+`mix` combined), and
# the raw key `lock` displays as "HOT" -- the raw key literally named `hot` displays as
# "WARMING" instead, a genuine naming collision. All anchored to BACKTEST.base_pct (10.92%),
# the same denominator every other lift in this app uses.
#   pow  (UI: POWER)      1.43x, n=1,727
#   lock (UI: HOT)        1.45x, n=802
#   mix  (UI: PITCH EDGE) 1.51x, n=103
#   hrbp (UI: HR vs PEN)  1.21x, n=1,232
#   12%+ barrel (min 15 BBE, a raw metric threshold, not a badge): 1.80x, n=340
# Platoon (`plat`) and the raw `hot` key are deliberately NOT in this set -- the officially
# greenlit criteria are exactly these badges plus the barrel threshold.
#
# `pen` (UI: WEAK PEN) is a real, defined badge in this app's own BKLABEL mapping -- but checked
# directly against the full tracked history and it has ZERO holders (n=0) in the entire graded
# window. There is no real number to put here. Included at a neutral 1.0x (contributes nothing,
# doesn't count toward the convergence total) rather than a plausible-sounding guess, so this
# stops being silently missing without becoming silently fabricated.
GS_MATCHUP_LIFT = {"pow": 1.43, "lock": 1.45, "mix": 1.51, "hrbp": 1.21, "pen": 1.0}
GS_BARREL_THRESHOLD_LIFT = 1.80


def matchup_lift_multiplier(badges: set, l14_barrel_pct: float = None,
                            l14_bbe: int = None) -> tuple:
    """Grand Slam Part B -- how many of the real, currently-graded matchup edges this batter
    holds against tonight's specific pitcher, and the resulting HR-probability multiplier.

    Returns (multiplier, n_criteria_hit) -- n_criteria_hit feeds Part C's convergence check.
    Each real lift is damped to 0.5 of its full measured strength before compounding, the same
    damping principle the Cross-Game HR Optimizer already uses when stacking multiple
    correlated-with-power signals: these badges and barrel% correlate with each other and with
    a hitter's own general power level, so applying all of a hitter's real edges at full,
    undamped strength simultaneously would overstate the combination.
    """
    mult = 1.0
    n_hit = 0
    for bk, lift in GS_MATCHUP_LIFT.items():
        if badges and bk in badges and lift != 1.0:
            # lift==1.0 means no real graded evidence exists yet (currently only `pen`, which
            # has zero holders in the tracked history) -- included for completeness but must
            # NOT count toward the convergence total, or a batter with zero real evidence
            # behind a badge could still help trigger the 1.98x bonus.
            mult *= 1.0 + (lift - 1.0) * 0.5
            n_hit += 1
    if l14_barrel_pct is not None and l14_bbe is not None and l14_bbe >= 15 and l14_barrel_pct >= 12.0:
        mult *= 1.0 + (GS_BARREL_THRESHOLD_LIFT - 1.0) * 0.5
        n_hit += 1
    return mult, n_hit


def grand_slam_convergence_multiplier(n_criteria_hit: int) -> float:
    """Grand Slam Part C -- reuses this app's real 1.98x '4 families converging' number
    (validated for a DIFFERENT, general-HR-outcome signal composition, n=278) once a batter
    hits 3+ of Part B's specific criteria. This is a deliberate reuse of a real number across a
    related but different composition, not an independent backtest of THIS exact combination
    -- said plainly here rather than implied as separately proven.
    """
    return 1.98 if n_criteria_hit >= 3 else 1.0


def vuln_probability_boost(vuln_score: float) -> float:
    """Task 3 -- scales the HR-probability side of the Grand Slam equation using the real,
    already-computed 0-100 SP HR Vulnerability score (pe['vuln']['score'] on pitcher_edges;
    >=70 = 'elite_target', >=50 = 'strong', confirmed directly against the live payload and
    its own documented point budget: ERA 30, park 20, HR-splits by hand 15, WHIP 15, zone
    damage 12, dangerous bats 8).

    The vuln score itself is real and already validated for what it measures (general HR
    susceptibility). The SPECIFIC multiplier magnitude below for grand-slam probability
    specifically is a new composition -- this app has not independently backtested "grand slam
    rate by vuln score" -- so this is a reasoned scaling, stated as such, not a second
    claimed-backtested number.
    """
    if vuln_score is None:
        return 1.0
    if vuln_score >= 70:
        return 1.25
    if vuln_score >= 50:
        return 1.10
    return 1.0


def bullpen_exploit_multiplier(rank_val: float, label: str = None) -> float:
    """Task 2a -- the real bullpen_rankings payload (rank_val 0-100, label e.g. 'WORN'/'GASSED')
    already built by the main pipeline's bullpen loop, scaling Grand Slam probability.

    Continuous scaling above the 50 midpoint rather than an arbitrary cliff, since rank_val is
    itself a continuous composite (ERA/HR9/fatigue/platoon-gap components, confirmed directly
    against a live entry: KC = 60.6). A rank_val of 100 caps at +30%; the label check adds a
    further +8% specifically when the pen is confirmed WORN or GASSED, since workload state is
    real information rank_val alone doesn't fully capture (rank_val is season-long; label
    reflects live/recent workload).

    This is DIFFERENT from and additional to the existing pen_boost in grand_slam_score() --
    pen_boost (late_hr feature + pen_state's simpler bb_pct/fatigue) adjusts the ORDINAL 0-100
    ranking score; this adjusts the REAL p_slam probability, using the more comprehensive
    bullpen_rankings payload specifically. Not backtested as a grand-slam-specific multiplier
    magnitude -- rank_val itself is real and already used elsewhere, this specific scaling of
    it for grand slam probability is a reasoned composition, stated as such.
    """
    if rank_val is None:
        return 1.0
    mult = 1.0 + max(0.0, (float(rank_val) - 50.0)) / 50.0 * 0.30
    if str(label or "").upper() in ("WORN", "GASSED"):
        mult *= 1.08
    return mult


def signal_tiebreaker_multiplier(signal_count: int) -> float:
    """Task 4 -- a strict, fractional tie-breaker ONLY, per spec: 1.0 + signal_count*0.015,
    capped so it can nudge between near-equal candidates but can never let a weak signal-heavy
    hitter leapfrog a genuine power threat. signal_count is measured+provisional combined --
    the exact same count docs/index.html's own cvMeasured/provisional computation displays as
    the "+N" tag (checked directly: `(c.measured||[]).length + (c.provisional||[]).length`).

    Capped at 8 signals (~12% max) even if a hitter somehow carries more -- the UI's own stated
    context is "heat separates roughly twice the spread convergence does," so this must stay a
    genuine tie-breaker, not a real driver of the ranking.
    """
    if not signal_count:
        return 1.0
    return 1.0 + min(int(signal_count), 8) * 0.015


def slam_probability(p_loaded, hr_per_pa, park_mult=1.0, near_miss_boost=0.0,
                     traffic_mult=1.0, matchup_lift_mult=1.0, convergence_mult=1.0,
                     vuln_mult=1.0, signal_mult=1.0, team_traffic_mult=1.0):
    """P(grand slam this game) = P(bases loaded in a PA) x P(he homers in it), across ~4.3 PAs.

    traffic_mult: Part A -- this specific pitcher's own bases-loaded rate vs league average
    (pitcher_traffic_profile() in statcast_data.py), applied to p_loaded.
    team_traffic_mult: Part A2 -- ADDED this session, per Travis's direct question: does this
    batting TEAM'S real, recent (14-day) bases-loaded frequency deserve real weight, separate
    from Part A (the pitcher's season-long tendency) and separate from traffic_score() (which
    only looks at the 3 hitters batting immediately ahead of THIS one, in today's specific
    lineup)? A genuinely different question -- whether the WHOLE order has been generating more
    traffic lately, for reasons neither of those existing signals would catch (the lineup
    running hot together, a soft recent stretch of opposing pitching, real situational-hitting
    form). Combined with traffic_mult via a damped blend, not simple multiplication:
    traffic_mult stays at full strength (pitcher_traffic_profile is an established, stable,
    season-scale signal); team_traffic_mult is damped to 50% since it is NEW and has not yet
    been backtest-validated against real grand slam outcomes specifically (see
    replay_team_traffic in backtest.py) -- stacking a real, proven signal at full strength
    against a real-but-unproven one at full strength would risk exactly the kind of
    overconfidence this session already found and fixed once in the Genius Pairing stack.
    matchup_lift_mult: Part B -- this batter's real, currently-graded edges against THIS
    pitcher (badges/barrel%), applied to hr_per_pa alongside the existing park factor.
    convergence_mult: Part C -- the "hits 3+ of the Part B criteria" bonus. Reuses the real
    1.98x number this app already validated for "4 measured signal families converging" on
    general HR outcomes (n=278) -- a deliberate reuse across a related but different
    composition, not an independent validation of this exact combination.
    vuln_mult: Task 3 -- the opposing starter's real 0-100 HR Vulnerability score, scaled.
    signal_mult: Task 4 -- the measured+provisional signal count, strict tie-breaker only.

    Reported as a real probability so it can be compared against a book price.
    """
    if p_loaded is None or not hr_per_pa:
        return None
    _combined_traffic = float(traffic_mult or 1.0) * (1.0 + (float(team_traffic_mult or 1.0) - 1.0) * 0.5)
    p_loaded_adj = min(1.0, max(0.0, float(p_loaded) * _combined_traffic))
    hr = (float(hr_per_pa) * float(park_mult or 1.0) * (1.0 + float(near_miss_boost or 0.0))
         * float(matchup_lift_mult or 1.0) * float(vuln_mult or 1.0))
    per_pa = p_loaded_adj * hr
    base_prob = max(0.0, min(0.08, 1.0 - (1.0 - per_pa) ** 4.3))
    final = base_prob * float(convergence_mult or 1.0) * float(signal_mult or 1.0)
    return max(0.0, min(0.08, final))


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
