"""
build_board.py — the one script the cron runs.

  python -m etl.build_board

It pulls the slate (StatsAPI) + Statcast season data (Savant), computes every
hitter in today's posted lineups, scores them, and writes docs/board.json.

Designed to fail soft: any single data source hiccup degrades that column to
null rather than crashing the whole run, so an unattended cron keeps producing
a board.
"""
from __future__ import annotations
from etl import kengine, hitmodel, arsenal as arsenal_mod, environment as env_mod
import json
import os
import time
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from etl import statsapi, statcast_data, parks, compute, park_model, props, targets
try:
    from etl import features                 # parallel feature-extraction edges (never touch heat)
except Exception:
    features = None
# microclimate temp-sensitivity profiles (built by the separate microclimate.py ETL)
_MICRO = {}
try:
    _mp = os.path.join(os.path.dirname(__file__), "..", "docs", "microclimate.json")
    with open(_mp) as _f:
        _MICRO = (json.load(_f) or {}).get("profiles", {})
except Exception:
    _MICRO = {}

try:                       # cache Savant pulls to disk so repeat runs are fast
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()
except Exception:
    pass

ET = ZoneInfo("America/New_York")
SEASON_START = os.environ.get("SEASON_START", "2026-03-26")
BUILD_HEALTH = []          # subsystem skip notes; shipped in board.json as build_health


def _hrr_p_over(lam, line=1.5):
    """P(HRR > line) for a projected total of `lam`.

    Two corrections sit on top of the raw projection, and they do different jobs.

    MEAN. Summing expected hits, runs and RBIs from per-PA rates runs hot — it puts the slate
    average at 1.92 HRR, which implies 49.2% of hitters clear 1.5 when the tracker has actually
    graded 40.6% across 29,694 player-games. No plausible dispersion absorbs an 8.6-point gap
    (even 2.5 leaves it 4.3 points high), so the mean is corrected directly. 0.817 was SOLVED
    against the graded rate rather than picked; re-fit it if that rate moves.

    SHAPE. HRR counts are positively correlated — a home run pays a hit, a run and an RBI in one
    event — so the distribution is wider than a Poisson of the same mean: more zeros and more big
    nights. A negative binomial carries exactly that extra parameter.
    """
    import math
    lam = max(1e-6, float(lam)) * 0.817          # mean calibration, see above
    DISPERSION = 1.8                             # var = lam * DISPERSION
    if DISPERSION <= 1.0:
        p0 = math.exp(-lam)
        p1 = lam * p0
    else:
        r = lam / (DISPERSION - 1.0)
        q = 1.0 / DISPERSION
        p0 = q ** r
        p1 = r * (1.0 - q) * p0
    return round(max(0.0, min(1.0, 1.0 - p0 - p1)), 4)


def _pct_scale(v):
    """Normalise a rate stat to percentage scale (16.0, not 0.16).

    The modern FanGraphs JSON API returns K-BB% as a raw fraction, not a percentage — real MLB
    K-BB% sits between roughly 3% and 35%, so a value under 1.0 is unambiguously a fraction that
    needs multiplying by 100, not a genuinely tiny percentage. Left un-normalised, this silently
    broke two downstream consumers: kengine's K-count nudge treated -14.84 (15 - 0.16) as the
    real gap and clamped to its maximum -5% penalty for every single pitcher regardless of their
    actual K-BB%, and the Parlay Scanner's `k_bb_pct > 15` gate excluded every arm on every
    slate, because no fraction is ever greater than 15.
    """
    if v is None:
        return None
    v = float(v)
    return round(v * 100.0, 2) if 0 < v < 1.0 else round(v, 2)


# ---------------------------------------------------------------------------
# CROSS-GAME 3-LEG HR PARLAY OPTIMIZER
# ---------------------------------------------------------------------------
#
# Weights below are graded on THIS app's own 118-day backtest / tracker history, not chosen to
# look right. Where the brief asked for a signal this app has actually tested and found NULL,
# that signal is deliberately excluded from scoring rather than "rewarded" — the brief's own
# "empirical freedom" directive says to tune based on what the data shows converts, and the data
# says these two specifically do not:
#
#   Command Risk (Location+ < 98) vs HR rate:  10.9% vs 10.9%  (lift 1.00x both sides, n=29,828)
#   Full FanGraphs run-mechanics nudge vs total-runs MAE: 0.008 runs, ~3% of the app's entire
#     usable accuracy range (floor ~3.30, baseline ~3.58) on 1,578 games
#
# Building a scorer that pays out on a signal already shown to be noise would be worse than
# building nothing — it would look data-driven while quietly not being data-driven at all.
#
# What IS weighted, with the real graded number next to each:
#
#   convergence families   0.75x / 1.07x / 1.32x / 1.56x / 1.98x  (0/1/2/3/4 families)
#   near misses (2+)       1.11x
#   barrel% (12%+)         1.80x  -- the single strongest individual metric graded
#   hand-first fit better  1.18x
#   lineup spot 1-4        1.17x-1.25x  vs  spot 9 at 0.60x        (tracker, not backtest)
#   sample gate (250+ BBE) required before ANY convergence credit applies -- thin-sample
#     convergence (conv1+ / <120 BBE) grades at 0.96x, i.e. WORSE than no signal at all
#
# FB%/GB% allowed and bullpen rank are included as small, explicitly-labelled UNTESTED
# secondary nudges -- real sabermetric reasoning (fly-ball rate correlates with HR/9), but this
# app has never graded them against its own outcomes the way the signals above are graded, and
# the scoring function says so in its own output rather than presenting them at equal confidence.

HRPO_FAMILY_LIFT = {0: 0.75, 1: 1.07, 2: 1.32, 3: 1.56, 4: 1.98, 5: 1.31}
HRPO_SPOT_LIFT = {1: 1.19, 2: 1.17, 3: 1.25, 4: 1.19, 5: 0.98, 6: 0.83, 7: 0.97, 8: 0.83, 9: 0.60}
HRPO_BASE_RATE = 0.109          # re-anchored from the live backtest's base_pct at call time
HRPO_MIN_BBE = 250               # below this, convergence is graded WORSE than no signal at all


def _longball_score(p):
    """Distance-ceiling score for the Fanatics Longest HR promo -- probability of a hit does not
    matter, only how far it goes IF it goes. Built entirely from real, measured metrics: no
    avg_ev-percentile approximation. max_hit_speed/ev50 come from a real Savant leaderboard
    (batter_exitvelo_barrels); launch35_pct/fbld_ev/avg_bat_speed/fast_swing_rate come from the
    shared Statcast frame (batter_ceiling_profile).

    Returns (score 0-100, drivers[], ceiling_ft) or (None, [], None) if there isn't enough real
    data to score this hitter at all.
    """
    lb = p.get("long_ball") or {}
    hp = (p.get("features") or {}).get("hr_power") or {}
    max_dist = hp.get("max_dist")

    def cl(v):
        return max(0.0, min(1.0, v))

    drivers = []

    # ---- Pillar 1: Absolute Power Ceiling (40%) ----
    max_ev = lb.get("max_hit_speed")
    ev50 = lb.get("ev50")
    bat_speed = lb.get("avg_bat_speed")
    fast_swing = lb.get("fast_swing_rate")
    if not any(v is not None for v in (max_ev, ev50, bat_speed)):
        return None, [], max_dist

    power_parts, power_n = 0.0, 0
    if max_ev is not None:
        power_parts += cl((max_ev - 105) / 13) * 1.4; power_n += 1.4   # weighted heaviest --
        if max_ev >= 115:                                               # the genuine "ceiling"
            drivers.append(f"Max EV: {max_ev:.1f} mph")                 # number this feature
    if ev50 is not None:                                                 # is named for
        power_parts += cl((ev50 - 92) / 10); power_n += 1
        if ev50 >= 98:
            drivers.append(f"EV50: {ev50:.1f} mph")
    if bat_speed is not None:
        power_parts += cl((bat_speed - 68) / 12); power_n += 1
        if bat_speed >= 76:
            drivers.append(f"{bat_speed:.1f} mph bat speed")
    if fast_swing is not None:
        power_parts += cl((fast_swing - 15) / 45) * 0.6; power_n += 0.6
    power = (power_parts / power_n) if power_n else 0.0

    # ---- Pillar 2: Trajectory & Elevated Contact (35%) ----
    launch35 = lb.get("launch35_pct")
    fbld_ev = lb.get("fbld_ev")
    if launch35 is None and fbld_ev is None:
        return None, [], max_dist       # no real trajectory data -- no distance case to make
    traj_parts, traj_n = 0.0, 0
    if launch35 is not None:
        traj_parts += cl((launch35 - 15) / 30) * 1.5; traj_n += 1.5   # primary trajectory
        if launch35 >= 35:                                              # signal per spec --
            drivers.append(f"Launch 20-35°: {launch35:.0f}%")           # weighted heaviest
    if fbld_ev is not None:
        traj_parts += cl((fbld_ev - 90) / 15); traj_n += 1
        if fbld_ev >= 100:
            drivers.append(f"{fbld_ev:.1f} mph on FB/LD")
    traj = (traj_parts / traj_n) if traj_n else 0.0

    # ---- Pillar 3: Environmental Physics (25%) -- the same park_hr_factor the rest of the app
    # already shows, not a separately re-derived environment score. ----
    pf = p.get("park_hr_factor")
    park_c = 0.0
    if pf is not None:
        park_c = cl((pf - 0.90) / 0.30)
        if pf >= 1.05:
            drivers.append(f"Carry: {pf:.2f}x")
        elif pf <= 0.93:
            park_c *= 0.5

    score = power * 40 + traj * 35 + park_c * 25
    return round(max(0, min(100, score)), 1), drivers[:5], max_dist


def build_long_ball_jackpot(players, lb_evbarrels=None):
    """Long Ball Jackpot -- Distance King / Weather-Altitude Play / Mega-Leverage Nuke.

    No book-odds comparison: "longest HR of the day" is a distance CONTEST, not a standard
    HR-prop market, and docs/odds.json has no equivalent line to compare against.
    """
    scored = []
    for p in players:
        score, drivers, ceiling = _longball_score(p)
        if score is None:
            continue
        lb = p.get("long_ball") or {}
        scored.append({
            "id": p.get("id"), "name": p.get("name"), "team": p.get("team"),
            "opp_team": p.get("opp_team"), "game_pk": p.get("game_pk"),
            "spot": p.get("lineup_spot"), "score": score, "drivers": drivers,
            "ceiling_ft": ceiling, "park_hr_factor": p.get("park_hr_factor"),
            "max_ev": lb.get("max_hit_speed"), "ev50": lb.get("ev50"),
        })
    scored.sort(key=lambda x: -x["score"])

    picks = []
    used = set()

    def _pick(pool, label, note=None):
        for x in pool:
            if x["id"] not in used:
                used.add(x["id"])
                picks.append({**x, "pick_label": label, "pick_note": note})
                return
    _pick(scored, "The Distance King")

    by_game_pf = {}
    for x in scored:
        by_game_pf.setdefault(x["game_pk"], []).append(x)
    best_game = max(by_game_pf.items(),
                    key=lambda kv: max((y.get("park_hr_factor") or 0) for y in kv[1]),
                    default=(None, []))[0]
    if best_game is not None:
        game_pool = sorted(by_game_pf.get(best_game, []),
                           key=lambda x: -(x.get("max_ev") or x.get("ev50") or 0))
        _pick(game_pool, "The Weather/Altitude Play")

    # Per-game Long Ball board -- every scored batter from BOTH teams, ranked together, same
    # shape as the Grand Slam Board. Tier cutoffs are score-based (this contest has no real
    # probability, only a ranking) -- readability buckets, not a validated system, same
    # disclaimer as Grand Slam's tiers.
    def _lb_tier(sc):
        if sc >= 75: return "STRONG"
        if sc >= 55: return "SOLID"
        if sc >= 35: return "LONGSHOT"
        return "DARTS"

    lb_board = []
    for gpk, rows in by_game_pf.items():
        rows_sorted = sorted(rows, key=lambda x: -x["score"])
        batters = [{"rank": i + 1, "id": x["id"], "name": x["name"], "team": x["team"],
                   "spot": x.get("spot"), "score": x["score"], "drivers": x["drivers"],
                   "ceiling_ft": x.get("ceiling_ft"), "max_ev": x.get("max_ev"),
                   "ev50": x.get("ev50"), "tier": _lb_tier(x["score"])}
                  for i, x in enumerate(rows_sorted)]
        if not batters:
            continue
        # Team order here reflects whichever team's hitter scored highest first, NOT true
        # away/home -- this function only sees batters, not the game schedule itself, so
        # labelling them "team_a"/"team_b" rather than "away"/"home" avoids implying a
        # direction this data doesn't actually carry.
        _teams = list(dict.fromkeys(x.get("team") for x in rows_sorted if x.get("team")))
        lb_board.append({"game_pk": gpk, "n_batters": len(batters),
                         "team_a": _teams[0] if len(_teams) > 0 else None,
                         "team_b": _teams[1] if len(_teams) > 1 else None,
                         "best_ceiling_ft": max((b.get("ceiling_ft") or 0) for b in batters),
                         "batters": batters})
    lb_board.sort(key=lambda g: -g["best_ceiling_ft"])

    # Mega-Leverage Nuke -- TRUE MLB-wide top 5% of max_hit_speed, from the full Savant
    # leaderboard (lb_evbarrels covers every qualified MLB batter, not just tonight's slate),
    # who is NOT one of the 10 highest-scored names on tonight's own board.
    top10_ids = {x["id"] for x in scored[:10]}
    mlb_max_evs = sorted((v.get("max_hit_speed") for v in (lb_evbarrels or {}).values()
                         if v.get("max_hit_speed") is not None))
    p95_threshold = (mlb_max_evs[int(len(mlb_max_evs) * 0.95)]
                    if len(mlb_max_evs) >= 20 else None)
    if p95_threshold is not None:
        nuke_pool = [x for x in scored
                    if x["id"] not in top10_ids and (x.get("max_ev") or 0) >= p95_threshold]
        nuke_pool.sort(key=lambda x: -(x.get("max_ev") or 0))
        _pick(nuke_pool, "The Mega-Leverage Nuke",
             note=f"Top 5% MLB max EV ({p95_threshold:.1f}+ mph), off the radar")

    return {"picks": picks, "candidates_scored": len(scored),
            "mlb_p95_max_ev": p95_threshold, "board": lb_board,
            "notes": ["Absolute Power Ceiling uses real max_hit_speed/ev50/bat speed from a "
                     "Savant leaderboard fetch and the shared Statcast frame -- not an "
                     "avg_ev-percentile approximation.",
                     "No book-odds comparison: this is a distance contest, not a standard HR "
                     "prop market, and there is no equivalent line in docs/odds.json to check "
                     "against.",
                     "Mega-Leverage Nuke uses max_hit_speed's TRUE MLB-wide percentile (from "
                     "the full Savant leaderboard, not tonight's slate)."]}


def _hrpo_calibrated_prob(heat, backtest_calib):
    """Same heat-band calibration the frontend's calibratedHRprob() uses, computed server-side.
    Used as the FLOOR/anchor, not the whole score -- see _hrpo_raw_signals for why this
    function stopped being the only thing that mattered."""
    if heat is None or not backtest_calib:
        return None
    b = max(0, min(90, int(heat // 10) * 10))
    lo = backtest_calib.get(str(b)) or {}
    hi = backtest_calib.get(str(min(90, b + 10))) or {}
    rlo = (lo["hr"] / lo["n"]) if lo.get("n", 0) >= 25 else None
    rhi = (hi["hr"] / hi["n"]) if hi.get("n", 0) >= 25 else None
    if rlo is None and rhi is None:
        return None
    if rlo is None:
        return rhi
    if rhi is None:
        return rlo
    frac = (heat - b) / 10.0
    return rlo + (rhi - rlo) * frac


# Badge conversion rates, pulled from the SAME season-scope aggregation the Calendar's "Trends &
# what's working" panel computes live from HISTORY.days[].by_badge (n>=25 filter, matching that
# panel exactly). Re-checked directly against the tracker before this rewrite:
#   lock  15.5%  n=766   pow  15.2%  n=1,673   mix  17.0%  n=100 (thin)   hrsp  12.6%  n=1,051
# Base rate is ~11.8-10.9% depending on window, so lock/pow sit at a real, consistent ~1.3x;
# mix is real but thin enough that n itself should gate how much weight it gets; hrsp is only
# marginally above base rate, not the strong signal its raw percentage makes it look like next
# to lock/pow -- included at much lower weight than the other three for exactly that reason.
HRPO_BADGE_LIFT = {"lock": 1.31, "pow": 1.29, "mix": 1.20, "hrsp": 1.07}
HRPO_MIX_MIN_N = 100     # mix's own graded sample -- weight scaled down below this, not just gated

HRPO_SPOT_LIFT = {1: 1.19, 2: 1.17, 3: 1.25, 4: 1.19, 5: 0.98, 6: 0.83, 7: 0.97, 8: 0.83, 9: 0.60}
HRPO_BASE_RATE = 0.109
HRPO_MIN_BBE = 250

# Ideal launch angle: the one traditional "underlying mechanic" signal that actually holds up at
# BOTH scopes when checked directly against the tracker -- +12% season (n=2,981), +31% this
# month (n=750). Barrel% deliberately does NOT get an equivalent standalone weight here: it
# shows +31% this month (n=193) but -2% for the full season (n=1,126) -- the larger, more
# reliable sample says it adds nothing once heat is already known, which makes sense, since
# barrel_pct is already 22 of the 100 points heat itself is built from (see the double-counting
# fix from the previous version of this function, which this rewrite does not reintroduce).
HRPO_IDEAL_AA_LIFT = 1.12


def _hrpo_raw_signals(p, pp_by_id, pen_by_team, backtest_calib, base_rate):
    """Every dimensional input this scorer draws on, computed ONCE per hitter and handed to
    BOTH ticket formulas below -- so 'Arsenal & Lock' and 'Air-Power & Volume' are two genuinely
    different weightings of the same real inputs, not two cosmetic views of one score.

    Returns None if the hitter can't be scored at all (no heat).
    """
    heat = p.get("heat")
    if heat is None:
        return None

    base_prob = _hrpo_calibrated_prob(heat, backtest_calib)
    if base_prob is None:
        base_prob = base_rate * max(0.6, min(1.6, heat / 55.0))

    bbe = ((p.get("sample") or {}).get("season")) or 0
    thin = bbe < HRPO_MIN_BBE

    conv = ((p.get("converge") or {}).get("hr")) or {}
    fams = len(conv.get("measured", [])) + len(conv.get("provisional", []))
    nm = ((p.get("near_miss") or {}).get("near")) or 0

    # Dimensional matchup -- arsenal_fit is a direct, already-computed, already-graded 0-15ish
    # score (11+ = 1.39x, real, n=7,462). zone_edge/overlap are matchup-specific, attached to
    # THIS hitter's own record only when he faces tonight's specific arm with real zone data --
    # "one hitter faces one arm, so this is unambiguous" per the code that attaches it.
    arsenal_fit = p.get("arsenal_fit")
    zprof = ((p.get("features") or {}).get("zone_profile")) or {}
    _ze = zprof.get("zone_edge")
    zone_edge = _ze.get("edge_score") if isinstance(_ze, dict) else _ze
    zone_overlap_n = ((zprof.get("overlap") or {}).get("count")) if zprof.get("overlap") else None

    # Badges actually carried tonight, weighted by their real season conversion rate above.
    badges = {b.get("k") for b in (p.get("badges") or []) if b.get("k")}

    # Ideal launch angle: whether THIS hitter clears the anchor tonight -- the literal boolean
    # the tracker's own by_signal grading is built from, not a re-derived approximation.
    sig_flags = ((p.get("score_breakdown") or {}).get("signals")) or {}
    ideal_aa_clear = sig_flags.get("ideal_aa_pct") is True

    spot = p.get("lineup_spot")

    _arm_id = (p.get("opp_pitcher") or {}).get("id")
    _arm_pp = pp_by_id.get(_arm_id) if _arm_id is not None else None
    fb_pct_allowed = _arm_pp.get("fb_pct") if _arm_pp else None
    _pen = pen_by_team.get(p.get("opp_team"))
    pen_worn = bool(_pen and str(_pen.get("label") or "").upper() in ("WORN", "GASSED"))

    return {
        "heat": heat, "base_prob": base_prob, "thin": thin, "proven": not thin,
        "fams": fams, "near_miss": nm, "arsenal_fit": arsenal_fit,
        "zone_edge": zone_edge, "zone_overlap_n": zone_overlap_n,
        "badges": badges, "ideal_aa_clear": ideal_aa_clear, "spot": spot,
        "fb_pct_allowed": fb_pct_allowed, "pen_worn": pen_worn,
    }


def _hrpo_combine_arsenal_lock(sig):
    """Ticket 1 -- 'Arsenal & Lock Matchup'. Weighted toward dimensional matchup fit and
    conversion badges, explicitly allowed to surface a 55-68 heat hitter over a higher-heat one
    with a weaker matchup -- the whole point of this ticket per the brief.

    Damping (0.45) on the correlated-with-heat pieces carries over from the previous version's
    fix: family/near-miss lift was measured as marginal lift over a flat base rate, not lift
    conditional on heat already being known, and stacking undamped multipliers overstated the
    combination badly enough to need catching in testing last time.
    """
    DAMP = 0.45
    prob = sig["base_prob"]
    drivers = []

    af = sig["arsenal_fit"]
    if af is not None and af >= 11:
        prob *= 1.0 + (1.39 - 1.0) * 0.70          # arsenal fit is NOT heat-correlated the way
        drivers.append(f"arsenal fit {af:.0f} (elite, 1.39x graded)")   # families/near-miss are,
    elif af is not None and af >= 9:                                     # so it keeps more of
        prob *= 1.0 + (1.11 - 1.0) * 0.70                                # its measured strength
        drivers.append(f"arsenal fit {af:.0f} (good, 1.11x graded)")

    ze = sig["zone_edge"]
    zn = sig["zone_overlap_n"]
    if ze is not None and ze >= 70:
        prob *= 1.0 + (1.88 - 1.0) * 0.55           # thin sample (n=78) -- damped harder than
        drivers.append(f"zone edge {ze:.0f} (n=78 graded, damped)")      # arsenal fit specifically
    elif zn is not None and zn >= 5:
        prob *= 1.0 + (1.38 - 1.0) * 0.70
        drivers.append(f"{zn} zone overlaps (5+ premium, 1.38x graded)")
    elif zn is not None and zn >= 3:
        prob *= 1.0 + (1.18 - 1.0) * 0.70
        drivers.append(f"{zn} zone overlaps (viable, 1.18x graded)")

    badge_mult = 1.0
    for bk in ("lock", "mix"):
        if bk in sig["badges"]:
            lift = HRPO_BADGE_LIFT[bk]
            w = 0.70 if bk != "mix" else 0.70 * min(1.0, 250 / max(HRPO_MIX_MIN_N, 1))
            badge_mult *= 1.0 + (lift - 1.0) * w
            drivers.append(f"{bk} badge ({lift:.2f}x graded)")
    prob *= badge_mult

    if sig["fb_pct_allowed"] is not None and sig["fb_pct_allowed"] >= 42:
        prob *= 1.05          # untested interaction -- kept small and labelled, same as before
        drivers.append(f"FB-heavy arm ({sig['fb_pct_allowed']:.0f}%, untested matchup nudge)")

    if not sig["thin"] and sig["fams"] >= 2:
        prob *= 1.0 + (HRPO_FAMILY_LIFT_LOOKUP(sig["fams"]) - 1.0) * DAMP
        drivers.append(f"{sig['fams']} families (damped)")

    spot = sig["spot"]
    if spot is not None:
        mult = HRPO_SPOT_LIFT.get(int(spot), 0.85)
        prob *= 1.0 + (mult - 1.0) * DAMP
        drivers.append(f"spot #{int(spot)}")
    else:
        prob *= 0.92

    return max(0.01, min(0.27, prob)), drivers[:4]


def _hrpo_combine_air_power(sig):
    """Ticket 2 -- 'Air-Power & Volume'. Weighted toward ideal launch angle (the one traditional
    power signal that actually holds up season-long), the pow badge, and top-of-order volume.

    Barrel% is deliberately absent here even though the brief asked for it as this ticket's
    namesake signal -- checked directly against the tracker and it shows -2% lift for the full
    season (n=1,126), essentially nothing once heat (which is already 22% barrel_pct) is known.
    Ideal launch angle is the real version of what this ticket is trying to capture: +12%
    season, +31% this month, both on real sample sizes.
    """
    DAMP = 0.45
    prob = sig["base_prob"]
    drivers = []

    if sig["ideal_aa_clear"]:
        prob *= 1.0 + (HRPO_IDEAL_AA_LIFT - 1.0) * 0.75
        drivers.append(f"ideal launch angle clears ({HRPO_IDEAL_AA_LIFT:.2f}x graded, season)")

    if "pow" in sig["badges"]:
        lift = HRPO_BADGE_LIFT["pow"]
        prob *= 1.0 + (lift - 1.0) * 0.70
        drivers.append(f"pow badge ({lift:.2f}x graded)")
    if "hrsp" in sig["badges"]:
        lift = HRPO_BADGE_LIFT["hrsp"]
        prob *= 1.0 + (lift - 1.0) * 0.40      # weakest of the four graded badges -- weighted down
        drivers.append(f"hrsp badge ({lift:.2f}x graded, modest)")

    if not sig["thin"] and sig["near_miss"] >= 2:
        prob *= 1.0 + (1.11 - 1.0) * DAMP
        drivers.append(f"{sig['near_miss']} near misses (damped)")

    spot = sig["spot"]
    if spot is not None:
        mult = HRPO_SPOT_LIFT.get(int(spot), 0.85)
        # Top-of-order volume weighted less-damped here specifically -- this ticket's own name
        # is "...& Volume", so PA count matters more to it than to the matchup-first ticket.
        prob *= 1.0 + (mult - 1.0) * 0.60
        drivers.append(f"spot #{int(spot)} ({mult:.2f}x graded)")
    else:
        prob *= 0.88

    if sig["pen_worn"]:
        prob *= 1.05
        drivers.append("worn/gassed pen (untested nudge)")

    return max(0.01, min(0.27, prob)), drivers[:4]


def HRPO_FAMILY_LIFT_LOOKUP(fams):
    return {0: 0.75, 1: 1.07, 2: 1.32, 3: 1.56, 4: 1.98, 5: 1.31}.get(min(fams, 5), 1.0)



def build_cross_game_hr_parlays(players, backtest_calib, odds_prices, games,
                                pitcher_props=None, bullpen_rankings=None):
    """The Anchor and Overlooked +EV 3-leg tickets. Every leg comes from a distinct game_pk --
    the whole point is eliminating the sportsbook's same-game-parlay correlation tax, which only
    works if the legs genuinely do not correlate with each other, i.e. different games.

    Book odds: HR prop odds live in docs/odds.json under `prices`, name-matched the same way
    FanGraphs is name-matched elsewhere in this file. When odds.json has not been fetched yet
    (or a specific player has no line), the ticket still builds -- fair odds from this app's own
    model are always shown; the live-book / +EV comparison is shown only when a real price
    exists, never estimated or invented.
    """
    base_rate = HRPO_BASE_RATE
    pp_by_id = {a.get("id"): a for a in (pitcher_props or []) if a.get("id") is not None}
    pen_by_team = {r.get("team"): r for r in (bullpen_rankings or [])}

    def _odds_for(name):
        if not odds_prices:
            return None
        return odds_prices.get(statcast_data._norm_name(name))

    env_ok = {}
    for g in games or []:
        gpk = g.get("game_pk")
        temp = g.get("temp_f")
        wind_out = g.get("wind_out")
        try:
            carry = env_mod.carry_multiplier(
                temp if temp is not None else 70.0,
                elevation_ft=(g.get("elevation_ft") or 0.0))
            if wind_out is False:
                carry *= 0.94        # wind blowing IN specifically, not just "not out"
        except Exception:
            carry = 1.0
        # Physics-based gate, not an invented empirical threshold: this app has no backtested
        # temperature signal (checked -- by_edge has no weather bucket at all), so the cutoff is
        # the carry model's own output, not a guessed degree number. 0.92x is a real, meaningful
        # suppression -- roughly what carry_multiplier gives a 45F night at sea level.
        env_ok[gpk] = carry >= 0.92

    candidates_al = []      # scored via the Arsenal & Lock formula
    candidates_ap = []      # scored via the Air-Power & Volume formula
    for p in players:
        gpk = p.get("game_pk")
        if gpk is None or not env_ok.get(gpk, True):
            continue
        sig = _hrpo_raw_signals(p, pp_by_id, pen_by_team, backtest_calib, base_rate)
        if sig is None:
            continue
        odds = _odds_for(p.get("name"))
        book_prob = None
        if odds and odds.get("best") is not None:
            am = odds["best"]
            book_prob = (100.0 / (am + 100.0)) if am > 0 else (-am / (-am + 100.0))

        def _record(prob, drivers):
            edge = round(prob - book_prob, 4) if book_prob else None
            # A model/book ratio this extreme is more likely leftover model overconfidence than a
            # real edge a heuristic scorer of this size can reliably claim to have found.
            edge_extreme = bool(book_prob and prob > 2.5 * book_prob)
            return {
                "id": p.get("id"), "name": p.get("name"), "team": p.get("team"),
                "opp_team": p.get("opp_team"), "game_pk": gpk, "spot": p.get("lineup_spot"),
                "prob": round(prob, 4), "drivers": drivers, "proven": sig["proven"],
                "fams": sig["fams"], "near_miss": sig["near_miss"],
                "book_odds": odds.get("best") if odds else None,
                "book_prob": round(book_prob, 4) if book_prob else None,
                "edge": edge, "edge_extreme": edge_extreme,
            }

        prob_al, drv_al = _hrpo_combine_arsenal_lock(sig)
        candidates_al.append(_record(prob_al, drv_al))
        prob_ap, drv_ap = _hrpo_combine_air_power(sig)
        candidates_ap.append(_record(prob_ap, drv_ap))

    def _fair(prob):
        if not prob or prob <= 0:
            return None
        dec = 1.0 / prob
        return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))

    def _pick_distinct_games(ranked, n=3):
        picked, used_games = [], set()
        for c in ranked:
            if c["game_pk"] in used_games:
                continue
            picked.append(c)
            used_games.add(c["game_pk"])
            if len(picked) == n:
                break
        return picked

    def _ticket(label, subtitle, ranked):
        legs = _pick_distinct_games(ranked, 3)
        if len(legs) < 3:
            return {"label": label, "subtitle": subtitle, "legs": legs,
                    "combined_prob": None, "fair_odds": None,
                    "book_combined_prob": None, "book_fair_odds": None, "ev_pct": None,
                    "incomplete": True, "ev_extreme": False,
                    "note": f"Only {len(legs)} qualifying leg(s) across distinct games tonight — "
                            "not enough of the slate cleared the bar to build a full 3-leg ticket."}
        combined = 1.0
        for x in legs:
            combined *= x["prob"]
        book_combined = None
        if all(x.get("book_prob") for x in legs):
            book_combined = 1.0
            for x in legs:
                book_combined *= x["book_prob"]
        ev = None
        if book_combined:
            ev = round(100 * (combined - book_combined) / book_combined, 1)
        return {"label": label, "subtitle": subtitle, "legs": legs,
                "combined_prob": round(combined, 5), "fair_odds": _fair(combined),
                "book_combined_prob": round(book_combined, 5) if book_combined else None,
                "book_fair_odds": _fair(book_combined) if book_combined else None,
                "ev_pct": ev, "incomplete": False,
                "ev_extreme": any(x.get("edge_extreme") for x in legs)}

    # ARSENAL & LOCK MATCHUP: proven-sample hitters, ranked by the matchup-first formula --
    # allowed to surface a 55-68 heat hitter with a strong arsenal/zone/badge case over a
    # higher-heat hitter with a weaker one, per the brief.
    al_pool = [c for c in candidates_al if c["proven"]]
    al_pool.sort(key=lambda x: -x["prob"])
    arsenal_lock = _ticket("Arsenal & Lock Matchup",
                          "Pitch-arsenal fit, zone overlap, and lock/mix badges -- allowed to "
                          "outrank raw heat when the matchup case is strong",
                          al_pool)

    # AIR-POWER & VOLUME: proven-sample hitters, ranked by the ideal-launch/badge/volume
    # formula. Both pools are scored on EVERY qualifying hitter -- a hitter can legitimately
    # appear on both tickets if he is strong on both dimensions, or only one if he is not.
    ap_pool = [c for c in candidates_ap if c["proven"]]
    ap_pool.sort(key=lambda x: -x["prob"])
    air_power = _ticket("Air-Power & Volume",
                        "Ideal launch angle, the pow badge, and top-of-order PA volume in a "
                        "positive-carry park",
                        ap_pool)

    # Alternates for the quick-swap button -- pulled from EACH ticket's own pool, so cycling
    # a leg in "Arsenal & Lock" offers arsenal/zone alternates, not air-power ones.
    by_game_al, by_game_ap = {}, {}
    for c in candidates_al:
        by_game_al.setdefault(c["game_pk"], []).append(c)
    for c in candidates_ap:
        by_game_ap.setdefault(c["game_pk"], []).append(c)
    alternates_by_game = {}
    for gpk in {x["game_pk"] for x in arsenal_lock["legs"]}:
        alternates_by_game.setdefault(str(gpk), sorted(by_game_al.get(gpk, []),
                                                        key=lambda x: -x["prob"])[:3])
    for gpk in {x["game_pk"] for x in air_power["legs"]}:
        k = str(gpk)
        alt = sorted(by_game_ap.get(gpk, []), key=lambda x: -x["prob"])[:3]
        if k not in alternates_by_game:
            alternates_by_game[k] = alt

    return {"anchor": arsenal_lock, "overlooked": air_power,
            "candidates_scored": len(candidates_al),
            "games_env_filtered": sum(1 for v in env_ok.values() if not v),
            "alternates_by_game": alternates_by_game,
            "notes": [
                "Command Risk (Location+<98) and the xFIP-vs-ERA regression nudge are "
                "DELIBERATELY EXCLUDED from this scorer -- this app backtested both against "
                "real outcomes and found no measurable effect (Location+: 10.9% vs 10.9% HR "
                "rate, z=0.04, n=29,828). FB% and bullpen state are included as small, "
                "explicitly-labelled untested nudges, not proven signals.",
                "Barrel% is deliberately absent as a standalone bonus in BOTH tickets -- it "
                "showed -2% lift for the full season (n=1,126) once heat (already 22% "
                "barrel_pct) is known, vs. +31% this single month (n=193). Ideal launch angle "
                "replaces it as this app's real, both-scopes-consistent power signal.",
                "The base probability is the real heat-decile calibration curve from the "
                "backtest, not an open-ended formula -- barrel% is NOT applied a second time on "
                "top of it, since barrel% is already 22% of the heat score itself.",
                "Each remaining multiplier (families, near-miss, lineup spot) is individually "
                "graded on its own in the backtest. Their PRODUCT together -- stacking all "
                "three on one hitter -- has not itself been backtested as a compound formula, "
                "only each factor in isolation. Treat a maxed-out driver list as a strong case "
                "built from real evidence, not as a number proven to that precision.",
                "Combined probabilities assume the three legs are INDEPENDENT because they are "
                "forced into three different games. Real same-game correlation is not what "
                "this ticket is testing -- that is a different, currently-unmeasured question "
                "the app's HR log only started tracking game_pk for as of this build.",
            ]}


def _fg_with_defaults(fg_entry):
    """Fill a per-pitcher FanGraphs record with neutral defaults so every consumer — ETL or
    frontend — can read `.fg.stuff_plus` etc. without a None-guard at every call site.

    Plus-stats (stuff_plus, location_plus, pitching_plus) default to 100, their own genuine
    league-average value — NOT a sentinel, an honest neutral. Rate stats (xfip, siera, k_bb_pct)
    have no equivalent honest default (there is no "average" xFIP that means "unknown" without
    being mistaken for a real number), so those stay None and every caller already treats a rate
    of None as "no adjustment", which is correct.
    """
    e = fg_entry or {}
    return {
        "xfip": e.get("xfip"), "siera": e.get("siera"),
        "stuff_plus": e.get("stuff_plus") if e.get("stuff_plus") is not None else 100,
        "location_plus": e.get("location_plus") if e.get("location_plus") is not None else 100,
        "pitching_plus": e.get("pitching_plus") if e.get("pitching_plus") is not None else 100,
        "k_bb_pct": _pct_scale(e.get("k_bb_pct")), "ip": e.get("ip"),
    }


def _hnote(sub, err):
    BUILD_HEALTH.append({"sub": sub, "issue": f"{type(err).__name__}: {err}"[:160]})


RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "45"))  # window for L5/L15/L30
OUT_PATH = os.environ.get("BOARD_OUT", "docs/board.json")
MIN_STATCAST_ROWS = int(os.environ.get("MIN_STATCAST_ROWS", "5000"))
PULL_RETRIES = int(os.environ.get("PULL_RETRIES", "3"))


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ").strip()


def build(date_str: str | None = None) -> dict:
    now = datetime.now(ET)
    date_str = date_str or now.strftime("%Y-%m-%d")
    print(f"[build] slate {date_str}")

    slate = statsapi.get_slate(date_str)
    games = slate["games"]
    print(f"[build] {len(games)} games, lineups posted for {len(slate['lineups'])}")

    # Fall back to each team's last batting order (projected) where today's
    # lineup isn't posted yet, so the board isn't blank in the morning.
    yest = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    _recent_cache = {}

    def _recent(team_id):
        if team_id not in _recent_cache:
            _recent_cache[team_id] = statsapi.get_recent_lineup(team_id, yest)
        return _recent_cache[team_id]

    projected_sides = set()
    for g in games:
        pk = g["game_pk"]
        lu = slate["lineups"].get(pk) or {}
        away = lu.get("away") or None
        home = lu.get("home") or None
        # fill EACH side independently — a posted away lineup must not block a
        # projected home lineup (partial postings otherwise erase a whole team)
        if not away:
            away = _recent(g["away_id"])
            if away:
                projected_sides.add((pk, "away"))
        if not home:
            home = _recent(g["home_id"])
            if home:
                projected_sides.add((pk, "home"))
        if away or home:
            slate["lineups"][pk] = {"away": away or [], "home": home or []}
    proj_game_pks = {pk for (pk, _s) in projected_sides}
    print(f"[build] projected {len(projected_sides)} lineup side(s) across {len(proj_game_pks)} game(s)")

    # collect batter ids from posted lineups
    batter_ids, game_of_batter, side_of_batter, spot_of_batter, status_of_batter = [], {}, {}, {}, {}
    for pk, lu in slate["lineups"].items():
        gmeta = next((g for g in games if g["game_pk"] == pk), None)
        if not gmeta:
            continue
        for i, bid in enumerate(lu.get("away", [])):
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "away"; spot_of_batter[bid] = i + 1
            status_of_batter[bid] = "projected" if (pk, "away") in projected_sides else "confirmed"
        for i, bid in enumerate(lu.get("home", [])):
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "home"; spot_of_batter[bid] = i + 1
            status_of_batter[bid] = "projected" if (pk, "home") in projected_sides else "confirmed"
    batter_ids = list(dict.fromkeys(batter_ids))

    pitcher_ids = [p for g in games for p in (g["away_pitcher_id"], g["home_pitcher_id"])]

    # handedness for everyone
    hands = statsapi.get_handedness(batter_ids + [p for p in pitcher_ids if p])

    # one big Statcast pull -> recent windows + season + pitcher allowed
    end = date_str
    print(f"[build] pulling Statcast {SEASON_START}..{end}")
    df = statcast_data.pd.DataFrame()
    for attempt in range(1, PULL_RETRIES + 1):
        try:
            df = statcast_data.pull_season(SEASON_START, end)
        except Exception as e:
            print(f"[build] statcast pull attempt {attempt} failed: {e}")
            df = statcast_data.pd.DataFrame()
        if len(df) >= MIN_STATCAST_ROWS:
            break
        if attempt < PULL_RETRIES:
            wait = 30 * attempt
            print(f"[build] got {len(df)} rows (<{MIN_STATCAST_ROWS}); retrying in {wait}s")
            time.sleep(wait)

    # GUARD: if Savant came back empty/short, do NOT zero out a good board.
    if len(df) < MIN_STATCAST_ROWS:
        print(f"[build] Statcast insufficient ({len(df)} rows). Keeping last good board.")
        raise statcast_data.StatcastUnavailable(len(df))

    profiles = statcast_data.batter_profiles(df, batter_ids, date_str)
    bb_samples = statcast_data.batted_ball_sample(df, batter_ids)
    bb_logs = statcast_data.batted_ball_log(df, batter_ids)
    try:
        bat_tables = statcast_data.batter_pitch_tables(df, batter_ids)     # full BvP line by pitch
        arm_tables = statcast_data.pitcher_pitch_tables(df, pitcher_ids)
        pitch_hist = statcast_data.pitch_history(df, pitcher_ids)          # usage per start
        team_ks = statcast_data.team_k_splits(df)                          # lineup K% by context
        sprint = statcast_data.sprint_speeds()                             # hit model: infield singles

        # ---- Phase-2 dependencies ----
        # CSW and pitches-per-PA come from the frame already in memory; framing is a separate
        # leaderboard. Each is non-fatal on its own so one missing source cannot take the
        # board down — the engines fall back to their league defaults.
        csw_map = kengine.csw_from_statcast(df, pitcher_ids)
        framing = kengine.catcher_framing()
        pitch_limits, velo_drops = {}, {}
        for _pid in pitcher_ids:
            try:
                pl = kengine.pitch_limit_from_starts(df, _pid)
                if pl:
                    pitch_limits[int(_pid)] = round(pl, 1)
                vd, _flag = kengine.velocity_flag(df, _pid, date_str)
                if vd is not None:
                    velo_drops[int(_pid)] = vd
            except Exception:
                continue
        # directional defense: expected hits vs actual hits allowed, by spray bucket
        dir_def = env_mod.directional_defense_proxy(df, days=30, asof=date_str)

        # Bat tracking per hitter (Statcast 2024+). Swings only — a null bat_speed means the
        # batter did not swing, not that he swung slowly, so including them would drag every
        # average toward zero.
        bat_track = {}
        try:
            _bt = df[df["bat_speed"].notna()] if "bat_speed" in df.columns else None
            if _bt is not None and not _bt.empty:
                for _bid, _g in _bt.groupby("batter"):
                    if int(_bid) not in set(batter_ids):
                        continue
                    prof = arsenal_mod.bat_tracking_profile(
                        _g["bat_speed"].tolist(),
                        _g["swing_length"].tolist() if "swing_length" in _g.columns else None)
                    if prof and prof.get("n_swings", 0) >= 25:
                        bat_track[int(_bid)] = prof
        except Exception as _e:
            print(f"[build] bat tracking skipped: {_e}")

        # Geometry-aware near misses over the trailing 14 days
        near_miss = {}
        try:
            near_miss = env_mod.near_miss_log(df, batter_ids, days=14, asof=date_str)
        except Exception as _e:
            print(f"[build] near-miss log skipped: {_e}")

        # Trailing 3-day reliever pitch logs -> availability. Without this the fatigue model
        # has no input and every bullpen looks fresh.
        pen_logs, pen_state = {}, {}
        try:
            _tids = {g.get("home_id") for g in games} | {g.get("away_id") for g in games}
            pen_logs = statsapi.get_reliever_pitch_logs(_tids, date_str, days=3)
            print(f"[build] reliever pitch logs: {len(pen_logs)} arms")
        except Exception as _e:
            print(f"[build] reliever pitch logs skipped: {_e}")

        print(f"[build] bat tracking {len(bat_track)} hitters | near-miss {len(near_miss)}")
        print(f"[build] CSW arms {len(csw_map)} | framing {len(framing)} | "
              f"pitch limits {len(pitch_limits)} | velo drops {len(velo_drops)} | "
              f"directional def {len(dir_def)} teams")
        fg_pitch = statcast_data.fangraphs_pitching()                      # xFIP/SIERA/Stuff+
        first_inn = statcast_data.first_inning_splits(df, pitcher_ids)     # F3/F5 slow starters
        print(f"[build] fangraphs arms: {len(fg_pitch)} | first-inning splits: {len(first_inn)}")
        # Long Ball Jackpot ceiling metrics -- launch35_pct/fbld_ev/bat speed computed from the
        # SAME shared `df` already in memory (no second Statcast pull), ev50/max_hit_speed from
        # a small, separate Savant leaderboard fetch (same shape as the FanGraphs call above).
        try:
            lb_ceiling = statcast_data.batter_ceiling_profile(df, batter_ids)
            lb_evbarrels = statcast_data.batter_exitvelo_barrels()
            print(f"[build] long ball ceiling: {len(lb_ceiling)} batters (trajectory/bat speed) "
                  f"| {len(lb_evbarrels)} batters (MLB-wide ev50/max EV leaderboard)")
        except Exception as e:
            lb_ceiling, lb_evbarrels = {}, {}
            _hnote("long ball ceiling metrics", e)
            print(f"[build] long ball ceiling metrics skipped: {e}")
        print(f"[build] bvp tables: {len(bat_tables)} hitters, {len(arm_tables)} arms, "
              f"{len(pitch_hist)} histories, {len(team_ks)} team K splits")
    except Exception as e:
        print(f"[build] BvP tables skipped (non-fatal): {e}")
        bat_tables, arm_tables, pitch_hist, team_ks, sprint = {}, {}, {}, {}, {}
        fg_pitch, first_inn = {}, {}
        csw_map, framing, pitch_limits, velo_drops, dir_def = {}, {}, {}, {}, {}
        bat_track, near_miss, pen_logs, pen_state = {}, {}, {}, {}
    try:
        arsenals = statcast_data.pitcher_arsenal(df, pitcher_ids)     # usage % by batter hand
        vs_pitch = statcast_data.batter_vs_pitch(df, batter_ids)      # hitter vs specific pitch types
        print(f"[build] pitch mix: {len(arsenals)} arsenals, {len(vs_pitch)} hitter splits")
    except Exception as e:
        print(f"[build] pitch-mix data skipped (non-fatal): {e}"); arsenals, vs_pitch = {}, {}
    try:
        team_def = statcast_data.team_defense(df)          # OAA-style runs saved/game, {} if thin
        if team_def:
            print(f"[build] team defense computed for {len(team_def)} teams")
    except Exception as e:
        print(f"[build] team defense skipped (non-fatal): {e}"); team_def = {}
    try:
        _pit_ids = {b["pit"] for lg in bb_logs.values() for b in lg if b.get("pit") is not None}
        bb_pit_names = statcast_data.player_names(_pit_ids)
    except Exception as e:
        print(f"[build] batted-ball pitcher names skipped: {e}"); bb_pit_names = {}
    pitch_profiles = statcast_data.pitcher_profiles(df, pitcher_ids, date_str)

    # Pitcher ROLE (season-long), computed once and threaded through every bullpen call.
    # This is what keeps real starters out of the bullpen — a starter's spot relief
    # appearance, or a bulk starter following an opener, used to land him in the pen pool
    # and his HRs-allowed-as-a-starter got tagged "HR vs PEN."
    try:
        p_roles = statcast_data.pitcher_roles(df)
    except Exception as e:
        p_roles = {}
        _hnote("pitcher roles", e); print(f"[build] pitcher roles skipped: {e}")

    bullpens = statcast_data.bullpen_profiles(df, date_str, roles=p_roles)
    # Bullpen AVAILABILITY: who's burnt from recent usage. The full-roster pen number
    # silently includes a closer who threw the last two nights and can't go tonight.
    try:
        pen_avail = statcast_data.bullpen_availability(df, date_str, roles=p_roles)
        pens_avail = statcast_data.bullpen_profiles_available(df, date_str, pen_avail, roles=p_roles)
        _gassed = [t for t, v in pen_avail.items() if v.get("label") == "GASSED"]
        _gassed_txt = (" — GASSED: " + ", ".join(sorted(_gassed))) if _gassed else ""
        print(f"[build] bullpen availability: {len(pen_avail)} pens analyzed{_gassed_txt}")
    except Exception as e:
        pen_avail, pens_avail = {}, {}
        _hnote("bullpen availability", e); print(f"[build] bullpen availability skipped: {e}")
    # Traded players: trailing profile is park-contaminated until the new sample builds
    try:
        traded = statcast_data.team_changes(df, date_str)
        if traded:
            print(f"[build] team changes detected: {len(traded)} player(s)")
    except Exception as e:
        traded = {}
        _hnote("team changes", e); print(f"[build] team changes skipped: {e}")
    career = statcast_data.career_table(2015, now.year)

    # season batter-vs-pitcher (for the Matchup tab): has this hitter homered off today's
    # starter, or off any active arm in the opponent's pen? Computed from the slate frame.
    bvp = {}
    bvp_pen = {}
    pen_arms = {}
    pen_names = {}
    try:
        # Two BvP tables, deliberately different:
        #   bvp     — ALL PAs. Correct for "HR vs SP": you homered off that guy, period.
        #   bvp_pen — RELIEF PAs only. Correct for "HR vs PEN": the bullpen is a ROLE,
        #             not a person. A homer off a starter in his start is not a bullpen
        #             homer, even if that starter later appears in the pen pool.
        bvp = statcast_data.bvp_table(df)
        bvp_pen = statcast_data.bvp_table(df, relief_only=True)
        pen_arms = statcast_data.bullpen_arms(df, date_str, roles=p_roles)
        all_arms = sorted({pid for arms in pen_arms.values() for pid in arms})
        if all_arms:
            try:
                pen_names = {int(k): v.get("name", "") for k, v in
                             statsapi.get_handedness(all_arms).items()}
            except Exception as e:
                _hnote("bullpen name lookup", e); print(f"[build] bullpen name lookup skipped: {e}")
        _sp_ct = sum(1 for r in p_roles.values() if r["role"] == "SP")
        _rp_ct = sum(1 for r in p_roles.values() if r["role"] == "RP")
        _sw_ct = sum(1 for r in p_roles.values() if r["role"] == "SWING")
        print(f"[build] pitcher roles: {_sp_ct} SP / {_rp_ct} RP / {_sw_ct} SWING")
        print(f"[build] BvP: {len(bvp)} matchups ({len(bvp_pen)} relief-only), "
              f"{sum(len(a) for a in pen_arms.values())} active arms")
    except Exception as e:
        _hnote("BvP", e); print(f"[build] BvP skipped: {e}")

    # career BvP vs the starter (MLB Stats API), cached so the day's builds share one fetch
    import time as _t
    bvp_cache_path = os.path.join(os.path.dirname(__file__), "..", "docs", "bvp_career.json")
    try:
        bvp_career_cache = json.load(open(bvp_cache_path)).get("pairs", {})
    except Exception:
        bvp_career_cache = {}
    _bvp_now = _t.time(); _bvp_ttl = 64800; _bvp_fetched = [0]; _BVP_MAX = 340

    try:                                           # season HR distribution by batting-order slot
        hr_spot = statcast_data.hr_by_lineup_spot(df)
    except Exception as e:
        hr_spot = {}; _hnote("hr_by_spot", e); print(f"[build] hr_by_spot skipped: {e}")

    try:                                           # opener detection: how deep starters really go
        start_lens = statcast_data.starter_lengths(df)
        p_apps = statcast_data.pitcher_appearances(df)
    except Exception as e:
        start_lens, p_apps = {}, {}; _hnote("starter lengths", e); print(f"[build] starter lengths skipped: {e}")

    # ---- PHASE C precomputes: Quick Target, Day/Night, TTO, pitcher ERA/WHIP ----
    # ---- LEAGUE-WIDE STATCAST PERCENTILES (true MLB percentiles from the full season pull) ----
    league_pctl = {}
    try:
        _lg = features.league_batter_stats(df)
        league_pctl = features.league_percentiles(_lg)
        print(f"[build] league percentiles: {len(league_pctl)} qualified MLB batters")
    except Exception as e:
        _hnote("league percentiles", e); print(f"[build] league percentiles skipped: {e}")

    # ---- BALLPARK PAL park factors (primary source for park/weather effect) ----
    # Modeled on ~1M batted balls, with PER-HITTER factors based on each batter's own spray
    # profile. Falls back to the local park model automatically when the key is missing or
    # the API is down. NOTE: BPP is today-and-future only, so the backtest keeps using the
    # local model — expect a small live-vs-backtest divergence on park terms.
    BPP = {"ok": False}
    try:
        from etl import ballparkpal
        BPP = ballparkpal.fetch_all(date_str)
        if BPP.get("ok"):
            print(f"[build] park factors: Ballpark Pal ({len(BPP.get('hitters') or {})} hitter-level)")
        else:
            print(f"[build] park factors: local model ({BPP.get('reason','api unavailable')})")
    except Exception as e:
        BPP = {"ok": False}
        _hnote("ballparkpal", e); print(f"[build] ballparkpal skipped: {e}")

    try:                                           # QUICK TARGET: dangerous lineup spots per arm
        spot_damage = statcast_data.pitcher_damage_by_spot(df)
        print(f"[build] quick target: {len(spot_damage)} arms scored by lineup slot")
    except Exception as e:
        spot_damage = {}; _hnote("quick target", e); print(f"[build] quick target skipped: {e}")

    try:                                           # day/night splits (from sv_id timestamps)
        dn_splits = statcast_data.day_night_splits(df)
        print(f"[build] day/night splits: {len(dn_splits)} batters")
    except Exception as e:
        dn_splits = {}; _hnote("day/night", e); print(f"[build] day/night skipped: {e}")

    try:                                           # times-through-order vulnerability per arm
        tto_by_pid = statcast_data.tto_vulnerability(df)
        print(f"[build] TTO vulnerability: {len(tto_by_pid)} arms")
    except Exception as e:
        tto_by_pid = {}; _hnote("tto", e); print(f"[build] TTO skipped: {e}")

    try:                                           # season ERA/WHIP for today's starters
        _sp_ids = []
        for _g in games:
            for _k in ("away_pitcher_id", "home_pitcher_id"):
                if _g.get(_k):
                    _sp_ids.append(_g[_k])
        pstats = statsapi.get_pitcher_stats(_sp_ids) if _sp_ids else {}
        print(f"[build] pitcher season stats: {len(pstats)}/{len(_sp_ids)} starters (ERA/WHIP)")




    except Exception as e:
        pstats = {}; _hnote("pitcher stats", e); print(f"[build] pitcher stats skipped: {e}")

    # Aggregate each team's bullpen into usable rates + a blended pitch mix. Unavailable
    # arms are dropped entirely rather than averaged down — those innings genuinely fall to
    # someone else, so keeping them would describe a bullpen that will not appear.
    try:
        for _g in games:
            for _side in ("home", "away"):
                _tm = _g.get(_side)
                _tid = _g.get(f"{_side}_id")
                if not _tm or _tm in pen_state:
                    continue
                _arms = []
                for _pid, _log in (pen_logs or {}).items():
                    # Match on the numeric team id carried by each appearance.
                    #
                    # This previously filtered on `pstats[pid]["team"]`, which fails twice over:
                    # pitcher stats expose `teamId` rather than `team`, and they are fetched for
                    # STARTERS only, so a reliever is absent from that table entirely. The
                    # comparison was None != "DET" for every arm, so every bullpen came back
                    # empty and the Bullpens tab silently kept using the older Statcast-inferred
                    # availability while this ran to completion with no error.
                    if not _tid or not any(_e.get("team_id") == _tid for _e in (_log or [])):
                        continue
                    _prof = (pstats or {}).get(int(_pid)) or {}
                    _arms.append({"id": _pid, "pitch_log": _log,
                                  "k_pct": (_prof.get("k_pct_allowed") or 22.5) / 100.0,
                                  "bb_pct": (_prof.get("bb_pct_allowed") or 8.5) / 100.0,
                                  "xwoba": _prof.get("xwobacon_allowed"),
                                  "leverage": 2,
                                  "arsenal": ((arsenals.get(int(_pid)) or {}).get("R") or [])})
                if _arms:
                    _st = env_mod.bullpen_state(_arms)
                    _st["arsenal"] = arsenal_mod.bullpen_arsenal(
                        [a for a in _arms
                         if env_mod.reliever_status(a["pitch_log"])[0] != env_mod.UNAVAILABLE])
                    pen_state[_tm] = _st
        # Loud when it produces nothing. The previous failure ran to completion, printed no
        # error, and left a stale tab — the worst kind of bug because everything looks fine.
        if pen_logs and not pen_state:
            print(f"[build] WARNING: {len(pen_logs)} reliever logs fetched but ZERO bullpens "
                  f"built — team matching is broken, Bullpens tab will use stale data")
            _hnote("bullpen state", "no pens matched from reliever logs")
        if pen_state:
            print(f"[build] bullpen state: {len(pen_state)} teams, "
                  f"{sum(s.get('n_out', 0) for s in pen_state.values())} arms unavailable")
    except Exception as _e:
        print(f"[build] bullpen state skipped: {_e}")



    # ---- FEATURE EDGES (parallel track, never touches heat) ----
    # Reliever cumulative fatigue: per-arm leverage-weighted workload over trailing 5 days.
    fatigue_log = {}
    fatigue_by_pid = {}
    if features is not None:
        try:
            fatigue_log = statcast_data.reliever_appearance_log(df, roles=p_roles)
            for _pid, _apps in fatigue_log.items():
                fatigue_by_pid[_pid] = features.reliever_fatigue(_apps, date_str)
            print(f"[build] reliever fatigue: {len(fatigue_by_pid)} arms scored")
        except Exception as e:
            _hnote("reliever fatigue", e); print(f"[build] reliever fatigue skipped: {e}")
    # Pitcher arsenals for the pitch-matchup join (starters facing today's hitters).
    arsenal_by_pid = {}
    if features is not None:
        try:
            for _pid in pitcher_ids:
                _rows = statcast_data.pitcher_arsenal_rows(df, _pid, date_str)
                _ars = features.pitcher_arsenal(_rows)
                if _ars:
                    arsenal_by_pid[_pid] = _ars
            print(f"[build] pitcher arsenals: {len(arsenal_by_pid)} built")
        except Exception as e:
            _hnote("pitcher arsenals", e); print(f"[build] pitcher arsenals skipped: {e}")
    # Today's forecast temp per game, for the microclimate flag (built early so the row loop
    # can use it). Reuses the same weather source as the Weather dashboard.
    _game_temp = {}
    if _MICRO:
        try:
            from etl import weather as _W, park_geometry as _PG
            for g in games:
                _lat, _lon, _ = _PG.park_coords(g["park"])
                _wx = _W.get_weather(_lat, _lon, g["time"], venue=g["park"])
                _tf = (_wx or {}).get("temp_f")
                if _tf is not None:
                    _game_temp[g["game_pk"]] = round(_tf)
        except Exception as e:
            _hnote("microclimate temps", e); print(f"[build] microclimate temps skipped: {e}")

    try:                                           # B2B: homered in his most recent game AND it was actually yesterday
        b2b_set = statcast_data.hr_last_game(df, slate_date=date_str)
    except Exception as e:
        b2b_set = set(); _hnote("hr_last_game", e); print(f"[build] hr_last_game skipped: {e}")

    try:                                           # per-park wind sensitivity (weekly, archived wx)
        from etl import wind_sens as WS
        ws_cache = os.path.join(os.path.dirname(__file__), "..", "docs", "wind_sens.json")
        park_model.set_wind_sens(WS.load_wind_sensitivity(ws_cache, df=df))
    except Exception as e:
        _hnote("wind sensitivity", e); print(f"[build] wind sensitivity skipped: {e}")

    try:                                           # pitcher batted-ball mix allowed (FB% = target)
        p_batted = statcast_data.pitcher_batted_profile(df)
    except Exception as e:
        p_batted = {}; _hnote("pitcher batted profile", e); print(f"[build] pitcher batted profile skipped: {e}")

    try:                                           # PF-style profile labels (trailing 14d)
        _lab_start = str(statcast_data.game_day_cutoff(df, date_str, 14).date())
        hit_labels = statcast_data.hitter_labels(df, _lab_start)   # same game-day window as the model
        print(f"[build] labels: {sum(1 for v in hit_labels.values() if v=='elite')} elite, "
              f"{sum(1 for v in hit_labels.values() if v=='fb')} fb, "
              f"{sum(1 for v in hit_labels.values() if v=='ld')} ld")
    except Exception as e:
        hit_labels = {}; _hnote("labels", e); print(f"[build] labels skipped: {e}")

    # statsapi and Statcast mostly share team abbreviations; a few differ.
    _TEAM_ALIAS = {"AZ": "ARI", "ARI": "AZ", "CWS": "CHW", "CHW": "CWS",
                   "WSH": "WSN", "WSN": "WSH", "KC": "KCR", "KCR": "KC",
                   "SD": "SDP", "SDP": "SD", "SF": "SFG", "SFG": "SF",
                   "TB": "TBR", "TBR": "TB"}

    def _bullpen_for(abbr):
        if abbr in bullpens:
            return bullpens[abbr]
        return bullpens.get(_TEAM_ALIAS.get(abbr))

    # score every probable pitcher's HR vulnerability once
    pitcher_hr = {}
    for pid, prof_p in pitch_profiles.items():
        pitcher_hr[pid] = compute.pitcher_hr_score(prof_p.get("recent", {}), prof_p.get("season", {}))

    # 2-year HR-by-hand per starter (cached in a repo file so we don't re-pull hourly)
    # _HAND2YR_V invalidates old cache entries when the computation changes.
    # v2 = regular-season-only filter + two-calendar-season window (was trailing 730d
    # with spring training + postseason contamination).
    _HAND2YR_V = 2
    _HAND2YR_PATH = os.path.join(os.path.dirname(OUT_PATH) or ".", "hand2yr.json")
    try:
        with open(_HAND2YR_PATH) as _f:
            hand2yr_cache = json.load(_f)
    except Exception:
        hand2yr_cache = {}
    hand2yr = {}
    for pid in {p for p in pitcher_ids if p}:
        key = str(pid)
        ent = hand2yr_cache.get(key)
        fresh = False
        if ent and ent.get("asof") and ent.get("v") == _HAND2YR_V:
            try:
                fresh = 0 <= (datetime.strptime(date_str, "%Y-%m-%d") -
                              datetime.strptime(ent["asof"], "%Y-%m-%d")).days <= 10
            except Exception:
                fresh = False
        if fresh:
            hand2yr[pid] = ent.get("data")
        else:
            data = statcast_data.pitcher_hand_hr_2yr(pid, date_str)
            if data is not None:
                hand2yr_cache[key] = {"asof": date_str, "v": _HAND2YR_V, "data": data}
                hand2yr[pid] = data
            elif ent and ent.get("v") == _HAND2YR_V:
                # pull failed but we have an older same-version value — keep it
                hand2yr[pid] = ent.get("data")
            # old-version data on a failed pull: drop rather than serve contaminated numbers

    # opposing pitcher lookup per batter
    def opp_pitcher(pk, side):
        g = next((x for x in games if x["game_pk"] == pk), None)
        if not g:
            return None, None
        pid = g["home_pitcher_id"] if side == "away" else g["away_pitcher_id"]
        return pid, g

    players = []
    _discipline_raw = {}      # {batter_id: {chase_pct, zcontact_pct...}} -> percentiled after loop
    def _gated_hit_for(bid, recent, pprof, prow):
        """Contact-gated 1+ hit probability with directional defense and pitch-shape splits."""
        try:
            bats = prow.get("bats") or "R"
            throws = (prow.get("opp_pitcher") or {}).get("throws") or "R"
            eff = arsenal_mod.effective_batter_hand(bats, throws)
            # handedness FIRST, then the usage cutoff — a pitch that is 6% overall can be 22%
            # to one side, and a usage-first filter would drop exactly that putaway pitch
            arm_id = (prow.get("opp_pitcher") or {}).get("id")
            by_hand = arsenals.get(int(arm_id)) if (arsenals and arm_id) else None
            side = arsenal_mod.handedness_first_arsenal(by_hand or {}, eff) if by_hand else None
            shapes = None
            if side and prow.get("vs_pitch"):
                shapes = hitmodel.PitchShapeSplits(side, prow["vs_pitch"])
            # directional defense: stadium LF/CF/RF mapped onto this hitter's pull/oppo frame
            team_def = dir_def.get(prow.get("opp_team")) if dir_def else None
            spray = None
            _w = (prow.get("windows") or {}).get("L14d") or {}
            if _w.get("pull_pct") is not None and _w.get("oppo_pct") is not None:
                pull = float(_w["pull_pct"]) / 100.0
                oppo = float(_w["oppo_pct"]) / 100.0
                spray = {"pull": pull, "oppo": oppo,
                         "center": max(0.0, 1.0 - pull - oppo)}
            zone_def = (env_mod.defense_for_hitter(team_def, spray, bats=eff)
                        if (team_def and spray) else None)
            prob, bd = props.hit_prob_gated(
                recent, pprof, shapes=shapes, spray_profile=spray, def_by_zone=zone_def,
                sprint_speed=(sprint or {}).get(int(bid)),
                lineup_spot=prow.get("lineup_spot"),
                implied_team_total=prow.get("implied_team_total"))
            if prob is None:
                return None
            return {"p_hit": prob, "xpa": bd.get("xpa"), "bip_rate": bd.get("bip_rate"),
                    "xba_con": bd.get("xba_con"), "xba_eff": bd.get("xba_eff"),
                    "whiff": bd.get("whiff"), "shapes": bd.get("shapes")}
        except Exception as _e:
            print(f"[build] gated hit skipped for {bid}: {_e}")
            return None

    def _kengine_for(pid, meta):
        """New K projection for one arm. Returns None if inputs are too thin to trust."""
        try:
            cs = csw_map.get(int(pid))
            if not cs:
                return None
            lk = meta.get("opp_lineup_k_rates")
            if not lk:
                _lk = meta.get("opp_lineup_k_pct")
                lk = [(_lk or 22.0) / 100.0] * 9
            ars = (arsenals.get(int(pid)) or {}) if arsenals else {}
            flat = []
            for _h in ("R", "L"):
                for _p in (ars.get(_h) or []):
                    flat.append(_p)
            # Stuff+ / K-BB% — bounded minority nudges inside kengine.py, never touching CSW
            # itself. Looked up by normalised name (FanGraphs has no MLBAM id to join on).
            # Raw (non-defaulted) lookup here specifically: kengine's stuff_plus->k_bb_pct
            # fallback needs to see a genuine None to know Stuff+ itself is unavailable and
            # fall through to K-BB%. Using the neutral-100-defaulted version would make
            # stuff_plus always "present" (as 100) and silently discard a real k_bb_pct signal
            # on the rare pitcher who has one but not the other.
            _fge_raw = (fg_pitch.get(statcast_data._norm_name(meta.get("name") or ""))
                       if meta.get("name") else None) or {}
            res = kengine.predict_pitcher_k_count(
                cs, lk, arsenal=flat or None,
                framing_runs=framing.get(int(meta.get("catcher_id") or 0)),
                pitch_limit=pitch_limits.get(int(pid)),
                velo_drop=velo_drops.get(int(pid)),
                trailing_k_pct=(meta.get("k_pct") or None),
                stuff_plus=_fge_raw.get("stuff_plus"),
                k_bb_pct=_pct_scale(_fge_raw.get("k_bb_pct")))
            return {"exp_k": res["exp_k"], "xbf": res["xbf"], "csw": res["csw"],
                    "csw_adj": res["csw_adj"], "arsenal_depth": res["arsenal_depth"],
                    "velo_drop": res["velo_drop"], "velo_flag": res["velo_flag"],
                    "framing_runs": res["framing_runs"],
                    "dist": {str(k): v for k, v in (res.get("dist") or {}).items()},
                    "o45": kengine.prob_over_ks(res["dist"], 4.5),
                    "o55": kengine.prob_over_ks(res["dist"], 5.5),
                    "o65": kengine.prob_over_ks(res["dist"], 6.5),
                    "o75": kengine.prob_over_ks(res["dist"], 7.5)}
        except Exception as _e:
            print(f"[build] kengine skipped for {pid}: {_e}")
            return None

    for bid in batter_ids:
        prof = profiles.get(bid, {})
        recent = prof.get("recent", {})
        season = prof.get("season", {})
        hand = hands.get(bid, {})
        bats = hand.get("bats", "R")
        name = hand.get("name", str(bid))
        car = career.get(_norm(name), {})

        pk = game_of_batter[bid]
        side = side_of_batter[bid]
        pid, g = opp_pitcher(pk, side)
        pprof = pitch_profiles.get(pid, {}) if pid else {}
        phr = pitcher_hr.get(pid, {}) if pid else {}
        meta = slate["pitchers"].get(pid, {}) if pid else {}
        throws = hands.get(pid, {}).get("throws", "") if pid else ""

        # switch hitters bat opposite the pitcher's hand — use that side for park factor
        eff_side = bats
        if bats == "S":
            eff_side = "L" if throws == "R" else "R"
        # Park factor: prefer Ballpark Pal's per-HITTER model (built on this batter's own
        # spray profile), then their per-game number, then our local geometry model.
        _pf_local = parks.park_factor(g["park"], eff_side) if g else 1.0
        pf, pf_src = _pf_local, "local"
        try:
            from etl import ballparkpal as _BPP
            pf, pf_src = _BPP.resolve_hr_mult(
                BPP, player_id=bid, player_name=name,
                game_id=(g or {}).get("game_pk"),
                away=(g or {}).get("away"), home=(g or {}).get("home"),
                fallback=_pf_local)
        except Exception:
            pf, pf_src = _pf_local, "local"

        # doubles/triples (XBH) park factor for this hitter, if BallparkPal supplies it
        _xbh = None
        try:
            _h = (BPP.get("hitters") or {}).get(bid) or (BPP.get("hitters") or {}).get(str(bid))
            if _h and _h.get("xbh_mult") is not None:
                _xbh = round(_h["xbh_mult"], 2)
        except Exception:
            _xbh = None
        score, breakdown = compute.heat_score(recent, phr.get("score"))

        # vs-pitch-mix variant: re-weight the two pitch-dependent power signals —
        # avg EV and barrel% — to THIS arm's pitch mix (last 2wk), then recompute Heat.
        # Barrel varies most by pitch type and is the heaviest signal, so this is what
        # actually moves the ranking. Bidirectional: weak-vs-mix hitters drop too.
        mix_prof = compute.pitch_mix_profile(prof.get("pitch_splits_recent"), pprof.get("usage"))
        pmatch = compute.pitch_matchup(prof.get("pitch_splits"), pprof.get("usage"), season.get("barrel_pct"))
        heat_mix = score
        if mix_prof and mix_prof.get("avg_ev") is not None:
            recent_mix = dict(recent)
            recent_mix["avg_ev"] = mix_prof["avg_ev"]
            if mix_prof.get("barrel_pct") is not None:
                recent_mix["barrel_pct"] = mix_prof["barrel_pct"]
            heat_mix, _ = compute.heat_score(recent_mix, phr.get("score"))

        # trend (contact-quality) + the synthesized one-line read
        tr = compute.trend(prof.get("windows", {}).get("L5", {}),
                           prof.get("windows", {}).get("L30", {}),
                           mid_w=prof.get("windows", {}).get("L15", {}))
        eff_hand = eff_side if bats == "S" else bats
        angle = compute.read_angle(
            hand=bats, trend=tr, pitch_matchup=pmatch,
            luck_gap=recent.get("luck_gap"), xwobacon=recent.get("xwobacon"),
            opp_form=(phr.get("form") or {}).get("label"),
            hand_hr=(hand2yr.get(pid) or {}).get("two_yr"), eff_hand=eff_hand)
        badges = compute.player_badges(
            opp_form=(phr.get("form") or {}).get("label"),
            hand_hr=(hand2yr.get(pid) or {}).get("two_yr"), eff_hand=eff_hand,
            pitch_matchup=pmatch, luck_gap=recent.get("luck_gap"), trend=tr,
            xwobacon=recent.get("xwobacon"),
            max_ev=(prof.get("season", {}) or {}).get("max_ev"))

        pr = pprof.get("recent", {})

        # ---- FEATURE EDGES per hitter (parallel track, never touches heat) ----
        feat = {}
        if features is not None:
            try:
                # 1. pitch-type matchup: hitter's family power+whiff profile vs THIS arm's arsenal
                ars = arsenal_by_pid.get(pid) if pid else None
                _hp_rows = None
                if ars:
                    _hp_rows = statcast_data.batter_pitch_rows(df, bid, date_str)
                    hprof = features.hitter_pitch_profile(_hp_rows)
                    if hprof:
                        pm = features.pitch_matchup(hprof, ars)
                        if pm:
                            feat["pitch_matchup"] = pm
                # HR power profile: raw batted-ball power lens (barrel/dist/hr-swing). Reuses the
                # rows already pulled above; only pulls its own if pitch_matchup didn't.
                try:
                    _pp_rows = _hp_rows if _hp_rows is not None else statcast_data.batter_pitch_rows(df, bid, date_str)
                    hpp = features.hr_power_profile(_pp_rows)
                    if hpp:
                        feat["hr_power"] = hpp
                    sqr = features.square_up_rating(_pp_rows)
                    if sqr:
                        feat["square_up"] = sqr
                    # plate discipline (raw chase% + z-contact%); percentile-ranked after the loop
                    try:
                        pdr = features.plate_discipline_raw(_pp_rows)
                        if pdr:
                            _discipline_raw[bid] = pdr
                    except Exception:
                        pass
                except Exception:
                    pass
                # 4. late-HR context: short starter + gassed pen behind the OPPOSING arm.
                # (elevated late HR expectancy this hitter benefits from when facing this team)
                opp_abbr = g["home"] if side == "away" else g["away"] if g else None
                pen_obj = _bullpen_for(opp_abbr) if opp_abbr else None
                if pen_obj and pid:
                    _sl = start_lens.get(pid)
                    exp_ip = _sl.get("med_len") if isinstance(_sl, dict) else _sl
                    pen_pids = pen_obj.get("arm_ids") or []
                    fidx = [fatigue_by_pid[a]["index"] for a in pen_pids if a in fatigue_by_pid]
                    if exp_ip and fidx:
                        lc = features.late_hr_context(exp_ip, fidx)
                        if lc:
                            feat["late_hr"] = lc
                # 2. MICROCLIMATE: this hitter's temp-sensitivity vs today's forecast temp.
                # The horse-genetics edge — flags when tonight's conditions favor/hurt a
                # temperature-fragile hitter, using real per-pitch-timestamp history.
                mp = _MICRO.get(str(bid))
                if mp and mp.get("temp_sensitivity_ev") is not None:
                    today_temp = _game_temp.get(g["game_pk"]) if g else None
                    sens = mp["temp_sensitivity_ev"]        # +ve = loses EV when cool
                    micro = {"sensitivity_ev": sens, "median_temp": mp.get("median_temp"),
                             "warm": mp.get("warm"), "cool": mp.get("cool"),
                             "n": mp.get("n_total"), "today_temp": today_temp}
                    if today_temp is not None and abs(sens) >= 2.0:
                        cold = today_temp < mp.get("median_temp", 70)
                        # fragile hitter (sens>0) in the cold = downgrade; in the warmth = boost
                        if sens >= 2.0:
                            micro["flag"] = "COLD FADE" if cold else "WARM BOOST"
                        elif sens <= -2.0:   # rare: hits better cold
                            micro["flag"] = "COLD BOOST" if cold else "WARM FADE"
                    feat["microclimate"] = micro
            except Exception:
                pass   # feature failure never breaks a row
        ps = pprof.get("season", {})
        opp_pitcher_obj = {
            "id": pid,
            "name": meta.get("name", ""),
            "throws": throws,
            "hr_score": phr.get("score"),
            "recent_score": phr.get("recent_score"),
            "season_score": phr.get("season_score"),
            "form": phr.get("form"),
            "flags": phr.get("flags", []),
            "fg": _fg_with_defaults(fg_pitch.get(statcast_data._norm_name(meta.get("name") or ""))),
            "recent": {
                "barrel_pct_allowed": pr.get("barrel_pct_allowed"),
                "hardhit_pct_allowed": pr.get("hardhit_pct_allowed"),
                "avg_ev_allowed": pr.get("avg_ev_allowed"),
                "hr_per_pa": pr.get("hr_per_pa"),
                "ideal_aa_allowed": pr.get("ideal_aa_allowed"),
                "pull_air_allowed": pr.get("pull_air_allowed"),
                "swstr_pct_allowed": pr.get("swstr_pct_allowed"),
                "fb_velo": pr.get("fb_velo"),
                "velo_trend": pr.get("velo_trend"),
                "bbe": pr.get("bbe"),
            },
            "season": {
                "barrel_pct_allowed": ps.get("barrel_pct_allowed"),
                "hardhit_pct_allowed": ps.get("hardhit_pct_allowed"),
                "avg_ev_allowed": ps.get("avg_ev_allowed"),
                "hr_per_pa": ps.get("hr_per_pa"),
                "ideal_aa_allowed": ps.get("ideal_aa_allowed"),
                "pull_air_allowed": ps.get("pull_air_allowed"),
                "swstr_pct_allowed": ps.get("swstr_pct_allowed"),
                "fb_velo": ps.get("fb_velo"),
            },
        }

        # opener detection: listed SP whose real starts run 1-2 innings, or a pure
        # reliever getting the "start". Downstream, BvP-vs-SP matters less (one look)
        # and the bullpen matters much more.
        _bat = p_batted.get(pid)
        if _bat:
            opp_pitcher_obj["fb_pct"] = _bat["fb_pct"]
            opp_pitcher_obj["ld_pct"] = _bat.get("ld_pct")
            opp_pitcher_obj["gb_pct"] = _bat["gb_pct"]
        _sl = start_lens.get(pid)
        if _sl and _sl["starts"] >= 2 and _sl["med_len"] <= 2.0:
            opp_pitcher_obj["opener"] = True
            opp_pitcher_obj["start_len"] = round(_sl["med_len"], 1)
        elif _sl is None and p_apps.get(pid, 0) >= 5:
            opp_pitcher_obj["opener"] = True          # relieves all year, "starting" today
            opp_pitcher_obj["start_len"] = None
        else:
            opp_pitcher_obj["opener"] = False
            opp_pitcher_obj["start_len"] = round(_sl["med_len"], 1) if _sl else None

        # pitcher platoon splits — what he allows vs this hitter's hand
        eff_hand = eff_side if bats == "S" else bats
        psplits = pprof.get("splits") or {}
        opp_pitcher_obj["platoon"] = compute.platoon_note(psplits)
        opp_pitcher_obj["hr_by_hand"] = {
            "R_hr": (psplits.get("R") or {}).get("season", {}).get("hr_allowed"),
            "R_pa": (psplits.get("R") or {}).get("season", {}).get("pa"),
            "L_hr": (psplits.get("L") or {}).get("season", {}).get("hr_allowed"),
            "L_pa": (psplits.get("L") or {}).get("season", {}).get("pa"),
        }
        opp_pitcher_obj["hr_by_hand_2yr"] = hand2yr.get(pid)
        vh = compute.hand_vuln(psplits.get(eff_hand)) if eff_hand in ("R", "L") else None
        opp_pitcher_obj["vs_hand"] = eff_hand
        opp_pitcher_obj["vs_hand_score"] = vh["score"] if vh else None
        _vhs = (psplits.get(eff_hand) or {})
        opp_pitcher_obj["vs_hand_metrics"] = {
            "barrel_pct_allowed": (_vhs.get("season") or {}).get("barrel_pct_allowed"),
            "hr_per_pa": (_vhs.get("season") or {}).get("hr_per_pa"),
            "bbe": (_vhs.get("season") or {}).get("bbe"),
        }

        # opposing BULLPEN vulnerability (overall + vs this hitter's hand)
        opp_abbr = g["home"] if side == "away" else g["away"]
        opp_bullpen = compute.bullpen_vuln(_bullpen_for(opp_abbr), eff_hand) if g else None

        # The pen he'll ACTUALLY face: same score recomputed from available arms only.
        # If the closer and setup man threw the last two nights, they're not in tonight's
        # game, and the arms that remain are more HR-prone. Full-pen score kept as
        # opp_bullpen; this is the honest one.
        pen_live = None
        try:
            av = pen_avail.get(opp_abbr) or pen_avail.get(_TEAM_ALIAS.get(opp_abbr)) or {}
            prof_av = pens_avail.get(opp_abbr) or pens_avail.get(_TEAM_ALIAS.get(opp_abbr))
            if av:
                pen_live = {
                    "fatigue": av.get("fatigue"),
                    "label": av.get("label"),
                    "n_out": av.get("n_out"),
                    "n_arms": av.get("n_arms"),
                    "pen_pitches_l2": av.get("pen_pitches_l2"),
                }
                if prof_av:
                    vuln_av = compute.bullpen_vuln(prof_av, eff_hand)
                    if vuln_av:
                        pen_live["score"] = vuln_av.get("score")
                        pen_live["vs_hand"] = vuln_av.get("vs_hand")
                        base = (opp_bullpen or {}).get("score")
                        if base is not None and vuln_av.get("score") is not None:
                            pen_live["delta"] = round(vuln_av["score"] - base, 1)
        except Exception:
            pen_live = None

        metrics = {}
        # the four headline signals first (in your order), then context metrics
        for key in ("pull_air_pct", "avg_ev", "barrel_pct", "ideal_aa_pct",
                    "bat_speed", "hardhit_pct", "iso", "slg", "launch_angle",
                    "fb_pct", "pull_pct", "swstr_pct", "k_pct"):
            metrics[key] = {
                "recent": recent.get(key),
                "season": season.get(key),
                "career": car.get(key),
            }

        # ---- ELITE gate: the four headline thresholds, applied app-wide. A hitter is "elite"
        # when he clears all four on his season profile (season = stable enough to gate on):
        #   pull_air_pct >= 33  (66% of HR are pulled; 33 is the floor, <33 removed)
        #   avg_ev       >= 90  (harder contact = more distance)
        #   barrel_pct   >= 10  (80-86% of HR are barreled; 10 is the good/elite line)
        #   ideal_aa_pct >= league-ish (upward attack angle at contact, launch 5-20)
        # ideal_aa has no universal cutoff yet (new metric); we gate the other three hard and
        # treat ideal_aa as a bonus tier. Each check reports which it cleared so the UI can show it.
        def _egate(src):
            pa_ = src.get("pull_air_pct"); ev_ = src.get("avg_ev")
            br_ = src.get("barrel_pct"); ia_ = src.get("ideal_aa_pct")
            checks = {
                "pull": (pa_ is not None and pa_ >= 33, pa_),
                "ev":   (ev_ is not None and ev_ >= 90, ev_),
                "barrel": (br_ is not None and br_ >= 10, br_),
                "ideal_aa": (ia_ is not None and ia_ >= 55, ia_),   # 55%+ = strong upward-AA rate
            }
            core = ["pull", "ev", "barrel"]                 # the three hard gates
            cleared_core = sum(1 for k in core if checks[k][0])
            is_elite = cleared_core == 3
            return {
                "elite": is_elite,
                "cleared_core": cleared_core,
                "ideal_aa_bonus": checks["ideal_aa"][0],
                "checks": {k: {"ok": v[0], "val": v[1]} for k, v in checks.items()},
                # a tier label for quick scanning
                "tier": ("ELITE+" if is_elite and checks["ideal_aa"][0]
                         else "ELITE" if is_elite
                         else "NEAR" if cleared_core == 2 else "BELOW"),
            }
        elite = _egate(season)
        elite["recent"] = _egate(recent)      # also flag if he's elite on recent form specifically

        # auto "why" line — the cleared signals + arm read, for instant scanning
        why_bits = []
        if recent.get("pull_air_pct") is not None and recent["pull_air_pct"] >= 40:
            why_bits.append(f"{recent['pull_air_pct']:.0f}% air-pull")
        if recent.get("barrel_pct") is not None and recent["barrel_pct"] >= 11:
            why_bits.append(f"{recent['barrel_pct']:.0f}% brl")
        if recent.get("avg_ev") is not None and recent["avg_ev"] >= 88.5:
            why_bits.append(f"{recent['avg_ev']:.1f} EV")
        if recent.get("ideal_aa_pct") is not None and recent["ideal_aa_pct"] >= 58:
            why_bits.append(f"{recent['ideal_aa_pct']:.0f}% IAA")
        if recent.get("iso") is not None and recent["iso"] >= 0.200:
            why_bits.append(f".{int(round(recent['iso']*1000)):03d} ISO")
        oform = (opp_pitcher_obj.get("form") or {}).get("label", "")
        why = " · ".join(why_bits[:3])
        if oform in ("SHELLABLE", "STEADY-BAD", "SLIPPING", "HITTABLE"):
            why = (why + " · " if why else "") + f"vs {oform} arm"

        opp_abbr = g["home"] if side == "away" else g["away"]
        sp_bvp = bvp.get((bid, pid)) if pid else None
        bp_list = []
        for apid in pen_arms.get(opp_abbr, []):
            # relief-only table: a HR this hitter hit off this pitcher WHILE HE WAS
            # STARTING must not count as a bullpen HR
            rec = bvp_pen.get((bid, apid))
            if rec and rec[0] > 0:
                bp_list.append({"name": pen_names.get(apid, ""), "pa": rec[0], "hr": rec[1]})
        bp_list.sort(key=lambda x: (x["hr"], x["pa"]), reverse=True)
        player_bvp = {
            "sp": {"name": opp_pitcher_obj.get("name", ""),
                   "pa": sp_bvp[0] if sp_bvp else 0, "hr": sp_bvp[1] if sp_bvp else 0},
            "bp": [a for a in bp_list if a["name"]][:12],
            "bp_hr": any(a["hr"] > 0 for a in bp_list),
        }
        if pid:                                   # career vs today's starter (cached)
            _k = f"{bid}-{pid}"
            _c = bvp_career_cache.get(_k)
            _sc = None
            if _c and (_bvp_now - _c.get("ts", 0) < _bvp_ttl):
                _sc = {"pa": _c["pa"], "hr": _c["hr"]}
            elif _bvp_fetched[0] < _BVP_MAX:
                _r = statsapi.bvp_career(bid, pid)
                _bvp_fetched[0] += 1
                if _r is not None:
                    bvp_career_cache[_k] = {"pa": _r["pa"], "hr": _r["hr"], "ts": _bvp_now}
                    _sc = {"pa": _r["pa"], "hr": _r["hr"]}
                _t.sleep(0.02)
            if _sc is not None:
                player_bvp["sp_career"] = {"name": opp_pitcher_obj.get("name", ""), **_sc}

        # tags for having gone deep off today's arms (flow into tracker + parlay weighting)
        _bvp_badges = []
        if player_bvp.get("sp_career", {}).get("hr", 0) or player_bvp["sp"]["hr"]:
            _bvp_badges.append({"t": "HR vs SP", "k": "hrsp"})
        if player_bvp["bp_hr"]:
            _bvp_badges.append({"t": "HR vs PEN", "k": "hrbp"})
        badges = _bvp_badges + badges

        players.append({
            "id": bid,
            "name": name,
            "bats": bats,
            "bvp": player_bvp,
            "hr_by_spot": hr_spot.get(bid, {}),
            "hr_last_game": bid in b2b_set,
            "day_night": (g or {}).get("day_night") or "",     # today's game condition
            "dn_split": dn_splits.get(bid),                     # his day vs night history
            "hit_label": hit_labels.get(bid),
            "lineup_spot": spot_of_batter.get(bid),
            "lineup_status": status_of_batter.get(bid, "confirmed"),
            "trend": tr,
            "angle": angle,
            "badges": badges,
            "team": g["away"] if side == "away" else g["home"],
            "opp_team": g["home"] if side == "away" else g["away"],
            "side": side,          # "away"/"home" — used by the Edges tab to match arm vs lineup
            "game_pk": pk,
            "time": g["time"],
            "park": g["park"],
            "park_hr_factor": round(pf, 2),
            "xbh_factor": _xbh,             # BallparkPal doubles/triples park factor (or None)
            "park_src": pf_src,          # 'bpp_hitter' | 'bpp_game' | 'local'
            "why": why,
            "tier": breakdown.get("tier"),
            "cleared": breakdown.get("cleared"),
            "sample": {                       # batted-ball counts so tiny windows are obvious
                "L5": (prof.get("windows", {}).get("L5", {}) or {}).get("bb_count"),
                "L15": (prof.get("windows", {}).get("L15", {}) or {}).get("bb_count"),
                "L30": (prof.get("windows", {}).get("L30", {}) or {}).get("bb_count"),
                "season": season.get("bb_count"),
            },
            "opp_pitcher": opp_pitcher_obj,
            "opp_bullpen": opp_bullpen,
            "opp_pen_live": pen_live,
            "traded": traded.get(bid),
            "pitch_splits": prof.get("pitch_splits"),
            "vs_pitch": vs_pitch.get(int(bid)) if vs_pitch else None,
            "pitch_table": bat_tables.get(int(bid)) if bat_tables else None,
            "pitch_usage": pprof.get("usage"),
            "pitch_matchup": pmatch,
            "features": feat,          # parallel edges: pitch_matchup, late_hr (never touch heat)
            "elite": elite,            # four-threshold elite gate (pull/EV/barrel/ideal-AA)
            "_season_metrics": {k: season.get(k) for k in (
                "pull_air_pct", "avg_ev", "barrel_pct", "ideal_aa_pct", "bb_pct", "iso",
                "xwoba", "hardhit_pct", "bat_speed")},
            # Long Ball Jackpot's real ceiling metrics -- launch35_pct/fbld_ev/avg_bat_speed/
            # fast_swing_rate from the shared Statcast frame (batter_ceiling_profile), ev50/
            # max_hit_speed from the true MLB-wide Savant leaderboard (batter_exitvelo_barrels).
            "long_ball": {**(lb_ceiling.get(int(bid)) or {}), **(lb_evbarrels.get(int(bid)) or {})},
            "heat_mix": heat_mix,
            "mix": mix_prof,
            "ev_overall": recent.get("avg_ev"),
            "luck": {
                "recent": {k: recent.get(k) for k in ("xwobacon", "wobacon", "luck_gap", "barrel_pct", "hr", "bb_count")},
                "season": {k: season.get(k) for k in ("xwobacon", "wobacon", "luck_gap", "barrel_pct", "hr", "bb_count")},
            },
            "max_ev": {"recent": recent.get("max_ev"), "season": season.get("max_ev")},
            "heat": score,
            "score_breakdown": breakdown,
            # Props scores — parallel track for the Other Props tab, NEVER touch heat.
            # hrr_heat needs lineup_spot + HR heat; computed as a post-attach step.
            "hit_heat": props.hit_heat(recent, pprof, sprint_speed=(sprint or {}).get(int(bid)))[0],
            "bat_tracking": (bat_track.get(int(bid)) if bat_track else None),
            "near_miss": (near_miss.get(int(bid)) if near_miss else None),
            "k_heat_bat": props.k_heat_hitter(recent, pprof)[0],
            "hrr_heat": props.hrr_heat(recent, pprof,
                lineup_spot=spot_of_batter.get(bid), hr_heat=score)[0],
            "metrics": metrics,
            "windows": prof.get("windows", {}),
            "hr_recent": {w: prof.get("windows", {}).get(w, {}).get("hr") for w in ("L5", "L15", "L30")},
        })

    # Contact-gated hit projection. Runs as a POST-ATTACH step because it needs the assembled
    # row — handedness, opposing arm, lineup spot, vs_pitch and the trailing window all live on
    # the player record, and none of them exist while the dict literal is still being built.
    # (An earlier version tried to pass the row into its own literal, which is impossible and
    # crashed the whole build with an UnboundLocalError.)
    for _pl in players:
        try:
            _bid = int(_pl.get("id") or 0)
            _rec = (_pl.get("windows") or {}).get("L14d") or {}
            _pp = (_pl.get("opp_pitcher") or {}).get("profile") or {}
            _pl["hit_gated"] = _gated_hit_for(_bid, _rec, _pp, _pl)
        except Exception:
            _pl["hit_gated"] = None
    _ng = sum(1 for _pl in players if _pl.get("hit_gated"))
    print(f"[build] contact-gated hit projections: {_ng}/{len(players)}")

    # Per-player HRR. The tab previously showed one number per heat tier — the tier's historical
    # rate — so every hitter in a band read identically and the card could not help you choose
    # between them. This produces a genuine per-hitter projection: expected hits + runs + RBIs
    # built volume-first (xPA scaled by the implied team total), then the probability of clearing
    # the 1.5 line.
    #
    # HRR is a sum of correlated counts — one home run pays a hit, a run AND an RBI at once — so
    # a plain Poisson understates how often a hitter lands on 2+ and how often on 0. The
    # dispersion term below widens the distribution to account for that; it is calibrated so the
    # slate-wide average matches the rate the backtest actually graded, which is the only anchor
    # available. Treated as a display lens: it never feeds heat.
    _by_team_spot = {}
    for _pl in players:
        _t, _s = _pl.get("team"), _pl.get("lineup_spot")
        if _t and _s:
            _by_team_spot.setdefault(_t, {})[int(_s)] = _pl

    def _neighbors(team, spot, offsets):
        out = []
        lineup = _by_team_spot.get(team) or {}
        for off in offsets:
            s = ((int(spot) - 1 + off) % 9) + 1
            nb = lineup.get(s)
            if nb:
                w = (nb.get("windows") or {}).get("L30d") or (nb.get("windows") or {}).get("L14d") or {}
                out.append({"obp_30": w.get("obp"), "woba_30": w.get("xwobacon"),
                            "slg_30": w.get("slg")})
        return out

    _nhrr = 0
    for _pl in players:
        try:
            _hg = _pl.get("hit_gated")
            _spot = _pl.get("lineup_spot")
            if not _hg or not _spot:
                _pl["hrr_proj"] = None
                continue
            _itt = _pl.get("implied_team_total") or (_pl.get("game") or {}).get("implied_total")
            _hp = props.hrr_projection(
                _hg, int(_spot), implied_team_total=_itt,
                batters_ahead=_neighbors(_pl.get("team"), _spot, [-2, -1]),
                batters_behind=_neighbors(_pl.get("team"), _spot, [1, 2]))
            _lam = float(_hp.get("hrr") or 0.0)
            _hp["p_over15"] = _hrr_p_over(_lam, 1.5)
            _pl["hrr_proj"] = _hp
            _nhrr += 1
        except Exception:
            _pl["hrr_proj"] = None
    if _nhrr:
        _ps = [(_pl["hrr_proj"] or {}).get("p_over15") for _pl in players
               if (_pl.get("hrr_proj") or {}).get("p_over15") is not None]
        _avg = sum(_ps) / len(_ps) if _ps else 0
        print(f"[build] HRR projections: {_nhrr}/{len(players)} · "
              f"slate mean P(2+) {100*_avg:.1f}% · expected HRR "
              f"{sum((_pl['hrr_proj'] or {}).get('hrr') or 0 for _pl in players if _pl.get('hrr_proj'))/max(1,_nhrr):.2f}")

    players.sort(key=lambda p: p["heat"], reverse=True)

    # Plate discipline + Statcast percentile card, both from TRUE MLB percentiles (computed
    # over every qualified batter in the season pull, not just today's slate).
    try:
        for _p in players:
            cats = league_pctl.get(_p["id"])
            if not cats:
                continue
            _p.setdefault("features", {})["percentiles"] = cats
            ch = cats.get("chase_pct"); zc = cats.get("zcontact_pct")
            eye = bool(ch and ch["pctl"] >= 75)
            crosshair = bool(zc and zc["pctl"] >= 75)
            warning = bool(ch and ch["pctl"] <= 25)
            mult = 1.0; grade = 0
            if eye: mult *= 1.06; grade += 1
            if crosshair: mult *= 1.05; grade += 1
            if warning: mult *= 0.94; grade -= 1
            _p["features"]["discipline"] = {
                "chase_pct": ch["value"] if ch else None,
                "chase_pctl": ch["pctl"] if ch else None,
                "zcontact_pct": zc["value"] if zc else None,
                "zcontact_pctl": zc["pctl"] if zc else None,
                "eye": eye, "crosshair": crosshair, "warning": warning,
                "hr_mult": round(mult, 3), "grade_delta": grade,
                "scope": "mlb",          # true league percentiles, not slate-relative
            }
        _n_pc = sum(1 for _p in players if (_p.get("features") or {}).get("percentiles"))
        print(f"[build] percentile cards: {_n_pc} hitters (true MLB percentiles)")
    except Exception as e:
        _hnote("percentiles", e); print(f"[build] percentiles skipped: {e}")

    # ---- GRAND SLAM generator: score each hitter's GS likelihood = traffic (bases loaded when
    # he bats) x punish (can he golf a grooved in-zone fastball) + pen/shift amplifiers. Parallel
    # to heat. Attaches p["grand_slam"], and board["grand_slam"] = top 3 for the DK jackpot. ----
    try:
        from etl import grandslam
        # on-base proxy per player from season profile (OBP ~ (H+BB)/PA; we approximate with
        # a blend of bb_pct and batting-average-ish signal available on the row).
        for p in players:
            szn = (p.get("_season_metrics") or {})
            # reconstruct a rough OBP: league-avg baseline nudged by walk rate + on-base skill
            bb = szn.get("bb_pct")
            # we stored iso/slg but not AVG directly; use a coarse OBP proxy from bb_pct + xwoba-ish
            # fallback: heat-independent. If bb missing, use .315 league-ish.
            ob = 0.300
            if bb is not None:
                ob = 0.300 + (bb - 8.5) * 0.006      # each pt of BB% over league ~+6 OBP pts
            xw = szn.get("xwoba")
            if xw is not None:
                ob = max(ob, xw - 0.030)              # xwOBA is a strong OBP proxy
            p["_ob"] = round(max(0.230, min(0.470, ob)), 3)

        # group hitters by game to model each lineup's traffic
        by_game_lu = {}
        for p in players:
            by_game_lu.setdefault(p.get("game_pk"), []).append(p)

        gs_all = []
        for gpk, lineup in by_game_lu.items():
            # by_game_lu deliberately mixes BOTH teams (needed for a combined per-game board),
            # but every batting-order spot 1-9 exists on BOTH teams -- confirmed on a real game:
            # spot 3 has one hitter per team in the SAME list. Without a team filter,
            # traffic_score()'s "who bats 3 spots ahead" lookup could silently grab the wrong
            # team's hitter for roughly half of all traffic computations -- verified this
            # produces a real, large divergence (score 10 vs 80 on the same synthetic inputs,
            # opposite team ordering).
            _by_team = {}
            for x in lineup:
                _by_team.setdefault(x.get("team"), []).append(x)
            for p in lineup:
                own_lineup = _by_team.get(p.get("team")) or lineup
                opp = p.get("opp_pitcher") or {}
                opp_bb = (opp.get("season") or {}).get("bb_pct_allowed") or (opp.get("recent") or {}).get("bb_pct_allowed")
                traffic = grandslam.traffic_score(p.get("lineup_spot"), own_lineup, opp_bb)
                # in-zone fastball punish, if the feature pipeline computed it
                izfb = (p.get("features") or {}).get("in_zone_fb")
                _badges = {b.get("k") for b in (p.get("badges") or []) if b.get("k")}
                _ideal_clear = (((p.get("score_breakdown") or {}).get("signals") or {})
                               .get("ideal_aa_pct") is True)
                punish = grandslam.punish_score(p.get("_season_metrics"), p.get("elite"), izfb,
                                                badges=_badges, ideal_aa_clear=_ideal_clear)
                # pen amplifier: gassed pen / short starter behind the arm (late_hr feature),
                # PLUS the opposing bullpen's own leverage-weighted walk rate and fatigue --
                # pen_state is built earlier in this function (bullpen_state()), the same source
                # the Bullpens tab reads, so this is real data, not a new invented number.
                pen_boost = 0.0
                lc = (p.get("features") or {}).get("late_hr")
                if lc and lc.get("score"):
                    pen_boost = min(8.0, lc["score"] * 0.10)
                _ps = pen_state.get(p.get("opp_team")) or {}
                if _ps.get("bb_pct") is not None:
                    pen_boost += max(0.0, min(4.0, (_ps["bb_pct"] * 100 - 9.5) * 1.3))
                if _ps.get("fatigue") is not None:
                    pen_boost += min(3.0, _ps["fatigue"] * 3.0)
                # pen_state (bullpen_state()) has no "label" field -- that's bullpen_rankings'
                # shape, a different dict not built until much later in this function. Derived
                # here directly from what pen_state actually has, for the driver display text.
                _fat, _nout = _ps.get("fatigue"), _ps.get("n_out")
                _pen_label = ("GASSED" if (_fat is not None and _fat >= 0.5) or
                                         (_nout is not None and _nout >= 3)
                             else "WORN" if (_fat is not None and _fat >= 0.25) or
                                            (_nout is not None and _nout >= 1)
                             else None)
                # Feed the new engines into the slam probability.
                #
                # The three hitters ahead now supply per-PA on-base numbers from the
                # contact-gated model instead of an OBP proxy, the park's carry multiplier
                # scales the home-run half, and a hitter with repeated near misses gets a small
                # credit because those are the balls that clear on a different night. Each of
                # these already ships elsewhere in the board; none of them reached this model.
                _p_slam = None
                try:
                    _sp = p.get("lineup_spot")
                    if _sp:
                        _ahead = []
                        for _k in range(1, 4):
                            _s2 = ((int(_sp) - 1 - _k) % 9) + 1
                            _pl2 = next((x for x in lineup if x.get("lineup_spot") == _s2), None)
                            if not _pl2:
                                continue
                            _hg2 = _pl2.get("hit_gated") or {}
                            _xpa2, _ph2 = _hg2.get("xpa"), _hg2.get("p_hit")
                            if _xpa2 and _ph2 and _ph2 < 1:
                                # per-PA hit chance, then add walks to get on-base
                                _perpa = 1.0 - (1.0 - _ph2) ** (1.0 / _xpa2)
                                _bb2 = ((_pl2.get("windows") or {}).get("L14d") or {}).get("bb_pct")
                                _ahead.append(_perpa + (float(_bb2 or 8.5) / 100.0))
                        if len(_ahead) >= 3:
                            _wild = float(opp_bb or 8.5) / 100.0
                            _pl_prob = grandslam.loaded_bases_prob(_ahead, wildness=_wild)
                            _w14 = (p.get("windows") or {}).get("L14d") or {}
                            _hrpa = None
                            if _w14.get("pa") and _w14.get("hr") is not None and _w14["pa"] >= 20:
                                _hrpa = float(_w14["hr"]) / float(_w14["pa"])
                                _hrpa = min(_hrpa, 0.032 * 3.0)      # cap, same reasoning as runs.py
                                _hrpa = 0.6 * _hrpa + 0.4 * 0.032    # regress a 2-week sample
                            _pk2 = ((p.get("park_hr") or {}).get("boost") or 0) / 100.0
                            _nm2 = (p.get("near_miss") or {}).get("near") or 0
                            _p_slam = grandslam.slam_probability(
                                _pl_prob, _hrpa, park_mult=1.0 + _pk2 * 0.45,
                                near_miss_boost=min(0.12, 0.04 * _nm2))
                except Exception:
                    _p_slam = None
                gs = grandslam.grand_slam_score(traffic, punish, pen_boost=pen_boost,
                                                p_slam=_p_slam, pen_label=_pen_label)
                if gs:
                    p["grand_slam"] = gs
                    gs_all.append((gs["score"], p))
        # board-level top 3 for the DK jackpot
        # Rank on the calibrated probability where it exists, falling back to the ordinal score.
        # The score and the probability can disagree: the score is a geometric blend of two
        # 0-100 components, so it rewards being decent at both, while the probability answers
        # the question actually being asked — how often does this end in a slam. Sorting on the
        # score meant the top three surfaced to the jackpot were not the three likeliest.
        gs_all.sort(key=lambda x: -((x[1].get("grand_slam") or {}).get("p_slam") or (x[0] / 1e6)))
        board_gs = []
        for sc, p in gs_all[:12]:
            board_gs.append({
                "id": p["id"], "name": p["name"], "team": p.get("team"),
                "opp_team": p.get("opp_team"), "spot": p.get("lineup_spot"),
                "game_pk": p.get("game_pk"), "time": p.get("time"),
                "score": sc,
                "p_slam": p["grand_slam"].get("p_slam"),
                "fair_odds": p["grand_slam"].get("fair_odds"),
                "traffic": p["grand_slam"]["traffic"],
                "punish": p["grand_slam"]["punish"], "drivers": p["grand_slam"]["drivers"],
                "heat": p.get("heat"), "elite": (p.get("elite") or {}).get("tier"),
            })
        _np = sum(1 for _x in board_gs if _x.get("p_slam") is not None)

        # Pick 1/2/3, searched over the FULL scored pool (gs_all), not just the top-12 slice
        # board_gs keeps -- a genuine deep-leverage play can rank #15 overall by raw p_slam and
        # still be the single best spot-6+ candidate, and restricting the search to board_gs
        # would silently hide it behind higher-probability top-of-order bats.
        def _gs_dict(sc, p):
            return {"id": p["id"], "name": p["name"], "team": p.get("team"),
                   "opp_team": p.get("opp_team"), "spot": p.get("lineup_spot"),
                   "game_pk": p.get("game_pk"), "time": p.get("time"), "score": sc,
                   "p_slam": p["grand_slam"].get("p_slam"),
                   "fair_odds": p["grand_slam"].get("fair_odds"),
                   "traffic": p["grand_slam"]["traffic"], "punish": p["grand_slam"]["punish"],
                   "drivers": p["grand_slam"]["drivers"], "heat": p.get("heat"),
                   "elite": (p.get("elite") or {}).get("tier")}

        def _gs_pick(pool, used_ids):
            for sc, p in pool:
                if p["id"] not in used_ids:
                    return _gs_dict(sc, p)
            return None
        _used = set()
        gs_picks = []
        primary = _gs_pick(gs_all, _used)
        if primary:
            gs_picks.append({**primary, "pick_label": "Primary Target"})
            _used.add(primary["id"])
        top_pool = [(sc, p) for sc, p in gs_all if p.get("lineup_spot") and p["lineup_spot"] <= 5]
        topmash = _gs_pick(top_pool, _used)
        if topmash:
            gs_picks.append({**topmash, "pick_label": "Top-of-Order Mash"})
            _used.add(topmash["id"])
        deep_pool = [(sc, p) for sc, p in gs_all if p.get("lineup_spot") and p["lineup_spot"] >= 6]
        deep = _gs_pick(deep_pool, _used)
        if deep:
            gs_picks.append({**deep, "pick_label": "Mega-Leverage Deep Target"})
            _used.add(deep["id"])
        board_gs_jackpot = {"picks": gs_picks, "candidates_scored": len(gs_all),
                            "notes": ["Traffic comes from the on-base bats ahead in the order + "
                                     "the starter's wildness + the opposing bullpen's own "
                                     "leverage-weighted walk rate and fatigue; punish from "
                                     "barrel/EV/pull, badges, ideal launch angle, and in-zone "
                                     "fastball damage.",
                                     "Grand slams are rare and high-variance by nature -- score "
                                     "ranks relative likelihood, while p_slam/fair_odds is a "
                                     "real calibrated probability where enough data exists to "
                                     "compute one. Not CLV-validated."]}

        # Per-game Grand Slam Board -- every batter from BOTH teams in a game, ranked together
        # by p_slam, not just the slate-wide top-12/top-3.
        def _gs_tier(ps):
            if ps is None:
                return "DARTS"
            if ps >= 0.0045: return "STRONG"
            if ps >= 0.0030: return "SOLID"
            if ps >= 0.0015: return "LONGSHOT"
            return "DARTS"

        _by_game_gs = {}
        for sc, p in gs_all:
            _by_game_gs.setdefault(p.get("game_pk"), []).append((sc, p))
        board_gs_board = []
        for gpk, rows in _by_game_gs.items():
            rows.sort(key=lambda x: -((x[1]["grand_slam"] or {}).get("p_slam") or (x[0] / 1e6)))
            _g = next((gg for gg in (games or []) if gg.get("game_pk") == gpk), {})
            batters = []
            _slam_complement = 1.0
            for i, (sc, p) in enumerate(rows):
                ps = (p["grand_slam"] or {}).get("p_slam")
                if ps:
                    _slam_complement *= (1.0 - ps)
                batters.append({
                    "rank": i + 1, "id": p["id"], "name": p["name"], "team": p.get("team"),
                    "spot": p.get("lineup_spot"), "score": sc,
                    "p_slam": ps, "fair_odds": (p["grand_slam"] or {}).get("fair_odds"),
                    "tier": _gs_tier(ps),
                })
            board_gs_board.append({
                "game_pk": gpk, "away": _g.get("away"), "home": _g.get("home"),
                "park": _g.get("park"), "time": _g.get("time"),
                "game_slam_prob": round(1.0 - _slam_complement, 4) if batters else None,
                "best_individual_prob": batters[0]["p_slam"] if batters else None,
                "n_batters": len(batters), "batters": batters,
            })
        board_gs_board.sort(key=lambda g: -(g.get("game_slam_prob") or 0))

        print(f"[build] grand slam: {len(gs_all)} hitters scored, top {len(board_gs)} surfaced "
              f"({_np} with a calibrated probability, {len(gs_picks)}/3 picks selected, "
              f"{len(board_gs_board)} per-game boards)")
    except Exception as e:
        board_gs = []
        board_gs_jackpot = {"picks": [], "candidates_scored": 0, "notes": []}
        board_gs_board = []
        _hnote("grand slam", e); print(f"[build] grand slam skipped: {e}")

    try:                                           # persist career-BvP cache for the next build
        json.dump({"pairs": bvp_career_cache, "updated": _bvp_now}, open(bvp_cache_path, "w"))
        print(f"[build] career BvP: {_bvp_fetched[0]} fetched this run, {len(bvp_career_cache)} cached")
    except Exception as e:
        print(f"[build] bvp cache write failed: {e}")

    # ---- park + weather HR model: a separate lens, computed per game in one vectorized
    # pass, attached as p["park_hr"]. Never feeds the heat score or the grader. The park's
    # overall HR level comes from Baseball Savant (auto-pulled, rolling, handedness-split);
    # the physics only adds per-hitter spray + live weather on top, anchored to Savant so a
    # dimension/orientation that's slightly off can't make the park factor wrong. Wrapped so
    # a Savant/weather outage degrades gracefully instead of breaking the board.
    try:
        from etl import park_factors
        pf_cache = os.path.join(os.path.dirname(__file__), "..", "docs", "park_factors.json")
        savant = park_factors.load_park_factors(pf_cache, df=df, year=now.year)
        gpk_home = {g["game_pk"]: g["home"] for g in games}     # durable key for the factor lookup

        try:                                       # spray chart uses last 14d for volume
            _drv_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
            drives_map = statcast_data.recent_drives(df, _drv_start)
            # robbed scan: the PRIOR DAY only — "did he just miss last night?" A flyout from
            # three days ago isn't actionable context for today's slate.
            _rob_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            robbed_drives = statcast_data.recent_drives(df, _rob_start, min_dist=320)
        except Exception as e:
            drives_map = {}; robbed_drives = {}
            _hnote("recent drives", e); print(f"[build] recent drives skipped: {e}")

        by_game = {}
        for p in players:
            by_game.setdefault((p["park"], p["time"], p["game_pk"]), []).append(p)
        for (venue, gtime, pk), ps in by_game.items():
            # tonight's conditions once per game, reused for every hitter's robbed check
            try:
                rho_g, wind_g = park_model.game_conditions(venue, gtime)
            except Exception:
                rho_g, wind_g = None, None
            for p in ps:
                # spray chart: 2-week volume so the picture has data points
                drv = drives_map.get(p["id"]) or []
                # robbed scan: only last 3 days — recent-recent contact only
                rob_drv = robbed_drives.get(p["id"]) or []
                flags = [0] * len(drv)
                # mark HRs in the spray drives
                for j, d in enumerate(drv):
                    if d["hr"]:
                        flags[j] = 1
                # robbed check runs on the tight 3-day set, ignoring HRs; flagged
                # results get folded back into the spray chart flags by (date, dist)
                rob_hits = []
                if rho_g is not None and rob_drv:
                    deep = [(j, d) for j, d in enumerate(rob_drv)
                            if (not d["hr"]) and d["dist"] >= 320 and 15 <= d["la"] <= 45]
                    if deep:
                        cl = park_model.clears_here([d["ev"] for _, d in deep],
                                                    [d["la"] for _, d in deep],
                                                    [d["spray"] for _, d in deep],
                                                    venue, rho_g, wind_g)
                        for (j, d), c in zip(deep, cl):
                            if bool(c):
                                rob_hits.append(d)
                                # mark in spray chart too if same ball is present
                                for jj, sd in enumerate(drv):
                                    if sd["date"] == d["date"] and abs(sd["dist"] - d["dist"]) <= 2:
                                        flags[jj] = 2
                if drv:
                    p["drives"] = [[d["spray"], d["dist"], flags[j]] for j, d in enumerate(drv)]
                if rob_hits:
                    best = max(rob_hits, key=lambda x: x["dist"])
                    p["robbed"] = {"n": len(rob_hits),
                                   "best_ft": best["dist"], "best_date": best["date"],
                                   "inning": best.get("inning"), "half": best.get("half"),
                                   # every near-miss, so the UI can list them with their inning
                                   "hits": [{"ft": h["dist"], "inning": h.get("inning"),
                                             "half": h.get("half"), "ev": h.get("ev")}
                                            for h in sorted(rob_hits, key=lambda x: -x["dist"])[:4]]}
            evs, las, sprays, spans = [], [], [], []
            for p in ps:
                s = bb_samples.get(p["id"])
                n = 0 if (s is None) else len(s["ev"])
                spans.append((p, n))
                if n:
                    evs.append(s["ev"]); las.append(s["la"]); sprays.append(s["spray"])
            if not evs:
                continue
            ev_all = np.concatenate(evs); la_all = np.concatenate(las); sp_all = np.concatenate(sprays)
            hr_park, hr_neut, meta = park_model.evaluate_game(ev_all, la_all, sp_all, venue, gtime)
            i = 0
            for p, n in spans:
                if not n:
                    continue
                hand = p.get("bats", "R")
                sav = park_factors.factor_for(savant, venue, hand, team=gpk_home.get(pk))
                anchor = park_model.savant_anchor(venue, hand, sav)
                agg = park_model.aggregate_hitter(hr_park[i:i+n], hr_neut[i:i+n], meta,
                                                  anchor=anchor, savant_factor=sav)
                if agg:
                    p["park_hr"] = agg
                # "Would clear in X of 30 parks" — a geometry read on this hitter's recent
                # batted balls. Restrict to genuinely deep/liftable balls so it measures HR
                # potential, not weak grounders. Cheap (ms/hitter) and never touches heat.
                try:
                    ev_s = ev_all[i:i+n]; la_s = la_all[i:i+n]; sp_s = sp_all[i:i+n]
                    deep_m = (ev_s >= 95.0) & (la_s >= 18.0) & (la_s <= 42.0)
                    ndeep = int(deep_m.sum())
                    if ndeep >= 3:
                        pc = park_model.parks_cleared(ev_s[deep_m], la_s[deep_m], sp_s[deep_m])
                        per = pc["per_ball"]
                        # a ball is "borderline" if it clears some parks but not most —
                        # those are the ones where tonight's park actually decides it
                        borderline = sum(1 for c in per if 1 <= c <= 24)
                        here = int(np.asarray(
                            park_model.clears_here(ev_s[deep_m], la_s[deep_m], sp_s[deep_m],
                                                   venue, *park_model.game_conditions(venue, gtime)),
                            dtype=bool).sum())
                        p["parks30"] = {
                            "n_deep": ndeep,
                            "out_here": here,               # clear tonight's park
                            "any": pc["any"],               # clear at least one park
                            "avg_parks": pc["avg_parks"],   # mean parks a clearing ball clears
                            "borderline": borderline,       # park-dependent balls
                            "per_ball": per,
                        }
                except Exception:
                    pass
                # batted-ball log (the detailed table view) — attach display rows with pitcher name
                # and the X/30-parks read per ball; drop raw helper fields to keep board.json lean
                try:
                    _log = bb_logs.get(int(p.get("id")))
                    if _log:
                        _ev = np.array([b["_ev"] for b in _log]); _la = np.array([b["_la"] for b in _log])
                        _sp = np.array([b["_sp"] for b in _log])
                        _pc = park_model.parks_cleared(_ev, _la, _sp)
                        _per = _pc.get("per_ball") or []
                        _rows = []
                        for _j, _b in enumerate(_log):
                            _row = {_k: _b[_k] for _k in ("date","arm","pt","ev","la","dist","bs","pv","res","traj","brl")}
                            _row["pit"] = bb_pit_names.get(_b.get("pit"))
                            _row["x30"] = int(_per[_j]) if _j < len(_per) else None
                            _rows.append(_row)
                        p["bb_log"] = _rows
                except Exception:
                    pass
                i += n
    except Exception as e:
        _hnote("park model", e); print(f"[build] park model skipped: {e}")

    # ---- decision helpers ----
    def _thin(p):
        return any(str(f).startswith("small sample") for f in p["score_breakdown"].get("flags", []))

    # Top Plays: strongest, non-thin hitters not facing a DEALING arm
    top_plays = []
    for p in players:
        if _thin(p) or p["heat"] < 60:
            continue
        if (p["opp_pitcher"].get("form") or {}).get("label") == "DEALING":
            continue
        top_plays.append({
            "name": p["name"], "team": p["team"], "opp_team": p["opp_team"],
            "heat": p["heat"], "tier": p["tier"], "why": p["why"],
            "spot": p["lineup_spot"], "time": p["time"],
            "arm": p["opp_pitcher"].get("name"),
            "arm_form": (p["opp_pitcher"].get("form") or {}).get("label"),
            "arm_score": p["opp_pitcher"].get("hr_score"),
        })
        if len(top_plays) >= 12:
            break

    # Stacks (pairing): a target for the whole lineup — either a vulnerable arm with
    # 2+ strong hitters facing him, OR a bullpen game (opener "start" + weak pen),
    # which the old form filter wrongly excluded. Sorted by a blended stack score so
    # a slightly-less-bad arm facing five monsters can outrank a worse arm facing two.
    from collections import defaultdict
    groups = defaultdict(list)
    for p in players:
        groups[(p["game_pk"], p["opp_pitcher"].get("name"))].append(p)
    stacks = []
    for (gpk, arm), ps in groups.items():
        if not arm:
            continue
        op = ps[0]["opp_pitcher"]
        form = (op.get("form") or {}).get("label", "")
        pen_sc = (ps[0].get("opp_bullpen") or {}).get("score")
        opener = bool(op.get("opener"))
        targetable_arm = form in ("SHELLABLE", "STEADY-BAD", "SLIPPING", "HITTABLE")
        pen_game = opener and (pen_sc or 0) >= 55
        if not (targetable_arm or pen_game):
            continue
        strong = sorted([x for x in ps if x["heat"] >= 55], key=lambda x: x["heat"], reverse=True)
        if len(strong) < 2:
            continue
        vuln = max(op.get("hr_score") or 0, (pen_sc or 0) if opener else 0)
        top3 = sum(x["heat"] for x in strong[:3]) / min(3, len(strong))
        stacks.append({
            "arm": arm,
            "form": op.get("form"),
            "arm_score": op.get("hr_score"),
            "pen_score": pen_sc,
            "opener": opener,
            "pen_game": pen_game,
            "stack_score": int(round(0.55 * vuln + 0.45 * top3)),
            "park": strong[0].get("park"),
            "park_factor": strong[0].get("park_hr_factor"),
            "game_pk": gpk,
            "team": strong[0]["team"], "opp_team": strong[0]["opp_team"],
            "time": strong[0]["time"],
            "hitters": [{
                "id": x["id"], "name": x["name"], "heat": x["heat"], "tier": x["tier"],
                "spot": x["lineup_spot"], "bats": x["bats"],
                "b2b": bool(x.get("hr_last_game")),
                "owns": any(b.get("k") in ("hrsp", "hrbp") for b in (x.get("badges") or [])),
            } for x in strong[:6]],
        })
    stacks.sort(key=lambda s: (s["stack_score"], len(s["hitters"])), reverse=True)

    try:                                       # career HR milestone watch
        _id_to_team = {p["id"]: p.get("team") for p in players}
        mstones = statcast_data.career_hr_milestones(
            [p["id"] for p in players], id_to_team=_id_to_team)
        for p in players:
            if p["id"] in mstones:
                p["milestone"] = mstones[p["id"]]
    except Exception as e:
        _hnote("milestones", e); print(f"[build] milestones skipped: {e}")

    # ---- fences for the spray chart + the morning briefing ----
    fences = {}
    try:
        for g in games:
            if g["park"] not in fences:
                fences[g["park"]] = park_model.fence_polyline(g["park"])
    except Exception as e:
        _hnote("fences", e); print(f"[build] fences skipped: {e}")

    briefing = []
    try:
        boosts = [(p["park_hr"]["boost"], p) for p in players
                  if p.get("park_hr") and p["park_hr"].get("boost") is not None]
        if boosts:
            b, p = max(boosts, key=lambda x: x[0])
            if b >= 8:
                w = (p["park_hr"].get("wind_mph"))
                briefing.append(f"Best environment: {p['park']} at +{b}%"
                                + (f" with wind {w} mph" if w else "") + ".")
        opener_teams = sorted({p["opp_team"] for p in players
                               if (p.get("opp_pitcher") or {}).get("opener")})
        if opener_teams:
            briefing.append(("Bullpen game" if len(opener_teams) == 1 else "Bullpen games")
                            + f" vs {', '.join(opener_teams)} — weigh the pen, not the listed arm.")
        rb = sorted([p for p in players if p.get("robbed")],
                    key=lambda p: (-p["robbed"]["n"], -p["robbed"]["best_ft"]))[:2]
        for p in rb:
            r = p["robbed"]
            briefing.append(f"Robbed watch: {p['name']} — {r['best_ft']}ft out on {r['best_date'][5:]} "
                            f"clears here tonight" + (f" ({r['n']} such balls)." if r["n"] > 1 else "."))
        b2b = [p for p in players if p.get("hr_last_game") and p["heat"] >= 70]
        if b2b:
            names = ", ".join(p["name"] for p in b2b[:2])
            briefing.append(f"B2B fade: {names} homered last night — bases over HR by your rules.")
        ms1 = [p for p in players if (p.get("milestone") or {}).get("away") == 1]
        for p in ms1[:2]:
            m = p["milestone"]
            briefing.append(f"Milestone watch: {p['name']} sits at {m['career_hr']} career HR — "
                            f"one swing from {m['next']}.")
        nlab = {"elite": 0, "fb": 0, "ld": 0}
        for p in players:
            if p.get("hit_label"): nlab[p["hit_label"]] += 1
        if sum(nlab.values()):
            briefing.append(f"Profiles on slate: {nlab['elite']} ELITE · {nlab['fb']} FB · {nlab['ld']} LD.")
    except Exception as e:
        _hnote("briefing", e); print(f"[build] briefing skipped: {e}")

    # per-game weather summaries for the Weather dashboard (roof call, disruption status,
    # wind rendered relative to each park's actual orientation)
    wx_list = []
    try:
        from etl import weather as W, park_geometry as PG
        for g in games:
            lat, lon, _ = PG.park_coords(g["park"])
            wx = W.get_weather(lat, lon, g["time"], venue=g["park"])
            roof = W.roof_call(g["park"], wx)
            pp = (wx or {}).get("precip_prob")
            if roof in ("dome", "closed", "canopy") or pp is None:
                status = "clear" if roof else "unknown"
            elif pp < 20:
                status = "clear"
            elif pp < 45:
                status = "chance"
            elif pp < 70:
                status = "likely"
            else:
                status = "postpone"
            cf = PG.cf_bearing(g["park"])
            frm = (wx or {}).get("wind_from_deg")
            rel = round(((frm + 180.0) - cf) % 360.0) if (frm is not None and cf is not None) else None
            wx_list.append({
                "game_pk": g["game_pk"], "away": g["away"], "home": g["home"],
                "park": g["park"], "time": g["time"],
                "temp_f": round((wx or {}).get("temp_f")) if (wx or {}).get("temp_f") is not None else None,
                "rh_pct": round((wx or {}).get("rh_pct")) if (wx or {}).get("rh_pct") is not None else None,
                "precip_prob": round(pp) if pp is not None else None,
                "wind_mph": round((wx or {}).get("wind_mph")) if (wx or {}).get("wind_mph") is not None else None,
                "wind_rel_deg": rel, "roof": roof, "status": status,
            })
    except Exception as e:
        _hnote("weather summaries", e); print(f"[build] weather summaries skipped: {e}")

    # ---- PITCHER EDGES: per-arm arsenal + zone heatmap + ranked opposing batters who
    # punish his zones. Powers the Edges tab (pitcher-first drill-down). Parallel track. ----
    pitcher_edges = []
    if features is not None:
        try:
            # index built players by (game_pk, batter side) so we can find each arm's opponents
            for g in games:
                for pid, p_side, opp_side in ((g["home_pitcher_id"], "home", "away"),
                                              (g["away_pitcher_id"], "away", "home")):
                    if not pid:
                        continue
                    ars = arsenal_by_pid.get(pid)
                    arows = statcast_data.pitcher_arsenal_rows(df, pid, date_str)
                    zgrid = features.pitcher_zone_grid(arows)
                    pzdmg = features.pitcher_zone_damage(arows)   # MEATBALL zones (damage allowed)
                    meta = slate["pitchers"].get(pid, {})
                    # opposing batters: those in this game on the other side
                    opp_batters = []
                    for pl in players:
                        if pl.get("game_pk") != g["game_pk"] or pl.get("side") != opp_side:
                            continue
                        bz = None
                        bhr = None
                        try:
                            brows = statcast_data.batter_pitch_rows(df, pl["id"], date_str)
                            bz = features.batter_zone_damage(brows)
                            bhr = features.batter_hr_zones(brows)      # actual HR locations
                        except Exception:
                            bz = None; bhr = None
                        # the pitcher attacks lefties and righties differently — grade this
                        # hitter against the grid for HIS side, falling back to the overall grid
                        _hand = pl.get("bats")
                        if _hand == "S":
                            _hand = "L" if meta.get("throws", "R") == "R" else "R"
                        zgrid_h = zgrid
                        if _hand in ("R", "L"):
                            try:
                                _gh = features.pitcher_zone_grid(arows, hand=_hand)
                                if _gh and _gh.get("n", 0) >= 150:
                                    zgrid_h = _gh
                            except Exception:
                                pass
                        vz = features.batter_vs_pitcher_zones(bz, zgrid_h) if bz else {}
                        zedge = features.zone_matchup_edges(bz, zgrid_h) if bz else {}
                        pm = (pl.get("features") or {}).get("pitch_matchup") or {}
                        # zones this batter "crushes" = cells with xwOBAcon >= 0.400 on adequate
                        # sample. Count is shown as a badge; the map colors these for THIS hitter.
                        crushed = []
                        zdmg_full = {}
                        _zedge = zedge
                        if bz:
                            for zk, zv in bz.items():
                                xw = zv.get("xwobacon"); n = zv.get("n")
                                # carry barrel% + air distance so the map shows HR-power context
                                zdmg_full[zk] = {"xw": xw, "n": n,
                                                 "brl": zv.get("barrel_pct"),
                                                 "dist": zv.get("avg_dist"),
                                                 "pw": zv.get("power"),
                                                 "hr": zv.get("hr_rate"),
                                                 "slg": zv.get("slg"),
                                                 "iso": zv.get("iso"),
                                                 "hh": zv.get("hh_pct"),
                                                 "ev": zv.get("avg_ev")}
                                if xw is not None and xw >= 0.400 and (n or 0) >= 6:
                                    crushed.append(int(zk))
                        # ZONE SIGNAL: HRs this batter hit in this pitcher's meatball zones.
                        overlap = features.zone_overlap(bhr, pzdmg) if bhr else {"count":0,"cells":[],"hr_by_cell":{},"center_count":0,"badge":None,"meatballs":[]}
                        opp_batters.append({
                            "id": pl["id"], "name": pl["name"], "spot": pl.get("lineup_spot"),
                            "bats": pl.get("bats"), "heat": pl.get("heat"),
                            "zone_score": vz.get("score"), "hot_zone": vz.get("hot_zone"),
                            "hot_xw": vz.get("hot_xw"),
                            "matchup_score": pm.get("score"),
                            # full per-zone {xw, n} for the interactive per-hitter heatmap
                            "zone_dmg": zdmg_full or None,
                            "crushed_zones": sorted(crushed),      # e.g. [4,7,8] -> badge "3"
                            "overlap": overlap,                     # HRs in his meatball zones
                            "hr_zones": bhr or None,                # his HR count per zone
                            "zone_edge": _zedge or None,            # pitcher-usage-weighted matchup
                            "elite": (pl.get("elite") or {}).get("tier"),
                        })
                    # rank opponents by zone score (who punishes where he lives)
                    opp_batters.sort(key=lambda b: (b["zone_score"] is not None, b["zone_score"] or 0), reverse=True)
                    fat = fatigue_by_pid.get(pid) if p_roles.get(pid, {}).get("role") != "SP" else None
                    # exploitability: how much the top opponents punish his zones. Only count
                    # batters with ADEQUATE total zone sample (>=20 batted balls across their
                    # graded zones) so a hitter with 5 BBE doesn't inflate an arm's rank.
                    def _bat_sample(b):
                        zd = b.get("zone_dmg") or {}
                        return sum((zv.get("n") or 0) for zv in zd.values())
                    qualified = [b for b in opp_batters
                                 if b["zone_score"] is not None and _bat_sample(b) >= 20]
                    top_scores = [b["zone_score"] for b in qualified[:5]]
                    exploit = round(sum(top_scores) / len(top_scores), 3) if top_scores else None

                    # ---- HR VULNERABILITY SCORE (additional pitcher score, 0-100) ----
                    # ERA 30 · park 20 · hand-splits 15 · WHIP 15 · zone damage 12 · danger 8
                    _ps = (pstats or {}).get(pid) or {}
                    # park HR factor: average the L/R factors for a general park read
                    # park HR factor: Ballpark Pal's per-game model, else local L/R average
                    _pf = None
                    try:
                        from etl import ballparkpal as _BPP2
                        _pf, _ = _BPP2.resolve_hr_mult(
                            BPP, game_id=g.get("game_pk"),
                            away=g.get("away"), home=g.get("home"), fallback=None)
                    except Exception:
                        _pf = None
                    if _pf is None:
                        try:
                            _pf_l = parks.park_factor(g["park"], "L")
                            _pf_r = parks.park_factor(g["park"], "R")
                            _pf = (float(_pf_l) + float(_pf_r)) / 2.0
                        except Exception:
                            _pf = None
                    # dangerous bats today: elite-gated or genuinely hot, in this lineup
                    _danger = sum(1 for b in opp_batters
                                  if (b.get("elite") in ("ELITE", "ELITE+"))
                                  or (b.get("heat") or 0) >= 65
                                  or ((b.get("overlap") or {}).get("count", 0) >= 3))
                    vuln = features.vuln_score(
                        era=_ps.get("era"), whip=_ps.get("whip"),
                        park_factor=_pf,
                        hand_hr=(hand2yr.get(pid) or {}).get("two_yr"),
                        zone_damage=pzdmg,
                        danger_count=_danger)

                    # batted-ball profile allowed (GB/FB, contact quality, velo) for the arm card
                    _pp = pitch_profiles.get(pid) or {}
                    _pseason = _pp.get("season") or {}
                    _precent = _pp.get("recent") or {}
                    _arm_profile = {
                        "gb_pct": _pseason.get("gb_pct"), "fb_pct": _pseason.get("fb_pct"),
                        "ld_pct": _pseason.get("ld_pct"),
                        "avg_ev_allowed": _pseason.get("avg_ev_allowed"),
                        "barrel_pct_allowed": _pseason.get("barrel_pct_allowed"),
                        "hardhit_pct_allowed": _pseason.get("hardhit_pct_allowed"),
                        "fb_velo": _pseason.get("fb_velo") or _precent.get("fb_velo"),
                        "velo_trend": _precent.get("velo_trend"),
                        "recent_barrel": _precent.get("barrel_pct_allowed"),
                        "recent_ev": _precent.get("avg_ev_allowed"),
                    }
                    # HR HOT SPOTS: hitters who have taken this arm deep (from the BvP table)
                    _hot_spots = []
                    try:
                        for (_b, _p2), _v in (bvp or {}).items():
                            if _p2 != pid:
                                continue
                            _hr = _v[1] if isinstance(_v, (list, tuple)) and len(_v) > 1 else 0
                            if _hr and _hr > 0:
                                _nm = (hands.get(_b, {}) or {}).get("name") or str(_b)
                                _hot_spots.append({"id": int(_b), "name": _nm, "hr": int(_hr),
                                                   "pa": int(_v[0]) if _v else 0})
                        _hot_spots.sort(key=lambda x: -x["hr"])
                        _hot_spots = _hot_spots[:6]
                    except Exception:
                        _hot_spots = []

                    # per-hand zone usage + damage-allowed, computed from this arm's own rows
                    _zgrid_hand, _pzd_hand = {}, {}
                    try:
                        for _h in ("R", "L"):
                            _rh = arows[arows["stand"] == _h] if "stand" in arows.columns else None
                            if _rh is None or len(_rh) < 150:
                                continue
                            _g = features.pitcher_zone_grid(_rh, hand=_h)
                            if _g:
                                _zgrid_hand[_h] = _g
                            _d = features.pitcher_zone_damage(_rh)
                            if _d:
                                _pzd_hand[_h] = _d
                    except Exception:
                        _zgrid_hand, _pzd_hand = {}, {}

                    pitcher_edges.append({
                        "id": pid, "name": meta.get("name") or f"#{pid}",
                        "team": g["home"] if p_side == "home" else g["away"],
                        "opp_team": g["away"] if p_side == "home" else g["home"],
                        "game_pk": g["game_pk"], "time": g["time"], "park": g["park"],
                        "throws": (hands.get(pid, {}) or {}).get("throws", ""),
                        "arsenal": ars, "zone_grid": zgrid,
                        # per-hand usage grids (13 cells incl. the chase quadrants) + the pitch
                        # mix he throws each side, so the heatmap can answer "where does he live
                        # against THIS batter's side, and with what?"
                        "zone_grid_R": _zgrid_hand.get("R"),
                        "zone_grid_L": _zgrid_hand.get("L"),
                        "meatball_R": _pzd_hand.get("R"),
                        "meatball_L": _pzd_hand.get("L"),
                        "arsenal_hand": (arsenals.get(int(pid)) if arsenals else None),
                        "meatball_zones": pzdmg,     # per-zone damage allowed (darkest = meatball)
                        "season": pstats.get(pid),   # ERA / WHIP / IP / HR allowed
                        "vuln": vuln,                # 0-100 HR vulnerability composite
                        "quick_target": spot_damage.get(pid),   # dangerous lineup slots vs him
                        "tto": tto_by_pid.get(pid),  # times-through-order vulnerability
                        "profile": _arm_profile,     # GB/FB/EV/barrel/hardhit allowed + velo
                        "hand_splits": (hand2yr.get(pid) or {}),   # HR allowed by batter hand
                        "fg": _fg_with_defaults(
                            fg_pitch.get(statcast_data._norm_name(meta.get("name") or ""))),
                        "park_hr_factor": round(_pf, 2) if _pf else None,
                        "fatigue": fat, "exploit_score": exploit,
                        "batters": opp_batters,
                    })
            # rank arms by exploitability (most exploitable first)
            pitcher_edges.sort(key=lambda e: (e["exploit_score"] is not None, e["exploit_score"] or 0), reverse=True)
            print(f"[build] pitcher edges: {len(pitcher_edges)} arms")
            # Unify edge surfacing: copy each hitter's zone profile (vs today's opposing arm) back
            # onto their own player row, so the BOARD CARD can show which zones they crush without
            # the user drilling into Edges. One hitter faces one arm, so this is unambiguous.
            _pid_map = {p["id"]: p for p in players}
            for pe in pitcher_edges:
                for b in pe.get("batters") or []:
                    tgt = _pid_map.get(b["id"])
                    if tgt is not None and (b.get("hr_zones") or b.get("crushed_zones")):
                        tgt.setdefault("features", {})["zone_profile"] = {
                            "crushed": b["crushed_zones"],
                            "zone_dmg": b.get("zone_dmg"),
                            "vs_arm": pe["name"],
                            "zone_score": b.get("zone_score"),
                            "overlap": b.get("overlap"),          # true ZONE-badge overlap count
                            "hr_zones": b.get("hr_zones"),        # his HR count per zone
                            "zone_edge": b.get("zone_edge"),      # matchup edges vs his hand
                            "arm_grid": pe.get("zone_grid"),      # pitcher usage per zone
                            "arm_throws": pe.get("throws"),
                            "meatball_zones": pe.get("meatball_zones"),  # for the amber-dot map
                        }

            # ---- SIGNAL CONVERGENCE, per prop type ----
            # Weighted by what the backtest ACTUALLY measured for each prop, which differs a lot:
            # the HR heat bands separate hard (+47/+30/+14%), but 1+hit is only +11% at the very
            # top and flat below it. So a band only counts as a "measured signal" for a prop where
            # it showed real lift — otherwise convergence would just be counting noise.
            # Badge lift (POW +76%, LOCK +36%) was measured against HR outcomes only, so badges
            # count as measured for HR and stay provisional everywhere else. DUE (+1%) and COOL
            # (-2%) are excluded from every prop. Parallel lens; never feeds any heat model.
            try:
                # Lift is READ FROM THE LATEST BACKTEST, not hardcoded — so when you re-run the
                # backtest, convergence re-weights itself automatically. A band only counts as a
                # measured signal if it CURRENTLY shows real lift (>=8% over the pool base rate);
                # if a re-run flattens a band, it silently stops counting instead of quietly
                # asserting a number that's no longer true. Falls back to the last-known values
                # only when a backtest file isn't present at all.
                def _lift_bands(block, key="hit", min_lift=8.0):
                    """[(heat_cut, lift_pct)] for bands that currently beat the base rate."""
                    try:
                        bt_ = (block or {}).get("by_tier") or {}
                        tot_n = sum(v.get("n", 0) for v in bt_.values())
                        tot_h = sum(v.get(key, 0) for v in bt_.values())
                        if not tot_n or not tot_h:
                            return None
                        base = tot_h / tot_n
                        cuts = {"70+": 70, "55-69": 55, "40-54": 40}
                        out = []
                        for tier, cut in cuts.items():
                            v = bt_.get(tier) or {}
                            if v.get("n", 0) < 150:
                                continue
                            rate = v.get(key, 0) / v["n"]
                            lift = 100.0 * (rate / base - 1) if base else 0
                            if lift >= min_lift:
                                out.append((cut, int(round(lift))))
                        return sorted(out, key=lambda x: -x[0]) or None
                    except Exception:
                        return None
                _BT = {}
                try:
                    _btp = os.path.join(os.path.dirname(__file__), "..", "docs", "backtest.json")
                    if os.path.exists(_btp):
                        with open(_btp) as _f:
                            _BT = json.load(_f) or {}
                except Exception:
                    _BT = {}
                _props_bt = _BT.get("props") or {}
                _LIFT = {
                    "hr":  _lift_bands({"by_tier": _BT.get("by_tier")}, key="hr") or [(70, 47), (55, 30), (40, 14)],
                    "hit": _lift_bands(_props_bt.get("hit1")) or [(70, 11)],
                    "hrr": _lift_bands(_props_bt.get("hrr")) or [(70, 13)],
                }
                # badge lift, likewise read live from the backtest's measured table
                _BADGES = {}
                try:
                    for _k, _v in (_BT.get("by_badge") or {}).items():
                        _l = _v.get("lift")
                        if _v.get("n", 0) >= 300 and _l and _l >= 1.15:
                            _BADGES[str(_k).lower()] = int(round(100 * (_l - 1)))
                except Exception:
                    pass
                if not _BADGES:
                    _BADGES = {"pow": 76, "lock": 36}
                # pitcher K bands, same treatment
                _K_BANDS = None
                try:
                    _pk = (_props_bt.get("pk") or {}).get("by_tier") or {}
                    _tn = sum(v.get("n", 0) for v in _pk.values())
                    _tk = sum(v.get("total_ks", 0) for v in _pk.values())
                    if _tn and _tk:
                        _kbase = _tk / _tn
                        _K_BANDS = []
                        for _tier, _cut in (("70+", 70), ("55-69", 55)):
                            _v = _pk.get(_tier) or {}
                            if _v.get("n", 0) >= 150:
                                _l = 100.0 * ((_v["total_ks"] / _v["n"]) / _kbase - 1)
                                if _l >= 8:
                                    _K_BANDS.append((_cut, int(round(_l))))
                except Exception:
                    _K_BANDS = None
                if not _K_BANDS:
                    _K_BANDS = [(70, 28), (55, 10)]
                print(f"[build] convergence lift from backtest: hr={_LIFT['hr']} hit={_LIFT['hit']} "
                      f"hrr={_LIFT['hrr']} badges={_BADGES} k={_K_BANDS}")
                globals()["_CONVERGE_LIFT"] = {
                    "hr": _LIFT["hr"], "hit": _LIFT["hit"], "hrr": _LIFT["hrr"],
                    "badges": _BADGES, "k": _K_BANDS,
                    "source": ("backtest" if _BT else "fallback"),
                }
                _HEAT_KEY = {"hr": "heat", "hit": "hit_heat", "hrr": "hrr_heat"}
                # 75th percentile of tonight's zone-edge scores — "clearly better than the rest
                # of this slate" rather than an absolute number that may not match the scale.
                def _ze_of(x):
                    return (((x.get("features") or {}).get("zone_profile") or {}).get("zone_edge") or {}).get("edge_score")
                _zevals = sorted(v for v in (_ze_of(x) for x in players) if v is not None)
                _ZE_CUT = (_zevals[int(0.75 * (len(_zevals) - 1))] if len(_zevals) >= 12 else 1e9)

                # Launch-profile gates at the 85th percentile of tonight's board. Fixed cutoffs
                # fired for ~90% of hitters, which is decoration; p85 keeps it to the genuine top.
                def _pctl_cut(_vals, _q=0.85):
                    _v = sorted(x for x in _vals if x is not None)
                    return _v[int(_q * (len(_v) - 1))] if len(_v) >= 20 else 1e9
                _LAUNCH_CUT = {}
                for _mk in ("pull_air_pct", "ideal_aa_pct"):
                    _LAUNCH_CUT[_mk] = _pctl_cut([
                        ((x.get("metrics") or {}).get(_mk) or {}).get("recent")
                        or ((x.get("metrics") or {}).get(_mk) or {}).get("season")
                        for x in players])
                _pbvals = []
                for _x in players:
                    _v2 = _x.get("vs_pitch") or {}
                    _pn = sum((_r[8] or 0) for _r in _v2.values() if len(_r) > 8)
                    _bn = sum((_r[4] or 0) for _r in _v2.values() if len(_r) > 8)
                    if _bn >= 60:
                        _pbvals.append(100.0 * _pn / _bn)
                _LAUNCH_CUT["pull_barrel"] = _pctl_cut(_pbvals)
                _PARK_CUT = _pctl_cut([x.get("park_hr_factor") for x in players], 0.75)
                _P30_CUT = _pctl_cut([(x.get("parks30") or {}).get("out_here") for x in players])
                _AP_CUT  = _pctl_cut([(x.get("parks30") or {}).get("avg_parks") for x in players])
                _VH_CUT  = _pctl_cut([(x.get("opp_pitcher") or {}).get("vs_hand_score") for x in players])
                _BPH_CUT = _pctl_cut([(x.get("opp_bullpen") or {}).get("vs_hand") for x in players])
                _LATE_CUT= _pctl_cut([((x.get("features") or {}).get("late_hr") or {}).get("score") for x in players])
                print(f"[build] launch gates (p85): {_LAUNCH_CUT}")
                print(f"[build] new gates (p85): parks30={_P30_CUT} avg_parks={_AP_CUT} "
                      f"vs_hand={_VH_CUT} pen_hand={_BPH_CUT} late={_LATE_CUT}")
                if _zevals:
                    print(f"[build] zone-edge gate (p75 of slate) = {_ZE_CUT} "
                          f"(range {_zevals[0]}-{_zevals[-1]}, n={len(_zevals)})")

                for _p in players:
                    _bset = {str(b).lower() for b in (_p.get("badges") or [])}
                    # ---- the full provisional evidence set ----
                    # Grouped into FAMILIES because these reads are not independent: barrel rate,
                    # hard-hit%, EV percentile and Square Up are all measuring the same thing.
                    # Counting them separately would let one piece of evidence pose as four.
                    # Each family contributes at most 1 to the count, and the detail is kept so
                    # the card can show what fired inside it.
                    _fam = {}                       # family -> [labels]
                    def _add(family, label):
                        _fam.setdefault(family, []).append(label)

                    _F = _p.get("features") or {}
                    # -- contact quality (raw power / earned contact) --
                    _pc = _F.get("percentiles") or {}
                    _hi_pctl = [f"{v.get('label') or k} P{v['pctl']}"
                                for k, v in _pc.items()
                                if isinstance(v, dict) and (v.get("pctl") or 0) >= 80]
                    if len(_hi_pctl) >= 3:        # a cluster, not a single lucky metric
                        _add("contact", "MLB pctl: " + ", ".join(sorted(_hi_pctl)[:3]))
                    _sq = _F.get("square_up") or {}
                    if (_sq.get("rating") or 0) >= 30:
                        _add("contact", f"Square Up {_sq['rating']}")
                    _hp = _F.get("hr_power") or {}
                    if (_hp.get("barrel_pct") or 0) >= 10:
                        _add("contact", f"Barrel {_hp['barrel_pct']}%")
                    if (_hp.get("max_dist") or 0) >= 440:
                        _add("contact", f"Ceiling {_hp['max_dist']}ft")
                    # Bat tracking joins Contact Quality rather than forming its own family: it
                    # measures the same underlying thing as exit velo and barrel rate — how well
                    # this hitter strikes a baseball. A separate family would let one piece of
                    # evidence count twice, which is the collinearity the family structure
                    # exists to prevent. Credit is conditional on the arm's velocity, because a
                    # compact 78-mph swing is an edge against 97 and irrelevant against 89.
                    try:
                        _bt = _p.get("bat_tracking")
                        _fbv = ((_p.get("opp_pitcher") or {}).get("fb_velo")
                                or (_p.get("opp_pitcher") or {}).get("season", {}).get("fb_velo"))
                        _cred = arsenal_mod.bat_speed_vs_velocity_credit(_bt, _fbv)
                        if _cred is not None and _cred >= 0.60:
                            _lab = f"Bat speed {_bt.get('avg_bat_speed')} vs {float(_fbv):.0f} mph"
                            if _bt.get("short_fast_rate") is not None:
                                _lab += f" ({_bt['short_fast_rate']}% short+fast)"
                            _add("contact", _lab)
                    except Exception:
                        pass
                    # -- launch profile: does he actually get the ball AIRBORNE TO THE PULL SIDE? --
                    # Kept as its own family on purpose. Measured across tonight's board, pull-air
                    # correlates r=-0.11 with exit velo and r=-0.01 with barrel rate, and ideal
                    # launch angle r=+0.01 with EV — i.e. launch geometry is nearly INDEPENDENT of
                    # raw contact quality. (Hard-hit%, by contrast, is r=+0.81 with EV, which is
                    # why it belongs inside "contact" rather than here.) The classic failure mode
                    # this separates: the guy who scorches everything into the ground.
                    _mm = _p.get("metrics") or {}
                    def _lv(_k):
                        _v = _mm.get(_k) or {}
                        return _v.get("recent") if _v.get("recent") is not None else _v.get("season")
                    _pa = _lv("pull_air_pct")
                    if _pa is not None and _pa >= _LAUNCH_CUT.get("pull_air_pct", 1e9):
                        _add("launch", f"Pull-air {round(_pa,1)}%")
                    _ia = _lv("ideal_aa_pct")
                    if _ia is not None and _ia >= _LAUNCH_CUT.get("ideal_aa_pct", 1e9):
                        _add("launch", f"Ideal angle {round(_ia,1)}%")
                    # Batted-ball profile: a fly-ball hitter is a different animal from a
                    # ground-ball hitter no matter how hard either one hits it. Tracked at 1.30x
                    # over 576 player-games (z=+2.31). It belongs here rather than in "contact"
                    # because it describes HOW he launches the ball, not how hard.
                    _hl = str(_p.get("hit_label") or "").lower()
                    if _hl == "fb":
                        _add("launch", "Fly-ball hitter")
                    elif _hl == "ld":
                        _add("launch", "Line-drive hitter")

                    # pull-BARREL: a barrel hit to the pull side is about as close to a guaranteed
                    # home run as a batted ball gets. Aggregated across every pitch type he's seen.
                    try:
                        _vp2 = _p.get("vs_pitch") or {}
                        _pb = sum((_v[8] or 0) for _v in _vp2.values() if len(_v) > 8)
                        _bb2 = sum((_v[4] or 0) for _v in _vp2.values() if len(_v) > 8)
                        if _bb2 >= 60:
                            _pbr = round(100.0 * _pb / _bb2, 1)
                            _p["pull_barrel_pct"] = _pbr
                            if _pbr >= _LAUNCH_CUT.get("pull_barrel", 1e9):
                                _add("launch", f"Pull-barrel {_pbr}%")
                    except Exception:
                        pass

                    # -- form: is the 2-week window beating his season baseline? --
                    _m = _p.get("metrics") or {}
                    _tr = []
                    for _k, _lab, _min in (("barrel_pct", "barrel", 2.0), ("avg_ev", "EV", 1.0),
                                           ("pull_air_pct", "pull-air", 4.0),
                                           ("ideal_aa_pct", "ideal-angle", 4.0),
                                           ("hardhit_pct", "hard-hit", 4.0),
                                           ("iso", "ISO", 0.040)):
                        _v = _m.get(_k) or {}
                        _r, _s = _v.get("recent"), _v.get("season")
                        if _r is not None and _s is not None and (_r - _s) >= _min:
                            _tr.append(f"{_lab} +{round(_r - _s, 1)}")
                    if len(_tr) >= 2:               # need 2 of 3 rising, not one noisy metric
                        _add("form", "2wk > season: " + ", ".join(_tr))
                    # -- location: zone overlap --
                    _ze = (_F.get("zone_profile") or {}).get("zone_edge") or {}
                    # Gate RELATIVE to tonight's slate. A fixed cutoff of 62 was used here
                    # originally, but edge_score is a usage-weighted average of zone power and
                    # actually ranges ~1-33 — so the gate never once fired and this family was
                    # dead. A percentile gate is also self-correcting if the scale ever changes.
                    if ((_F.get("discipline") or {}).get("crosshair")) is True:
                        _add("location", "Crosshair: his zones = the arm's meatballs")
                    if _ze.get("edge_score") is not None and _ze["edge_score"] >= _ZE_CUT:
                        _add("location", f"Zone edge {round(_ze['edge_score'],1)}"
                             + (f" ({_ze.get('hot_in_top')}/{_ze.get('top_k')} hot)" if _ze.get("top_k") else ""))
                    # -- arsenal: does he punish what this arm throws? --
                    _pm = _F.get("pitch_matchup") or {}
                    if (_pm.get("score") or 0) >= 65:
                        _add("arsenal", f"Pitch matchup {round(_pm['score'])}")
                    # Late-inning blend: plate appearances past the starter's xBF are faced
                    # against the AVAILABLE bullpen's aggregate mix, not the starter's. A hitter
                    # batting 2nd sees the starter three times and the pen once; a hitter batting
                    # 8th may see the pen twice. Evaluating every PA against the starter
                    # overstates the matchup for exactly the hitters who face him least.
                    try:
                        _armid = (_p.get("opp_pitcher") or {}).get("id")
                        _xbf = ((_p.get("opp_pitcher") or {}).get("kengine") or {}).get("xbf")
                        _spot = _p.get("lineup_spot")
                        _penst = pen_state.get(_p.get("opp_team")) if pen_state else None
                        if _armid and _xbf and _spot and _penst and _penst.get("arsenal"):
                            _handk = _p.get("bats")
                            if _handk == "S":
                                _handk = "L" if (_p.get("opp_pitcher") or {}).get("throws") == "R" else "R"
                            _sp_ars = arsenal_mod.handedness_first_arsenal(
                                (arsenals.get(int(_armid)) or {}), _handk)
                            # PAs this spot gets: 1st, 2nd, 3rd, 4th trip = spot, spot+9, ...
                            _late = [i for i in range(int(_spot) - 1, 40, 9) if i >= _xbf]
                            if _sp_ars and _late:
                                _blend = arsenal_mod.blend_arsenal_for_pa(
                                    _sp_ars, _penst["arsenal"], _late[0], _xbf)
                                _top = ", ".join(f"{pt} {u}%" for pt, u, *_ in _blend[:2])
                                _add("arsenal",
                                     f"PA {_late[0]+1}+ vs bullpen mix ({_top})")
                            if _penst.get("n_out"):
                                _add("arm", f"{_penst['n_out']} pen arm(s) unavailable")
                    except Exception:
                        pass
                    _fit = None
                    try:
                        _arm = (_p.get("opp_pitcher") or {}).get("id")
                        _hand = _p.get("bats")
                        if _hand == "S":
                            _hand = "L" if (_p.get("opp_pitcher") or {}).get("throws") == "R" else "R"
                        _ars = (arsenals.get(int(_arm)) or {}).get(_hand) if _arm else None
                        _vp = _p.get("vs_pitch") or {}
                        if _ars and _vp:
                            # The full line against THIS ARM'S ACTUAL MIX — every pitch he throws
                            # this hand >=10% of the time. Counts are summed first and divided
                            # once, so a 200-pitch sample can't be outweighed by a 20-pitch one.
                            # This is the number the write-up is really making: "against the
                            # pitches he lives on, this hitter owns N homers and a .xxx SLG".
                            _mix = [(pt, pct) for pt, pct, _n in _ars if pct >= 10.0]
                            _T = [0.0] * 18
                            _used = []
                            for _pt, _pct in _mix:
                                _v2 = _vp.get(_pt)
                                if _v2 and len(_v2) >= 18:
                                    _used.append((_pt, _pct))
                                    for _i in range(18):
                                        _T[_i] += (_v2[_i] or 0)
                            if _used and _T[4] >= 25:          # need real batted-ball volume
                                (_pit, _wh, _pa3, _k3, _bbe3, _nev, _evs,
                                 _brl3, _pbrl, _air3, _pair, _hard3,
                                 _hr3, _ab3, _tb3, _hits3, _d350, _near) = _T
                                _slg = (_tb3 / _ab3) if _ab3 else None
                                _iso = ((_tb3 - _hits3) / _ab3) if _ab3 else None   # SLG - AVG
                                _mixline = {
                                    "pitches": [f"{_ptn}" for _ptn, _ in _used],
                                    "usage": round(sum(_p2 for _, _p2 in _used), 1),
                                    "bbe": int(_bbe3), "hr": int(_hr3),
                                    "near": int(_near), "d350": int(_d350),
                                    "slg": round(_slg, 3) if _slg is not None else None,
                                    "iso": round(_iso, 3) if _iso is not None else None,
                                    "avg": round(_hits3 / _ab3, 3) if _ab3 else None,
                                    "avg_ev": round(_evs / _nev, 1) if _nev else None,
                                    "hard_pct": round(100.0 * _hard3 / _nev, 1) if _nev else None,
                                    "barrel_pct": round(100.0 * _brl3 / _bbe3, 1) if _bbe3 else None,
                                    "pullair_pct": round(100.0 * _pair / _air3, 1) if _air3 else None,
                                    "k_pct": round(100.0 * _k3 / _pa3, 1) if _pa3 else None,
                                }
                                _p["vs_mix"] = _mixline
                                _fit = _mixline["barrel_pct"]
                    except Exception:
                        _fit = None
                    # ---- PUNISH SCORE: green-on-green overlap ----
                    # vs_mix above weights purely by how OFTEN he throws a pitch. This asks the
                    # sharper question: does the hitter mash the pitches this arm both throws a
                    # lot AND gets hurt on? A pitch thrown 40% of the time that nobody touches is
                    # worth far less than one thrown 25% that leaves the yard.
                    # weight(pitch) = usage x vulnerability(SLG + HR rate + barrel allowed)
                    # score        = weighted average of the hitter's damage on those same pitches
                    # 50 = neutral. Both sides are handedness-correct: the arm's line vs THIS
                    # batter's side, and the batter's line vs pitches from THIS arm's side.
                    try:
                        _at = (arm_tables.get(int(_arm)) or {}).get(_hand) if _arm else None
                        _vp3 = _p.get("vs_pitch") or {}
                        if _at and _vp3:
                            _num = _den = 0.0
                            _drivers = []
                            for _pt, _line in _at.items():
                                if not _line or len(_line) < 23:
                                    continue
                                _usage = _line[1] or 0.0
                                if _usage < 8.0:
                                    continue
                                # how badly does he get hurt on this pitch (0-1)
                                _vSlg = min(1.0, max(0.0, ((_line[5] or 0.35) - 0.330) / 0.320))
                                _vBrl = min(1.0, max(0.0, ((_line[16] or 6.0) - 5.0) / 13.0))
                                _vHr  = min(1.0, (float(_line[7] or 0) / max(1.0, float(_line[2] or 1))) / 0.06)
                                _vuln = 0.45 * _vSlg + 0.35 * _vBrl + 0.20 * _vHr
                                _w = (_usage / 100.0) * (0.35 + 0.65 * _vuln)
                                # how well does the hitter do on that same pitch (0-1)
                                _bv = _vp3.get(_pt)
                                if not _bv or len(_bv) < 18 or (_bv[4] or 0) < 8:
                                    continue
                                _bSlg = (_bv[14] / _bv[13]) if _bv[13] else None
                                _bBrl = (100.0 * _bv[7] / _bv[4]) if _bv[4] else None
                                _bHard = (100.0 * _bv[11] / _bv[5]) if _bv[5] else None
                                _sS = min(1.0, max(0.0, ((_bSlg or 0.38) - 0.330) / 0.320))
                                _sB = min(1.0, max(0.0, ((_bBrl or 6.0) - 4.0) / 14.0))
                                _sH = min(1.0, max(0.0, ((_bHard or 35.0) - 30.0) / 25.0))
                                _strength = 0.40 * _sS + 0.35 * _sB + 0.25 * _sH
                                _num += _w * _strength
                                _den += _w
                                _drivers.append((_pt, round(_usage, 1), round(100 * _vuln),
                                                 round(100 * _strength), int(_bv[12] or 0)))
                            if _den > 0 and len(_drivers) >= 2:
                                _score = int(round(100.0 * _num / _den))
                                _drivers.sort(key=lambda d: -(d[1] * d[2] * d[3]))
                                _p["mix_punish"] = {
                                    "score": _score,
                                    "drivers": [{"pt": d[0], "usage": d[1], "vuln": d[2],
                                                 "strength": d[3], "hr": d[4]} for d in _drivers[:4]],
                                }
                    except Exception:
                        pass

                    if _fit is not None:
                        _p["arsenal_fit"] = _fit
                        _vm = _p.get("vs_mix") or {}
                        if _fit >= 9.0:
                            _bits = []
                            if _vm.get("hr"):
                                _bits.append(f"{_vm['hr']} HR")
                            if _vm.get("near"):
                                _bits.append(f"{_vm['near']} near")
                            if _vm.get("slg") is not None:
                                _bits.append(f"{_vm['slg']:.3f}".replace("0.", ".") + " SLG")
                            _add("arsenal", "vs his mix: " + (", ".join(_bits) if _bits
                                                              else f"{_fit}% barrel"))
                    # -- opposing arm: vulnerability + the rate that matters, HR per 9 --
                    _op = _p.get("opp_pitcher") or {}
                    # Prefer the arm's vulnerability TO THIS BATTER'S SIDE over his overall score.
                    # It's the sharper number (wider spread on the board: max 70 vs 58) and it's
                    # the one that actually applies to this matchup.
                    _vh = _op.get("vs_hand_score")
                    if _vh is not None and _vh >= _VH_CUT:
                        _add("arm", f"Arm vuln vs {_op.get('vs_hand') or 'hand'} {int(_vh)}")
                    elif (_op.get("hr_score") or 0) >= 60:
                        _add("arm", f"Arm vuln {int(_op['hr_score'])}")
                    # bullpen he'd meet late, and whether it's exploitable tonight
                    _bp = _p.get("opp_bullpen") or {}
                    if (_bp.get("vs_hand") or 0) >= _BPH_CUT:
                        _add("arm", f"Pen weak vs {_bp.get('hand') or 'hand'} {int(_bp['vs_hand'])}")
                    _lh = (_F.get("late_hr") or {})
                    if (_lh.get("score") or 0) >= _LATE_CUT:
                        _add("arm", f"Late-game pen edge {round(_lh['score'],1)}")
                    try:
                        _hpa = (_op.get("recent") or {}).get("hr_per_pa")
                        if _hpa is not None:
                            _hr9 = round(float(_hpa) / 100.0 * 38.0 * 9.0 / 9.0, 2)  # ~38 PA/9ip
                            _p["opp_hr9"] = _hr9
                            if _hr9 >= 1.4:
                                _add("arm", f"Allows {_hr9} HR/9")
                    except Exception:
                        pass
                    if str((_op.get("form") or {}).get("label") or "").upper() in ("SLIPPING", "SHELLABLE", "HITTABLE"):
                        _add("arm", f"Arm {_op['form']['label'].title()}")
                    # -- park fit: would his OWN recent contact leave THIS yard? --
                    # park_hr_factor says "this park is friendly"; parks30 says "these specific
                    # batted balls clear this specific fence", which is a hitter x park
                    # interaction rather than a property of the park. Kept separate from
                    # Environment for exactly that reason.
                    _p30 = _p.get("parks30") or {}
                    _oh = _p30.get("out_here")
                    if _oh is not None and _oh >= _P30_CUT:
                        _add("parkfit", f"{int(_oh)} recent balls clear this park")
                    # Geometry-aware near misses: balls that died within 5 ft of the wall AT
                    # THEIR OWN SPRAY ANGLE. This is contact that clears on a warmer night or in
                    # a different park, which is exactly the part of a profile most likely to
                    # convert — and a flat 350-ft rule cannot express it.
                    _nm = (near_miss.get(int(bid)) if near_miss else None)
                    if _nm and _nm.get("near", 0) >= 2:
                        _add("parkfit", f"{_nm['near']} near misses (within 5ft of the wall)")
                    if _nm and _nm.get("best_delta") is not None and _nm["best_delta"] >= -2.0:
                        _add("parkfit", f"Best ball {_nm['best_delta']:+.0f}ft vs the wall")
                    _ap = _p30.get("avg_parks")
                    if _ap is not None and _ap >= _AP_CUT:
                        _add("parkfit", f"Avg ball clears {round(_ap,1)}/30 parks")

                    # -- opportunity: how many cracks at it does he actually get? --
                    # Spots 1-4 all tracked 1.23-1.30x (z = +2.1 to +2.7) while spot 9 sat at 7%.
                    # This is NOT a hitter-quality proxy: median heat is essentially flat across
                    # spots 1-7 on the board, so the effect is the extra plate appearance, which
                    # is genuinely independent evidence. Counted as CONTEXT, since it describes
                    # the slot rather than the hitter.
                    _sp2 = _p.get("lineup_spot")
                    if _sp2 and int(_sp2) <= 3:
                        _sfx = {1: "1st", 2: "2nd", 3: "3rd"}[int(_sp2)]
                        _add("opportunity", f"Bats {_sfx}")

                    # -- environment --
                    _pf = _p.get("park_hr_factor")
                    if _pf is not None and _pf >= _PARK_CUT:
                        _add("env", f"Park {_pf:.2f}x")
                    _mc = _F.get("microclimate") or {}
                    if (_mc.get("boost") or 0) >= 5:
                        _add("env", f"Wind/air +{int(_mc['boost'])}%")
                    # this hitter is personally temp-sensitive AND tonight's air suits him
                    if str(_mc.get("flag") or "") in ("WARM BOOST", "COLD BOOST"):
                        _add("env", f"{_mc['flag'].title()} ({int(_mc.get('today_temp') or 0)}F)")
                    # -- overall matchup grade (a composite, so its own family) --
                    _mg = _p.get("matchup_grade") or {}
                    if str(_mg.get("grade") or "").upper() in ("ELITE", "STRONG"):
                        _add("grade", f"Grade {_mg['grade'].title()}")
                    _bs = _p.get("bomb_score") or {}
                    if (_bs.get("score") or 0) >= 65:
                        _add("grade", f"Bomb {int(_bs['score'])}")

                    _FAM_LABEL = {"contact": "Contact quality", "launch": "Launch profile",
                                  "parkfit": "Fits this park", "opportunity": "Top of the order",
                                  "form": "Form trending up",
                                  "location": "Zone matchup", "arsenal": "Arsenal matchup",
                                  "arm": "Opposing arm", "env": "Environment",
                                  "grade": "Matchup grade"}
                    # Families split into two kinds, because they answer different questions:
                    #   HITTER families say "is this guy actually good/hot right now"
                    #   CONTEXT families say "is tonight a good spot" — true of everyone in the game
                    # A cold hitter in a great park facing a bad arm can rack up context families
                    # while carrying no evidence that HE will do anything. Ranking on the combined
                    # count rewarded exactly that, so the two are now tracked separately.
                    # parkfit is a hitter x park interaction, so it counts as hitter evidence,
                    # unlike raw park factor which is true of everyone in the game.
                    _CONTEXT = {"arm", "env", "opportunity"}
                    _shared = [{"k": _k, "lab": _FAM_LABEL.get(_k, _k.title()),
                                "detail": _v, "n": len(_v),
                                "kind": ("context" if _k in _CONTEXT else "hitter")}
                               for _k, _v in _fam.items()]

                    # --- reliability: how much data is actually behind this hitter's signals? ---
                    # Rate stats on a thin denominator are the classic way a nobody floats to the
                    # top of a leaderboard. This is reported, not hidden, so you can filter on it.
                    _bbe_season = ((_p.get("sample") or {}).get("season")
                                   or (_F.get("hr_power") or {}).get("n") or 0)
                    _bbe_recent = ((_p.get("sample") or {}).get("L15")
                                   or (_p.get("sample") or {}).get("L14d") or 0)
                    if _bbe_season >= 250:
                        _rel, _rel_lab = 3, "Full season"
                    elif _bbe_season >= 120:
                        _rel, _rel_lab = 2, "Adequate"
                    elif _bbe_season >= 60:
                        _rel, _rel_lab = 1, "Thin"
                    else:
                        _rel, _rel_lab = 0, "Very thin"
                    _p["reliability"] = {"score": _rel, "label": _rel_lab,
                                         "bbe_season": int(_bbe_season), "bbe_recent": int(_bbe_recent),
                                         "confirmed": (_p.get("lineup_status") == "confirmed")}
                    _conv = {}
                    for _prop, _bands in _LIFT.items():
                        _h = _p.get(_HEAT_KEY[_prop])
                        _meas, _prov = [], list(_shared)
                        if _h is not None:
                            for _cut, _lift in _bands:
                                if _h >= _cut:
                                    _meas.append({"k": "heat", "lab": f"Heat {int(_h)}", "lift": _lift})
                                    break
                        for _bk, _bl in _BADGES.items():
                            if _bk in _bset:
                                if _prop == "hr":
                                    _meas.append({"k": _bk, "lab": _bk.upper(), "lift": _bl})
                                else:
                                    _prov.append({"k": _bk, "lab": _bk.upper()})
                        _n_hit = sum(1 for s in _prov if s.get("kind") != "context")
                        _n_ctx = sum(1 for s in _prov if s.get("kind") == "context")
                        _conv[_prop] = {
                            "measured": _meas, "provisional": _prov,
                            "n_measured": len(_meas), "n_provisional": len(_prov),
                            "n_hitter": _n_hit, "n_context": _n_ctx,
                            "lift": sum(m["lift"] for m in _meas),
                        }
                    _p["converge"] = _conv
                # ---- second pass: the punish score needs the whole slate before it can be
                # gated, since "punishes his mix" only means something relative to the other
                # matchups on the board tonight. Runs here, after every player has a score.
                try:
                    _pvals = sorted(v for v in
                                    ((x.get("mix_punish") or {}).get("score") for x in players)
                                    if v is not None)
                    if len(_pvals) >= 20:
                        _PUNISH_CUT = _pvals[int(0.75 * (len(_pvals) - 1))]
                        print(f"[build] punish-score gate (p75 of slate) = {_PUNISH_CUT} "
                              f"(range {_pvals[0]}-{_pvals[-1]}, n={len(_pvals)})")
                        for _p in players:
                            _mp = (_p.get("mix_punish") or {}).get("score")
                            if _mp is None or _mp < _PUNISH_CUT:
                                continue
                            _c2 = (_p.get("converge") or {}).get("hr")
                            if not isinstance(_c2, dict):
                                continue
                            _drv = (_p.get("mix_punish") or {}).get("drivers") or []
                            _lab = ", ".join(f"{d['pt']} {d['usage']}%" for d in _drv[:2])
                            _entry = {"k": "arsenal", "lab": "Arsenal matchup",
                                      "detail": [f"Punishes his mix {_mp}/100"
                                                 + (f" (on {_lab})" if _lab else "")],
                                      "n": 1, "kind": "hitter"}
                            _existing = next((s for s in _c2.get("provisional") or []
                                              if s.get("k") == "arsenal"), None)
                            if _existing:
                                _existing.setdefault("detail", []).append(_entry["detail"][0])
                                _existing["n"] = len(_existing["detail"])
                            else:
                                _c2.setdefault("provisional", []).append(_entry)
                                _c2["n_provisional"] = len(_c2["provisional"])
                                _c2["n_hitter"] = sum(1 for s in _c2["provisional"]
                                                      if s.get("kind") != "context")
                except Exception as _e:
                    print(f"[build] punish gate skipped (non-fatal): {_e}")

                # ---- COV rank: position on the Signal Convergence board with HEAT EXCLUDED ----
                # Ranked here (not just at grading time) so the live board can badge it. Heat is
                # dropped from the ordering on purpose: COV answers "what does the evidence
                # OUTSIDE the core model like tonight", which is the whole point of the no-heat
                # view. Reliability is applied the same way the board applies it, so a thin-sample
                # hitter can't take COV1 on 40 batted balls.
                try:
                    _REL_W = {3: 1.0, 2: 0.66, 1: 0.49, 0: 0.63}   # measured lift by sample band

                    def _covscore(_x):
                        _c = (_x.get("converge") or {}).get("hr") or {}
                        _m = [s for s in (_c.get("measured") or []) if s.get("k") != "heat"]
                        _pv = _c.get("provisional") or []
                        _nh = sum(1 for s in _pv if s.get("kind") != "context")
                        _nc = sum(1 for s in _pv if s.get("kind") == "context")
                        _lift = sum(s.get("lift") or 0 for s in _m)
                        _rel = ((_x.get("reliability") or {}).get("score"))
                        _rel = 3 if _rel is None else _rel
                        return (len(_m) * 1000 + _lift + _nh * 0.6 + _nc * 0.2) * _REL_W.get(_rel, 1.0)

                    _ranked = sorted(
                        [x for x in players if (x.get("converge") or {}).get("hr")],
                        key=_covscore, reverse=True)
                    for _i, _x in enumerate(_ranked):
                        if _covscore(_x) <= 0:
                            break
                        _x["cov_rank"] = _i + 1

                    # Heat-INCLUSIVE convergence rank. The no-heat rank above answers "what does
                    # the evidence outside the core model like"; this one answers "what does all
                    # the evidence together like". Both are worth showing side by side — when a
                    # hitter is high on both, every read agrees; when the two diverge sharply,
                    # that's the interesting case.
                    def _covscore_heat(_x):
                        _c = (_x.get("converge") or {}).get("hr") or {}
                        _m = _c.get("measured") or []
                        _pv = _c.get("provisional") or []
                        _nh = sum(1 for s in _pv if s.get("kind") != "context")
                        _nc = sum(1 for s in _pv if s.get("kind") == "context")
                        _lift = sum(s.get("lift") or 0 for s in _m)
                        _rel = ((_x.get("reliability") or {}).get("score"))
                        _rel = 3 if _rel is None else _rel
                        return (len(_m) * 1000 + _lift + _nh * 0.6 + _nc * 0.2) * _REL_W.get(_rel, 1.0)

                    _rankedH = sorted(
                        [x for x in players if (x.get("converge") or {}).get("hr")],
                        key=_covscore_heat, reverse=True)
                    for _i, _x in enumerate(_rankedH):
                        if _covscore_heat(_x) <= 0:
                            break
                        _x["cov_rank_heat"] = _i + 1
                    print(f"[build] COV ranks assigned to {sum(1 for x in players if x.get('cov_rank'))} hitters")
                except Exception as _e:
                    print(f"[build] COV rank skipped (non-fatal): {_e}")
            except Exception as _e:
                print(f"[build] convergence scoring skipped (non-fatal): {_e}")

            # ---- BOMB SCORE: the composite batter-vs-pitcher matchup score (replaces the old
            # max-EV "Bomb" sort). Needs zone overlap, so it runs after the edges block. ----
            try:
                _era_by_pid = {int(k): v.get("era") for k, v in (pstats or {}).items()}
                # vuln tier per arm, for the Matchup Grade's pitcher-vulnerability factor
                _vuln_by_pid = {}
                for _pe in pitcher_edges:
                    if _pe.get("vuln") and _pe.get("id") is not None:
                        _vuln_by_pid[_pe["id"]] = _pe["vuln"]
                for _p in players:
                    _f = _p.get("features") or {}
                    _zp = _f.get("zone_profile") or {}
                    _ov = (_zp.get("overlap") or {}).get("count", 0)
                    _w = (_p.get("windows") or {}).get("L14d") or {}
                    _opp = _p.get("opp_pitcher") or {}
                    _opp_id = _opp.get("id")
                    # platoon edge: LHB vs RHP or RHB vs LHP (switch hitters always have it)
                    _bats = (_p.get("bats") or "").upper()
                    _thr = (_opp.get("throws") or "").upper()
                    _plat = None
                    if _bats and _thr:
                        _plat = True if (_bats == "S" or _bats != _thr) else False
                    _tto = (tto_by_pid.get(_opp_id) or {}).get("score") if _opp_id else None
                    bs = features.bomb_score(
                        iso=_w.get("iso"), slg=_w.get("slg"),
                        overlap_count=_ov,
                        park_boost=(_p.get("park_hr") or {}).get("boost"),
                        platoon=_plat,
                        pitcher_era=_era_by_pid.get(_opp_id) if _opp_id else None,
                        hot_streak=_p.get("trend"),
                        tto_score=_tto)
                    if bs:
                        _p["bomb_score"] = bs

                    # ---- MATCHUP GRADE: factor convergence across the five inputs ----
                    _vt = (_vuln_by_pid.get(_opp_id) or {}).get("tier") if _opp_id else None
                    _dd = ((_f.get("discipline") or {}).get("grade_delta")) or 0
                    mg = features.matchup_grade(
                        iso=_w.get("iso"),
                        overlap_count=_ov,
                        vuln_tier=_vt,
                        park_factor=_p.get("park_hr_factor"),
                        hot_form=_p.get("trend"),
                        discipline_delta=_dd)
                    if mg:
                        _p["matchup_grade"] = mg
                _n_bs = sum(1 for _p in players if _p.get("bomb_score"))
                _n_mg = sum(1 for _p in players if _p.get("matchup_grade"))
                _elite = sum(1 for _p in players if (_p.get("matchup_grade") or {}).get("grade") == "ELITE")
                print(f"[build] bomb score: {_n_bs} hitters · matchup grade: {_n_mg} ({_elite} ELITE)")
            except Exception as e:
                _hnote("bomb score", e); print(f"[build] bomb score skipped: {e}")
        except Exception as e:
            _hnote("pitcher edges", e); print(f"[build] pitcher edges skipped: {e}")

    # ---- TOP PLAYS v2: now that grade / bomb / ZONE / vuln all exist, rebuild the panel so it
    # ranks by CONVERGENCE rather than heat alone. A hitter with an ELITE grade and 5 HRs in the
    # arm's meatball zones outranks a hotter bat in a neutral spot. Falls back to the heat-only
    # list if the edge data didn't populate. ----
    try:
        _vuln_tier_by_pid = {}
        for _pe in (pitcher_edges or []):
            if _pe.get("vuln") and _pe.get("id") is not None:
                _vuln_tier_by_pid[_pe["id"]] = _pe["vuln"].get("tier")

        def _tp_rank(p):
            mg = p.get("matchup_grade") or {}
            gr = {"ELITE": 3, "STRONG": 2, "MOD": 1}.get(mg.get("grade"), 0)
            bs = (p.get("bomb_score") or {}).get("score") or 0
            zc = (((p.get("features") or {}).get("zone_profile") or {}).get("overlap") or {}).get("count", 0)
            return (gr, zc, bs, p.get("heat") or 0)

        _cands = [p for p in players
                  if not _thin(p)
                  and (p.get("heat") or 0) >= 55
                  and (p["opp_pitcher"].get("form") or {}).get("label") != "DEALING"]
        _cands.sort(key=_tp_rank, reverse=True)
        _tp2 = []
        for p in _cands:
            mg = p.get("matchup_grade") or {}
            bs = p.get("bomb_score") or {}
            zov = (((p.get("features") or {}).get("zone_profile") or {}).get("overlap") or {})
            d = (p.get("features") or {}).get("discipline") or {}
            # only surface plays that have SOME edge beyond heat
            if not (mg.get("grade") in ("ELITE", "STRONG") or zov.get("count", 0) >= 3
                    or (bs.get("score") or 0) >= 55 or (p.get("heat") or 0) >= 70):
                continue
            _tp2.append({
                "id": p["id"], "name": p["name"], "team": p["team"], "opp_team": p["opp_team"],
                "heat": p["heat"], "tier": p["tier"], "why": p.get("why"),
                "spot": p.get("lineup_spot"), "time": p.get("time"),
                "arm": p["opp_pitcher"].get("name"),
                "arm_form": (p["opp_pitcher"].get("form") or {}).get("label"),
                "arm_score": p["opp_pitcher"].get("hr_score"),
                # --- new signals ---
                "grade": mg.get("grade"), "aligned": mg.get("aligned"),
                "bomb": bs.get("score"), "bomb_tier": bs.get("tier"),
                "zone": zov.get("count", 0), "zone_cells": zov.get("cells") or [],
                "vuln_tier": _vuln_tier_by_pid.get(p["opp_pitcher"].get("id")),
                "elite": (p.get("elite") or {}).get("tier"),
                "eye": bool(d.get("eye")), "crosshair": bool(d.get("crosshair")),
                "warning": bool(d.get("warning")),
            })
            if len(_tp2) >= 12:
                break
        if _tp2:
            top_plays = _tp2
            print(f"[build] top plays v2: {len(top_plays)} (ranked by grade/zone/bomb)")
    except Exception as e:
        _hnote("top plays v2", e); print(f"[build] top plays v2 skipped: {e}")

    # ---- BULLPEN RANKINGS (renovated): rank every pen on the slate from most exploitable to
    # toughest on the stats that actually decide it — bullpen ERA, season HRs allowed (as a rate),
    # how worn down the pen is (recent workload + arms unavailable), and its platoon split. The
    # statcast HR-vulnerability read is kept as a smaller supporting term. Degrades gracefully to
    # the statcast/fatigue terms when the traditional season stats can't be fetched. ----
    bullpen_rankings = []
    try:
        slate_teams = set()
        for g in games:
            slate_teams.add(g["away"]); slate_teams.add(g["home"])

        # season ERA/HR/IP for every reliever who's pitched recently, per team
        pen_arm_ids, team_arm_ids = set(), {}
        for team in slate_teams:
            av = (pen_avail or {}).get(team) or (pen_avail or {}).get(_TEAM_ALIAS.get(team)) or {}
            ids = [int(x) for x in (av.get("available", []) + av.get("unavailable", [])) if x]
            team_arm_ids[team] = ids
            pen_arm_ids.update(ids)
        # Cache season stats per date so we don't re-fetch ~200 relievers every build.
        # Mirrors the hand2yr.json cache pattern: entries are stamped with today's date and
        # only arms missing/stale for today are pulled. Season stats barely move intraday.
        _PEN_STATS_PATH = os.path.join(os.path.dirname(OUT_PATH) or ".", "pen_season.json")
        try:
            with open(_PEN_STATS_PATH) as _f:
                _pen_cache = json.load(_f) or {}
        except Exception:
            _pen_cache = {}
        pen_stats = {}
        _need = []
        for _pid in pen_arm_ids:
            ent = _pen_cache.get(str(_pid))
            if ent and ent.get("asof") == date_str and ent.get("data") is not None:
                pen_stats[_pid] = ent["data"]
            else:
                _need.append(_pid)
        try:
            fetched = statsapi.get_pitcher_stats(sorted(_need)) if _need else {}
            for _pid, _st in fetched.items():
                pen_stats[int(_pid)] = _st
                _pen_cache[str(int(_pid))] = {"asof": date_str, "data": _st}
            print(f"[build] bullpen season stats: {len(pen_stats)}/{len(pen_arm_ids)} relievers "
                  f"({len(pen_stats) - len(fetched)} cached, {len(fetched)} fetched)")
            try:
                with open(_PEN_STATS_PATH, "w") as _f:
                    json.dump(_pen_cache, _f, separators=(",", ":"))
            except Exception as _e:
                print(f"[build] pen_season cache write skipped: {_e}")
        except Exception as e:
            _hnote("bullpen season stats", e); print(f"[build] bullpen season stats fetch skipped: {e}")

        def _clamp01(x): return max(0.0, min(1.0, x))

        for team in slate_teams:
            pen = _bullpen_for(team)
            avail = (pen_avail or {}).get(team) or (pen_avail or {}).get(_TEAM_ALIAS.get(team)) or {}
            vuln = compute.bullpen_vuln(pen) if pen else None
            hr_score = (vuln or {}).get("score")          # statcast HR-vulnerability (contact quality)
            fatigue = avail.get("fatigue")                # 0-100, higher = more gassed
            n_avail = len(avail.get("available", [])) if avail else None
            n_out = len(avail.get("unavailable", [])) if avail else None
            label = avail.get("label")
            platoon = (vuln or {}).get("platoon") or {}

            # aggregate traditional season stats across this pen's arms (IP-weighted ERA, HR rate)
            arms = [pen_stats[i] for i in team_arm_ids.get(team, []) if i in pen_stats]
            bp_era = season_hr = total_ip = hr9 = None
            if arms:
                ip_sum = sum((a.get("ip") or 0) for a in arms)
                era_ip = [(a.get("era"), a.get("ip")) for a in arms if a.get("era") is not None and a.get("ip")]
                if era_ip:
                    _w = sum(ip for _, ip in era_ip)
                    if _w > 0:
                        bp_era = round(sum(e * ip for e, ip in era_ip) / _w, 2)
                _hr = sum((a.get("hr") or 0) for a in arms)
                season_hr = _hr if _hr else None
                total_ip = round(ip_sum, 1) if ip_sum else None
                if season_hr is not None and total_ip and total_ip > 0:
                    hr9 = round(season_hr / (total_ip / 9.0), 2)

            # score components (higher = more exploitable)
            parts = {}
            rank_val = 0.0
            if bp_era is not None:
                era_pts = _clamp01((bp_era - 3.2) / 2.0) * 30       # 3.2 ERA -> 0, 5.2+ -> 30
                rank_val += era_pts; parts["era"] = round(era_pts, 1)
            if hr9 is not None:
                hr_pts = _clamp01((hr9 - 0.9) / 0.8) * 30           # 0.9 HR/9 -> 0, 1.7+ -> 30
                rank_val += hr_pts; parts["hr"] = round(hr_pts, 1)
            if fatigue is not None:
                wear_pts = _clamp01(fatigue / 100.0) * 16 + min(6, (n_out or 0) * 3)
                rank_val += wear_pts; parts["wear"] = round(wear_pts, 1)
            gap = platoon.get("gap")
            if gap is not None:
                plat_pts = _clamp01(gap / 25.0) * 10
                rank_val += plat_pts; parts["platoon"] = round(plat_pts, 1)
            if hr_score is not None:
                sc_pts = _clamp01((hr_score - 35) / 40.0) * 12       # supporting contact-quality term
                rank_val += sc_pts; parts["contact"] = round(sc_pts, 1)
            if not parts:
                continue

            # human-readable reasons
            reasons = []
            if bp_era is not None:
                reasons.append(f"{bp_era} pen ERA")
            if hr9 is not None and season_hr is not None:
                reasons.append(f"{hr9} HR/9 ({season_hr} HR)")
            if label == "GASSED":
                reasons.append(f"gassed — {n_out or '?'} down, {avail.get('pen_pitches_l1','?')} pitches yesterday")
            elif label == "WORN":
                reasons.append(f"worn — {n_out or 0} arm{'s' if (n_out or 0) != 1 else ''} down, {avail.get('pen_pitches_l2','?')} pitches over 2 days")
            if gap is not None and gap >= 8 and platoon.get("worse"):
                reasons.append(f"crushed by {'RHB' if platoon['worse'] == 'R' else 'LHB'} (gap {gap})")
            for fl in (vuln or {}).get("flags", [])[:1]:
                reasons.append(fl)
            if n_avail is not None and n_avail <= 4:
                reasons.append(f"only {n_avail} fresh arms")

            form = (vuln or {}).get("form") or {}
            bullpen_rankings.append({
                "team": team,
                "rank_val": round(rank_val, 1),
                "bp_era": bp_era,
                "hr9": hr9,
                "season_hr": season_hr,
                "total_ip": total_ip,
                "hr_score": hr_score,
                "fatigue": fatigue,
                "label": label,
                "n_available": n_avail,
                "n_unavailable": n_out,
                "platoon": ({"worse": platoon.get("worse"), "gap": platoon.get("gap"),
                             "R": platoon.get("R"), "L": platoon.get("L")} if platoon else None),
                "form": form.get("label") if isinstance(form, dict) else form,
                "parts": parts,
                "reasons": reasons or ["about average tonight"],
            })
        # sort worst (highest rank_val = most exploitable) first
        # Fold in the boxscore-based availability read. Two systems were measuring the same
        # thing from different sources and only one of them reached a screen: the ranking below
        # infers who pitched from the Statcast pitch-by-pitch frame, while environment's
        # reliever_status() reads actual pitch counts from MLB boxscores over the trailing three
        # days. The boxscore is the more direct evidence and covers arms the Statcast pull can
        # miss, so it is attached ALONGSIDE the existing numbers — where the two disagree that
        # is worth seeing, not worth hiding behind one silently overwriting the other.
        for _r in bullpen_rankings:
            _st = (pen_state or {}).get(_r.get("team"))
            if not _st:
                continue
            _r["live_pen"] = {
                "n_out": _st.get("n_out"),
                "n_tired": _st.get("n_tired"),
                "n_ok": _st.get("n_ok"),
                "k_pct": _st.get("k_pct"),
                "xwoba": _st.get("xwoba"),
                "source": "boxscore pitch logs (3d)",
            }
            # Flag a disagreement rather than resolving it silently.
            _old_out = _r.get("n_unavailable")
            if _old_out is not None and _st.get("n_out") is not None and abs(_old_out - _st["n_out"]) >= 2:
                _r["live_pen"]["disagrees"] = f"statcast says {_old_out} down, boxscores say {_st['n_out']}"
        _lp = sum(1 for _r in bullpen_rankings if _r.get("live_pen"))
        if _lp:
            print(f"[build] bullpen rankings: live availability attached to {_lp}/{len(bullpen_rankings)} pens")

        bullpen_rankings.sort(key=lambda x: -x["rank_val"])
        print(f"[build] bullpen rankings: {len(bullpen_rankings)} pens ranked (renovated: ERA/HR/wear/platoon)")
    except Exception as e:
        _hnote("bullpen rankings", e); print(f"[build] bullpen rankings skipped: {e}")

    # ---- PARK RANKS: best/worst HR park on tonight's slate, for the Weather view's ranking.
    # Prefer Ballpark Pal's per-game HR factor (the authoritative park+weather model); fall back
    # to the local park model so the ranking ALWAYS renders even when the BPP key isn't set.
    # Each entry carries the park name so the view can label it. ----
    park_ranks = []
    try:
        _pk_by_teams = {f'{g["away"]}@{g["home"]}': g.get("park") for g in games}
        if BPP.get("ok") and BPP.get("by_teams"):
            for v in sorted(BPP["by_teams"].values(), key=lambda x: -(x.get("hr_mult") or 0)):
                park_ranks.append({
                    "away": v.get("away"), "home": v.get("home"),
                    "park": _pk_by_teams.get(f'{v.get("away")}@{v.get("home")}'),
                    "hr_mult": v.get("hr_mult"), "hr_pct": v.get("hr_pct"),
                    "runs_mult": v.get("runs_mult"), "runs_pct": v.get("runs_pct"),
                    "hr_amount": v.get("hr_amount"), "game_time": v.get("game_time"),
                    "src": "bpp"})
        else:
            for g in games:
                _pk = g.get("park")
                _l = parks.park_factor(_pk, "L"); _r = parks.park_factor(_pk, "R")
                _m = round((_l + _r) / 2.0, 3)
                park_ranks.append({
                    "away": g.get("away"), "home": g.get("home"), "park": _pk,
                    "hr_mult": _m, "hr_pct": round((_m - 1.0) * 100, 1),
                    "runs_mult": None, "runs_pct": None, "hr_amount": None,
                    "game_time": g.get("time"), "src": "local"})
            park_ranks.sort(key=lambda x: -(x.get("hr_mult") or 0))
        print(f"[build] park ranks: {len(park_ranks)} ({'bpp' if BPP.get('ok') else 'local'})")
    except Exception as e:
        _hnote("park ranks", e); print(f"[build] park ranks skipped: {e}")

    # strip build-time helper fields that shouldn't ship in the JSON
    for _p in players:
        _p.pop("_ob", None)
        _p.pop("_season_metrics", None)

    # ---- TEAM TARGETS: which defenses concede tonight, ranked ----
    # The board ranks hitters; this ranks the other side, so a matchup can be picked first and
    # the hitters second. Scored from the same pieces already on the board — no new data source.
    team_targets = []
    try:
        _pen_by_team = {r.get("team"): r for r in (bullpen_rankings or [])}
        # pitcher_edges is keyed by the pitcher's own TEAM, and the games list carries no
        # starter id at all — so matching had nothing to match on and every card read "SP: TBD".
        # This is the same list the Vulnerable Arms tab renders, so the two screens now name the
        # same starter by construction rather than by coincidence.
        _arm_by_team = {}
        for _e in (pitcher_edges or []):
            if _e.get("team") and _e["team"] not in _arm_by_team:
                _arm_by_team[_e["team"]] = _e
                # xfip/fb_pct/gb_pct — read from the RAW sources (fg_pitch keyed by normalised name,
        # p_batted keyed by pitcher id), not from the local `pitcher_props` list: that list is
        # not assembled until later in this function, and reading it here would have been a
        # use-before-definition (pyflakes caught this before it shipped).

        def _lineup_hand_pct(team, opp_throws):
            """Share of tonight's confirmed lineup that bats RIGHT-handed, switch hitters
            resolved to the side opposite the starter's throwing hand — the same convention
            used elsewhere in this file for arsenal blending. Returns None (not 0.5) when no
            lineup has posted yet, so the platoon bump inside targets.py stays neutral rather
            than being computed off an empty/partial roster."""
            lineup = [x for x in players if x.get("team") == team and x.get("lineup_spot")]
            if len(lineup) < 7:          # a real lineup is 9; require most of it to trust the split
                return None
            r_n = 0
            for x in lineup:
                b = x.get("bats")
                if b == "S":
                    b = "L" if opp_throws == "R" else "R"
                if b == "R":
                    r_n += 1
            return round(r_n / len(lineup), 3)

        for _g in (games or []):
            for _bat, _def in (("away", "home"), ("home", "away")):
                _bt, _dt = _g.get(_bat), _g.get(_def)
                if not _bt or not _dt:
                    continue
                _edge = _arm_by_team.get(_dt) or {}
                _vs = (_edge.get("vuln") or {}).get("score")
                _form = ((_edge.get("profile") or {}).get("form") or {}).get("label") \
                    if isinstance(_edge.get("profile"), dict) else None
                _pen = _pen_by_team.get(_dt)
                _hr9 = (_pen or {}).get("hr9")
                # park + weather come off any hitter in this game
                _pk = _wx = _wo = None
                for _pl in players:
                    if _pl.get("game_pk") == _g.get("game_pk"):
                        _ph = _pl.get("park_hr") or {}
                        _pk = _ph.get("boost") if _pk is None else _pk
                        _wx = _ph.get("temp_f") if _wx is None else _wx
                        _wo = _ph.get("wind_out") if _wo is None else _wo
                        if _pk is not None and _wx is not None:
                            break
                # Starter enrichment inputs — every one optional; a TBD starter (_edge == {})
                # means _arm_id is None, every lookup below short-circuits to None, and
                # targets.py's own fallback logic (already built to handle this) takes over.
                _arm_id = _edge.get("id")
                _era = (_edge.get("season") or {}).get("era")
                _throws = _edge.get("throws")
                _fg_e = _fg_with_defaults(
                    fg_pitch.get(statcast_data._norm_name(_edge.get("name") or ""))
                    if _edge.get("name") else None)
                _xfip = _fg_e.get("xfip")
                _loc_plus = _fg_e.get("location_plus")
                _bat_e = p_batted.get(_arm_id) if _arm_id is not None else None
                _fb_pct = (_bat_e or {}).get("fb_pct")
                _gb_pct = (_bat_e or {}).get("gb_pct")
                _hand_splits = _edge.get("hand_splits")
                _lhp = _lineup_hand_pct(_bt, _throws)
                _sc = targets.target_score(starter_vuln=_vs, arm_form=_form, pen=_pen,
                                           team_hr9=_hr9, park_boost=_pk,
                                           temp_f=_wx, wind_out=_wo,
                                           xfip=_xfip, era=_era, fb_pct=_fb_pct, gb_pct=_gb_pct,
                                           hand_splits=_hand_splits,
                                           lineup_hand_pct=_lhp, location_plus=_loc_plus)
                if not _sc:
                    continue
                # the hitters you'd actually be targeting, best first
                _bats = sorted(
                    [x for x in players if x.get("team") == _bt and x.get("heat") is not None],
                    key=lambda x: -x["heat"])[:4]
                team_targets.append({
                    "bat_team": _bt, "def_team": _dt, "game_pk": _g.get("game_pk"),
                    "time": _g.get("time"), "park": _g.get("park"),
                    "sp_name": _edge.get("name"), "sp_throws": _edge.get("throws"),
                    "score": _sc["score"], "tier": targets.tier(_sc["score"]),
                    "components": _sc["components"], "weights": _sc["weights"],
                    "drivers": _sc["drivers"], "coverage": _sc["coverage"],
                    "pills": _sc.get("pills") or [], "flags": _sc.get("flags") or [],
                    "pen_label": (_pen or {}).get("label"),
                    "top_bats": [{"id": x.get("id"), "name": x.get("name"),
                                  "heat": x.get("heat"), "spot": x.get("lineup_spot")}
                                 for x in _bats[:3]],
                })
        team_targets.sort(key=lambda x: -x["score"])
        print(f"[build] team targets: {len(team_targets)} matchups ranked"
              + (f", top {team_targets[0]['bat_team']} vs {team_targets[0]['def_team']} "
                 f"({team_targets[0]['score']})" if team_targets else ""))
    except Exception as e:
        team_targets = []
        _hnote("team targets", e); print(f"[build] team targets skipped: {e}")

    board = {
        "generated_at": now.isoformat(timespec="seconds"),
        "slate_date": date_str,
        "model_version": compute.MODEL_VERSION,
        "games": [{
            "game_pk": g["game_pk"], "away": g["away"], "home": g["home"],
            "park": g["park"], "time": g["time"],
        } for g in games],
        "lineups_pending": [g["game_pk"] for g in games if g["game_pk"] not in slate["lineups"]],
        "projected_games": [
            {"game_pk": g["game_pk"], "away": g["away"], "home": g["home"]}
            for g in games if g["game_pk"] in proj_game_pks
        ],
        "recent_window": {
            "days": 14,
            # v2: the window is now the last 14 GAME-days, not calendar days. Across the
            # All-Star break a calendar window silently held ~10 game-days and compressed
            # every heat score. Stamped so the tracker can segment pre/post-change days.
            "basis": "game_days",
            "v": 2,
            "start": str(statcast_data.game_day_cutoff(df, date_str, 14).date()),
            "end": date_str,
        },
        "players": players,
        "team_targets": team_targets,
        "converge_lift": globals().get("_CONVERGE_LIFT"),   # lift convergence is CURRENTLY using
        "arsenals": arsenals,               # starter pitch usage % split by batter handedness
        "arm_tables": arm_tables,           # per-arm full BvP stat line by pitch type & batter hand
        "pitch_hist": pitch_hist,           # per-arm pitch usage per start (arsenal drift)
        "team_ks": team_ks,                 # lineup strikeout rate + league rank by context
        "pitcher_edges": pitcher_edges,     # Edges tab: per-arm zone heatmap + ranked batters
        "bullpen_rankings": bullpen_rankings,   # slate-wide pen ranking, worst to best
        "park_source": ("ballparkpal" if BPP.get("ok") else "local"),
        "park_ranks": park_ranks,           # best/worst HR park tonight (BPP live, local fallback)
        "grand_slam": board_gs,             # top GS-jackpot candidates (traffic x punish)
        "grand_slam_jackpot": board_gs_jackpot,  # Primary / Top-of-Order Mash / Mega-Leverage Deep
        "grand_slam_board": board_gs_board,      # full per-game ranked board, both teams
        "top_plays": top_plays,
        "wx": wx_list,
        "fences": fences,
        "briefing": briefing,
        "build_health": {
            "df_rows": int(len(df)) if df is not None else 0,
            "players": len(players), "arms_ok": True,
            "labeled": sum(1 for p in players if p.get("hit_label")),
            "b2b": sum(1 for p in players if p.get("hr_last_game")),
            "openers": sum(1 for p in players if (p.get("opp_pitcher") or {}).get("opener")),
            "stacks": len(stacks), "wx": len(wx_list),
            "issues": BUILD_HEALTH,
        },
        "arms": sorted([
            {
                "id": pid,
                "name": slate["pitchers"].get(pid, {}).get("name", str(pid)),
                "throws": hands.get(pid, {}).get("throws", ""),
                "team": next((g["home"] if g["home_pitcher_id"] == pid else g["away"]
                              for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "opp": next((g["away"] if g["home_pitcher_id"] == pid else g["home"]
                             for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "park": next((g["park"] for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "time": next((g["time"] for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "hr_score": phr.get("score"),
                "recent_score": phr.get("recent_score"),
                "season_score": phr.get("season_score"),
                "delta": phr.get("delta"),
                "form": phr.get("form"),
                "flags": phr.get("flags", []),
                "opener": bool(
                    (start_lens.get(pid) and start_lens[pid]["starts"] >= 2
                     and start_lens[pid]["med_len"] <= 2.0)
                    or (start_lens.get(pid) is None and p_apps.get(pid, 0) >= 5)),
                "fb_pct": (p_batted.get(pid) or {}).get("fb_pct"),
                "ld_pct": (p_batted.get(pid) or {}).get("ld_pct"),
                "gb_pct": (p_batted.get(pid) or {}).get("gb_pct"),
                "start_len": (round(start_lens[pid]["med_len"], 1)
                              if start_lens.get(pid) else None),
                # heaviest 2yr HR-by-hand side, raw numbers for the strip
                "hand_hr": (lambda ty: (max(
                    ({"side": h, "hr": s["hr"], "pa": s["pa"]}
                     for h, s in (ty or {}).items() if s and s.get("pa", 0) >= 100),
                    key=lambda x: x["hr"] / max(1, x["pa"]), default=None)))(
                        (hand2yr.get(pid) or {}).get("two_yr")),
                # same heaviest-side but for this season only — so users can compare
                # against reference tools (PropFinder etc) that default to season-only
                "hand_hr_ytd": (lambda ty: (max(
                    ({"side": h, "hr": s["hr"], "pa": s["pa"]}
                     for h, s in (ty or {}).items() if s and s.get("pa", 0) >= 40),
                    key=lambda x: x["hr"] / max(1, x["pa"]), default=None)))(
                        (hand2yr.get(pid) or {}).get("this_yr")),
                # full raw splits (both R and L, both windows) so the expanded arm view
                # can show a proper table without more data fetches
                "hand_hr_full": hand2yr.get(pid),
                "badges": compute.pitcher_badges(
                    recent=pitch_profiles.get(pid, {}).get("recent", {}),
                    score=phr.get("score"), recent_score=phr.get("recent_score"),
                    season_score=phr.get("season_score"),
                    two_yr=(hand2yr.get(pid) or {}).get("two_yr")),
                "platoon": compute.platoon_note(pitch_profiles.get(pid, {}).get("splits")),
            }
            for pid, phr in pitcher_hr.items()
        ], key=lambda a: (a["hr_score"] is not None, a["hr_score"] or 0), reverse=True),
    }

    # persist the 2-year HR-by-hand cache so future builds reuse it (avoids hourly re-pulls)
    try:
        with open(_HAND2YR_PATH, "w") as _f:
            json.dump(hand2yr_cache, _f)
    except Exception as _e:
        print(f"[build] hand2yr cache write failed (non-fatal): {_e}")

    # ---- Pitcher K props (Ks tab in Other Props) ----
    # MUST live inside build() so pitch_profiles is in scope. Each starter gets a
    # k_heat computed from their 14-day K stuff blended toward season, weighted
    # against opposing lineup K vulnerability. Openers get a hard downgrade.
    pitcher_props = []
    try:
        opp_batters = {}   # pitcher_id -> list of hitter k_pct values (opposing lineup)
        seen_pitcher = {}  # pitcher_id -> game/team metadata for the pitcher himself
        for hp in board["players"]:
            op = hp.get("opp_pitcher") or {}
            pid = op.get("id")
            if not pid:
                continue
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            k_pct = ((hp.get("windows") or {}).get("L14d") or {}).get("k_pct")
            if k_pct is not None:
                opp_batters.setdefault(pid, []).append(k_pct)
            if pid not in seen_pitcher:
                seen_pitcher[pid] = {
                    "id": pid,
                    "name": op.get("name") or "",
                    "throws": op.get("throws"),
                    "game_pk": hp.get("game_pk"),
                    "opp_team": hp.get("team"),
                    "team": hp.get("opp_team"),
                    "time": hp.get("time"),
                    "park": hp.get("park"),
                    "opener": bool(op.get("opener")),
                    "form": op.get("form"),
                    "hr_score": op.get("hr_score"),
                    "recent_score": op.get("recent_score"),
                    "season_score": op.get("season_score"),
                }
        for pid, meta in seen_pitcher.items():
            pprof = pitch_profiles.get(pid) or {}
            opp_ks = opp_batters.get(pid) or []
            opp_lineup_k = round(sum(opp_ks) / len(opp_ks), 1) if opp_ks else None
            k_sc, k_br = props.pitcher_k_heat(pprof, opp_lineup_k, opener=meta["opener"])
            if k_sc is None:
                continue
            # Estimated K total for the pitcher today. Two ingredients:
            #  1. Effective K% — blend of pitcher's own K% and the opposing lineup's K%
            #     (both matter; a ~28% K arm vs a 22% K lineup lands around 25%).
            #  2. Expected batters faced: ~24 for a healthy starter, ~4 for a listed
            #     opener. When start_len data is available, we lean on that; otherwise
            #     use median start_len from _pitcher_metrics.
            est_ks = None
            # K rate: blend recent form into the season baseline IN PROPORTION TO ITS SAMPLE, then
            # regress a thin overall sample toward league. Using the raw 2-week number was the bug
            # that put small-sample arms (a 45% K rate over ~40 PA) at the top of the list — they
            # regress hard, which is why the "best" K plays kept losing.
            _rec = pprof.get("recent") or {}
            _szn = pprof.get("season") or {}
            k_rec, k_szn = _rec.get("k_pct_allowed"), _szn.get("k_pct_allowed")
            pa_rec, pa_szn = float(_rec.get("pa") or 0), float(_szn.get("pa") or 0)
            LG_K_RATE = 22.0
            if k_rec is not None and k_szn is not None:
                w = min(1.0, pa_rec / 120.0)              # ~120 PA to fully trust the recent window
                pitcher_k_pct = w * k_rec + (1 - w) * k_szn
            else:
                pitcher_k_pct = k_szn if k_szn is not None else k_rec
            if pitcher_k_pct is not None and pa_szn > 0:
                w2 = min(1.0, pa_szn / 200.0)             # thin season -> pull toward league
                pitcher_k_pct = w2 * pitcher_k_pct + (1 - w2) * LG_K_RATE
            if pitcher_k_pct is not None:
                if opp_lineup_k is not None:
                    # odds-ratio (log5) matchup instead of a linear blend: combine the pitcher's
                    # K rate and the opposing lineup's K rate against the league baseline so the
                    # extremes compound correctly — a high-K arm vs a high-K lineup lands ABOVE
                    # both (not between them), and an ace vs an elite-contact lineup lands below.
                    # A linear blend flattens exactly the matchups where the K edge lives.
                    LG_K = 22.0                                  # league strikeout rate, ~2024-25
                    P = min(0.60, max(0.03, pitcher_k_pct / 100.0))
                    L = min(0.60, max(0.03, opp_lineup_k / 100.0))
                    G = LG_K / 100.0
                    odds = (P / (1 - P)) * (L / (1 - L)) / (G / (1 - G))
                    eff_k = 100.0 * odds / (1.0 + odds)
                    eff_k = max(10.0, min(42.0, eff_k))          # keep it in a sane band
                else:
                    eff_k = pitcher_k_pct
                bf = 4 if meta["opener"] else 22
                # if we know his typical start length, refine BF (~4.3 BF per IP). The modern
                # starter averages ~5.1 IP (~22 BF); the old default of 24 quietly added most of
                # half a strikeout to every projection.
                sl = start_lens.get(pid, {}).get("med_len") if start_lens else None
                if sl and not meta["opener"]:
                    bf = max(6, min(27, sl * 4.3))
                est_ks = round(eff_k / 100.0 * bf, 1)
            pitcher_props.append({
                **meta,
                "k_heat": k_sc,
                "k_signals": k_br.get("signals"),
                "recent_weight": k_br.get("pitcher_recent_weight"),
                "opp_lineup_k_pct": opp_lineup_k,
                "opp_lineup_n": len(opp_ks),
                "k_pct": pitcher_k_pct,
                "swstr_pct": (pprof.get("recent") or {}).get("swstr_pct_allowed") or
                             (pprof.get("season") or {}).get("swstr_pct_allowed"),
                # kengine projection: CSW true-talent, dynamic xBF, arsenal-depth TTOP, and
                # the exact Poisson-binomial distribution. Emitted alongside the legacy
                # `est_ks` rather than replacing it, so the backtest can grade both on the same
                # historical days and the swap is made on evidence rather than assumption.
                "kengine": _kengine_for(pid, meta),
                "fg": _fg_with_defaults(
                    fg_pitch.get(statcast_data._norm_name(meta.get("name") or ""))),
                "first_inn": (first_inn.get(int(pid)) if first_inn else None),
                "est_ks": est_ks,
                # the OVER target the model actually likes = the half-line just BELOW the estimate
                "est_line_over": (float(np.ceil(est_ks - 0.5) - 0.5) if est_ks is not None else None),
            })
        pitcher_props.sort(key=lambda x: -(x["k_heat"] or 0))

        # ---- STRIKEOUT CONVERGENCE ----
        # Runs HERE, after pitcher_props exists — it was previously placed hundreds of lines
        # earlier where the list was still undefined, so the Ks tab silently had nothing.
        # Measured: only the K-heat bands the backtest actually graded. Provisional: the reads
        # that make sense for a strikeout prop but haven't been graded yet — the arm's own whiff
        # rate, and three things about the LINEUP he's facing (how much it whiffs, how little
        # damage it does in his zones, how poorly the individual matchups grade).
        try:
            _K_BANDS_P = globals().get("_CONVERGE_LIFT", {}).get("k") or [(70, 28), (55, 10)]
            _by_game_bat = {}
            for _pl in board.get("players", []):
                _by_game_bat.setdefault(_pl.get("game_pk"), []).append(_pl)
            for _a in pitcher_props:
                _meas, _prov = [], []
                _kh = _a.get("k_heat")
                if _kh is not None:
                    for _cut, _lift in _K_BANDS_P:
                        if _kh >= _cut:
                            _meas.append({"k": "heat", "lab": f"K-heat {int(_kh)}", "lift": _lift})
                            break
                _fam = {}
                def _addk(fam, lab):
                    _fam.setdefault(fam, []).append(lab)
                # -- the arm himself --
                _sw = _a.get("swstr_pct")
                if _sw is not None and _sw >= 12.5:
                    _addk("stuff", f"SwStr {_sw}%")
                _kp = _a.get("k_pct")
                if _kp is not None and _kp >= 26:
                    _addk("stuff", f"K% {round(_kp,1)}")
                # -- the lineup he's facing --
                _ol = _a.get("opp_lineup_k_pct")
                if _ol is not None and _ol >= 23:
                    _addk("lineup", f"Lineup K {round(_ol,1)}%")
                try:
                    _tk = (team_ks.get(_a.get("opp_team")) or {})
                    _season = _tk.get("season") or {}
                    if _season.get("rank") and _season["rank"] >= 20:
                        _addk("lineup", f"K-rank {_season['rank']}/30")
                    _hand_key = "vs_rhp" if (_a.get("throws") == "R") else "vs_lhp"
                    _hs = _tk.get(_hand_key) or {}
                    if _hs.get("rank") and _hs["rank"] >= 20:
                        _addk("lineup", f"{'vs RHP' if _a.get('throws')=='R' else 'vs LHP'} rank {_hs['rank']}/30")
                except Exception:
                    pass
                # -- do the individual matchups grade badly for the hitters? --
                # A lineup that can't reach his zones, and grades poorly across the board, is a
                # different (and better) K spot than one that just whiffs a lot in aggregate.
                try:
                    _opp = [x for x in _by_game_bat.get(_a.get("game_pk"), [])
                            if x.get("team") == _a.get("opp_team") and x.get("lineup_spot")]
                    if len(_opp) >= 5:
                        _zs, _gr = [], 0
                        for _x in _opp:
                            _ze2 = ((_x.get("features") or {}).get("zone_profile") or {}).get("zone_edge") or {}
                            if _ze2.get("edge_score") is not None:
                                _zs.append(float(_ze2["edge_score"]))
                            _g = str(((_x.get("matchup_grade") or {}).get("grade") or "")).upper()
                            if _g in ("ELITE", "STRONG"):
                                _gr += 1
                        if _zs and (sum(_zs) / len(_zs)) <= 48:
                            _addk("zones", f"Lineup zone edge {round(sum(_zs)/len(_zs),1)} (weak)")
                        if _gr == 0:
                            _addk("matchups", "No hitter grades strong")
                        elif _gr <= 1 and len(_opp) >= 7:
                            _addk("matchups", f"Only {_gr} strong matchup")
                        _cold = sum(1 for _x in _opp if (_x.get("heat") or 0) < 40)
                        if _cold >= max(4, int(0.6 * len(_opp))):
                            _addk("matchups", f"{_cold}/{len(_opp)} hitters cold")
                except Exception:
                    pass
                _KFAM = {"stuff": "Arm's stuff", "lineup": "Lineup whiffs",
                         "zones": "Lineup can't reach his zones", "matchups": "Matchups grade poorly"}
                _prov = [{"k": _k, "lab": _KFAM.get(_k, _k), "detail": _v, "n": len(_v),
                          "kind": ("context" if _k in ("lineup", "zones", "matchups") else "hitter")}
                         for _k, _v in _fam.items()]
                _a["converge"] = {
                    "measured": _meas, "provisional": _prov,
                    "n_measured": len(_meas), "n_provisional": len(_prov),
                    "n_hitter": sum(1 for s in _prov if s["kind"] != "context"),
                    "n_context": sum(1 for s in _prov if s["kind"] == "context"),
                    "lift": sum(m["lift"] for m in _meas),
                }
            print(f"[build] K convergence attached to {len(pitcher_props)} arms")
        except Exception as _e:
            print(f"[build] K convergence skipped (non-fatal): {_e}")

        board["pitcher_props"] = pitcher_props
        print(f"[build] pitcher K props: {len(pitcher_props)} arms ranked "
              f"(out of {len(seen_pitcher)} distinct pitchers seen)")
    except Exception as e:
        board["pitcher_props"] = []
        _hnote("pitcher K props", e); print(f"[build] pitcher K props skipped: {e}")

    # ---- Heat history: 10-day heat trajectory per player (sparkline data) ----
    # Walks the most recent snapshots and attaches heat_history:[...] per player.
    # Cheap because snapshots are already on disk and we only need the (id, heat)
    # tuples. Displayed as an inline sparkline on the expanded card.
    try:
        snap_dir = os.path.join(os.path.dirname(OUT_PATH) or ".", "snapshots")
        import glob
        recent_snaps = sorted(glob.glob(os.path.join(snap_dir, "20*.json")))[-14:]
        heat_trail = {}   # {id: [(date, heat), ...]}
        for spath in recent_snaps:
            try:
                with open(spath) as sf:
                    sd = json.load(sf)
                sdate = sd.get("date")
                for pl in sd.get("players", []):
                    pid_pl = pl.get("id")
                    heat_val = pl.get("heat")
                    if pid_pl is None or heat_val is None:
                        continue
                    heat_trail.setdefault(int(pid_pl), []).append((sdate, heat_val))
            except Exception:
                continue
        attached = 0
        for p in board["players"]:
            trail = heat_trail.get(p["id"])
            if trail:
                # keep only most recent 10, chronological
                trail = sorted(trail, key=lambda x: x[0])[-10:]
                p["heat_history"] = [round(h, 1) for _, h in trail]
                attached += 1
        print(f"[build] heat_history attached to {attached} of {len(board['players'])} hitters "
              f"(from {len(recent_snaps)} snapshots)")
    except Exception as e:
        _hnote("heat_history", e); print(f"[build] heat_history skipped: {e}")

    # ---- Game projections: expected runs, moneyline, total, F5 ----
    # Uses the run-expectancy engine (etl/runs.py). Completely separate from the HR
    # heat model and from props. INFORMATIONAL until backtested — the moneyline is
    # the sharpest market in baseball and this model has no defense, no true park
    # run factor, and no bullpen-availability data.
    # Season-long team quality (run differential, not W-L) as a prior for the win model.
    try:
        _team_rec = statsapi.get_team_records()
        if _team_rec:
            print(f"[build] team records: {len(_team_rec)} teams (Pythagorean prior active)")
    except Exception as _e:
        print(f"[build] team records skipped (non-fatal): {_e}"); _team_rec = {}

    game_projections = []
    try:
        from etl import runs as RUNS
        # group hitters by game + side
        by_game = {}
        for p in board["players"]:
            if p.get("lineup_status") == "out":
                continue
            gpk = p.get("game_pk")
            if not gpk:
                continue
            g = by_game.setdefault(gpk, {"home": [], "away": [], "meta": None})
            gm = next((x for x in games if x["game_pk"] == gpk), None)
            if gm and g["meta"] is None:
                g["meta"] = gm
            side = "home" if p.get("team") == (gm or {}).get("home") else "away"
            g[side].append(p)

        for gpk, g in by_game.items():
            gm = g["meta"]
            if not gm:
                continue
            hp_id, ap_id = gm.get("home_pitcher_id"), gm.get("away_pitcher_id")
            if not hp_id or not ap_id:
                continue
            # lineups in batting order (fall back to heat order if spots missing)
            def _order(lst):
                sp = [x for x in lst if x.get("lineup_spot")]
                if len(sp) >= 8:
                    return sorted(sp, key=lambda x: x["lineup_spot"])[:9]
                return sorted(lst, key=lambda x: -(x.get("heat") or 0))[:9]
            home_order = _order(g["home"])
            away_order = _order(g["away"])
            home_l = [(x.get("windows") or {}).get("L14d") or {} for x in home_order]
            away_l = [(x.get("windows") or {}).get("L14d") or {} for x in away_order]
            home_hands = [x.get("bats") for x in home_order]
            away_hands = [x.get("bats") for x in away_order]
            if len(home_l) < 6 or len(away_l) < 6:
                continue
            home_sp = pitch_profiles.get(hp_id) or {}
            away_sp = pitch_profiles.get(ap_id) or {}
            # prefer the AVAILABLE-arms pen — that's who actually pitches tonight
            home_pen = (pens_avail.get(gm.get("home"))
                        or _bullpen_for(gm.get("home")) or {})
            away_pen = (pens_avail.get(gm.get("away"))
                        or _bullpen_for(gm.get("away")) or {})
            # expected batters faced per starter (~4.3 BF per inning)
            def _bf(pid):
                sl = (start_lens.get(pid) or {}).get("med_len")
                return max(6.0, min(30.0, sl * 4.3)) if sl else 24.0
            # Park RUN factor. Prefer BallparkPal's TRUE runs multiplier (it models runs separately
            # from HRs and already bakes in today's weather). Only if that's missing do we fall back
            # to the damped HR-boost proxy — and in THAT case add our own temperature adjustment
            # (warmer air = more offense), so we don't double-count weather when BPP already has it.
            pf = None; temp_f = None; _elev = None; _hum = None
            for x in g["home"] + g["away"]:
                ph = x.get("park_hr") or {}
                if pf is None and ph.get("boost") is not None: pf = ph["boost"]
                if temp_f is None and ph.get("temp_f") is not None: temp_f = ph["temp_f"]
                if _elev is None and ph.get("elevation_ft") is not None: _elev = ph["elevation_ft"]
                if _hum is None and ph.get("humidity_pct") is not None: _hum = ph["humidity_pct"]
            hr_proxy = 1.0 + (pf / 100.0) * 0.45 if pf is not None else 1.0
            try:
                rm, _rm_src = ballparkpal.resolve_runs_mult(
                    BPP, away=gm.get("away"), home=gm.get("home"),
                    game_id=gm.get("game_pk"), fallback=None)
            except Exception:
                rm = None
            if rm is not None:
                park_mult = rm                       # BPP runs factor (weather already included)
                park_src = "bpp_runs"
            else:
                # Air density from the Nathan carry model, not a linear temperature nudge.
                #
                # The old line applied a flat 0.3% per degree and knew nothing about ELEVATION,
                # which is the single biggest environmental factor in the sport: Coors sits a
                # mile up and the thin air adds roughly 6-7% of carry regardless of temperature.
                # A 400-ft fly there travels about 427 ft. Approximating that with a temperature
                # slope understates Denver and overstates warm sea-level parks.
                #
                # carry_multiplier() was written, unit-tested against the published Coors effect,
                # and then never called — the ETL kept using the linear stand-in. Only reached
                # here in the LOCAL fallback: when BallparkPal returns a runs factor it already
                # has weather folded in, and applying this on top would double-count it.
                wx = 1.0
                try:
                    wx = env_mod.carry_multiplier(
                        temp_f if temp_f is not None else 70.0,
                        elevation_ft=_elev if _elev is not None else 0.0,
                        humidity_pct=_hum if _hum is not None else 50.0)
                except Exception:
                    if temp_f is not None:
                        wx = max(0.94, min(1.08, 1.0 + (temp_f - 72.0) * 0.003))
                park_mult = hr_proxy * wx            # local HR-proxy + air-density carry
                park_src = "local"
            park_mult = max(0.82, min(1.28, park_mult))

            def _pen_fatigue(team):
                """0-1 how gassed this bullpen is, from the live pen tracker."""
                try:
                    for _pl in board.get("players", []):
                        if _pl.get("opp_team") == team:
                            _pl2 = _pl.get("opp_pen_live") or {}
                            _f = _pl2.get("fatigue")
                            if _f is not None:
                                return max(0.0, min(1.0, float(_f)))
                except Exception:
                    pass
                return None

            _dalias = {"AZ":"ARI","ARI":"AZ","CWS":"CHW","CHW":"CWS","WSH":"WSN","WSN":"WSH","SD":"SDP","SDP":"SD","SF":"SFG","SFG":"SF","TB":"TBR","TBR":"TB","KC":"KCR","KCR":"KC"}
            def _def_for(ab):
                if not ab: return 0.0
                v = team_def.get(ab)
                if v is None: v = team_def.get(_dalias.get(ab))
                return float(v or 0.0)
            proj = RUNS.project_game(
                home_l, away_l, home_sp, away_sp, home_pen, away_pen,
                home_bf=_bf(hp_id), away_bf=_bf(ap_id), park_mult=park_mult,
                home_hands=home_hands, away_hands=away_hands,
                home_def=_def_for(gm.get("home")), away_def=_def_for(gm.get("away")),
                home_pyth=(_team_rec.get(gm.get("home")) or {}).get("pyth"),
                away_pyth=(_team_rec.get(gm.get("away")) or {}).get("pyth"),
                home_fg=fg_pitch.get(statcast_data._norm_name(gm.get("home_sp") or "")),
                away_fg=fg_pitch.get(statcast_data._norm_name(gm.get("away_sp") or "")),
                home_first_inn=(first_inn.get(hp_id) if first_inn else None),
                away_first_inn=(first_inn.get(ap_id) if first_inn else None),
                home_pen_fatigue=_pen_fatigue(gm.get("home")),
                away_pen_fatigue=_pen_fatigue(gm.get("away")))
            if not proj:
                continue
            game_projections.append({
                "game_pk": gpk,
                "home": gm.get("home"), "away": gm.get("away"),
                "park": gm.get("park"), "time": gm.get("time"),
                "home_sp": (slate["pitchers"].get(hp_id) or {}).get("name", ""),
                "away_sp": (slate["pitchers"].get(ap_id) or {}).get("name", ""),
                "home_sp_id": hp_id, "away_sp_id": ap_id,
                "lineups_confirmed": all(x.get("lineup_spot") for x in g["home"][:9])
                                     and all(x.get("lineup_spot") for x in g["away"][:9]),
                "park_run_src": park_src,   # 'bpp_runs' (true runs factor) | 'local' (HR proxy + wx)
                **proj,
            })
        game_projections.sort(key=lambda x: -(x.get("total") or 0))

        # ---- MONEYLINE CONVERGENCE ----
        # Deliberately more conservative than the hitter version. The backtest DOES grade the run
        # model (Brier 0.2473 vs a 0.2497 baseline, calibration tight), so the win-probability
        # edge counts as measured. Everything else stays provisional, and the honest framing
        # matters here: the moneyline is the sharpest market in baseball, so agreement between
        # our own sub-models is NOT the same as an edge over the price.
        try:
            _ml_lift = None
            try:
                _r = (globals().get("_BT_RUNS") or {})
                if not _r:
                    _btp2 = os.path.join(os.path.dirname(__file__), "..", "docs", "backtest.json")
                    if os.path.exists(_btp2):
                        with open(_btp2) as _f2:
                            _r = (json.load(_f2) or {}).get("runs") or {}
                        globals()["_BT_RUNS"] = _r
                _br, _bb = _r.get("brier"), _r.get("baseline_brier")
                if _br and _bb and _br < _bb:
                    _ml_lift = int(round(100 * (_bb - _br) / _bb))   # % better than baseline
            except Exception:
                _ml_lift = None
            for _g in game_projections:
                for _side in ("home", "away"):
                    _wp = _g.get(f"{_side}_wp")
                    if _wp is None:
                        continue
                    _meas, _fam = [], {}
                    def _addm(fam, lab):
                        _fam.setdefault(fam, []).append(lab)
                    if _wp >= 0.58 and _ml_lift:
                        _meas.append({"k": "runmodel", "lab": f"Model {round(100*_wp)}%", "lift": _ml_lift})
                    _bd = _g.get(f"{_side}_breakdown") or {}
                    _obd = _g.get(f"{'away' if _side=='home' else 'home'}_breakdown") or {}
                    # starter edge
                    _sx, _ox = _bd.get("sp_xwoba_allowed"), _obd.get("sp_xwoba_allowed")
                    if _sx is not None and _ox is not None and (_ox - _sx) >= 0.020:
                        _addm("starter", f"SP edge {round(_ox-_sx,3)} xwOBA")
                    # bullpen edge
                    _px, _opx = _bd.get("pen_xwoba_allowed"), _obd.get("pen_xwoba_allowed")
                    if _px is not None and _opx is not None and (_opx - _px) >= 0.020:
                        _addm("bullpen", f"Pen edge {round(_opx-_px,3)}")
                    # defense
                    _df = _obd.get("opp_def")
                    if _df is not None and _df >= 0.2:
                        _addm("defense", f"Defense +{round(_df,2)} runs saved")
                    # platoon
                    if (_bd.get("platoon_spots") or 0) >= 5:
                        _addm("platoon", f"{_bd['platoon_spots']} platoon spots")
                    # run margin
                    _rr, _orr = _g.get(f"{_side}_runs"), _g.get(f"{'away' if _side=='home' else 'home'}_runs")
                    if _rr is not None and _orr is not None and (_rr - _orr) >= 0.6:
                        _addm("runs", f"+{round(_rr-_orr,2)} run edge")
                    _MLFAM = {"starter": "Starter edge", "bullpen": "Bullpen edge",
                              "defense": "Defense", "platoon": "Platoon", "runs": "Run margin"}
                    _prov = [{"k": _k, "lab": _MLFAM.get(_k, _k), "detail": _v, "n": len(_v),
                              "kind": "hitter"} for _k, _v in _fam.items()]
                    _g[f"{_side}_converge"] = {
                        "measured": _meas, "provisional": _prov,
                        "n_measured": len(_meas), "n_provisional": len(_prov),
                        "n_hitter": len(_prov), "n_context": 0,
                        "lift": sum(m["lift"] for m in _meas),
                    }
            print(f"[build] ML convergence attached to {len(game_projections)} games")
        except Exception as _e:
            print(f"[build] ML convergence skipped (non-fatal): {_e}")

        board["game_projections"] = game_projections
        print(f"[build] game projections: {len(game_projections)} games modeled")
    except Exception as e:
        board["game_projections"] = []
        _hnote("game projections", e); print(f"[build] game projections skipped: {e}")

    # ---- CROSS-GAME 3-LEG HR PARLAY OPTIMIZER ----
    try:
        _bt_calib = {}
        try:
            with open("docs/backtest.json") as _f:
                _bt_calib = (json.load(_f).get("calib")) or {}
        except Exception:
            pass
        _odds_prices = {}
        try:
            with open("docs/odds.json") as _f:
                _odds_prices = {statcast_data._norm_name(k): v
                               for k, v in (json.load(_f).get("prices") or {}).items()}
        except Exception:
            pass
        _hrpo_games = []
        for _g in (games or []):
            _hrpo_games.append({"game_pk": _g.get("game_pk"),
                               "temp_f": None, "wind_out": None, "elevation_ft": None})
        # pull temp/wind straight off any player in that game -- same source Target Teams uses
        for _pl in players:
            for _hg in _hrpo_games:
                if _hg["game_pk"] == _pl.get("game_pk") and _hg["temp_f"] is None:
                    _ph = _pl.get("park_hr") or {}
                    _hg["temp_f"] = _ph.get("temp_f")
                    _hg["wind_out"] = _ph.get("wind_out")
        board["cross_game_parlays"] = build_cross_game_hr_parlays(
            players, _bt_calib, _odds_prices, _hrpo_games,
            pitcher_props=pitcher_props, bullpen_rankings=bullpen_rankings)
        _cgp = board["cross_game_parlays"]
        print(f"[build] cross-game HR parlays: {_cgp['candidates_scored']} candidates scored, "
              f"anchor {'complete' if not _cgp['anchor']['incomplete'] else 'incomplete'}, "
              f"overlooked {'complete' if not _cgp['overlooked']['incomplete'] else 'incomplete'}")
    except Exception as e:
        board["cross_game_parlays"] = None
        _hnote("cross-game parlays", e); print(f"[build] cross-game parlays skipped: {e}")

    try:
        board["long_ball_jackpot"] = build_long_ball_jackpot(players, lb_evbarrels)
        _lb = board["long_ball_jackpot"]
        print(f"[build] long ball jackpot: {_lb['candidates_scored']} candidates scored, "
              f"{len(_lb['picks'])}/3 picks selected, MLB p95 max EV: {_lb['mlb_p95_max_ev']}")
    except Exception as e:
        board["long_ball_jackpot"] = {"picks": [], "candidates_scored": 0, "notes": []}
        _hnote("long ball jackpot", e); print(f"[build] long ball jackpot skipped: {e}")

    return board


def main():
    try:
        board = build()
    except statcast_data.StatcastUnavailable as e:
        # Savant was unavailable/throttled. Leave the existing board.json in place
        # (no write) so the page keeps serving the last good data instead of zeros.
        print(f"[build] SKIPPED write — Statcast unavailable ({e}). Last good board preserved.")
        return
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        # compact: the client parses/holds this in mobile memory, so drop pretty-print whitespace
        json.dump(board, f, separators=(",", ":"), default=str)

    # slate-level SMASH selection, mirrored from the UI's convergence scorer so the
    # grader can measure the flag's real conversion rate (the whole point of the flag).
    # Uses standard heat (the UI's default view).
    def _smash_score(p):
        H = p.get("heat") or 0
        s = max(0.0, min(3.0, (H - 45) / 10.0)); r = 0
        ks = {b["k"] for b in (p.get("badges") or [])}
        tr = p.get("trend") or {}
        if "lock" in ks: s += 1.5; r += 1
        elif "hot" in ks: s += 0.75; r += 1
        if "due" in ks: s += 1.0; r += 1
        if tr.get("dir") == "up": s += 1.0; r += 1
        pb = (p.get("park_hr") or {}).get("boost") or 0
        if pb >= 12: s += 1.5; r += 1
        elif pb >= 6: s += 0.75; r += 1
        opnr = bool((p.get("opp_pitcher") or {}).get("opener"))
        if "hrsp" in ks: s += (0.75 if opnr else 1.5); r += 1
        if "hrbp" in ks: s += (1.5 if opnr else 1.0); r += 1
        spot_hr = (p.get("hr_by_spot") or {}).get(p.get("lineup_spot") or 0, 0)
        if spot_hr >= 3: s += 1.5; r += 1
        elif spot_hr >= 2: s += 0.75; r += 1
        mixd = (p.get("heat_mix") - p["heat"]) if (p.get("heat_mix") is not None and p.get("heat") is not None) else 0
        if mixd >= 6: s += 1.0; r += 1
        elif mixd >= 4: s += 0.5; r += 1
        arm = (p.get("opp_pitcher") or {}).get("hr_score") or 0
        if arm >= 65 or "arm" in ks: s += 1.0; r += 1
        if "mix" in ks: s += 0.75; r += 1
        if "pow" in ks: s += 0.5; r += 1
        if "plat" in ks: s += 0.5; r += 1
        return s, r
    smash_ids = set()
    try:
        cand = []
        for p in board["players"]:
            if p.get("lineup_status") == "out":
                continue
            sc, nr = _smash_score(p)
            if sc >= 6.5 and nr >= 3 and (p.get("heat") or 0) >= 55:
                cand.append((sc, p["id"]))
        cand.sort(reverse=True)
        smash_ids = {pid for _, pid in cand[:3]}
        print(f"[build] SMASH: {len(smash_ids)} flagged")
        if smash_ids:
            _nm = [p["name"] for p in board["players"] if p["id"] in smash_ids]
            board.setdefault("briefing", []).insert(0, "SMASH today: " + " · ".join(_nm) + ".")
    except Exception as e:
        _hnote("smash calc", e); print(f"[build] smash calc skipped: {e}")

    # ---- Auto-tracked parlays: pick server-side so the grader can score them ----
    # Uses simplified rules that mirror the app's UI logic (heat as the ranking
    # score in place of the client-side blend, but same filters/constraints).
    # Records get flagged by strategy so we can measure whether each type actually
    # pays off vs the base rate — closes the "are parlays actually winning?" question.
    parlay_picks = []
    try:
        live_p = [p for p in board["players"] if p.get("lineup_status") != "out"]

        # Sample-aware heat for PARLAY RANKING ONLY — never touches the real heat field or the
        # model. A recent HR legitimately raises the power metrics, but in a thin window (few
        # batted balls) one or two big games spike the score disproportionately. For parlays we
        # shrink heat toward a neutral 50 when the recent sample is small, so a genuine 14-day
        # streak on solid sample ranks ahead of a "2 HRs in 15 BBE" mirage. Full trust at 25+
        # batted balls; linearly reduced below that.
        def _adj_heat(p):
            h = p.get("heat") or 0
            bbe = (((p.get("windows") or {}).get("L14d") or {}).get("bb_count")
                   or (p.get("sample") or {}).get("L15") or 0)
            # shrink factor: 1.0 at 25+ BBE, ramps down to 0.55 at 0 BBE
            trust = max(0.55, min(1.0, 0.55 + 0.45 * (bbe / 25.0)))
            return 50 + (h - 50) * trust      # pull toward 50 when sample is thin
        by_heat = sorted(live_p, key=lambda p: -_adj_heat(p))

        # Jackpot: top-3 mid-tier by (max_ev + park_hr + iso), not B2B, heat>=45
        def _mev(p):
            m = p.get("max_ev") or {}
            return m.get("season") or m.get("recent") or 0
        def _rk(p):
            for i, x in enumerate(by_heat):
                if x["id"] == p["id"]:
                    return i
            return 99
        jp_pool = [p for p in live_p
                   if not p.get("hr_last_game")
                   and _rk(p) >= 8
                   and 45 <= (p.get("heat") or 0) <= 78
                   and _mev(p) >= 108]
        def _dist(p):
            ev = (min(118.0, max(105.0, _mev(p))) - 105.0) / 13.0
            pk = (p.get("park_hr") or {}).get("boost") or 0
            pk = max(-0.5, min(1.0, pk / 20.0))
            iso = ((p.get("windows") or {}).get("L14d") or {}).get("iso") or 0
            return ev * 0.5 + pk * 0.3 + min(1.0, iso / 0.30) * 0.2
        jp = sorted(jp_pool, key=_dist, reverse=True)[:3]
        if jp:
            parlay_picks.append({
                "kind": "jackpot",
                "legs": [{"id": p["id"], "name": p["name"], "team": p["team"], "heat": p.get("heat")} for p in jp],
            })

        # Best3: top-3 by heat, one per team, at most 1 B2B
        best3, seen_teams, b2b_used = [], set(), False
        for p in by_heat:
            if len(best3) >= 3:
                break
            if p["team"] in seen_teams:
                continue
            if p.get("hr_last_game"):
                if b2b_used:
                    continue
                b2b_used = True
            best3.append(p); seen_teams.add(p["team"])
        if len(best3) == 3:
            parlay_picks.append({
                "kind": "best3",
                "legs": [{"id": p["id"], "name": p["name"], "team": p["team"], "heat": p.get("heat")} for p in best3],
            })

        # Round Robin by 2s: 5 legs across tiers, 5 different games,
        # max 1 chalk (top-3 board), max 1 B2B
        def _pick_from(pool, avoid_ids, avoid_games, chalk_used, b2b_used):
            for p in pool:
                if p["id"] in avoid_ids:
                    continue
                if p["game_pk"] in avoid_games:
                    continue
                is_chalk = _rk(p) < 3
                if is_chalk and chalk_used:
                    continue
                if p.get("hr_last_game") and b2b_used:
                    continue
                return p
            return None

        rr_slots = [
            lambda p: _adj_heat(p) >= 75,
            lambda p: 55 <= _adj_heat(p) < 75,
            lambda p: 55 <= _adj_heat(p) < 75,
            lambda p: 40 <= _adj_heat(p) < 55,
            lambda p: _adj_heat(p) < 55 and _rk(p) >= 20,
        ]
        rr_picks = []
        used_ids, used_games = set(), set()
        chalk_used = b2b_used = False
        for slot in rr_slots:
            pool = [p for p in by_heat if slot(p)]
            p = _pick_from(pool, used_ids, used_games, chalk_used, b2b_used)
            if p is None:
                break
            rr_picks.append(p)
            used_ids.add(p["id"]); used_games.add(p["game_pk"])
            if _rk(p) < 3: chalk_used = True
            if p.get("hr_last_game"): b2b_used = True
        if len(rr_picks) == 5:
            parlay_picks.append({
                "kind": "rr5",
                "legs": [{"id": p["id"], "name": p["name"], "team": p["team"], "heat": p.get("heat")} for p in rr_picks],
            })

        board["parlay_picks"] = parlay_picks
        print(f"[build] parlay picks: {len(parlay_picks)} strategies "
              f"({', '.join(pk['kind'] for pk in parlay_picks) or 'none — thin slate'})")
    except Exception as e:
        _hnote("parlay picks", e); print(f"[build] parlay picks skipped: {e}")

    # slim daily snapshot so the grader can grade this day even after the live
    # board rolls over to tomorrow's slate
    try:
        snap_dir = os.path.join(os.path.dirname(OUT_PATH) or ".", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        # vuln tier per arm, so each hitter's snapshot records how vulnerable his opponent was
        _vuln_tier_snap = {}
        for _pe in (board.get("pitcher_edges") or []):
            if _pe.get("vuln") and _pe.get("id") is not None:
                _vuln_tier_snap[_pe["id"]] = _pe["vuln"].get("tier")
        snap = {
            "date": board["slate_date"],
            "window_v": (board.get("recent_window") or {}).get("v", 1),
            "parlay_picks": parlay_picks,
            "pitcher_props": board.get("pitcher_props", []),
            "players": [{
                "id": p["id"], "name": p["name"], "team": p["team"],
                "heat": p["heat"], "tier": p.get("tier"), "cleared": p.get("cleared"),
                "signals": p["score_breakdown"].get("signals", {}),
                "opp_form": (p["opp_pitcher"].get("form") or {}).get("label"),
                "iso": (p.get("windows", {}).get("L14d", {}) or {}).get("iso"),
                "barrel_pct": (p.get("windows", {}).get("L14d", {}) or {}).get("barrel_pct"),
                # ---- enrichment: context that can't be backfilled later ----
                "badges": [b["k"] for b in (p.get("badges") or [])],
                "bp_score": (p.get("opp_bullpen") or {}).get("score"),
                "sp_vuln": (p["opp_pitcher"].get("hr_score")),
                "luck_gap": (((p.get("luck") or {}).get("recent")) or {}).get("luck_gap"),
                "heat_mix": p.get("heat_mix"),
                "spot": p.get("lineup_spot"),
                "park_boost": (p.get("park_hr") or {}).get("boost"),
                "trend": (p.get("trend") or {}).get("dir"),
                "b2b": p.get("hr_last_game"),
                "smash": p["id"] in smash_ids,
                "opener": bool((p.get("opp_pitcher") or {}).get("opener")),
                "hlabel": p.get("hit_label"),
                # Props-scoring fields (parallel to heat, never fed back in)
                "hit_heat": p.get("hit_heat"),
                "hrr_heat": p.get("hrr_heat"),
                "k_heat_bat": p.get("k_heat_bat"),
                # ---- NEW SIGNALS (graded by track.py; each is a testable hypothesis) ----
                "zone": ((((p.get("features") or {}).get("zone_profile") or {}).get("overlap")) or {}).get("count", 0),
                "grade": (p.get("matchup_grade") or {}).get("grade"),
                "grade_aligned": (p.get("matchup_grade") or {}).get("aligned"),
                "bomb": (p.get("bomb_score") or {}).get("score"),
                "vuln": _vuln_tier_snap.get((p.get("opp_pitcher") or {}).get("id")),
                "elite_tier": (p.get("elite") or {}).get("tier"),
                "sq_up": ((p.get("features") or {}).get("square_up") or {}).get("rating"),
                "eye": bool(((p.get("features") or {}).get("discipline") or {}).get("eye")),
                "crosshair": bool(((p.get("features") or {}).get("discipline") or {}).get("crosshair")),
                "chaser": bool(((p.get("features") or {}).get("discipline") or {}).get("warning")),
                "hr_power": ((p.get("features") or {}).get("hr_power") or {}).get("barrel_pct"),
                "micro": ((p.get("features") or {}).get("microclimate") or {}).get("flag"),
                "late_hr": ((p.get("features") or {}).get("late_hr") or {}).get("label"),
                "pitch_mix": ((p.get("features") or {}).get("pitch_matchup") or {}).get("score"),
                "day_night": p.get("day_night"),
            } for p in board["players"]],
        }
        with open(os.path.join(snap_dir, f"{board['slate_date']}.json"), "w") as f:
            json.dump(snap, f, default=str)
        import glob
        for old in sorted(glob.glob(os.path.join(snap_dir, "20*.json")))[:-16]:
            os.remove(old)
    except Exception as e:
        print(f"[build] snapshot write failed (non-fatal): {e}")

    # ---- Re-write board.json with all post-build modifications ----
    # The initial write happens right after build() returns, but SMASH briefing,
    # pitcher_props, and parlay_picks are all computed in main() AFTER that write.
    # Without this second serialize, none of that surfaces on the live UI (they
    # only make it into the snapshot). This second write is what makes the Pitcher
    # Ks tab actually get its data.
    try:
        with open(OUT_PATH, "w") as f:
            json.dump(board, f, separators=(",", ":"), default=str)
        print(f"[build] re-wrote {OUT_PATH} with pitcher_props ({len(board.get('pitcher_props',[]))} arms), "
              f"parlay_picks ({len(board.get('parlay_picks',[]))} strategies)")
    except Exception as e:
        print(f"[build] board re-write failed (non-fatal): {e}")

    print(f"[build] wrote {OUT_PATH}: {len(board['players'])} hitters, "
          f"{len(board['games'])} games")

    # ---- first-run smell test: do the right names surface? ----
    ps = board["players"]
    def topby(key, label):
        ranked = sorted(
            [p for p in ps if (p["metrics"].get(key) or {}).get("recent") is not None],
            key=lambda p: p["metrics"][key]["recent"], reverse=True)[:5]
        print(f"  top {label}:")
        for p in ranked:
            print(f"    {p['metrics'][key]['recent']:>6}  {p['name']}")
    print("[sanity] eyeball these — known pull sluggers should be high on pull-air:")
    topby("pull_air_pct", "pull-air%")
    topby("ideal_aa_pct", "ideal AA%")
    thin = [p["name"] for p in ps
            if any(str(f).startswith("small sample") for f in p.get("score_breakdown", {}).get("flags", []))]
    if thin:
        print(f"[sanity] {len(thin)} thin-sample hitters flagged (dimmed on board): {', '.join(thin[:8])}")


if __name__ == "__main__":
    main()
