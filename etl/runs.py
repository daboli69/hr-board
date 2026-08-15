"""
runs.py — game-level run expectancy and win probability.

SEPARATE from the HR heat model and from props.py. Reads the same batter/pitcher
profiles and produces expected runs per side, then a win probability, total, and
first-5 line. Nothing here feeds back into heat_score or any prop score.

METHOD (all standard sabermetric machinery, no fitted parameters):

  1. Expected wOBA per hitter
       xwOBA ≈ BB% · wBB  +  (1 − K% − BB%) · xwOBAcon
     xwOBAcon is Statcast's expected wOBA on contact — quality of contact, stripped
     of defense and luck. Adding walks and removing strikeouts turns it into a
     full-PA rate.

  2. Same construction on the pitcher side (xwOBA allowed), and on the bullpen.

  3. Matchup via the odds-ratio (log5) form:
       matchup = batter · pitcher / league
     The standard way to combine a hitter rate and a pitcher rate.

  4. Runs from wOBA via linear weights:
       R/PA = lgR_PA + (xwOBA − lgwOBA) / wOBA_scale
     This is the canonical wRAA conversion.

  5. Innings split: the starter faces ~BF batters (from his median start length),
     the bullpen absorbs the rest of the ~38 PA a team sends up.

  6. Park + weather multiplier on the run environment.

  7. Win probability: runs are OVERDISPERSED relative to Poisson (variance ≈ 2.1×
     mean), so each side is modeled as a negative binomial and the two distributions
     are convolved. Ties go to extra innings, which the home side wins ~52% of.

WHAT THIS MODEL DOES NOT KNOW — read before betting anything off it:
  * Defense. No DRS/OAA. A great defensive team suppresses runs in a way xwOBAcon
    cannot see (xwOBAcon is deliberately defense-independent).
  * Baserunning, catcher framing, umpire zone.
  * True park RUN factor (only an HR factor exists here; runs are approximated).
  * Bullpen availability — who threw last night, who is unavailable.
  * Injuries, weather changes after build, late scratches.

Those omissions are exactly what the moneyline market prices well. Treat the win
probability as INFORMATIONAL until the backtest says otherwise. The total and F5
lines are likely the more useful outputs.
"""
from __future__ import annotations

import os

try:
    from . import markov
except ImportError:                       # standalone import (tests)
    import markov

# League constants (2024-ish; stable year to year, not tuned to any sample)
LG_WOBA = 0.318
WOBA_SCALE = 1.24
W_BB = 0.69              # wOBA weight for a walk
LG_R_PER_PA = 0.117      # ≈ 4.45 runs / 38 PA
LG_PA_PER_TEAM_GAME = 38.0
HOME_FIELD_RUNS = 0.15   # home teams score ~0.15 more per game
EXTRA_INNING_HOME_WP = 0.52

# Runs are overdispersed vs Poisson. Empirically var ≈ 2.1 × mean for team-game runs.
RUN_VAR_RATIO = 2.1

# Regression constants. A 14-day window is a SMALL sample — an .520 xwOBAcon over
# 30 batted balls is mostly noise. Without shrinkage the odds-ratio model compounds
# nine hot hitters against one cold pitcher and produces 99% win probabilities, which
# is nonsense: the best team vs the worst in MLB is ~65-70%. These constants pull each
# estimate toward league mean in proportion to how little data backs it.
#   shrunk = (n·observed + K·league) / (n + K)
BATTER_REG_BBE = 60      # batted balls for a batter's xwOBAcon to get half weight
PITCHER_REG_PA = 120     # PA for a pitcher's rates to get half weight
PEN_REPLACEMENT_WOBA = 0.360   # what a 4th/5th reliever allows, roughly
PEN_FATIGUE_WEIGHT = 0.65      # how far toward replacement a fully-gassed pen slides

TEAM_RUNS_FLOOR = 1.6    # hard sanity clamp — no MLB team projects below this
TEAM_RUNS_CEIL = 8.0     # or above this

# The odds-ratio method is known to OVERSTATE extremes, and this model is blind to
# defense — which is precisely the force that drags real outcomes back toward the
# mean (xwOBAcon is defense-independent by construction, so a great defense is
# invisible here). Both push the same way, so the combined matchup rate is regressed
# toward league before it becomes runs. Without this the model posts 90%+ favorites,
# which no book ever does: MLB moneylines essentially never exceed ~-350 (78%).
MATCHUP_DAMP = 0.62
WP_FLOOR, WP_CEIL = 0.20, 0.80   # hard clamp; the market's own practical range


def _shrink(obs, league, n, k):
    """Regress an observed rate toward the league mean by sample size."""
    if obs is None:
        return league
    n = max(0.0, n or 0.0)
    return (n * obs + k * league) / (n + k)


def _batter_xwoba(recent):
    """Full-PA expected wOBA for a hitter, with the components regressed toward
    league mean by sample size. A 14-day line off 25 batted balls gets pulled hard;
    a full one off 80 barely moves."""
    if not recent:
        return None
    xwc = recent.get("xwobacon")
    k = recent.get("k_pct")
    bb = recent.get("bb_pct")
    if xwc is None or k is None or bb is None:
        return None
    bbe = recent.get("bb_count") or 0
    pa = recent.get("pa") or 0
    # league reference values for each component
    LG_XWOBACON, LG_K, LG_BB = 0.370, 22.0, 8.5
    xwc = _shrink(xwc, LG_XWOBACON, bbe, BATTER_REG_BBE)
    k = _shrink(k, LG_K, pa, BATTER_REG_BBE * 2)     # K% stabilizes faster than contact quality
    bb = _shrink(bb, LG_BB, pa, BATTER_REG_BBE * 2)
    k, bb = k / 100.0, bb / 100.0
    contact = max(0.0, 1.0 - k - bb)
    return bb * W_BB + contact * xwc


def _pitcher_xwoba_allowed(prof):
    """Full-PA expected wOBA allowed. prof is a {recent, season} wrapper; prefers the
    14-day window and blends toward season when recent PA is thin — same convention
    the HR model and props.py use."""
    if not prof:
        return None
    recent = prof.get("recent") or {}
    season = prof.get("season") or {}
    rpa = recent.get("pa") or 0
    conf = 0.0 if rpa < 10 else 1.0 if rpa >= 60 else (rpa - 10) / 50.0

    def blend(key):
        r, s = recent.get(key), season.get(key)
        if r is None and s is None:
            return None
        if r is None:
            return s
        if s is None:
            return r
        return conf * r + (1 - conf) * s

    xwc = blend("xwobacon_allowed")
    k = blend("k_pct_allowed")
    bb = blend("bb_pct_allowed")
    # Regress toward league by the pitcher's total PA in the blended window.
    # Season PA carries most arms; a callup with 30 PA gets pulled to league.
    tot_pa = (season.get("pa") or 0) or rpa
    LG_XWOBACON, LG_K, LG_BB = 0.370, 22.0, 8.5
    xwc = _shrink(xwc, LG_XWOBACON, tot_pa, PITCHER_REG_PA)
    k = _shrink(k, LG_K, tot_pa, PITCHER_REG_PA)
    bb = _shrink(bb, LG_BB, tot_pa, PITCHER_REG_PA)
    if xwc is None or k is None or bb is None:
        return None
    k, bb = k / 100.0, bb / 100.0
    contact = max(0.0, 1.0 - k - bb)
    return bb * W_BB + contact * xwc


def _pitcher_xwoba_allowed_vs(prof, hand):
    """Pitcher's expected wOBA allowed to a batter of a given hand ('R'/'L'), from his platoon
    split. Switch hitters ('S') bat with the platoon advantage, so they take the pitcher's weaker
    side. The split is blended toward his overall rate by how many PA it's built on, so a thin
    split can't swing the projection — this only moves the number when the platoon signal is real.
    Falls back to overall when there's no usable split. This is where non-obvious edges live: a
    lineup stacked against a starter's weak platoon side is something the market underweights."""
    overall = _pitcher_xwoba_allowed(prof)
    if not prof:
        return overall
    splits = prof.get("splits") or {}

    def _for(h):
        sp = splits.get(h)
        if not sp:
            return None
        val = _pitcher_xwoba_allowed(sp)          # split is itself a {season,recent} wrapper
        if val is None:
            return None
        if overall is None:
            return val
        pa = ((sp.get("recent") or {}).get("pa") or 0) + ((sp.get("season") or {}).get("pa") or 0)
        w = min(1.0, pa / 150.0)                   # ~150 PA vs a hand to fully trust the split
        return w * val + (1 - w) * overall

    if hand == "S":
        cand = [x for x in (_for("R"), _for("L")) if x is not None]
        return max(cand) if cand else overall
    v = _for(hand) if hand in ("R", "L") else None
    return v if v is not None else overall


def _matchup(batter_xwoba, pitcher_xwoba):
    """Odds-ratio (log5) combination of a hitter rate and a pitcher rate, then
    regressed toward league by MATCHUP_DAMP (see constant for why)."""
    if batter_xwoba is None and pitcher_xwoba is None:
        return LG_WOBA
    if batter_xwoba is None:
        raw = pitcher_xwoba
    elif pitcher_xwoba is None:
        raw = batter_xwoba
    else:
        raw = batter_xwoba * pitcher_xwoba / LG_WOBA
    return LG_WOBA + MATCHUP_DAMP * (raw - LG_WOBA)


# ---------------------------------------------------------------------------
# Markov event rates
# ---------------------------------------------------------------------------
# The Markov engine needs discrete PA outcomes (1B/2B/3B/HR/BB/K/Out); the profiles only carry
# rate stats. Rather than add a new ETL pull, the hit distribution is SOLVED from what is
# already stored. Given hits, total bases and home runs:
#     TB - H = 2B + 2*3B + 3*HR
# and triples run roughly a tenth as often as doubles league-wide, which closes the system.
# Deriving beats storing here: it stays consistent with the xBA/ISO/SLG the rest of the app
# already trusts, instead of introducing a second set of numbers that can drift apart.
TRIPLE_TO_DOUBLE = 0.10


def event_rates_from_profile(prof, lg_pa=None):
    """{1B,2B,3B,HR,BB,K} per PA from a stored batter window. None-safe."""
    if not prof:
        return None
    pa = float(prof.get("pa") or 0)
    ab = float(prof.get("ab") or 0)
    if pa < 20 or ab <= 0:
        return None
    k = (prof.get("k_pct") or 0) / 100.0
    bb = (prof.get("bb_pct") or 0) / 100.0
    xba = prof.get("xba")
    slg = prof.get("slg")
    hr_ct = float(prof.get("hr") or 0)
    if xba is None or slg is None:
        return None
    hits = float(xba) * ab
    tb = float(slg) * ab
    hr = hr_ct
    xb_non_hr = max(0.0, tb - hits - 3.0 * hr)          # doubles + 2*triples
    dbl = xb_non_hr / (1.0 + 2.0 * TRIPLE_TO_DOUBLE)
    tpl = TRIPLE_TO_DOUBLE * dbl
    sgl = max(0.0, hits - dbl - tpl - hr)
    raw = {
        "1B": sgl / pa, "2B": dbl / pa, "3B": tpl / pa, "HR": hr / pa,
        "BB": bb, "K": k,
    }

    # Regress toward league average by sample size.
    #
    # Without this the Markov engine takes short hot streaks literally. A hitter with 8 homers
    # in 40 plate appearances yields a 0.200 HR rate — six times league — and a lineup of such
    # profiles simulated to 32.9 runs against a real game total near 8.7. That single failure
    # mode put the backtest's markov MAE at 11.57 against the linear engine's 3.58, making the
    # simulation look far worse than it is.
    #
    # The linear engine never had this problem because it runs through a capped xwOBA, which
    # bounds the damage implicitly. Markov consumes the rates directly, so the bound has to be
    # explicit. K is expressed in plate appearances: an observed rate carries half its weight at
    # PA = K, so a full season barely moves while a 40-PA sample stays close to league.
    # Cap each rate at a multiple of league before regressing. The TB-minus-hits algebra is
    # sensitive to extreme SLG — a .798 slugging line implies a 19% double rate, four times any
    # real hitter — and regression alone does not fix that because it pulls a wrong number
    # toward the mean rather than rejecting it. Capping first, then regressing, does both.
    CAP = {"1B": 3.0, "2B": 3.5, "3B": 6.0, "HR": 4.0, "BB": 3.0, "K": 2.5}
    for ev in raw:
        lg = markov.LG.get(ev, 0.0)
        if lg > 0:
            raw[ev] = min(raw[ev], lg * CAP.get(ev, 3.0))

    K_PA = 200.0
    w = pa / (pa + K_PA)
    return {ev: w * raw[ev] + (1.0 - w) * markov.LG.get(ev, 0.0) for ev in raw}


def pitcher_event_rates(prof):
    """Pitcher's allowed event rates per PA, from the stored profile."""
    if not prof:
        return None
    src = prof.get("season") or prof.get("recent") or prof
    pa = float(src.get("pa") or 0)
    if pa < 50:
        return None
    k = (src.get("k_pct_allowed") or 22.0) / 100.0
    bb = (src.get("bb_pct_allowed") or 8.5) / 100.0
    hr = src.get("hr_per_pa")
    hr = (float(hr) / 100.0) if hr is not None and hr > 1 else (float(hr) if hr else 0.032)
    # contact quality sets the hit rates; xwOBAcon maps to a BABIP-like rate
    xwc = src.get("xwobacon_allowed") or 0.360
    scale = max(0.75, min(1.30, float(xwc) / 0.360))
    raw = {
        "1B": 0.142 * scale, "2B": 0.046 * scale, "3B": 0.004 * scale,
        "HR": hr, "BB": bb, "K": k,
    }

    # Cap and regress the PITCHER side too.
    #
    # The batter side got this treatment and the pitcher side did not, which is why the backtest
    # still showed a heavy right tail after the first fix: 11.9% of games projected above 15
    # runs, with the worst misses clustered in mid-April. Early in a season a starter has barely
    # cleared the 50-PA gate, so one bad outing dominates his profile — 4 home runs in 55 plate
    # appearances reads as a 0.073 HR rate, more than double league. Log5 then multiplies that
    # against the batter's rate and a single team projects 24 runs.
    #
    # Same shape as the batter fix: clip physically implausible rates first, then pull what is
    # left toward league in proportion to how little data supports it. A full season barely
    # moves; an April sample stays close to average, which is the honest read of what is known.
    CAP = {"1B": 2.5, "2B": 3.0, "3B": 5.0, "HR": 3.0, "BB": 3.0, "K": 2.2}
    for ev in raw:
        lg = markov.LG.get(ev, 0.0)
        if lg > 0:
            raw[ev] = min(raw[ev], lg * CAP.get(ev, 2.5))

    K_PA = 300.0                      # half weight at 300 batters faced
    w = pa / (pa + K_PA)
    return {ev: w * raw[ev] + (1.0 - w) * markov.LG.get(ev, 0.0) for ev in raw}


def _woba_to_r_per_pa(xwoba):
    return LG_R_PER_PA + (xwoba - LG_WOBA) / WOBA_SCALE


def _nb_pmf(mean, kmax=25):
    """Negative-binomial run distribution with variance = RUN_VAR_RATIO × mean.
    Falls back to a degenerate spike if the mean is non-positive."""
    import math
    if mean <= 0.05:
        p = [0.0] * (kmax + 1)
        p[0] = 1.0
        return p
    var = RUN_VAR_RATIO * mean
    if var <= mean:                      # can't be underdispersed; use Poisson
        out = []
        for k in range(kmax + 1):
            out.append(math.exp(-mean) * mean ** k / math.factorial(k))
        s = sum(out)
        return [x / s for x in out]
    r = mean * mean / (var - mean)       # NB size parameter
    p_succ = r / (r + mean)
    out = []
    for k in range(kmax + 1):
        # C(k+r-1, k) p^r (1-p)^k  using lgamma for non-integer r
        lc = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1))
        out.append(math.exp(lc + r * math.log(p_succ) + k * math.log(1 - p_succ)))
    s = sum(out)
    return [x / s for x in out]


def win_prob(home_runs, away_runs, kmax=25):
    """P(home wins) by convolving two negative-binomial run distributions.
    Ties are extra innings, which the home side takes ~52% of."""
    ph = _nb_pmf(home_runs, kmax)
    pa = _nb_pmf(away_runs, kmax)
    win = tie = 0.0
    for h in range(kmax + 1):
        for a in range(kmax + 1):
            j = ph[h] * pa[a]
            if h > a:
                win += j
            elif h == a:
                tie += j
    return max(WP_FLOOR, min(WP_CEIL, win + tie * EXTRA_INNING_HOME_WP))


def team_runs(lineup_recents, opp_sp_prof, opp_pen_prof, sp_bf=None,
              park_mult=1.0, is_home=False, lineup_hands=None, opp_def=0.0, opp_fg=None,
              opp_pen_fatigue=None):
    """Expected runs for one team.

    lineup_recents : list of trailing-14d batter profile dicts, in batting order
    opp_sp_prof    : opposing starter profile wrapper {recent, season, splits}
    opp_pen_prof   : opposing bullpen profile wrapper {recent, season}
    sp_bf          : batters the starter is expected to face (default 24)
    park_mult      : run-environment multiplier (1.0 = neutral)
    lineup_hands   : optional list of batter hands ('R'/'L'/'S'), parallel to lineup_recents,
                     enabling the platoon matchup vs the starter's split
    opp_def        : opponent's defense in runs saved per game (OAA proxy); subtracted from this
                     team's runs, since the fielding side suppresses balls in play
    """
    if not lineup_recents:
        return None, {}
    sp_x_overall = _siera_adjust(_pitcher_xwoba_allowed(opp_sp_prof), opp_fg)
    pen_x = _pitcher_xwoba_allowed(opp_pen_prof)
    if pen_x is None:
        pen_x = LG_WOBA                      # league-average pen if unknown
    # Availability penalty. If the leverage arms threw the last two days they are effectively
    # unavailable, and the innings they would have covered fall to the 4th and 5th men instead.
    # Rather than model a depth chart, degrade the pen's expected wOBA allowed toward a
    # replacement-level bullpen in proportion to how gassed it is. This is the spot the doc
    # calls the sharp edge: books are slow to move on bullpen availability.
    if opp_pen_fatigue is not None:
        try:
            _f = max(0.0, min(1.0, float(opp_pen_fatigue)))
            pen_x = pen_x + _f * (PEN_REPLACEMENT_WOBA - pen_x) * PEN_FATIGUE_WEIGHT
        except Exception:
            pass

    bf = sp_bf if sp_bf else 24.0
    bf = max(4.0, min(30.0, bf))
    total_pa = LG_PA_PER_TEAM_GAME
    pen_pa = max(0.0, total_pa - bf)

    # Walk the order: each spot gets its share of PA, split between SP and pen. The SP portion uses
    # the starter's platoon split for that batter's hand (the pen is a mix of arms, so it stays on
    # the overall rate).
    n = len(lineup_recents)
    runs = 0.0
    plat_used = 0
    spot_pa = _order_pa(total_pa, n)       # top of the order gets the extra trips
    for i, rec in enumerate(lineup_recents):
        b_x = _batter_xwoba(rec)
        hand = lineup_hands[i] if (lineup_hands and i < len(lineup_hands)) else None
        if hand:
            sp_x = _pitcher_xwoba_allowed_vs(opp_sp_prof, hand)
            if sp_x is not None and opp_sp_prof and (opp_sp_prof.get("splits")):
                plat_used += 1
        else:
            sp_x = sp_x_overall
        sp_x = _siera_adjust(sp_x, opp_fg) if sp_x is not None else sp_x_overall
        if sp_x is None:                     # no probable starter announced yet
            sp_x = LG_WOBA
        pa_i = spot_pa[i]
        pa_sp = pa_i * (bf / total_pa)
        pa_pen = pa_i * (pen_pa / total_pa)
        r_sp = _woba_to_r_per_pa(_matchup(b_x, sp_x))
        r_pen = _woba_to_r_per_pa(_matchup(b_x, pen_x))
        runs += pa_sp * r_sp + pa_pen * r_pen

    runs *= park_mult
    if is_home:
        runs += HOME_FIELD_RUNS
    try:
        runs -= max(-0.6, min(0.6, float(opp_def or 0.0)))   # clamp: defense is a modest, real effect
    except Exception:
        pass
    runs = max(TEAM_RUNS_FLOOR, min(TEAM_RUNS_CEIL, runs))
    return round(runs, 2), {
        "sp_xwoba_allowed": round(sp_x_overall, 3) if sp_x_overall is not None else None,
        "pen_xwoba_allowed": round(pen_x, 3),
        "sp_bf": round(bf, 1),
        "pen_pa": round(pen_pa, 1),
        "park_mult": round(park_mult, 3),
        "lineup_n": n,
        "platoon_spots": plat_used,
        "opp_def": round(float(opp_def or 0.0), 3),
    }


# Times-through-the-order penalty. A starter's wOBA allowed rises roughly .015-.020 the second
# time a lineup sees him and again the third — hitters adjust, and the book's line often doesn't
# fully price it. Applied per PLATE APPEARANCE rather than per inning, which is the mechanically
# correct unit: the 10th batter he faces is the start of the 2nd time through regardless of
# which inning it happens in.
TTO_PENALTY = (0.000, 0.017, 0.034)     # 1st, 2nd, 3rd+ time through


def _tto_for_pa(pa_index, lineup_n):
    """wOBA-allowed penalty for the (0-based) pa_index-th batter this starter faces."""
    if lineup_n <= 0:
        return 0.0
    turn = int(pa_index // lineup_n)
    return TTO_PENALTY[min(turn, len(TTO_PENALTY) - 1)]


def _order_pa(total_pa, n):
    """Plate appearances by lineup spot, using how a batting order actually turns over.

    The model previously split PAs evenly (total/9), which is fine over a full game but wrong
    for F5: through five innings roughly 20 batters come up, so the top of the order bats a
    third more often than the bottom. Spot 1 gets 3 PAs where spot 9 gets 2 — and those extra
    trips go to the best hitters, which systematically understated good offenses in the F5
    market. This returns the real per-spot counts rather than a flat average.
    """
    if n <= 0:
        return []
    base = int(total_pa // n)
    extra = int(round(total_pa - base * n))
    return [base + (1 if i < extra else 0) for i in range(n)]


def _siera_adjust(xwoba, fg):
    """Nudge the bottom-up pitcher number toward SIERA/xFIP, and reward genuine pitch quality.

    The Statcast term is built from contact quality; SIERA and xFIP are built from strikeouts,
    walks and batted-ball type with HR/FB luck stripped out. They disagree in useful ways, and
    where they do, the truth is usually between them. Pitch-quality (Stuff+, or Pitching+ as a
    fallback) earns a small separate nudge because it stabilises weeks before results do — that
    lag is precisely when the market is still pricing the old pitcher. Weights are small on
    purpose: this is a correction, not a replacement for the matchup model.

    Stuff+ and Pitching+ are NOT stacked when both are present — they are highly correlated
    (Pitching+ is Stuff+ blended with Location+), and adding both would double-count one signal
    under two names. Stuff+ is preferred as the more direct swing-and-miss read; Pitching+ is
    the fallback when Stuff+ specifically is missing.
    """
    if not fg or xwoba is None:
        return xwoba
    LG_SIERA = 4.10
    ra = fg.get("siera") if fg.get("siera") is not None else fg.get("xfip")
    adj = 0.0
    if ra is not None:
        # +/-1.0 run of SIERA (vs league) maps to +/-0.020 xwOBA allowed, clamped
        adj += max(-0.020, min(0.020, (float(ra) - LG_SIERA) * 0.020))
    pq = fg.get("stuff_plus")
    if pq is None:
        pq = fg.get("pitching_plus")
    if pq is not None:
        # 100 is average; the full +/-0.010 is reached exactly at the +/-10-point boundary
        # (Stuff+/Pitching+ 110 or 90), matching the documented "elite/poor" threshold rather
        # than only asymptotically approaching it.
        adj += max(-0.010, min(0.010, -(float(pq) - 100.0) * 0.0010))
    return max(0.230, min(0.480, xwoba + adj * 0.60))    # 60% weight on the correction


def first5_runs(lineup_recents, opp_sp_prof, park_mult=1.0, is_home=False, lineup_hands=None,
                opp_def=0.0, first_inn=None):
    """Expected runs through 5 innings — the starter faces roughly the first
    ~20 batters, so F5 is almost purely a starter-vs-lineup question. That makes
    it a cleaner read than the full game (no bullpen guesswork). Platoon matters most here."""
    if not lineup_recents:
        return None
    sp_x_overall = _pitcher_xwoba_allowed(opp_sp_prof)
    pa_f5 = 20.0
    n = len(lineup_recents)
    # Real order turnover, not a flat average: the top of the order gets the extra trips.
    spot_pa = _order_pa(pa_f5, n)
    # A starter who is measurably worse in the first inning costs runs specifically in the F5
    # window, and the full-game line hides it by averaging the first in with innings 2-6.
    _f1 = 0.0
    if first_inn and first_inn.get("vs_rest") is not None and (first_inn.get("pa") or 0) >= 40:
        _f1 = max(-0.35, min(0.35, float(first_inn["vs_rest"]) * 1.8))
    runs = 0.0
    for i, rec in enumerate(lineup_recents):
        b_x = _batter_xwoba(rec)
        hand = lineup_hands[i] if (lineup_hands and i < len(lineup_hands)) else None
        sp_x = _pitcher_xwoba_allowed_vs(opp_sp_prof, hand) if hand else sp_x_overall
        if sp_x is None:
            sp_x = sp_x_overall
        # A probable starter is often not announced when the morning board builds, so sp_x can
        # legitimately be None. The TTO penalty is an ADDITION to the pitcher's expected wOBA
        # allowed, and adding to None raised a TypeError that killed every game projection on
        # the slate. Fall back to league average and keep the penalty meaningful.
        if sp_x is None:
            sp_x = LG_WOBA
        # each trip this spot takes is a separate time through the order
        for _trip in range(int(spot_pa[i])):
            _pen = _tto_for_pa(_trip * n + i, n)
            runs += _woba_to_r_per_pa(_matchup(b_x, sp_x + _pen))
    runs += _f1                              # first-inning penalty/credit, F5 only
    runs *= park_mult
    if is_home:
        runs += HOME_FIELD_RUNS * 0.55
    try:
        runs -= float(opp_def or 0.0) * 0.55       # ~5/9 of the game defended by the starter's side
    except Exception:
        pass
    return round(max(0.2, runs), 2)


def fair_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    d = 1.0 / p
    return f"+{round((d - 1) * 100)}" if d >= 2 else f"-{round(100 / (d - 1))}"


# How much weight the season-long Pythagorean prior gets against the bottom-up projection.
# Deliberately modest: the lineup-and-starter model already sees most of what decides a game,
# and the prior's job is only to catch the durable team quality it can't — bench depth,
# baserunning, defense beyond the OAA term, manager. Too much weight here would just re-predict
# the standings and blunt the matchup edge, which is the whole reason to build the model.
# Two different weights, because the bet is a different question at each horizon.
#
# A caution on where these numbers came from: published guidance quotes ~35% (full game) and
# ~25% (F5) as the FEATURE IMPORTANCE a gradient-boosted model assigns to macro team strength.
# Feature importance and blend weight are not the same quantity — importance measures how often
# a feature drives a split, not how far to average toward it — so those figures are treated as
# direction, not gospel. They are set below the quoted values because this model is bottom-up:
# lineups, starters, bullpen and defense already carry real team quality, so a heavy macro blend
# would double-count it and blunt the matchup edge that is the entire point of building this.
# The backtest's Brier score is the arbiter; these are the knobs to turn.
PYTH_WEIGHT = 0.28        # full game
PYTH_WEIGHT_F5 = 0.20     # F5 — the starter dominates, so the macro anchor matters less


def _pyth_wp(home_pyth, away_pyth):
    """Head-to-head win probability implied by two Pythagorean win percentages (log5)."""
    if home_pyth is None or away_pyth is None:
        return None
    h = min(0.75, max(0.25, float(home_pyth)))
    a = min(0.75, max(0.25, float(away_pyth)))
    den = h * (1 - a) + a * (1 - h)
    if den <= 0:
        return None
    return h * (1 - a) / den


# Simulations per game. 10,000 gives a stable tail; the whole slate is ~15 games x 2 teams,
# so this is ~300k inning-simulations per ETL run — a few seconds, well inside the Actions
# budget. Dropping to 4,000 halves the time and roughly doubles the noise in the extreme tail,
# which is the part the engine exists to get right.
# 2,500 is the live-board default: it holds the mean to about +/-0.06 runs and the body of the
# distribution well, which is all the board needs. The tail (P of 11+ runs) is the part that
# stays noisy at this count, so the backtest — where tail accuracy is the thing being graded —
# raises it via MARKOV_SIMS rather than paying the cost on every six-hourly ETL run.
MARKOV_SIMS = int(os.environ.get("MARKOV_SIMS", "2500"))


def markov_project(home_lineup, away_lineup, home_sp, away_sp,
                   home_pen, away_pen, park_mult=1.0, home_hands=None, away_hands=None,
                   sp_bf_home=22, sp_bf_away=22, seed=None):
    """Full run DISTRIBUTIONS for both sides via the inning-state simulation.

    Returns None if the profiles cannot produce event rates, so the caller can fall back to the
    linear engine rather than the board losing its game lines.
    """
    def side(lineup, opp_sp, opp_pen, hands, is_home, sp_bf):
        sp_rates = pitcher_event_rates(opp_sp)
        pen_rates = pitcher_event_rates(opp_pen) or sp_rates
        if not sp_rates:
            return None
        sp_ev, pen_ev = [], []
        for i, rec in enumerate(lineup or []):
            br = event_rates_from_profile(rec)
            if br is None:
                br = dict(markov.LG)
                br.pop("OUT", None)
            sp_ev.append(markov.pa_event_probs(br, sp_rates, park_mult=park_mult))
            pen_ev.append(markov.pa_event_probs(br, pen_rates, park_mult=park_mult))
        if not sp_ev:
            return None
        mean, dist = markov.simulate_team_runs(
            sp_ev, n_sims=MARKOV_SIMS, seed=seed,
            pen_events=pen_ev, sp_batters=sp_bf)
        if is_home:
            mean += HOME_FIELD_RUNS
        return mean, dist

    h = side(home_lineup, away_sp, away_pen, home_hands, True, sp_bf_away)
    a = side(away_lineup, home_sp, home_pen, away_hands, False, sp_bf_home)
    if not h or not a:
        return None
    return {"home_mean": round(h[0], 2), "away_mean": round(a[0], 2),
            "home_dist": h[1], "away_dist": a[1],
            "total_dist": markov.game_totals(h[1], a[1]),
            "home_wp": round(markov.win_prob_from_dists(h[1], a[1]), 4)}


def project_game(home_lineup, away_lineup, home_sp, away_sp,
                 home_pen, away_pen, home_bf=None, away_bf=None, park_mult=1.0,
                 home_hands=None, away_hands=None, home_def=0.0, away_def=0.0,
                 home_pyth=None, away_pyth=None, home_fg=None, away_fg=None,
                 home_first_inn=None, away_first_inn=None,
                 home_pen_fatigue=None, away_pen_fatigue=None):
    """Full game projection. Returns runs, win prob, total, run line, F5.
    home_hands/away_hands: optional batter-hand lists (parallel to the lineups) for platoon.
    home_def/away_def: each team's defense in runs saved per game (OAA proxy); a team's fielding
    reduces the OTHER team's runs."""
    # away team bats against the HOME starter and HOME pen — and the HOME defense behind them
    away_r, away_bd = team_runs(away_lineup, home_sp, home_pen, home_bf,
                                park_mult, is_home=False, lineup_hands=away_hands,
                                opp_def=home_def, opp_fg=home_fg, opp_pen_fatigue=home_pen_fatigue)
    home_r, home_bd = team_runs(home_lineup, away_sp, away_pen, away_bf,
                                park_mult, is_home=True, lineup_hands=home_hands,
                                opp_def=away_def, opp_fg=away_fg, opp_pen_fatigue=away_pen_fatigue)
    if home_r is None or away_r is None:
        return None
    # ---- Markov engine: the run DISTRIBUTION, not just the mean ----
    # The linear projection above still runs and still supplies the mean, because it is what the
    # calibrated win-probability path was validated on. The simulation adds what the linear
    # model structurally cannot produce: the shape of the run distribution, which is what
    # decides an over/under at a given line. Where the two disagree on the mean, the linear
    # number is kept and the disagreement is reported — a silent swap would invalidate the
    # existing calibration with no way to notice.
    # Before lineups post, team_runs legitimately returns None. Everything below has to cope
    # with that rather than raising, otherwise the board ships zero game projections for the
    # entire morning — which is exactly what happened.
    if home_r is None or away_r is None:
        return {"pending": True,
                "note": "awaiting confirmed lineups",
                "home_runs": home_r, "away_runs": away_r,
                "total": None, "home_wp": None, "away_wp": None,
                "markov": None, "why": []}

    _mk = None
    try:
        _mk = markov_project(home_lineup, away_lineup, home_sp, away_sp,
                             home_pen, away_pen, park_mult=park_mult,
                             home_hands=home_hands, away_hands=away_hands,
                             sp_bf_home=(home_bf or 22), sp_bf_away=(away_bf or 22))
    except Exception as _e:
        print(f"[runs] markov skipped (non-fatal): {_e}")
        _mk = None

    hwp = win_prob(home_r, away_r)
    # Blend toward the season-long team-quality prior (run differential, not W-L).
    _pw = _pyth_wp(home_pyth, away_pyth)
    if _pw is not None:
        _pw_home = _pw + (EXTRA_INNING_HOME_WP - 0.50) * 0.5   # prior is neutral-site
        hwp = (1 - PYTH_WEIGHT) * hwp + PYTH_WEIGHT * min(0.85, max(0.15, _pw_home))
    f5_home = first5_runs(home_lineup, away_sp, park_mult, is_home=True, lineup_hands=home_hands,
                          opp_def=away_def, first_inn=away_first_inn)
    f5_away = first5_runs(away_lineup, home_sp, park_mult, is_home=False, lineup_hands=away_hands,
                          opp_def=home_def, first_inn=home_first_inn)
    # F5 win probability, with its own macro anchor. Previously F5 got no Pythagorean weight at
    # all, which left the shortest-horizon market running purely on the bottom-up projection.
    f5_hwp = win_prob(f5_home, f5_away) if (f5_home is not None and f5_away is not None) else None
    if f5_hwp is not None and _pw is not None:
        f5_hwp = (1 - PYTH_WEIGHT_F5) * f5_hwp + PYTH_WEIGHT_F5 * min(0.85, max(0.15, _pw_home))
    # ---- WHY: local attribution by ablation ----
    # Each factor is switched off and the model re-run; the shift in home win probability is
    # that factor's contribution for THIS game. It's the same idea as a SHAP local explanation
    # but computed exactly rather than approximated, which is affordable here because the model
    # is cheap to evaluate. The point is to let you see whether a number comes from a real
    # matchup edge or is just riding the macro anchor.
    _why = []
    try:
        def _wp_without(drop):
            _hs = home_sp if drop != "sp" else None
            _as_ = away_sp if drop != "sp" else None
            _hp = home_pen if drop != "pen" else None
            _ap = away_pen if drop != "pen" else None
            _hpf = None if drop in ("pen", "penfatigue") else home_pen_fatigue
            _apf = None if drop in ("pen", "penfatigue") else away_pen_fatigue
            _hd = 0.0 if drop == "def" else home_def
            _ad = 0.0 if drop == "def" else away_def
            _hh = None if drop == "platoon" else home_hands
            _ah = None if drop == "platoon" else away_hands
            _pm = 1.0 if drop == "park" else park_mult
            _hfg = None if drop == "stuff" else home_fg
            _afg = None if drop == "stuff" else away_fg
            _ar, _ = team_runs(away_lineup, _hs, _hp, home_bf, _pm, False, _ah, _hd, _hfg, _hpf)
            _hr, _ = team_runs(home_lineup, _as_, _ap, away_bf, _pm, True, _hh, _ad, _afg, _apf)
            if _hr is None or _ar is None:
                return None
            _w = win_prob(_hr, _ar)
            if drop != "macro" and _pw is not None:
                _w = (1 - PYTH_WEIGHT) * _w + PYTH_WEIGHT * min(0.85, max(0.15, _pw_home))
            return _w

        for _key, _label in (("macro", "Team quality (Pythagorean)"),
                             ("sp", "Starting pitching"),
                             ("pen", "Bullpen"),
                             ("penfatigue", "Bullpen availability"),
                             ("platoon", "Platoon matchup"),
                             ("def", "Defense"),
                             ("stuff", "Stuff+/SIERA"),
                             ("park", "Park & weather")):
            _w0 = _wp_without(_key)
            if _w0 is None:
                continue
            _delta = round(100.0 * (hwp - _w0), 1)
            if abs(_delta) >= 0.1:
                _why.append({"k": _key, "label": _label, "pts": _delta})
        _why.sort(key=lambda x: -abs(x["pts"]))
    except Exception:
        _why = []

    return {
        "why": _why,
        # Distribution-based outputs. `total_dist` is the payload the over/under grader needs;
        # `markov_delta` exposes how far the simulation's mean sits from the linear one, so a
        # systematic divergence shows up in the backtest instead of hiding.
        "markov": ({"home_mean": _mk["home_mean"], "away_mean": _mk["away_mean"],
                    "total_mean": round(_mk["home_mean"] + _mk["away_mean"], 2),
                    "home_wp": _mk["home_wp"],
                    "total_dist": {str(k): v for k, v in _mk["total_dist"].items() if v >= 1e-4},
                    # `markov_delta` is only meaningful when the linear engine also produced a
                    # number. team_runs returns None for an empty lineup — which is the NORMAL
                    # state before lineups post, not an error — so this must be guarded or the
                    # whole projection dies every morning.
                    "markov_delta": (round((_mk["home_mean"] + _mk["away_mean"])
                                           - (home_r + away_r), 2)
                                     if (home_r is not None and away_r is not None) else None)}
                   if _mk else None),
        "home_runs": home_r,
        "away_runs": away_r,
        "total": round(home_r + away_r, 2),
        "home_wp": round(hwp, 4),
        "away_wp": round(1 - hwp, 4),
        "home_fair": fair_american(hwp),
        "away_fair": fair_american(1 - hwp),
        "f5_home": f5_home,
        "f5_away": f5_away,
        "f5_total": round(f5_home + f5_away, 2) if (f5_home and f5_away) else None,
        "f5_home_wp": round(f5_hwp, 4) if f5_hwp else None,
        "home_breakdown": home_bd,
        "away_breakdown": away_bd,
    }
