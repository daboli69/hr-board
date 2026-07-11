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
              park_mult=1.0, is_home=False):
    """Expected runs for one team.

    lineup_recents : list of trailing-14d batter profile dicts, in batting order
    opp_sp_prof    : opposing starter profile wrapper {recent, season}
    opp_pen_prof   : opposing bullpen profile wrapper {recent, season}
    sp_bf          : batters the starter is expected to face (default 24)
    park_mult      : run-environment multiplier (1.0 = neutral)
    """
    if not lineup_recents:
        return None, {}
    sp_x = _pitcher_xwoba_allowed(opp_sp_prof)
    pen_x = _pitcher_xwoba_allowed(opp_pen_prof)
    if pen_x is None:
        pen_x = LG_WOBA                      # league-average pen if unknown

    bf = sp_bf if sp_bf else 24.0
    bf = max(4.0, min(30.0, bf))
    total_pa = LG_PA_PER_TEAM_GAME
    pen_pa = max(0.0, total_pa - bf)

    # Walk the order: each spot gets its share of PA, split between SP and pen
    n = len(lineup_recents)
    runs = 0.0
    spot_pa = [total_pa / n] * n           # even split is close enough at 9 spots
    for i, rec in enumerate(lineup_recents):
        b_x = _batter_xwoba(rec)
        pa_i = spot_pa[i]
        pa_sp = pa_i * (bf / total_pa)
        pa_pen = pa_i * (pen_pa / total_pa)
        r_sp = _woba_to_r_per_pa(_matchup(b_x, sp_x))
        r_pen = _woba_to_r_per_pa(_matchup(b_x, pen_x))
        runs += pa_sp * r_sp + pa_pen * r_pen

    runs *= park_mult
    if is_home:
        runs += HOME_FIELD_RUNS
    runs = max(TEAM_RUNS_FLOOR, min(TEAM_RUNS_CEIL, runs))
    return round(runs, 2), {
        "sp_xwoba_allowed": round(sp_x, 3) if sp_x is not None else None,
        "pen_xwoba_allowed": round(pen_x, 3),
        "sp_bf": round(bf, 1),
        "pen_pa": round(pen_pa, 1),
        "park_mult": round(park_mult, 3),
        "lineup_n": n,
    }


def first5_runs(lineup_recents, opp_sp_prof, park_mult=1.0, is_home=False):
    """Expected runs through 5 innings — the starter faces roughly the first
    ~20 batters, so F5 is almost purely a starter-vs-lineup question. That makes
    it a cleaner read than the full game (no bullpen guesswork)."""
    if not lineup_recents:
        return None
    sp_x = _pitcher_xwoba_allowed(opp_sp_prof)
    pa_f5 = 20.0
    n = len(lineup_recents)
    runs = 0.0
    for rec in lineup_recents:
        b_x = _batter_xwoba(rec)
        runs += (pa_f5 / n) * _woba_to_r_per_pa(_matchup(b_x, sp_x))
    runs *= park_mult
    if is_home:
        runs += HOME_FIELD_RUNS * 0.55
    return round(max(0.2, runs), 2)


def fair_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    d = 1.0 / p
    return f"+{round((d - 1) * 100)}" if d >= 2 else f"-{round(100 / (d - 1))}"


def project_game(home_lineup, away_lineup, home_sp, away_sp,
                 home_pen, away_pen, home_bf=None, away_bf=None, park_mult=1.0):
    """Full game projection. Returns runs, win prob, total, run line, F5."""
    # away team bats against the HOME starter and HOME pen
    away_r, away_bd = team_runs(away_lineup, home_sp, home_pen, home_bf,
                                park_mult, is_home=False)
    home_r, home_bd = team_runs(home_lineup, away_sp, away_pen, away_bf,
                                park_mult, is_home=True)
    if home_r is None or away_r is None:
        return None
    hwp = win_prob(home_r, away_r)
    f5_home = first5_runs(home_lineup, away_sp, park_mult, is_home=True)
    f5_away = first5_runs(away_lineup, home_sp, park_mult, is_home=False)
    f5_hwp = win_prob(f5_home, f5_away) if (f5_home and f5_away) else None
    return {
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
