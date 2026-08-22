"""Inning-state Markov chain + Monte Carlo run engine.

WHY THIS EXISTS — and why it is NOT about total MAE.

The linear wOBA->runs converter it replaces produces a single expected-runs number and then
assumes a negative-binomial shape around it with a fixed variance ratio. That is fine for a
point estimate but wrong for the thing you actually bet: P(total > line). Two lineups with
identical expected runs can have very different run DISTRIBUTIONS — a stacked lineup that
clusters baserunners produces more 7+ run games and more 2-run games than a flat lineup with
the same mean. A closed-form approximation cannot see that. A state simulation can, because
clustering is an emergent property of tracking who is on base.

On the MAE target specifically: game-total MAE is dominated by irreducible noise, not model
skill. With a game-total sd of ~4.4 and only ~1.5 runs of true between-game variation, a model
that knew every game's true mean exactly would still post ~3.30 MAE. A constant predictor posts
~3.51. So the realistic headroom from 3.60 is about 0.3 runs, not 1.0, and 2.60 is below the
noise floor for individual games. Judge this engine on the calibration of P(over/under) at the
posted line, not on MAE.

The 3.60 figure also came from a replay that omitted the bullpen entirely and had no defense or
true park run factor, so part of that gap is missing inputs rather than a bad converter.
"""
from __future__ import annotations

import numpy as np

# ---- league reference rates, per plate appearance ----
LG = {"BB": 0.085, "K": 0.225, "HR": 0.032, "1B": 0.142, "2B": 0.046, "3B": 0.004}
LG["OUT"] = 1.0 - sum(LG.values())

# Hit-by-pitch and reached-on-error put a runner on base and score runs, but they are not a
# batter "skill" the matchup model projects. Leaving them out cost ~0.5 R/G — the simulator
# came in at 3.91 against a real 4.45 — so they are added as a flat league constant on top of
# whatever the matchup produces. Modelling them as skill would be false precision; ignoring
# them entirely makes every projection systematically light.
FREE_BASE_RATE = 0.017      # HBP (~.011) + ROE (~.006) per PA

# Runner advancement on a single: the fraction of the time a runner takes the extra base.
# These are league-average behaviours; modelling them is what separates a state simulation
# from a rate model, because they decide whether baserunners convert into runs.
ADV_1B_FROM_1ST = 0.28     # first-to-third on a single
ADV_1B_FROM_2ND = 0.62     # second-to-home on a single
ADV_2B_FROM_1ST = 0.42     # first-to-home on a double
DP_RATE = 0.11             # of outs with a runner on first and <2 out
SF_RATE = 0.16             # of outs with a runner on third and <2 out


def _log5(bat_rate, pit_rate, lg_rate):
    """Bill James log5 for a single event rate.

    Rates COMPOUND rather than average: a high-strikeout hitter against a high-strikeout
    pitcher strikes out more often than either rate alone, which is exactly the matchup a
    linear blend flattens.
    """
    if bat_rate is None or pit_rate is None or not lg_rate:
        return bat_rate if bat_rate is not None else lg_rate
    b = min(0.95, max(0.001, float(bat_rate)))
    p = min(0.95, max(0.001, float(pit_rate)))
    g = min(0.95, max(0.001, float(lg_rate)))
    num = (b * p / g)
    den = num + ((1 - b) * (1 - p) / (1 - g))
    return num / den if den > 0 else b


def pa_event_probs(batter, pitcher, park_mult=1.0, tto_penalty=0.0):
    """Discrete PA outcome distribution for one batter-vs-pitcher matchup.

    Returns {"BB","K","HR","1B","2B","3B","OUT"} summing to 1.

    batter/pitcher are dicts of rates per PA. Missing rates fall back to league, so a thin
    profile degrades to average rather than to nonsense.
    """
    b = batter or {}
    p = pitcher or {}
    out = {}
    for ev in ("BB", "K", "HR", "1B", "2B", "3B"):
        out[ev] = _log5(b.get(ev), p.get(ev), LG[ev])

    # TTO penalty raises contact quality against the pitcher: applied to the hit events, since
    # the third time through a lineup does not mainly show up as fewer strikeouts.
    if tto_penalty:
        boost = 1.0 + float(tto_penalty) * 6.0     # .017 wOBA ~ +10% on hit rates
        for ev in ("HR", "1B", "2B", "3B"):
            out[ev] *= boost

    # Park acts on the ball leaving the yard, and much more weakly on other hits.
    if park_mult and park_mult != 1.0:
        out["HR"] *= float(park_mult)
        for ev in ("2B", "3B"):
            out[ev] *= 1.0 + (float(park_mult) - 1.0) * 0.35

    # league-constant free bases, treated as a walk for advancement purposes
    out["BB"] = out.get("BB", 0.0) + FREE_BASE_RATE

    # Bound the on-base rate to something a hitter can actually do.
    #
    # Without this the simulation trusts whatever the rate derivation hands it, and the
    # derivation can produce nonsense from extreme inputs: a .798 SLG run through the TB-minus-
    # hits algebra yields a 14.5% DOUBLE rate against a real league rate near 4.6%. Nine such
    # batters simulated to 33 runs, and across the backtest that drove markov MAE to 12.38
    # against the linear engine's 3.58 — the simulation looked broken when the inputs were.
    #
    # The old guard only triggered when events summed above 0.98, which let non-strikeout outs
    # collapse toward 2% of plate appearances. That is not a bound in any useful sense: an
    # inning at that out rate takes forty batters.
    #
    # 0.500 is deliberately generous — the best on-base seasons in history sit near .480, so
    # this clips only physically impossible lines and leaves every real hitter untouched.
    MAX_ON_BASE = 0.500
    on_base = sum(out.get(ev, 0.0) for ev in ("BB", "HR", "1B", "2B", "3B"))
    if on_base > MAX_ON_BASE:
        scale = MAX_ON_BASE / on_base
        for ev in ("BB", "HR", "1B", "2B", "3B"):
            out[ev] = out.get(ev, 0.0) * scale

    tot = sum(out.values())
    if tot >= 0.98:                      # keep room for outs
        for k in out:
            out[k] *= 0.97 / tot
        tot = sum(out.values())
    out["OUT"] = max(0.0, 1.0 - tot)
    return out


class InningState:
    """The 24-state base-out machine: 8 base configurations x 3 out counts.

    Bases are a 3-bit mask: bit0 = first, bit1 = second, bit2 = third.
    """

    __slots__ = ("bases", "outs", "runs")

    def __init__(self):
        self.bases = 0
        self.outs = 0
        self.runs = 0

    def reset(self):
        self.bases = 0
        self.outs = 0
        self.runs = 0

    def _score(self, n):
        self.runs += n

    def apply(self, event, rng):
        """Advance the state by one plate appearance. Returns runs scored on the play."""
        before = self.runs
        b = self.bases
        on1, on2, on3 = b & 1, (b >> 1) & 1, (b >> 2) & 1

        if event == "K":
            self.outs += 1

        elif event == "BB":
            # forced advance only — this is why walks cluster differently than singles
            if on1 and on2 and on3:
                self._score(1)
            elif on1 and on2:
                self.bases = 0b111
            elif on1:
                self.bases = b | 0b010
            else:
                self.bases = b | 0b001

        elif event == "HR":
            self._score(1 + on1 + on2 + on3)
            self.bases = 0

        elif event == "1B":
            scored = on3
            new = 0b001                                   # batter to first
            if on2:
                if rng.random() < ADV_1B_FROM_2ND:
                    scored += 1
                else:
                    new |= 0b100
            if on1:
                if rng.random() < ADV_1B_FROM_1ST:
                    new |= 0b100
                else:
                    new |= 0b010
            self._score(scored)
            self.bases = new

        elif event == "2B":
            scored = on3 + on2
            new = 0b010                                   # batter to second
            if on1:
                if rng.random() < ADV_2B_FROM_1ST:
                    scored += 1
                else:
                    new |= 0b100
            self._score(scored)
            self.bases = new

        elif event == "3B":
            self._score(on1 + on2 + on3)
            self.bases = 0b100

        else:  # OUT — where double plays and sac flies live
            if on1 and self.outs < 2 and rng.random() < DP_RATE:
                self.outs += 2
                self.bases = b & ~0b001                    # lead runner erased
            elif on3 and self.outs < 2 and rng.random() < SF_RATE:
                self.outs += 1
                self._score(1)
                self.bases = b & ~0b100
            else:
                self.outs += 1

        return self.runs - before


def simulate_team_runs(lineup_events, n_sims=10000, innings=9, seed=None,
                       start_spot=0, pen_events=None, sp_batters=22):
    """Monte Carlo a team's runs.

    lineup_events : list of 9 PA-probability dicts vs the STARTER, in batting order
    pen_events    : same list vs the BULLPEN; used once the starter's batters-faced is spent
    sp_batters    : how many batters the starter is expected to face before the pen takes over

    Returns (mean_runs, run_distribution) where the distribution is P(exactly k runs).
    Simulating rather than solving the chain analytically keeps the lineup ORDER intact, which
    is the entire point: the sequence of who bats after whom is what creates clustering.
    """
    rng = np.random.default_rng(seed)
    n = len(lineup_events)
    if n == 0:
        return 0.0, {}
    totals = np.zeros(n_sims, dtype=np.int16)
    st = InningState()

    # Pre-build cumulative distributions once — sampling is the hot loop.
    def cdf_of(ev):
        keys = ("BB", "K", "HR", "1B", "2B", "3B", "OUT")
        probs = np.array([max(0.0, ev.get(k, 0.0)) for k in keys], dtype=float)
        s = probs.sum()
        probs = probs / s if s > 0 else probs
        return keys, np.cumsum(probs)

    sp_cdf = [cdf_of(e) for e in lineup_events]
    pen_cdf = [cdf_of(e) for e in (pen_events or lineup_events)]

    for s in range(n_sims):
        spot = start_spot
        faced = 0
        total = 0
        for _ in range(innings):
            st.reset()
            while st.outs < 3:
                tbl = sp_cdf if faced < sp_batters else pen_cdf
                keys, cum = tbl[spot]
                r = rng.random()
                idx = int(np.searchsorted(cum, r))
                if idx >= len(keys):
                    idx = len(keys) - 1
                st.apply(keys[idx], rng)
                spot = (spot + 1) % n
                faced += 1
            total += st.runs
        totals[s] = total

    vals, counts = np.unique(totals, return_counts=True)
    dist = {int(v): float(c) / n_sims for v, c in zip(vals, counts)}
    return float(totals.mean()), dist


def game_totals(home_dist, away_dist):
    """Convolve two independent team run distributions into the game total distribution."""
    out = {}
    for h, ph in home_dist.items():
        for a, pa in away_dist.items():
            out[h + a] = out.get(h + a, 0.0) + ph * pa
    return out


def prob_over(total_dist, line):
    """P(total > line). For a half-line this is exact; for a whole number, pushes are excluded.

    This is the number to judge the engine on — not MAE. Totals MAE is dominated by
    irreducible variance, so a better distribution shows up as better over/under calibration
    long before it shows up as a lower MAE.
    """
    over = sum(p for k, p in total_dist.items() if k > line)
    push = sum(p for k, p in total_dist.items() if k == line)
    denom = 1.0 - push
    return over / denom if denom > 0 else 0.5


def win_prob_from_dists(home_dist, away_dist):
    """P(home wins), ties split as extra innings (slight home edge)."""
    hw = tie = 0.0
    for h, ph in home_dist.items():
        for a, pa in away_dist.items():
            if h > a:
                hw += ph * pa
            elif h == a:
                tie += ph * pa
    return hw + tie * 0.52


def run_line_prob(home_dist, away_dist, home_line):
    """P(home covers home_line), e.g. home_line=-1.5 -> P(home wins by 2+);
    home_line=+1.5 -> P(home doesn't lose by 2+, i.e. loses by <=1 or wins outright).

    ADDED this session. home_dist/away_dist (independent per-team run distributions from the
    same Monte Carlo simulation win_prob_from_dists and game_totals already use) were being
    computed every build and then discarded before reaching project_game's return dict -- this
    is the same independence assumption and the same joint-distribution approach those two
    functions already use, just asking a different question of the same real data (margin,
    not total or straight win). A half-line (the standard run-line format, e.g. -1.5) always
    has an exact answer with no push to handle, unlike prob_over's whole-number guard.
    """
    p = 0.0
    for h, ph in home_dist.items():
        for a, pa in away_dist.items():
            if (h - a) > -home_line:
                p += ph * pa
    return p
