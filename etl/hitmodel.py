"""Contact-gated hit model: Hit_Probability = (1 - matchup_whiff%) x xBA_on_contact.

WHY GATING MATTERS. The old model treated xBA and whiff as two weighted signals summed
together. That is wrong in a specific way: they are not additive, they are SEQUENTIAL. A hitter
cannot get a hit on a ball he never puts in play, so contact is a gate the at-bat must pass
through before contact quality is allowed to matter at all. A hitter with a .380 xBA-on-contact
who whiffs 35% of the time against this arm is worse than one at .320 who whiffs 18%, and a
weighted sum will rank them the other way round.

Every rate here is a MATCHUP rate via log5, not a batter-season rate, because whiff and contact
quality both compound against a specific pitcher.
"""
from __future__ import annotations

import numpy as np

LG_WHIFF = 0.245          # per swing
LG_XBA_CON = 0.330        # xBA on balls in play
LG_SWING = 0.470          # per pitch
LG_CONTACT_PA = 0.745     # share of PAs ending in a ball in play or a walk


def _log5(b, p, lg):
    if b is None or p is None or not lg:
        return b if b is not None else lg
    b = min(0.95, max(0.01, float(b)))
    p = min(0.95, max(0.01, float(p)))
    g = min(0.95, max(0.01, float(lg)))
    num = b * p / g
    den = num + (1 - b) * (1 - p) / (1 - g)
    return num / den if den else b


class PitchShapeSplits:
    """Batter xBA and whiff against the SPECIFIC pitch shapes an arm throws.

    Generic L/R splits blur the thing that actually decides the at-bat. A right-handed hitter
    who handles four-seams but is helpless against sweepers has one xBA vs RHP, and it is
    useless if tonight's righty throws 38% sweepers. This weights the batter's per-pitch-type
    performance by the arm's ACTUAL usage to that batter's side.

    Inputs come from data the app already builds:
      arsenal   : [(pitch_type, usage_pct, n), ...] for this arm vs this batter hand
      vs_pitch  : {pitch_type: [18 raw counts]}     for this batter vs this arm's hand
    """

    # indices into VS_PITCH_ORDER
    I_PITCHES, I_WHIFFS, I_PA, I_K = 0, 1, 2, 3
    I_BBE, I_NEV, I_EVSUM, I_BRL = 4, 5, 6, 7
    I_HARD, I_HR, I_AB, I_TB, I_HITS = 11, 12, 13, 14, 15

    def __init__(self, arsenal, vs_pitch, min_usage=8.0, min_pitches=20):
        self.rows = []
        total_w = 0.0
        for entry in (arsenal or []):
            pt, usage = entry[0], entry[1]
            if usage < min_usage:
                continue
            v = (vs_pitch or {}).get(pt)
            if not v or len(v) < 18 or v[self.I_PITCHES] < min_pitches:
                continue
            self.rows.append((pt, float(usage), v))
            total_w += float(usage)
        self.total_w = total_w

    def ok(self):
        return self.total_w > 0 and len(self.rows) >= 2

    def _weighted(self, fn):
        num = den = 0.0
        for pt, usage, v in self.rows:
            val = fn(v)
            if val is None:
                continue
            num += usage * val
            den += usage
        return (num / den) if den else None

    def xba_on_contact(self):
        """Batter's hits per ball in play, usage-weighted across the arm's real mix."""
        return self._weighted(lambda v: (v[self.I_HITS] / v[self.I_BBE]) if v[self.I_BBE] else None)

    def whiff_rate(self):
        """Whiffs per pitch seen, usage-weighted. Per-pitch rather than per-swing keeps it on
        the same denominator as usage, which is what the weighting is over."""
        return self._weighted(lambda v: (v[self.I_WHIFFS] / v[self.I_PITCHES])
                              if v[self.I_PITCHES] else None)

    def k_rate(self):
        return self._weighted(lambda v: (v[self.I_K] / v[self.I_PA]) if v[self.I_PA] else None)

    def hard_rate(self):
        return self._weighted(lambda v: (v[self.I_HARD] / v[self.I_NEV]) if v[self.I_NEV] else None)

    def detail(self):
        return [{"pt": pt, "usage": round(u, 1),
                 "xba_con": round(v[self.I_HITS] / v[self.I_BBE], 3) if v[self.I_BBE] else None,
                 "whiff": round(100.0 * v[self.I_WHIFFS] / v[self.I_PITCHES], 1)
                 if v[self.I_PITCHES] else None}
                for pt, u, v in self.rows]


def directional_defense(spray_profile, def_by_zone):
    """Multiplier on xBA-on-contact from where this hitter hits it vs where the defense is good.

    A team's overall defensive rating is the wrong resolution for a hit prop. What matters is
    whether the defenders standing where THIS hitter hits the ball are good. A pull-heavy lefty
    is punished by a strong right side and barely touched by a strong left side; a team rating
    averages those into nothing.

    spray_profile : {"pull": p, "center": p, "oppo": p}  shares summing to ~1
    def_by_zone   : {"pull": oaa_like, "center": ..., "oppo": ...} in runs-saved units,
                    positive = good defense
    Returns a multiplier, clamped, applied to xBA-on-contact.
    """
    if not spray_profile or not def_by_zone:
        return 1.0
    exposure = 0.0
    for zone in ("pull", "center", "oppo"):
        share = spray_profile.get(zone)
        d = def_by_zone.get(zone)
        if share is None or d is None:
            continue
        exposure += float(share) * float(d)
    # ~1 run-saved of directional exposure is worth roughly 1.5% of xBA
    return max(0.90, min(1.10, 1.0 - exposure * 0.015))


def contact_gated_hit_prob(batter, pitcher, shapes=None, spray_profile=None,
                           def_by_zone=None, sprint_speed=None, xpa=4.1):
    """P(at least one hit) for a full game.

    The chain, in order:
      1. matchup whiff / K rate  -> how often the PA even produces a ball in play
      2. xBA on contact          -> how often that ball becomes a hit
      3. directional defense     -> who is standing where he hits it
      4. sprint speed            -> infield singles on weak contact
      5. expected PAs            -> converts a per-PA rate into a game probability
    """
    # --- 1. contact gate -------------------------------------------------
    if shapes is not None and shapes.ok():
        whiff = shapes.whiff_rate()
        k_rate = shapes.k_rate()
        xba_con = shapes.xba_on_contact()
    else:
        whiff = _log5((batter or {}).get("whiff"), (pitcher or {}).get("whiff"), LG_WHIFF)
        k_rate = _log5((batter or {}).get("k_pct"), (pitcher or {}).get("k_pct"), 0.225)
        xba_con = _log5((batter or {}).get("xba_con"), (pitcher or {}).get("xba_con"), LG_XBA_CON)

    whiff = LG_WHIFF if whiff is None else float(whiff)
    k_rate = 0.225 if k_rate is None else float(k_rate)
    xba_con = LG_XBA_CON if xba_con is None else float(xba_con)

    # share of PAs that end with a ball in play. Walks are removed too: a walk is a fine
    # outcome for the hitter and a losing one for a 1+ hit ticket.
    bb_rate = _log5((batter or {}).get("bb_pct"), (pitcher or {}).get("bb_pct"), 0.085)
    bip_rate = max(0.05, 1.0 - k_rate - (bb_rate or 0.085))

    # --- 2/3. contact quality, adjusted for who is fielding it ------------
    xba_eff = xba_con * directional_defense(spray_profile, def_by_zone)

    # --- 4. legs: infield singles on weak contact ------------------------
    if sprint_speed is not None:
        try:
            xba_eff += max(-0.010, min(0.018, (float(sprint_speed) - 27.0) * 0.006))
        except Exception:
            pass

    p_hit_per_pa = max(0.001, min(0.60, bip_rate * xba_eff))

    # --- 5. per-PA -> per-game -------------------------------------------
    # Fractional PAs are handled by splitting the expectation between floor and ceiling rather
    # than rounding, so a 4.6-PA leadoff hitter isn't quietly treated as 4 or 5.
    n_lo = int(np.floor(xpa))
    frac = float(xpa) - n_lo
    q = 1.0 - p_hit_per_pa
    p_none = (q ** n_lo) * ((1 - frac) + frac * q)
    return {
        "p_hit": round(1.0 - p_none, 4),
        "p_hit_per_pa": round(p_hit_per_pa, 4),
        "whiff": round(whiff, 4),
        "k_rate": round(k_rate, 4),
        "bip_rate": round(bip_rate, 4),
        "xba_con": round(xba_con, 4),
        "xba_eff": round(xba_eff, 4),
        "xpa": round(float(xpa), 2),
        "shapes": shapes.detail() if (shapes is not None and shapes.ok()) else None,
    }


# ---------------------------------------------------------------------------
# HRR: volume first, then context
# ---------------------------------------------------------------------------

# Expected plate appearances by lineup spot, for a league-average team. The top of the order
# gets nearly a full extra PA over the bottom, which is a bigger driver of HRR volume than any
# rate stat — and it is the part most models skip.
XPA_BY_SPOT = {1: 4.65, 2: 4.55, 3: 4.44, 4: 4.34, 5: 4.23,
               6: 4.12, 7: 4.02, 8: 3.91, 9: 3.81}
LG_IMPLIED_TOTAL = 4.45      # league-average implied team runs


def expected_pa(lineup_spot, implied_team_total=None):
    """xPA for a spot, scaled by the team's Vegas implied run total.

    A team projected for 6 runs bats more often than one projected for 3 — more baserunners
    means more turns through the order. Using the implied total rather than a season average
    also folds in the opposing starter, the park and the weather for free, since the market has
    already priced all of it.
    """
    base = XPA_BY_SPOT.get(int(lineup_spot or 5), 4.2)
    if implied_team_total:
        scale = 1.0 + (float(implied_team_total) - LG_IMPLIED_TOTAL) * 0.045
        base *= max(0.88, min(1.14, scale))
    return base


def rbi_context(batters_ahead):
    """RBI opportunity from the two hitters directly AHEAD in the order.

    Uses trailing-30-day OBP, because driving in runs requires runners on base, and OBP is the
    single best measure of how often the men in front are standing there.
    Returns a multiplier around 1.0.
    """
    obps = [b.get("obp_30") for b in (batters_ahead or []) if b and b.get("obp_30") is not None]
    if not obps:
        return 1.0
    avg = sum(obps) / len(obps)
    return max(0.85, min(1.18, 1.0 + (avg - 0.320) * 1.6))


def run_context(batters_behind):
    """Run-scoring chance from the two hitters directly BEHIND in the order.

    Scoring a run after reaching base requires someone to drive you in, so this uses trailing
    30-day wOBA and SLG rather than OBP — a walk behind you advances you one base, an extra-base
    hit scores you from first.
    """
    vals = []
    for b in (batters_behind or []):
        if not b:
            continue
        w, s = b.get("woba_30"), b.get("slg_30")
        if w is not None and s is not None:
            vals.append(0.6 * float(w) + 0.4 * (float(s) * 0.42))
        elif w is not None:
            vals.append(float(w))
    if not vals:
        return 1.0
    avg = sum(vals) / len(vals)
    return max(0.85, min(1.18, 1.0 + (avg - 0.320) * 1.8))


def hrr_projection(hit_result, lineup_spot, implied_team_total=None,
                   batters_ahead=None, batters_behind=None, hr_rate_per_pa=0.032):
    """Expected hits + runs + RBIs.

    Volume comes first — xPA scaled by the implied total — and context multiplies it. That
    ordering is deliberate: no amount of lineup context rescues a hitter who only gets three
    trips, which is why bottom-of-the-order HRR tickets underperform their rate stats.
    """
    xpa = expected_pa(lineup_spot, implied_team_total)
    p_hit_pa = hit_result.get("p_hit_per_pa", 0.22)

    exp_hits = xpa * p_hit_pa
    # reaching base at all drives both runs and RBIs
    on_base_pa = min(0.60, p_hit_pa + 0.085)
    exp_runs = xpa * on_base_pa * 0.30 * run_context(batters_behind)
    exp_rbi = xpa * (0.115 * rbi_context(batters_ahead))
    exp_rbi += xpa * hr_rate_per_pa                      # a HR always drives in at least itself

    total = exp_hits + exp_runs + exp_rbi
    return {
        "xpa": round(xpa, 2),
        "exp_hits": round(exp_hits, 2),
        "exp_runs": round(exp_runs, 2),
        "exp_rbi": round(exp_rbi, 2),
        "hrr": round(total, 2),
        "rbi_ctx": round(rbi_context(batters_ahead), 3),
        "run_ctx": round(run_context(batters_behind), 3),
    }
