"""Pitcher strikeout engine — CSW true-talent, catcher framing, dynamic xBF, arsenal-depth TTOP.

WHY CSW% RATHER THAN K%. Strikeout rate is an OUTCOME; CSW% (called strikes + whiffs, per
pitch) is the process that produces it. CSW stabilises in roughly 50-100 pitches where K% needs
several starts, so on a 3-start sample CSW is measuring skill while K% is still measuring luck
and sequencing. Weighting CSW over trailing K% is the difference between reading a pitcher and
reading his last three box scores.

WHY THE OLD ENGINE RAN HOT. It used a static 22 batters faced for everyone. A pitcher on a
75-pitch leash averaging 4.3 pitches per PA faces 17 batters, not 22 — a 26% overstatement
before any rate question is asked. That constant was the largest single source of inflated
strikeout projections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LG_CSW = 0.285          # league-average called-strike + whiff rate, per pitch
LG_K_PCT = 0.225        # per PA
LG_P_PER_PA = 3.90

# CSW -> K%. Every point of CSW above average is worth roughly 1.35 points of K%. Kept as an
# explicit constant so it can be re-fit against graded data rather than buried in a formula.
CSW_TO_K_SLOPE = 1.35

# TTO penalty scaled by arsenal depth. A two-pitch starter has nothing new to show a lineup the
# third time through; a five-pitch starter can re-sequence. The published league-average effect
# (~.017 wOBA per turn) is the MIDDLE of this range, not a value to apply to everyone.
TTOP_BY_DEPTH = {
    2: (0.000, 0.030, 0.058),    # two-pitch: severe
    3: (0.000, 0.017, 0.034),    # league average
    4: (0.000, 0.012, 0.024),    # deep mix: muted
}

VELO_DROP_TRIGGER = 1.5          # mph off the season baseline
VELO_DROP_K_PENALTY = 0.90       # multiply K% by this when triggered


def csw_from_statcast(df, pitcher_ids, days=None, asof=None):
    """{pid: {csw, called, swstr, pitches, p_per_pa}} — the true-talent process metric.

    Also returns pitches-per-PA, which the dynamic xBF calculation needs. Computing both from
    the same slice keeps them consistent: an arm's leash and his efficiency are the same story.
    """
    need = {"pitcher", "description"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df
    if days and asof and "game_date" in d.columns:
        cut = pd.to_datetime(asof) - pd.Timedelta(days=days)
        d = d[pd.to_datetime(d["game_date"], errors="coerce") >= cut]
    if d.empty:
        return {}
    CALLED = {"called_strike"}
    WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
    wanted = {int(p) for p in pitcher_ids if p is not None}
    out = {}
    for pid, g in d.groupby("pitcher"):
        if int(pid) not in wanted:
            continue
        n = len(g)
        if n < 100:                       # below this CSW is still noise
            continue
        desc = g["description"].astype(str)
        called = int(desc.isin(CALLED).sum())
        whiff = int(desc.isin(WHIFF).sum())
        pa = int(g["events"].notna().sum()) if "events" in g.columns else 0
        out[int(pid)] = {
            "csw": round((called + whiff) / n, 4),
            "called": round(called / n, 4),
            "swstr": round(whiff / n, 4),
            "pitches": n,
            "p_per_pa": round(n / pa, 2) if pa else LG_P_PER_PA,
            "pa": pa,   # exposed so predict_pitcher_k_count can scale the Stuff+/K-BB% nudge
                        # down as the trailing sample grows — see that function's docstring.
        }
    return out


def catcher_framing(season=None, min_called="q"):
    """{catcher_id: framing_runs} from Statcast's framing leaderboard. Non-fatal.

    Framing is a real and under-priced strikeout input: the identical pitch in the identical
    location is a strike or a ball depending on who receives it, and an elite framer is worth
    roughly a point of CSW to every arm he catches.
    """
    import sys as _s
    import datetime as _dt
    yr = season or _dt.date.today().year
    df = None
    try:
        from pybaseball import statcast_catcher_framing
        df = statcast_catcher_framing(yr, min_called)
    except Exception as e:
        print(f"[framing] pybaseball fetch failed, trying direct request: {e}", file=_s.stderr)
    if df is None or df.empty:
        try:
            import io
            import requests
            url = (f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
                   f"?type=catcher&seasonStart={yr}&seasonEnd={yr}&team=&min={min_called}"
                   f"&csv=true")
            headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/124.0.0.0 Safari/537.36")}
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            # The prior fix (on_bad_lines="skip" alone, default C engine) still tripped on this
            # feed in production ("Expected 1 fields in line 38, saw 4") — the C parser's
            # bad-line recovery is less forgiving than the python engine's for some ragged-row
            # shapes. engine="python" + skipinitialspace=True is strictly more permissive and
            # is applied on top of, not instead of, on_bad_lines="skip".
            df = pd.read_csv(io.StringIO(resp.text), sep=",", engine="python",
                             on_bad_lines="skip", skipinitialspace=True)
        except Exception as e2:
            print(f"[framing] direct request also failed (non-fatal): {e2}", file=_s.stderr)
            return {}
    if df is None or df.empty:
        return {}
    try:
        cols = {c.lower(): c for c in df.columns}
        idc = next((cols[c] for c in ("player_id", "catcher", "playerid") if c in cols), None)
        runc = next((cols[c] for c in
                     ("runs_extra_strikes", "framing_runs", "runs") if c in cols), None)
        if not idc or not runc:
            return {}
        out = {}
        for _, r in df.iterrows():
            try:
                out[int(r[idc])] = float(r[runc])
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[framing] catcher framing unavailable (non-fatal): {e}", file=_s.stderr)
        return {}


def framing_csw_adjust(csw, framing_runs):
    """Shift a pitcher's CSW by his catcher's framing skill.

    ~10 framing runs over a season maps to about 1 point of CSW for the staff he catches.
    Clamped hard: framing is real, but it is not worth more than a point in either direction,
    and an unclamped adjustment would let a catcher outweigh the pitcher.
    """
    if csw is None or framing_runs is None:
        return csw
    delta = max(-0.012, min(0.012, float(framing_runs) * 0.0011))
    return max(0.18, min(0.42, float(csw) + delta))


def arsenal_depth(arsenal, viable_usage=10.0, dominance=85.0):
    """Count genuinely viable pitches, for TTOP scaling.

    An arm whose top two pitches account for >= `dominance` percent of everything he throws is
    a two-pitch pitcher regardless of how many show-me offerings appear in the log — a lineup
    stops respecting a 4%-usage curveball by the third look.
    """
    if not arsenal:
        return 3
    usages = sorted((float(a[1]) for a in arsenal), reverse=True)
    if len(usages) >= 2 and (usages[0] + usages[1]) >= dominance:
        return 2
    return max(2, min(4, sum(1 for u in usages if u >= viable_usage)))


def tto_penalty_for(depth, turn):
    """wOBA-allowed penalty for a given time through the order, scaled by arsenal depth."""
    table = TTOP_BY_DEPTH.get(int(depth), TTOP_BY_DEPTH[3])
    return table[min(int(turn), len(table) - 1)]


def dynamic_xbf(pitch_limit, p_per_pa, opener=False):
    """xBF = trailing 3-start pitch limit / pitches per PA.

    Replaces the static 22. A 105-pitch workhorse at 3.7 P/PA faces 28 batters; a 75-pitch arm
    at 4.3 faces 17. Treating those as identical is the biggest error in a K projection.
    """
    if opener:
        return 6.0
    pl = float(pitch_limit) if pitch_limit else 88.0
    ppa = max(3.2, min(4.8, float(p_per_pa) if p_per_pa else LG_P_PER_PA))
    return max(6.0, min(30.0, pl / ppa))


def pitch_limit_from_starts(df, pid, n_starts=3):
    """The leash, measured rather than assumed: mean pitch count over recent starts."""
    if df is None or df.empty or "pitcher" not in df.columns:
        return None
    g = df[df["pitcher"] == pid]
    if g.empty or "game_pk" not in g.columns:
        return None
    counts = g.groupby("game_pk").size().sort_index()
    counts = counts[counts >= 30]                 # real starts, not relief cameos
    return float(counts.tail(n_starts).mean()) if len(counts) else None


def velocity_flag(df, pid, asof, days=14):
    """(drop_mph, triggered) — trailing fastball velo vs the season baseline."""
    if df is None or df.empty or "pitcher" not in df.columns:
        return None, False
    g = df[df["pitcher"] == pid]
    if g.empty or "release_speed" not in g.columns:
        return None, False
    if "pitch_type" in g.columns:
        g = g[g["pitch_type"].isin({"FF", "SI", "FA", "FT"})]
    if g.empty:
        return None, False
    v = pd.to_numeric(g["release_speed"], errors="coerce")
    season = float(v.mean())
    if "game_date" in g.columns:
        cut = pd.to_datetime(asof) - pd.Timedelta(days=days)
        recent = v[pd.to_datetime(g["game_date"], errors="coerce") >= cut]
    else:
        recent = v.tail(200)
    if len(recent) < 40:
        return None, False
    drop = season - float(recent.mean())
    return round(drop, 2), bool(drop >= VELO_DROP_TRIGGER)


def _log5(b, p, lg):
    """Tango's log5 odds ratio — rates compound, they do not average."""
    if b is None or p is None or not lg:
        return b if b is not None else lg
    b = min(0.95, max(0.01, float(b)))
    p = min(0.95, max(0.01, float(p)))
    g = min(0.95, max(0.01, float(lg)))
    num = b * p / g
    den = num + (1 - b) * (1 - p) / (1 - g)
    return num / den if den else b


def predict_pitcher_k_count(pitcher_csw, lineup_k_rates, arsenal=None, framing_runs=None,
                            pitch_limit=None, velo_drop=None, opener=False,
                            trailing_k_pct=None, stuff_plus=None, k_bb_pct=None):
    """Expected strikeouts plus the exact distribution.

    Order of operations is deliberate:
      1. framing shifts CSW  (it changes the PROCESS, so it must land before the conversion)
      2. CSW -> true-talent K%, with trailing K% keeping a minority vote
      3. Stuff+ / K-BB% apply a BOUNDED minority nudge (see below)
      4. velocity override can veto everything above it
      5. dynamic xBF sets how many chances exist
      6. each PA is matched by log5 against that specific hitter, with TTOP by arsenal depth

    stuff_plus / k_bb_pct are OPTIONAL FanGraphs season-long signals, applied as small
    multiplicative nudges on top of k_base — never touching LG_CSW, CSW_TO_K_SLOPE, dynamic_xbf,
    or the exact Poisson-binomial distribution below, all of which stay exactly as calibrated.
    Each nudge is capped at +/-5%, and BOTH capped nudges are further scaled down as trailing_k_pct
    sample size grows — the whole point of these two is to sharpen a THIN sample's true-talent
    read, not to override a pitcher who already has a real trailing track record this season.
    """
    csw = (pitcher_csw or {}).get("csw")
    p_per_pa = (pitcher_csw or {}).get("p_per_pa", LG_P_PER_PA)
    trailing_pa = (pitcher_csw or {}).get("pa")   # optional, used only to scale the nudge

    csw_adj = framing_csw_adjust(csw, framing_runs)

    if csw_adj is not None:
        k_from_csw = LG_K_PCT + (csw_adj - LG_CSW) * CSW_TO_K_SLOPE
        # trailing K% still votes: it carries putaway-sequencing information CSW misses
        k_base = 0.70 * k_from_csw + 0.30 * (trailing_k_pct if trailing_k_pct is not None
                                             else k_from_csw)
    else:
        k_base = trailing_k_pct if trailing_k_pct is not None else LG_K_PCT
    k_base = max(0.08, min(0.45, k_base))

    # Stuff+ / K-BB% — bounded minority nudges, capped +/-5% total and scaled down as the
    # trailing-K% sample grows. `stuff_plus` is preferred when both are present, since it is the
    # more direct swing-and-miss signal; `k_bb_pct` (K% minus BB%, a FanGraphs season rate) is
    # the fallback command-adjusted read when Stuff+ specifically is unavailable. They are NOT
    # stacked — using both at once would double-count two metrics that are already correlated,
    # the same anti-double-counting principle applied elsewhere in this codebase (e.g. targets.py
    # keeping the pen's own HR/9 use separate from the staff-level HR/9 exponent term).
    _sample_w = 1.0
    if trailing_pa is not None:
        # full weight under 60 PA of trailing data, tapering to near-zero by 200 PA — a pitcher
        # with a real season's worth of trailing Ks does not need FanGraphs to tell him apart.
        _sample_w = max(0.0, min(1.0, 1.0 - (float(trailing_pa) - 60.0) / 140.0))
    _nudge = 0.0
    if stuff_plus is not None:
        _nudge = max(-0.05, min(0.05, (float(stuff_plus) - 100.0) * 0.0025))
    elif k_bb_pct is not None:
        # league K-BB% sits near 15%; every point above/below nudges k_base proportionally
        _nudge = max(-0.05, min(0.05, (float(k_bb_pct) - 15.0) * 0.003))
    if _nudge:
        k_base = max(0.08, min(0.45, k_base * (1.0 + _nudge * _sample_w)))

    velo_triggered = bool(velo_drop is not None and velo_drop >= VELO_DROP_TRIGGER)
    if velo_triggered:
        k_base *= VELO_DROP_K_PENALTY

    xbf = dynamic_xbf(pitch_limit, p_per_pa, opener=opener)

    depth = arsenal_depth(arsenal)
    rates = list(lineup_k_rates or [LG_K_PCT] * 9)
    n = max(1, len(rates))
    exp_k, per_pa, faced = 0.0, [], 0.0
    while faced < xbf:
        i = int(faced) % n
        turn = int(faced // n)
        # TTO raises contact quality, i.e. SUPPRESSES strikeouts — convert the wOBA-scale
        # penalty into a proportional K% haircut.
        haircut = max(0.60, 1.0 - tto_penalty_for(depth, turn) * 5.0)
        k_matchup = _log5(rates[i], k_base, LG_K_PCT) * haircut
        step = min(1.0, xbf - faced)
        exp_k += k_matchup * step
        per_pa.append(k_matchup)
        faced += 1.0

    return {
        "exp_k": round(exp_k, 2),
        "xbf": round(xbf, 1),
        "k_base": round(k_base, 4),
        "csw": round(csw, 4) if csw is not None else None,
        "csw_adj": round(csw_adj, 4) if csw_adj is not None else None,
        "framing_runs": framing_runs,
        "arsenal_depth": depth,
        "velo_drop": velo_drop,
        "velo_flag": velo_triggered,
        "stuff_plus": stuff_plus, "k_bb_pct": k_bb_pct,
        "dist": k_distribution(per_pa),
    }


def k_distribution(per_pa_rates):
    """Exact Poisson-binomial over the individual PA strikeout probabilities.

    Deliberately NOT Poisson. The PAs have different probabilities — a #9 hitter is not the
    leadoff man — and a Poisson assumes they are identical. That assumption is wrong exactly
    in the tails, which is where the over/under lines sit.
    """
    if not per_pa_rates:
        return {}
    dist = np.zeros(len(per_pa_rates) + 1)
    dist[0] = 1.0
    for p in per_pa_rates:
        p = max(0.0, min(1.0, float(p)))
        nxt = np.zeros_like(dist)
        nxt[0] = dist[0] * (1 - p)
        for k in range(1, len(dist)):
            nxt[k] = dist[k] * (1 - p) + dist[k - 1] * p
        dist = nxt
    return {int(k): round(float(v), 5) for k, v in enumerate(dist) if v >= 1e-4}


def prob_over_ks(dist, line):
    """P(strikeouts > line) from the exact distribution."""
    if not dist:
        return None
    return round(sum(p for k, p in dist.items() if k > line), 4)
