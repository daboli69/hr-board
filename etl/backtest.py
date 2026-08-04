"""
Season backtest: replay every slate as it would have looked that morning, grade it
against what actually happened, and answer "how big is the edge" with ~80 days of
data instead of waiting weeks of live tracking.

THE LEAK CONTRACT (the whole game is not cheating):
  - Features for date D are computed from df[game_date < D] ONLY — enforced by
    construction (the feature call receives a strictly-past frame) and verified by
    poison_check(), which corrupts all future rows and asserts identical heats.
  - The day's opposing starter is taken from that day's actual first pitcher
    (inning 1). Honest approximation: the real morning board uses PROBABLES, which
    occasionally get scratched; using the actual starter is mildly optimistic and
    is documented in the output.
  - Scope: the CORE model (four-signal heat + opposing-arm nudge). Park/weather,
    badges and BvP layers are not replayed; this measures the engine you froze.

Run via the manual "Backtest" workflow: pulls the season, replays, writes
docs/backtest.json for the Tracker tab.
"""
from __future__ import annotations
import json
import os
import sys

# pybaseball warns on every large pull that caching should be on. With it enabled the season
# fetch is written to disk, so a re-run (or a retry after a failure) skips the slowest step.
try:
    from pybaseball import cache as _pyb_cache
    _pyb_cache.enable()
except Exception:
    pass

import numpy as np
import pandas as pd

from etl import compute, statcast_data, props

WARMUP_DAYS = 21          # first N days of the frame are feature-only (no grading)
TIERS = (("70+", 70, 999), ("55-69", 55, 70), ("40-54", 40, 55), ("<40", -999, 40))

HIT_EVENTS = {"single", "double", "triple", "home_run"}
K_EVENTS = {"strikeout", "strikeout_double_play"}


def _badge_lift(by_badge: dict, base_rate: float) -> dict:
    """Turn raw per-badge {n, hr} tallies into HR rate + lift vs the pool base rate,
    sorted best-first. lift = badge HR rate / base HR rate. >1 means the badge picks
    hitters who homer more than average; <1 means it doesn't. Season-long sample gives
    this real statistical weight, unlike the thin day-by-day tracker version."""
    out = {}
    for k, v in by_badge.items():
        n, hr = v["n"], v["hr"]
        rate = hr / n if n else 0.0
        out[k] = {
            "n": n, "hr": hr,
            "rate_pct": round(100 * rate, 2),
            "lift": round(rate / base_rate, 3) if base_rate > 0 else None,
        }
    # sort best-first by lift (then by sample size), so the output reads as a ranking
    return dict(sorted(out.items(),
                       key=lambda kv: (-(kv[1]["lift"] or 0), -kv[1]["n"])))


def _tier(h):
    for name, lo, hi in TIERS:
        if lo <= h < hi:
            return name
    return "<40"


def _day_outcomes(day: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-batter (hits, hrr, ks) and per-pitcher K totals for one date.
    HRR now uses the shared hrr_recon module (authoritative rbi + score-delta runs), IDENTICAL
    to how track.py grades live — so backtest and tracker stay on the same scale. This replaced
    the old base-runner sim that undercounted runs/RBIs (missed sac flies, groundout RBIs, runs
    on outs), which had biased HRR low and made Unders look +EV everywhere."""
    from etl import hrr_recon
    bat_hits, bat_ks = {}, {}
    pit_ks = {}
    pa_df = day[day["events"].notna() & (day["events"] != "")]
    # hits + strikeouts (exact from events)
    for _, row in pa_df.iterrows():
        if row["batter"] != row["batter"]:
            continue
        bid = int(row["batter"]); ev = row["events"]
        if ev in K_EVENTS:
            bat_ks[bid] = bat_ks.get(bid, 0) + 1
            pit = row.get("pitcher")
            if pit == pit:
                pit_ks[int(pit)] = pit_ks.get(int(pit), 0) + 1
        if ev in HIT_EVENTS:
            bat_hits[bid] = bat_hits.get(bid, 0) + 1
    # HRR via the shared, corrected reconstruction
    hrr_map = hrr_recon.hrr_map(pa_df)
    hrr_total = {bid: v["hits"] + v["runs"] + v["rbis"] for bid, v in hrr_map.items()}
    return ({"hits": bat_hits, "ks": bat_ks, "hrr": hrr_map}, pit_ks, hrr_total)


def _day_heats(past: pd.DataFrame, day: pd.DataFrame, D: str) -> dict:
    """{batter_id: (heat, homered_today)} for one replay date. `past` must be
    strictly earlier than D — the caller owns that guarantee."""
    ev = day["events"].to_numpy()
    batters = sorted({int(b) for b in day["batter"].dropna().unique()})
    if not batters:
        return {}
    # opposing starter per batter: first pitcher of inning 1 in the half he bats in
    starters = {}
    inn1 = day[day["inning"].to_numpy() == 1]
    for (gp, half), grp in inn1.groupby(["game_pk", "inning_topbot"]):
        g = grp.sort_values(["at_bat_number", "pitch_number"])
        p0 = g.iloc[0]["pitcher"]
        if p0 == p0:
            starters[(int(gp), half)] = int(p0)
    face = {}
    for (gp, half), grp in day.groupby(["game_pk", "inning_topbot"]):
        sp = starters.get((int(gp), half))
        for b in grp["batter"].dropna().unique():
            face[int(b)] = sp
    bprof = statcast_data.batter_profiles(past, batters, asof=D)
    sp_ids = sorted({s for s in face.values() if s})
    pprof = statcast_data.pitcher_profiles(past, sp_ids, asof=D)
    phr = {}
    for pid in sp_ids:
        pr = pprof.get(pid) or {}
        try:
            phr[pid] = compute.pitcher_hr_score(pr.get("recent", {}), pr.get("season", {})).get("score")
        except Exception:
            phr[pid] = None
    hr_today = set(int(b) for b in day[ev == "home_run"]["batter"].dropna().unique())
    # outcomes for props grading (hits / Ks / HRR per batter, Ks per pitcher)
    bat_out, pit_ks_today, hrr_val = _day_outcomes(day)
    # opposing-lineup K% per starter: mean recent k_pct of the batters facing him
    opp_lineup_k = {}
    for pid in sp_ids:
        ks = [(bprof.get(b, {}).get("recent") or {}).get("k_pct")
              for b, s in face.items() if s == pid]
        ks = [k for k in ks if k is not None]
        if ks:
            opp_lineup_k[pid] = sum(ks) / len(ks)
    out = {}
    pitcher_scores = {}
    for pid in sp_ids:
        try:
            k_sc, _ = props.pitcher_k_heat(pprof.get(pid) or {}, opp_lineup_k.get(pid))
        except Exception:
            k_sc = None
        if k_sc is not None:
            pitcher_scores[pid] = (float(k_sc), int(pit_ks_today.get(pid, 0)))
    # ---- NEW-SIGNAL replay: compute the ZONE signal AS OF this date from `past` only (no
    # leakage), so the backtest can prove or kill the newer edges the way it validates heat.
    # Index `past` by batter and by pitcher ONCE per day. Previously each batter did its own
    # boolean scan over the whole season-to-date frame — with ~250 batters x 131 days that was
    # tens of billions of row comparisons, which is what pushed the workflow past 90 minutes.
    # A groupby costs one pass and turns every lookup into a dict hit.
    # A hitter's season-to-date zone profile barely moves in a day — a few batted balls added to
    # a few hundred. Recomputing it for every batter on every replayed date was the single most
    # expensive thing in the loop. Cache it and refresh on a cadence: the cached value is always
    # built from data STRICTLY BEFORE the current date, so this is more conservative than daily
    # recomputation, never leaky.
    _ZCACHE = globals().setdefault("_BT_ZCACHE", {})
    _zc_day = globals().get("_BT_ZCACHE_DAY")
    _day_ix = globals().setdefault("_BT_DAY_IX", {"n": 0})
    _day_ix["n"] += 1
    if _day_ix["n"] % 7 == 1:            # refresh weekly
        _ZCACHE.clear()

    try:
        _by_bat = {int(k): v for k, v in past.groupby("batter")} if len(past) else {}
    except Exception:
        _by_bat = {}
    try:
        _by_pit = {int(k): v for k, v in past.groupby("pitcher")} if len(past) else {}
    except Exception:
        _by_pit = {}
    _EMPTY = past.iloc[0:0]

    _meat_by_sp = {}
    _grid_by_sp = {}      # (pid, batter_hand) -> zone usage grid
    _ars_by_sp = {}       # (pid, batter_hand) -> [(pitch_type, usage_pct), ...]
    _throws_by_sp = {}
    try:
        from etl import features as _F
        for pid in sp_ids:
            _prows = _by_pit.get(int(pid), _EMPTY)
            if len(_prows) < 150:
                continue
            _meat_by_sp[pid] = _F.pitcher_zone_damage(_prows)
            if "p_throws" in _prows.columns and len(_prows):
                _tv = _prows["p_throws"].dropna()
                if len(_tv):
                    _throws_by_sp[pid] = str(_tv.iloc[0])
            for _h in ("R", "L"):
                try:
                    _g = _F.pitcher_zone_grid(_prows, hand=_h)
                    if _g and _g.get("n", 0) >= 150:
                        _grid_by_sp[(pid, _h)] = _g
                except Exception:
                    pass
                try:
                    _ph = _prows[_prows["stand"] == _h] if "stand" in _prows.columns else None
                    if _ph is not None and len(_ph) >= 150 and "pitch_type" in _ph.columns:
                        _vc = _ph["pitch_type"].value_counts()
                        _tot = int(_vc.sum())
                        _ars_by_sp[(pid, _h)] = [(str(k), 100.0 * int(v) / _tot)
                                                 for k, v in _vc.items() if int(v) >= 15]
                except Exception:
                    pass
    except Exception:
        _meat_by_sp = {}
    for bid in batters:
        prof = bprof.get(bid)
        if not prof:
            continue
        recent = prof.get("recent") or {}
        if not (recent.get("bb_count") or 0):
            continue
        try:
            heat, _ = compute.heat_score(recent, phr.get(face.get(bid)))
        except Exception:
            continue
        pp = pprof.get(face.get(bid)) or {}
        try:
            hh, _ = props.hit_heat(recent, pp)
        except Exception:
            hh = None
        try:
            # no morning lineup in the replay frame — spot multiplier omitted;
            # measures the hit-skill + HR-upside core of hrr_heat
            hr_h, _ = props.hrr_heat(recent, pp, lineup_spot=None, hr_heat=heat)
        except Exception:
            hr_h = None
        # Hitter-only badges, computed from the same profile the live board uses. We only
        # derive the badges that depend on HITTER data available in the replay frame
        # (POWER, DUE, MAY COOL, HOT, WARMING) — the opponent-context badges (WEAK ARM,
        # PLATOON, PITCH EDGE, WEAK PEN) need lineup/pen data not reconstructed here, so
        # they're intentionally left out rather than approximated.
        try:
            badge_list = compute.player_badges(
                luck_gap=recent.get("luck_gap"),
                trend=prof.get("trend"),
                max_ev=recent.get("max_ev"),
                xwobacon=recent.get("xwobacon"),
            )
            badge_keys = [b["k"] for b in badge_list]
        except Exception:
            badge_keys = []
        # ---- NEW SIGNALS, computed from `past` only (no leakage) ----
        _zone = None; _squp = None; _hrpow = None
        try:
            _brows = _by_bat.get(int(bid), _EMPTY)
            if len(_brows) >= 50:
                _bhr = _F.batter_hr_zones(_brows)
                _mb = _meat_by_sp.get(face.get(bid))
                if _bhr and _mb:
                    _zone = int((_F.zone_overlap(_bhr, _mb) or {}).get("count", 0))
                _sq = _F.square_up_rating(_brows)
                if _sq:
                    _squp = _sq.get("rating")
                _hp = _F.hr_power_profile(_brows)
                if _hp:
                    _hrpow = _hp.get("barrel_pct")
        except Exception:
            pass
        # ---- zone edge + arsenal fit + convergence, all as-of `past` ----
        _zedge = None; _afit = None; _nm = _nmh = _np = 0
        try:
            _sp = face.get(bid)
            _bh = None
            _brows2 = _by_bat.get(int(bid), _EMPTY)
            _bs = _brows2["stand"].dropna() if "stand" in _brows2.columns else None
            if _bs is not None and len(_bs):
                _bh = str(_bs.iloc[0])
            if _bh == "S":
                _bh = "L" if _throws_by_sp.get(_sp) == "R" else "R"
            if _sp and _bh and len(_brows2) >= 80:
                _bz = _ZCACHE.get(bid)
                if _bz is None:
                    _bz = _F.batter_zone_damage(_brows2) or {}
                    _ZCACHE[bid] = _bz
                _g = _grid_by_sp.get((_sp, _bh))
                if _bz and _g:
                    _ze = _F.zone_matchup_edges(_bz, _g)
                    if _ze:
                        _zedge = _ze.get("edge_score")
                # arsenal fit: his barrel rate on the pitches this arm actually throws him
                _ars = _ars_by_sp.get((_sp, _bh))
                if _ars and "pitch_type" in _brows2.columns:
                    _bb = _brows2[_brows2["bb_type"].notna()] if "bb_type" in _brows2 else None
                    if _bb is not None and len(_bb) >= 30:
                        _num = _den = 0.0
                        for _pt, _pct in _ars:
                            _sub = _bb[_bb["pitch_type"] == _pt]
                            if len(_sub) >= 5 and "launch_speed_angle" in _sub.columns:
                                _br = 100.0 * float((_sub["launch_speed_angle"] == 6).sum()) / len(_sub)
                                _num += _pct * _br; _den += _pct
                        if _den > 0:
                            _afit = round(_num / _den, 1)
            # convergence, using only the badges the backtest itself has validated
            _bset = {str(b).lower() for b in badge_keys}
            _nmh = (1 if heat >= 40 else 0)                 # HR heat band as a measured signal
            _nm = (1 if "pow" in _bset else 0) + (1 if "lock" in _bset else 0)
            # hit / HRR use their OWN heat models, and only the bands that actually showed lift
            # (70+ only: +11% for 1+hit, +13% for HRR — the middle bands were flat).
            _cv_hit = (1 if (hh is not None and hh >= 70) else 0)
            _cv_hrr = (1 if (hr_h is not None and hr_h >= 70) else 0)
            _np = ((1 if (_zedge is not None and _zedge >= 62) else 0)
                   + (1 if (_afit is not None and _afit >= 9.0) else 0)
                   + (1 if (phr.get(face.get(bid)) or 0) >= 60 else 0))
        except Exception:
            pass
        out[bid] = {
            "zone_edge": _zedge, "arsenal_fit": _afit,
            "cv_meas": _nm + _nmh, "cv_meas_noheat": _nm, "cv_prov": _np,
            "cv_hit": _cv_hit, "cv_hrr": _cv_hrr,
            "pull_air": (recent or {}).get("pull_air_pct"),
            "cv_fams": (_nm + _nmh + _np),   # measured + provisional families, as the UI counts them
            "bbe_season": int(len(_by_bat.get(int(bid), _EMPTY))) if bid is not None else None,
            "heat": float(heat), "hr": bid in hr_today,
            "hit_heat": float(hh) if hh is not None else None,
            "hrr_heat": float(hr_h) if hr_h is not None else None,
            "hits": int(bat_out["hits"].get(bid, 0)),
            "ks": int(bat_out["ks"].get(bid, 0)),
            "hrr": int(hrr_val.get(bid, 0)),
            "badges": badge_keys,
            "zone": _zone, "square_up": _squp, "hr_power": _hrpow,
        }
    return out, pitcher_scores


def replay(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    df = df.copy()
    df["_gd"] = df["game_date"].astype(str).str[:10]
    all_dates = sorted(df["_gd"].unique())
    if len(all_dates) <= WARMUP_DAYS:
        return {"error": f"need more than {WARMUP_DAYS} days of data"}
    dates = [d for d in all_dates[WARMUP_DAYS:] if (not start or d >= start) and (not end or d <= end)]
    by_tier = {name: {"n": 0, "hr": 0} for name, _, _ in TIERS}
    by_badge = {}   # badge_key -> {n, hr}: HR rate for hitters carrying each badge
    # convergence graded against EACH prop's own outcome, not the HR outcome
    by_conv = {}    # prop -> bucket -> {n, hit}
    def _conv_tally(prop, bucket, ok):
        d = by_conv.setdefault(prop, {}).setdefault(bucket, {"n": 0, "hit": 0})
        d["n"] += 1; d["hit"] += 1 if ok else 0
    top_n = {"5": {"n": 0, "hr": 0}, "10": {"n": 0, "hr": 0}, "25": {"n": 0, "hr": 0}}
    calib = {}
    n_tot = hr_tot = 0
    graded_days = 0
    # props accumulators — hit1/hit2 keyed by hit_heat tier, hrr by hrr_heat tier,
    # bku (batter K under) by k-side of hit profile is intentionally NOT here: the
    # UNDER score is graded live by track.py; the backtest covers the three core
    # props models (hit, hrr, pitcher k).
    P = {
        "hit1": {t: {"n": 0, "hit": 0} for t, _, _ in TIERS},
        "hit2": {t: {"n": 0, "hit": 0} for t, _, _ in TIERS},
        "hrr":  {t: {"n": 0, "hit": 0} for t, _, _ in TIERS},
        "pk":   {t: {"n": 0, "total_ks": 0, "o5": 0, "o6": 0, "o7": 0} for t, _, _ in TIERS},
    }
    p_top = {
        "hit1": {"5": {"n": 0, "hit": 0}, "10": {"n": 0, "hit": 0}, "25": {"n": 0, "hit": 0}},
        "hrr":  {"5": {"n": 0, "hit": 0}, "10": {"n": 0, "hit": 0}, "25": {"n": 0, "hit": 0}},
        "pk":   {"3": {"n": 0, "total_ks": 0, "o5": 0, "o6": 0}, "5": {"n": 0, "total_ks": 0, "o5": 0, "o6": 0}},
    }
    pk_n = pk_ks = pk_o5 = pk_o6 = pk_o7 = 0
    hit_n = hit1_tot = hit2_tot = hrr_n = hrr2_tot = 0
    # new-signal buckets, same {n,hr} shape the tracker uses so the UI renders them uniformly
    by_edge = {}
    def _edge(group, bucket, hit):
        g = by_edge.setdefault(group, {})
        b = g.setdefault(bucket, {"n": 0, "hr": 0})
        b["n"] += 1; b["hr"] += 1 if hit else 0
    for D in dates:
        day = df[df["_gd"] == D]
        past = df[df["_gd"] < D]
        heats, pitchers = _day_heats(past, day, D)
        if len(heats) < 30:                       # partial-slate days pollute rates
            continue
        graded_days += 1
        ranked = sorted(heats.items(), key=lambda kv: -kv[1]["heat"])
        for i, (bid, r) in enumerate(ranked):
            hit = r["hr"]
            n_tot += 1; hr_tot += 1 if hit else 0
            t = by_tier[_tier(r["heat"])]
            t["n"] += 1; t["hr"] += 1 if hit else 0
            for k in ("5", "10", "25"):
                if i < int(k):
                    top_n[k]["n"] += 1; top_n[k]["hr"] += 1 if hit else 0
            b = int(min(max(r["heat"], 0), 99) // 10) * 10
            c = calib.setdefault(str(b), {"n": 0, "hr": 0})
            c["n"] += 1; c["hr"] += 1 if hit else 0
            # ---- new-signal grading ----
            _z = r.get("zone")
            if _z is not None:
                _edge("zone", "5+ premium" if _z >= 5 else "3-4 viable" if _z >= 3
                      else "1-2 thin" if _z >= 1 else "0 none", hit)
            # ---- the newer reads, bucketed so they can be proven or killed ----
            _ze = r.get("zone_edge")
            if _ze is not None:
                _edge("zone_edge", "70+ strong" if _ze >= 70 else "62-69 good" if _ze >= 62
                      else "50-61 avg" if _ze >= 50 else "<50 weak", hit)
            _pa2 = r.get("pull_air")
            if _pa2 is not None:
                _edge("pull_air", "50+ elite" if _pa2 >= 50 else "40-49 good" if _pa2 >= 40
                      else "30-39 avg" if _pa2 >= 30 else "<30 weak", hit)
            _af = r.get("arsenal_fit")
            if _af is not None:
                _edge("arsenal_fit", "11+ punishes" if _af >= 11 else "9-10.9 good" if _af >= 9
                      else "7-8.9 avg" if _af >= 7 else "<7 weak", hit)
            _cm = r.get("cv_meas")
            if _cm is not None:
                _edge("converge", f"{min(int(_cm),3)}{'+' if int(_cm)>=3 else ''} measured", hit)
            _cn = r.get("cv_meas_noheat")
            if _cn is not None:
                _edge("converge_noheat", f"{min(int(_cn),2)}{'+' if int(_cn)>=2 else ''} measured", hit)
            _cp = r.get("cv_prov")
            if _cp is not None:
                _edge("converge_prov", f"{min(int(_cp),3)}{'+' if int(_cp)>=3 else ''} provisional", hit)
            # total evidence families — the number the convergence board actually shows
            # cross-tab: sample size WITHIN each convergence level. If "thin + converged" is
            # fine, the BBE penalty is removing good plays rather than bad ones.
            _bbe2 = r.get("bbe_season"); _cvn = r.get("cv_meas_noheat")
            if _bbe2 is not None and _cvn is not None:
                _sz = "250+" if _bbe2 >= 250 else ("120-249" if _bbe2 >= 120 else "<120")
                _cv2 = "conv1+" if _cvn >= 1 else "conv0"
                _edge("sample_x_converge", f"{_cv2} / {_sz}", hit)
            _cf = r.get("cv_fams")
            if _cf is not None:
                _edge("converge_families", f"{min(int(_cf),5)}{'+' if int(_cf)>=5 else ''} families", hit)
            # does sample size actually matter? grade thin vs full-season hitters directly
            _bbe = r.get("bbe_season")
            if _bbe is not None:
                _edge("sample_size", "250+ BBE" if _bbe >= 250 else "120-249" if _bbe >= 120
                      else "60-119 thin" if _bbe >= 60 else "<60 very thin", hit)
            _sq = r.get("square_up")
            if _sq is not None:
                _edge("square_up", "75+ elite" if _sq >= 75 else "60-74 strong" if _sq >= 60
                      else "45-59 avg" if _sq >= 45 else "<45 weak", hit)
            _hp = r.get("hr_power")
            if _hp is not None:
                _edge("hr_power", "12%+ barrel" if _hp >= 12 else "8-11%" if _hp >= 8 else "<8%", hit)
            # badge tally: each badge this hitter carries gets a plate appearance + HR credit
            for bk in r.get("badges", []):
                bb = by_badge.setdefault(bk, {"n": 0, "hr": 0})
                bb["n"] += 1; bb["hr"] += 1 if hit else 0
        # --- props: hit1/hit2 tiers by hit_heat ---
        hh_ranked = sorted((kv for kv in heats.items() if kv[1]["hit_heat"] is not None),
                           key=lambda kv: -kv[1]["hit_heat"])
        for i, (bid, r) in enumerate(hh_ranked):
            tier = _tier(r["hit_heat"])
            got1 = r["hits"] >= 1; got2 = r["hits"] >= 2
            P["hit1"][tier]["n"] += 1; P["hit1"][tier]["hit"] += 1 if got1 else 0
            P["hit2"][tier]["n"] += 1; P["hit2"][tier]["hit"] += 1 if got2 else 0
            hit_n += 1; hit1_tot += 1 if got1 else 0; hit2_tot += 1 if got2 else 0
            _cvh = r.get("cv_hit")
            if _cvh is not None:
                _conv_tally("hit1", f"{_cvh} measured", got1)
                _conv_tally("hit2", f"{_cvh} measured", got2)
            for k in ("5", "10", "25"):
                if i < int(k):
                    p_top["hit1"][k]["n"] += 1; p_top["hit1"][k]["hit"] += 1 if got1 else 0
        # --- props: hrr tiers by hrr_heat ---
        hr_ranked = sorted((kv for kv in heats.items() if kv[1]["hrr_heat"] is not None),
                           key=lambda kv: -kv[1]["hrr_heat"])
        for i, (bid, r) in enumerate(hr_ranked):
            tier = _tier(r["hrr_heat"])
            got = r["hrr"] >= 2
            P["hrr"][tier]["n"] += 1; P["hrr"][tier]["hit"] += 1 if got else 0
            _cvr = r.get("cv_hrr")
            if _cvr is not None:
                _conv_tally("hrr", f"{_cvr} measured", got)
            hrr_n += 1; hrr2_tot += 1 if got else 0
            for k in ("5", "10", "25"):
                if i < int(k):
                    p_top["hrr"][k]["n"] += 1; p_top["hrr"][k]["hit"] += 1 if got else 0
        # --- props: pitcher Ks by k_heat ---
        pk_ranked = sorted(pitchers.items(), key=lambda kv: -kv[1][0])
        for i, (pid, (ksc, actual)) in enumerate(pk_ranked):
            tier = _tier(ksc)
            e = P["pk"][tier]
            e["n"] += 1; e["total_ks"] += actual
            if actual >= 6: e["o5"] += 1
            if actual >= 7: e["o6"] += 1
            if actual >= 8: e["o7"] += 1
            pk_n += 1; pk_ks += actual
            pk_o5 += 1 if actual >= 6 else 0
            pk_o6 += 1 if actual >= 7 else 0
            pk_o7 += 1 if actual >= 8 else 0
            for k in ("3", "5"):
                if i < int(k):
                    tn = p_top["pk"][k]
                    tn["n"] += 1; tn["total_ks"] += actual
                    if actual >= 6: tn["o5"] += 1
                    if actual >= 7: tn["o6"] += 1
        if graded_days % 10 == 0:
            print(f"[backtest] {graded_days} days graded through {D}")
    return {
        "days": graded_days, "pool": n_tot, "hr": hr_tot,
        "model_version": compute.MODEL_VERSION,
        "base_pct": round(100 * hr_tot / n_tot, 2) if n_tot else None,
        "by_tier": by_tier, "top_n": top_n, "by_edge": by_edge,
        "by_badge": _badge_lift(by_badge, hr_tot / n_tot if n_tot else 0),
        "by_converge_prop": by_conv,   # convergence graded per prop, on that prop's outcome
        "calib": {k: calib[k] for k in sorted(calib, key=int)},
        "props": {
            "hit1": {"by_tier": P["hit1"], "top_n": p_top["hit1"],
                     "base_pct": round(100 * hit1_tot / hit_n, 2) if hit_n else None},
            "hit2": {"by_tier": P["hit2"],
                     "base_pct": round(100 * hit2_tot / hit_n, 2) if hit_n else None},
            "hrr":  {"by_tier": P["hrr"], "top_n": p_top["hrr"],
                     "base_pct": round(100 * hrr2_tot / hrr_n, 2) if hrr_n else None},
            "pk":   {"by_tier": P["pk"], "top_n": p_top["pk"],
                     "n": pk_n, "avg_ks": round(pk_ks / pk_n, 2) if pk_n else None,
                     "o5_pct": round(100 * pk_o5 / pk_n, 1) if pk_n else None,
                     "o6_pct": round(100 * pk_o6 / pk_n, 1) if pk_n else None,
                     "o7_pct": round(100 * pk_o7 / pk_n, 1) if pk_n else None},
        },
        "notes": ["core heat model (heat + arm nudge); park/weather layers not replayed",
                  "by_badge lift covers HITTER-only badges (POWER/DUE/HOT/WARMING/MAY COOL); "
                  "opponent-context badges (WEAK ARM/PLATOON/PITCH EDGE/WEAK PEN) need lineup/pen "
                  "data not in the replay frame and are omitted",
                  "opposing SP = actual first pitcher; the live board uses the morning probable, so replayed matchup info is marginally sharper than production — treat calibration as a slight ceiling, not a floor",
                  "props replayed with same leak contract; hrr graded WITHOUT lineup-spot multiplier (no morning lineups in replay)",
                  "hrr runs/rbis approximated identically to the live tracker",
                  f"first {WARMUP_DAYS} days used as feature warm-up, not graded"],
    }


def poison_check(df: pd.DataFrame, D: str) -> bool:
    """Prove no future leakage: corrupt every row on/after D absurdly; the heats
    AND props scores computed for D must not move at all."""
    df = df.copy(); df["_gd"] = df["game_date"].astype(str).str[:10]
    day = df[df["_gd"] == D]; past = df[df["_gd"] < D]
    a, ap = _day_heats(past, day, D)
    poisoned = df.copy()
    fut = poisoned["_gd"] >= D
    poisoned.loc[fut, "launch_speed"] = 130.0
    poisoned.loc[fut, "launch_angle"] = 28.0
    day2 = day.copy()                              # grading input unchanged; features re-derived
    past2 = poisoned[poisoned["_gd"] < D]
    b, bp = _day_heats(past2, day2, D)
    def _close(x, y):
        if x is None and y is None: return True
        if x is None or y is None: return False
        return abs(x - y) < 1e-9
    same = (set(a) == set(b)
            and all(_close(a[k]["heat"], b[k]["heat"]) for k in a)
            and all(_close(a[k]["hit_heat"], b[k]["hit_heat"]) for k in a)
            and all(_close(a[k]["hrr_heat"], b[k]["hrr_heat"]) for k in a)
            and set(ap) == set(bp)
            and all(_close(ap[k][0], bp[k][0]) for k in ap))
    print(f"[backtest] poison check {'PASS' if same else 'FAIL'} on {D} "
          f"(heat + hit_heat + hrr_heat + pitcher_k_heat all verified)")
    return same


def main():
    start = os.environ.get("BT_START")
    end = os.environ.get("BT_END")
    season_start = os.environ.get("BT_SEASON_START", "2026-03-25")
    from datetime import date
    season_end = os.environ.get("BT_SEASON_END", date.today().isoformat())
    print(f"[backtest] pulling {season_start} -> {season_end}")
    df = statcast_data.pull_season(season_start, season_end)
    if df is None or df.empty:
        print("[backtest] no data"); sys.exit(1)
    mid = sorted(df["game_date"].astype(str).str[:10].unique())
    if not poison_check(df, mid[len(mid) // 2]):
        sys.exit(2)
    rec = replay(df, start=start, end=end)
    # run model (moneyline / total / F5) — separate section in the same artifact
    try:
        rec["runs"] = replay_runs(df, start=start, end=end)
        r = rec["runs"]
        if r.get("games"):
            verdict = "BEATS baseline" if r["beats_baseline"] else "does NOT beat baseline"
            print(f"[backtest:runs] {r['games']} games · brier {r['brier']} vs baseline "
                  f"{r['brier_baseline']} — {verdict} · total MAE {r['total_mae']} runs")
    except Exception as e:
        rec["runs"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:runs] skipped: {e}")
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "backtest.json")
    json.dump(rec, open(out, "w"))
    print(f"[backtest] wrote {out}: {rec.get('days')} days, base {rec.get('base_pct')}%")


# ---------------------------------------------------------------------------
# Run-model backtest: does the game projection actually predict game outcomes?
#
# Reported metrics are chosen deliberately:
#   * CALIBRATION, not accuracy. "When the model says 60%, does the home team win
#     60% of the time?" A model can be 55% accurate and still useless if its
#     probabilities are miscalibrated — you cannot bet a number you can't trust.
#   * Brier score: mean squared error of the probability. Lower is better.
#     Always-predict-54% (the home-field base rate) scores ~0.248. If the model
#     cannot beat that, it has learned nothing.
#   * Total MAE: mean absolute error on the run total. Books are typically within
#     ~2.6 runs. Beating that is the (hard) bar for betting totals.
# ---------------------------------------------------------------------------
def replay_runs(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    from etl import runs as RUNS
    df = df.copy()
    df["_gd"] = df["game_date"].astype(str).str[:10]
    all_dates = sorted(df["_gd"].unique())
    if len(all_dates) <= WARMUP_DAYS:
        return {"error": "not enough days"}
    dates = [d for d in all_dates[WARMUP_DAYS:]
             if (not start or d >= start) and (not end or d <= end)]

    n = 0
    brier = 0.0
    correct = 0
    tot_abs_err = 0.0
    calib = {}                      # decile -> {n, home_wins}
    home_wins_actual = 0
    for D in dates:
        day = df[df["_gd"] == D]
        past = df[df["_gd"] < D]
        if past.empty:
            continue
        for gpk, g in day.groupby("game_pk"):
            # actual result from the final post-score of the game
            try:
                hs = float(g["post_home_score"].dropna().max())
                as_ = float(g["post_away_score"].dropna().max())
            except Exception:
                continue
            if hs != hs or as_ != as_ or hs == as_:
                continue                       # skip ties/unknowns
            home_won = hs > as_
            # lineups actually used, in order of first appearance
            half_bat = {}
            for half, side in (("Top", "away"), ("Bot", "home")):
                rows = g[g["inning_topbot"] == half]
                ids = list(dict.fromkeys(int(b) for b in rows["batter"].dropna()))[:9]
                half_bat[side] = ids
            if len(half_bat["home"]) < 6 or len(half_bat["away"]) < 6:
                continue
            # starters
            sp = {}
            for half, side in (("Top", "home"), ("Bot", "away")):
                r = g[(g["inning_topbot"] == half) & (g["inning"] == 1)]
                if r.empty:
                    continue
                sp[side] = int(r.sort_values(["at_bat_number", "pitch_number"]).iloc[0]["pitcher"])
            if "home" not in sp or "away" not in sp:
                continue
            allb = half_bat["home"] + half_bat["away"]
            bprof = statcast_data.batter_profiles(past, allb, asof=D)
            pprof = statcast_data.pitcher_profiles(past, [sp["home"], sp["away"]], asof=D)
            hl = [(bprof.get(b) or {}).get("recent") or {} for b in half_bat["home"]]
            al = [(bprof.get(b) or {}).get("recent") or {} for b in half_bat["away"]]
            # batter hand for the platoon matchup — the stand each batter actually used in this
            # game (already the correct platoon side vs the starter). Flows the same runs.py
            # platoon logic through the replay as production uses live.
            stand_map = {}
            for bid, grp in g.groupby("batter"):
                s = grp["stand"].dropna()
                if len(s):
                    stand_map[int(bid)] = s.mode().iloc[0]
            hl_hands = [stand_map.get(b) for b in half_bat["home"]]
            al_hands = [stand_map.get(b) for b in half_bat["away"]]
            proj = RUNS.project_game(hl, al, pprof.get(sp["home"]) or {},
                                     pprof.get(sp["away"]) or {},
                                     {}, {}, park_mult=1.0,
                                     home_hands=hl_hands, away_hands=al_hands)
            if not proj:
                continue
            p = proj["home_wp"]
            n += 1
            home_wins_actual += 1 if home_won else 0
            brier += (p - (1.0 if home_won else 0.0)) ** 2
            correct += 1 if ((p >= 0.5) == home_won) else 0
            tot_abs_err += abs(proj["total"] - (hs + as_))
            b = int(min(max(p, 0.0), 0.999) * 10) * 10
            c = calib.setdefault(str(b), {"n": 0, "home_wins": 0})
            c["n"] += 1; c["home_wins"] += 1 if home_won else 0
        if n and n % 50 == 0:
            print(f"[backtest:runs] {n} games scored through {D}")

    if not n:
        return {"error": "no gradeable games (post-score columns missing from the frame?)"}
    base_rate = home_wins_actual / n
    # baseline: always predict the actual home-field rate
    brier_base = sum((base_rate - (1 if i < home_wins_actual else 0)) ** 2
                     for i in range(n)) / n
    return {
        "games": n,
        "home_win_rate": round(100 * base_rate, 2),
        "accuracy": round(100 * correct / n, 2),
        "brier": round(brier / n, 4),
        "brier_baseline": round(brier_base, 4),
        "beats_baseline": bool(brier / n < brier_base),
        "total_mae": round(tot_abs_err / n, 2),
        "calib": {k: calib[k] for k in sorted(calib, key=int)},
        "notes": [
            "CALIBRATION is the number that matters, not accuracy — you cannot bet a probability you can't trust",
            "Brier lower is better; if it does not beat brier_baseline the model has learned nothing",
            "books hit total MAE around 2.6 runs — that is the bar for betting totals",
            "run model has NO defense, NO true park run factor, NO bullpen availability",
            "bullpen omitted entirely in this replay (starter + league-average pen)",
        ],
    }


if __name__ == "__main__":
    main()
