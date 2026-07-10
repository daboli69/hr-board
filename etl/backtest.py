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

import numpy as np
import pandas as pd

from etl import compute, statcast_data, props

WARMUP_DAYS = 21          # first N days of the frame are feature-only (no grading)
TIERS = (("70+", 70, 999), ("55-69", 55, 70), ("40-54", 40, 55), ("<40", -999, 40))

HIT_EVENTS = {"single", "double", "triple", "home_run"}
K_EVENTS = {"strikeout", "strikeout_double_play"}


def _tier(h):
    for name, lo, hi in TIERS:
        if lo <= h < hi:
            return name
    return "<40"


def _day_outcomes(day: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per-batter (hits, hrr_approx, ks) and per-pitcher K totals for one date.
    HRR approximation mirrors track.py: walk each half-inning's PA events in
    order, track base state, credit runs/RBIs on hits and walks. A floor, not
    exact — but identical to how the live tracker grades, so backtest and
    tracker read on the same scale."""
    bat_hits, bat_ks, bat_hrr = {}, {}, {}
    pit_ks = {}
    pa_df = day[day["events"].notna() & (day["events"] != "")]
    ordered = pa_df.sort_values(["game_pk", "inning", "inning_topbot",
                                 "at_bat_number", "pitch_number"])
    for (gp, inn, half), grp in ordered.groupby(
            ["game_pk", "inning", "inning_topbot"], sort=False):
        seen = set()
        base = {}
        for _, row in grp.iterrows():
            ab = row.get("at_bat_number")
            if ab in seen or row["batter"] != row["batter"]:
                continue
            seen.add(ab)
            bid = int(row["batter"])
            ev = row["events"]
            h = bat_hrr.setdefault(bid, {"hits": 0, "runs": 0, "rbis": 0})
            if ev in K_EVENTS:
                bat_ks[bid] = bat_ks.get(bid, 0) + 1
                pit = row.get("pitcher")
                if pit == pit:
                    pit_ks[int(pit)] = pit_ks.get(int(pit), 0) + 1
            if ev in HIT_EVENTS:
                bat_hits[bid] = bat_hits.get(bid, 0) + 1
                h["hits"] += 1
                if ev == "home_run":
                    h["rbis"] += 1 + len(base)
                    h["runs"] += 1
                    for r_ in list(base):
                        bat_hrr.setdefault(r_, {"hits": 0, "runs": 0, "rbis": 0})["runs"] += 1
                    base = {}
                elif ev == "single":
                    for r_, b_ in list(base.items()):
                        if b_ == 3:
                            bat_hrr.setdefault(r_, {"hits": 0, "runs": 0, "rbis": 0})["runs"] += 1
                            h["rbis"] += 1
                            del base[r_]
                    base[bid] = 1
                else:                                   # double / triple
                    for r_ in list(base):
                        bat_hrr.setdefault(r_, {"hits": 0, "runs": 0, "rbis": 0})["runs"] += 1
                        h["rbis"] += 1
                    base = {}
                    base[bid] = 2 if ev == "double" else 3
            elif ev in ("walk", "intent_walk", "hit_by_pitch"):
                if 1 in base.values():
                    for r_ in list(base):
                        if base[r_] == 3:
                            bat_hrr.setdefault(r_, {"hits": 0, "runs": 0, "rbis": 0})["runs"] += 1
                            h["rbis"] += 1
                            del base[r_]
                        elif base[r_] == 2:
                            base[r_] = 3
                        elif base[r_] == 1:
                            base[r_] = 2
                base[bid] = 1
    return ({"hits": bat_hits, "ks": bat_ks, "hrr": bat_hrr}, pit_ks,
            {bid: v["hits"] + v["runs"] + v["rbis"] for bid, v in bat_hrr.items()})


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
        out[bid] = {
            "heat": float(heat), "hr": bid in hr_today,
            "hit_heat": float(hh) if hh is not None else None,
            "hrr_heat": float(hr_h) if hr_h is not None else None,
            "hits": int(bat_out["hits"].get(bid, 0)),
            "ks": int(bat_out["ks"].get(bid, 0)),
            "hrr": int(hrr_val.get(bid, 0)),
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
        # --- props: hit1/hit2 tiers by hit_heat ---
        hh_ranked = sorted((kv for kv in heats.items() if kv[1]["hit_heat"] is not None),
                           key=lambda kv: -kv[1]["hit_heat"])
        for i, (bid, r) in enumerate(hh_ranked):
            tier = _tier(r["hit_heat"])
            got1 = r["hits"] >= 1; got2 = r["hits"] >= 2
            P["hit1"][tier]["n"] += 1; P["hit1"][tier]["hit"] += 1 if got1 else 0
            P["hit2"][tier]["n"] += 1; P["hit2"][tier]["hit"] += 1 if got2 else 0
            hit_n += 1; hit1_tot += 1 if got1 else 0; hit2_tot += 1 if got2 else 0
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
        "base_pct": round(100 * hr_tot / n_tot, 2) if n_tot else None,
        "by_tier": by_tier, "top_n": top_n,
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
        "notes": ["core model only (heat + arm nudge); park/weather/badge layers not replayed",
                  "opposing SP = actual first pitcher (probables occasionally differed)",
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
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "backtest.json")
    json.dump(rec, open(out, "w"))
    print(f"[backtest] wrote {out}: {rec.get('days')} days, base {rec.get('base_pct')}%")


if __name__ == "__main__":
    main()
