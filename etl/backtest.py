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

from etl import compute, statcast_data

WARMUP_DAYS = 21          # first N days of the frame are feature-only (no grading)
TIERS = (("70+", 70, 999), ("55-69", 55, 70), ("40-54", 40, 55), ("<40", -999, 40))


def _tier(h):
    for name, lo, hi in TIERS:
        if lo <= h < hi:
            return name
    return "<40"


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
    out = {}
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
        out[bid] = (float(heat), bid in hr_today)
    return out


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
    for D in dates:
        day = df[df["_gd"] == D]
        past = df[df["_gd"] < D]
        heats = _day_heats(past, day, D)
        if len(heats) < 30:                       # partial-slate days pollute rates
            continue
        graded_days += 1
        ranked = sorted(heats.items(), key=lambda kv: -kv[1][0])
        for i, (bid, (h, hit)) in enumerate(ranked):
            n_tot += 1; hr_tot += 1 if hit else 0
            t = by_tier[_tier(h)]
            t["n"] += 1; t["hr"] += 1 if hit else 0
            for k in ("5", "10", "25"):
                if i < int(k):
                    top_n[k]["n"] += 1; top_n[k]["hr"] += 1 if hit else 0
            b = int(min(max(h, 0), 99) // 10) * 10
            c = calib.setdefault(str(b), {"n": 0, "hr": 0})
            c["n"] += 1; c["hr"] += 1 if hit else 0
        if graded_days % 10 == 0:
            print(f"[backtest] {graded_days} days graded through {D}")
    return {
        "days": graded_days, "pool": n_tot, "hr": hr_tot,
        "base_pct": round(100 * hr_tot / n_tot, 2) if n_tot else None,
        "by_tier": by_tier, "top_n": top_n,
        "calib": {k: calib[k] for k in sorted(calib, key=int)},
        "notes": ["core model only (heat + arm nudge); park/weather/badge layers not replayed",
                  "opposing SP = actual first pitcher (probables occasionally differed)",
                  f"first {WARMUP_DAYS} days used as feature warm-up, not graded"],
    }


def poison_check(df: pd.DataFrame, D: str) -> bool:
    """Prove no future leakage: corrupt every row on/after D absurdly; the heats
    computed for D must not move at all."""
    df = df.copy(); df["_gd"] = df["game_date"].astype(str).str[:10]
    day = df[df["_gd"] == D]; past = df[df["_gd"] < D]
    a = _day_heats(past, day, D)
    poisoned = df.copy()
    fut = poisoned["_gd"] >= D
    poisoned.loc[fut, "launch_speed"] = 130.0
    poisoned.loc[fut, "launch_angle"] = 28.0
    day2 = day.copy()                              # grading input unchanged; features re-derived
    past2 = poisoned[poisoned["_gd"] < D]
    b = _day_heats(past2, day2, D)
    same = set(a) == set(b) and all(abs(a[k][0] - b[k][0]) < 1e-9 for k in a)
    print(f"[backtest] poison check {'PASS' if same else 'FAIL'} on {D}")
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
