"""
track.py — grades a completed day against actual HRs.

Grades YESTERDAY (ET) using that day's slim snapshot (written by build_board),
so it works even though the live board has rolled over to today. Records, for
the hitters we ranked: HR rate by Heat tier, by opposing-arm form, signal->HR
correlation (cleared vs not), top-N hit rate (top 5/10/25 by Heat), SP vs BP.

  python -m etl.track
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pybaseball import statcast

BOARD_PATH = os.environ.get("BOARD_OUT", "docs/board.json")
HISTORY_PATH = os.environ.get("HISTORY_OUT", "docs/history.json")
SNAP_DIR = os.path.join(os.path.dirname(BOARD_PATH) or ".", "snapshots")
MIN_ROWS = int(os.environ.get("MIN_DAY_ROWS", "200"))
SIGNALS = ["pull_air_pct", "avg_ev", "barrel_pct", "ideal_aa_pct", "iso", "slg"]


def _load_day(date):
    snap = os.path.join(SNAP_DIR, f"{date}.json")
    if os.path.exists(snap):
        with open(snap) as f:
            return json.load(f).get("players", [])
    if os.path.exists(BOARD_PATH):
        with open(BOARD_PATH) as f:
            b = json.load(f)
        if b.get("slate_date") == date:
            return [{
                "id": p["id"], "name": p["name"], "team": p.get("team"),
                "heat": p.get("heat"), "tier": p.get("tier"),
                "signals": p.get("score_breakdown", {}).get("signals", {}),
                "opp_form": (p.get("opp_pitcher") or {}).get("form", {}).get("label"),
                "iso": (p.get("windows", {}).get("L14d", {}) or {}).get("iso"),
                "barrel_pct": (p.get("windows", {}).get("L14d", {}) or {}).get("barrel_pct"),
                "badges": [bd["k"] for bd in (p.get("badges") or [])],
            } for p in b.get("players", [])]
    return None


def _starters(sc):
    starters = {}
    for (gp, half), grp in sc[sc["inning"] == 1].groupby(["game_pk", "inning_topbot"]):
        g = grp.sort_values(["at_bat_number", "pitch_number"])
        starters[(int(gp), half)] = int(g.iloc[0]["pitcher"])
    return starters


def _hr_map(sc):
    starters = _starters(sc)
    out = {}
    for _, row in sc[sc["events"] == "home_run"].iterrows():
        bid = int(row["batter"]); gp = int(row["game_pk"]); half = row["inning_topbot"]
        is_sp = starters.get((gp, half)) == int(row["pitcher"])
        rec = out.setdefault(bid, {"hr": 0, "sp": 0, "bp": 0})
        rec["hr"] += 1
        rec["sp" if is_sp else "bp"] += 1
    return out


def _tier(h):
    if h is None: return "n/a"
    return "70+" if h >= 70 else "55-69" if h >= 55 else "40-54" if h >= 40 else "<40"


def grade_date(date):
    """Grade a single date -> record dict, or None if it can't be graded yet (no snapshot,
    or results not posted). Pure: does not read or write history.json."""
    players = _load_day(date)
    if not players:
        print(f"[track] no snapshot for {date}; skip.")
        return None

    sc = None
    for attempt in range(1, 4):
        try:
            sc = statcast(start_dt=date, end_dt=date)
        except Exception as e:
            print(f"[track] {date} statcast attempt {attempt} failed: {e}"); sc = None
        if sc is not None and len(sc) >= MIN_ROWS:
            break
        time.sleep(20 * attempt)
    if sc is None or len(sc) < MIN_ROWS:
        print(f"[track] insufficient data for {date}; will retry next run."); return None

    hrmap = _hr_map(sc)
    def homered(p): return hrmap.get(p["id"], {}).get("hr", 0) > 0

    tiers, forms = {}, {}
    by_signal = {k: {"cleared": {"n": 0, "hr": 0}, "not": {"n": 0, "hr": 0}} for k in SIGNALS}
    by_badge = {}
    sp_hr = bp_hr = total_hr = 0
    badge_hits = 0   # total badges carried by HR hitters, for "badges per HR"
    hr_log = []
    for p in players:
        hit = homered(p)
        t = _tier(p.get("heat")); tt = tiers.setdefault(t, {"n": 0, "hr": 0})
        tt["n"] += 1; tt["hr"] += 1 if hit else 0
        form = p.get("opp_form") or "n/a"; ff = forms.setdefault(form, {"n": 0, "hr": 0})
        ff["n"] += 1; ff["hr"] += 1 if hit else 0
        sig = p.get("signals", {})
        for k in SIGNALS:
            b = "cleared" if sig.get(k) else "not"
            by_signal[k][b]["n"] += 1
            by_signal[k][b]["hr"] += 1 if hit else 0
        badges = p.get("badges") or []
        for k in badges:
            bb = by_badge.setdefault(k, {"n": 0, "hr": 0})
            bb["n"] += 1; bb["hr"] += 1 if hit else 0
        if hit:
            res = hrmap[p["id"]]
            total_hr += res["hr"]; sp_hr += res["sp"]; bp_hr += res["bp"]
            badge_hits += len(badges)
            hr_log.append({"name": p["name"], "heat": p.get("heat"), "tier": p.get("tier"),
                           "arm_form": p.get("opp_form"),
                           "off": "SP" if res["sp"] else "BP", "hr": res["hr"],
                           "badges": badges, "n_badges": len(badges)})

    # ---- the validation that matters: does Heat beat simple baselines? ----
    # Rank the same hitter pool by each method and check top-N HR rates head to head.
    def _ranked_topN(key):
        rk = sorted((p for p in players if p.get(key) is not None),
                    key=lambda p: p.get(key) or 0, reverse=True)
        return {str(n): {"n": min(n, len(rk)), "hr": sum(1 for p in rk[:n] if homered(p))}
                for n in (5, 10, 25)}

    ranked = sorted(players, key=lambda p: (p.get("heat") or 0), reverse=True)
    topN = {str(n): {"n": min(n, len(ranked)), "hr": sum(1 for p in ranked[:n] if homered(p))}
            for n in (5, 10, 25)}

    n_hit = sum(1 for p in players if homered(p))
    ranks = {
        "heat": topN,
        "iso": _ranked_topN("iso"),
        "barrel": _ranked_topN("barrel_pct"),
        # base rate = a random hitter from the slate (every method must beat this)
        "base": {str(n): {"n": len(players), "hr": n_hit} for n in (5, 10, 25)},
    }

    record = {
        "date": date, "players": len(players),
        "hitters_homered": n_hit,
        "total_hr": total_hr, "sp_hr": sp_hr, "bp_hr": bp_hr,
        "by_tier": tiers, "by_form": forms, "by_signal": by_signal, "by_badge": by_badge,
        "badges_on_hr": badge_hits, "top_n": topN,
        "ranks": ranks,
        "hr_log": sorted(hr_log, key=lambda x: (x["heat"] or 0), reverse=True),
    }

    print(f"[track] {date}: {record['hitters_homered']}/{record['players']} homered, "
          f"{total_hr} HR ({sp_hr} SP / {bp_hr} BP).")
    return record


LOOKBACK_DAYS = 10   # backfill any ungraded day this far back that still has a snapshot


def grade():
    """Backfill grader. Grades every past day within LOOKBACK that has a committed snapshot
    and isn't already in history. Idempotent and self-healing: a missed or failed day is
    picked up on the next run, so one skipped/delayed cron can never silently drop a day.
    Run it a few times a day and it stays caught up on its own."""
    tz = ZoneInfo("America/New_York")
    today = datetime.now(tz).date()

    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f).get("days", [])
        except Exception:
            history = []
    graded = {d.get("date") for d in history}

    added = []
    for off in range(1, LOOKBACK_DAYS + 1):     # offset 1 = yesterday (first complete day)
        date = (today - timedelta(days=off)).strftime("%Y-%m-%d")
        if date in graded:
            continue
        rec = grade_date(date)                  # only returns a record if it CAN be graded
        if rec:
            history.append(rec)
            graded.add(date)
            added.append(date)

    if not added:
        print("[track] nothing new to grade — all caught up (or snapshots not ready).")
        return

    history.sort(key=lambda d: d["date"])
    os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump({"updated": max(graded), "days": history}, f, indent=2, default=str)
    print(f"[track] graded {len(added)} day(s): {', '.join(sorted(added))}. "
          f"history={len(history)} days.")


if __name__ == "__main__":
    grade()
