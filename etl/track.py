"""
track.py — grades a day's board against what actually happened.

Run after games end (a separate scheduled workflow). It:
  1. Reads the live board.json (its slate_date is the day to grade).
  2. Pulls that day's Statcast, finds every HR, and classifies each as off the
     STARTING pitcher (SP) or a reliever (BP) by checking who threw inning 1.
  3. Joins results back to the board: for each hitter we ranked, did he homer,
     and against what (heat tier, opposing-arm form)?
  4. Appends a daily record to docs/history.json so the Tracker view can show
     whether high Heat / vulnerable arms actually produce HRs over time.

  python -m etl.track
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from pybaseball import statcast

BOARD_PATH = os.environ.get("BOARD_OUT", "docs/board.json")
HISTORY_PATH = os.environ.get("HISTORY_OUT", "docs/history.json")
MIN_ROWS = int(os.environ.get("MIN_DAY_ROWS", "200"))


def _starters(sc) -> dict:
    """(game_pk, inning_topbot) -> starting pitcher id, via who threw inning 1."""
    starters = {}
    inn1 = sc[sc["inning"] == 1]
    for (gp, half), grp in inn1.groupby(["game_pk", "inning_topbot"]):
        g = grp.sort_values(["at_bat_number", "pitch_number"])
        starters[(int(gp), half)] = int(g.iloc[0]["pitcher"])
    return starters


def _hr_map(sc) -> dict:
    """batter_id -> {'hr':n, 'sp':n, 'bp':n} for the day."""
    starters = _starters(sc)
    out = {}
    hrs = sc[sc["events"] == "home_run"]
    for _, row in hrs.iterrows():
        bid = int(row["batter"])
        gp = int(row["game_pk"])
        half = row["inning_topbot"]
        pid = int(row["pitcher"])
        is_sp = starters.get((gp, half)) == pid
        rec = out.setdefault(bid, {"hr": 0, "sp": 0, "bp": 0})
        rec["hr"] += 1
        rec["sp" if is_sp else "bp"] += 1
    return out


def _tier(heat):
    if heat is None:
        return "n/a"
    if heat >= 70:
        return "70+"
    if heat >= 55:
        return "55-69"
    if heat >= 40:
        return "40-54"
    return "<40"


def grade():
    with open(BOARD_PATH) as f:
        board = json.load(f)
    date = board["slate_date"]

    # only grade fully-completed days. If the board still shows today's (or a
    # future) slate, the games aren't final — grading happens the next morning.
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if date >= today:
        print(f"[track] {date} games not final yet (today is {today}); grading runs after completion. Skipping.")
        return

    print(f"[track] grading {date}")

    sc = None
    for attempt in range(1, 4):
        try:
            sc = statcast(start_dt=date, end_dt=date)
        except Exception as e:
            print(f"[track] statcast attempt {attempt} failed: {e}")
            sc = None
        if sc is not None and len(sc) >= MIN_ROWS:
            break
        time.sleep(20 * attempt)
    if sc is None or len(sc) < MIN_ROWS:
        print(f"[track] insufficient data for {date}; skipping (history unchanged)")
        return

    hrmap = _hr_map(sc)

    # per-tier and per-form tallies among the hitters we actually ranked
    tiers = {}
    forms = {}
    sp_hr = bp_hr = total_hr = 0
    hr_log = []
    for p in board.get("players", []):
        res = hrmap.get(p["id"], {"hr": 0, "sp": 0, "bp": 0})
        homered = res["hr"] > 0
        t = _tier(p.get("heat"))
        tt = tiers.setdefault(t, {"n": 0, "hr": 0})
        tt["n"] += 1
        tt["hr"] += 1 if homered else 0

        form = ((p.get("opp_pitcher") or {}).get("form") or {}).get("label", "n/a")
        ff = forms.setdefault(form, {"n": 0, "hr": 0})
        ff["n"] += 1
        ff["hr"] += 1 if homered else 0

        if homered:
            total_hr += res["hr"]
            sp_hr += res["sp"]
            bp_hr += res["bp"]
            hr_log.append({
                "name": p["name"], "team": p.get("team"),
                "heat": p.get("heat"),
                "arm": (p.get("opp_pitcher") or {}).get("name"),
                "arm_form": form,
                "arm_score": (p.get("opp_pitcher") or {}).get("hr_score"),
                "off": "SP" if res["sp"] else "BP",
                "hr": res["hr"],
            })

    record = {
        "date": date,
        "players": len(board.get("players", [])),
        "hitters_homered": sum(1 for p in board.get("players", []) if hrmap.get(p["id"], {}).get("hr", 0) > 0),
        "total_hr": total_hr,
        "sp_hr": sp_hr,
        "bp_hr": bp_hr,
        "by_tier": tiers,
        "by_form": forms,
        "hr_log": sorted(hr_log, key=lambda x: (x["heat"] or 0), reverse=True),
    }

    # append / replace by date
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH) as f:
                history = json.load(f).get("days", [])
        except Exception:
            history = []
    history = [d for d in history if d.get("date") != date]
    history.append(record)
    history.sort(key=lambda d: d["date"])

    os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump({"updated": date, "days": history}, f, indent=2, default=str)
    print(f"[track] {date}: {record['hitters_homered']} of {record['players']} ranked hitters homered "
          f"({total_hr} HR — {sp_hr} off SP / {bp_hr} off BP). history now {len(history)} days.")


if __name__ == "__main__":
    grade()
