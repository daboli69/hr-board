"""
backfill_calendar.py — fill the HR calendar's earlier months from Statcast.

The live tracker only has model-graded days from when it was deployed. This script rebuilds the
DATA-ONLY part of a day (how many HRs, who homered, and each HR's distance / exit velocity) for any
past date straight from Statcast, and merges those days into docs/history.json.

What it CAN reconstruct (from Statcast, for any past date):
    total_hr, hitters_homered, sp_hr, bp_hr, per-hitter hr_log with name + HR count + avg distance
    + avg EV, plus a day-level avg HR distance / EV — the distance trend is the sharpest signal a
    ball change actually happened.

What it CANNOT reconstruct (and doesn't try to fake):
    the heat / tier / badge model overlay. That came from the live board that day (live lineups,
    live park/weather feeds, as-of rolling windows). Backfilled days are marked {"backfilled": true}
    and carry no by_tier / by_badge breakdown — the calendar shows the HR facts, not invented badges.

It NEVER overwrites a real tracked day: if a date already exists in history.json without
"backfilled", it's left untouched.

Run it on GitHub Actions (Statcast isn't reachable from every environment). Month-by-month is
safest — a full season in one shot is a big, slow download:

    python etl/backfill_calendar.py --start 2026-03-20 --end 2026-03-31
    python etl/backfill_calendar.py --start 2026-04-01 --end 2026-04-30
    ... etc.

Or a whole range (it chunks internally by month):

    python etl/backfill_calendar.py --start 2026-03-20 --end 2026-06-25
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import date, datetime, timedelta

HISTORY_PATH = os.environ.get("HISTORY_OUT", "docs/history.json")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _days_from_hrs(hrs, name_map, starters=None):
    """Pure aggregation — takes a DataFrame of home_run rows and a {batter_id: name} map, returns
    {date: day_record}. Kept separate from the network fetch so it can be unit-tested with a mock
    frame. `starters` is an optional {(game_pk, inning_topbot): pitcher_id} map for SP/BP split."""
    import pandas as pd  # local import so the module loads even where pandas is absent
    starters = starters or {}
    out = {}
    if hrs is None or len(hrs) == 0:
        return out
    for game_date, g in hrs.groupby(hrs["game_date"].astype(str)):
        per = {}
        for _, r in g.iterrows():
            try:
                bid = int(r["batter"])
            except Exception:
                continue
            e = per.setdefault(bid, {"hr": 0, "dist": [], "ev": [], "sp": 0, "bp": 0})
            e["hr"] += 1
            dist = r.get("hit_distance_sc")
            ev = r.get("launch_speed")
            if pd.notna(dist):
                e["dist"].append(float(dist))
            if pd.notna(ev):
                e["ev"].append(float(ev))
            # SP/BP attribution when we have a starters map + pitcher/half
            gp = r.get("game_pk"); half = r.get("inning_topbot"); pit = r.get("pitcher")
            if pd.notna(gp) and pd.notna(pit):
                is_sp = starters.get((int(gp), half)) == int(pit)
                e["sp" if is_sp else "bp"] += 1
        hr_log = []
        for bid, e in sorted(per.items(), key=lambda kv: -kv[1]["hr"]):
            hr_log.append({
                "name": name_map.get(bid, str(bid)),
                "hr": e["hr"],
                "dist": round(_mean(e["dist"])) if e["dist"] else None,
                "ev": round(_mean(e["ev"]), 1) if e["ev"] else None,
                "backfilled": True,
            })
        all_dist = [d for e in per.values() for d in e["dist"]]
        all_ev = [v for e in per.values() for v in e["ev"]]
        total = sum(e["hr"] for e in per.values())
        out[str(game_date)] = {
            "date": str(game_date),
            "total_hr": total,
            "hitters_homered": len(per),
            "sp_hr": sum(e["sp"] for e in per.values()),
            "bp_hr": sum(e["bp"] for e in per.values()),
            "avg_hr_dist": round(_mean(all_dist)) if all_dist else None,
            "avg_hr_ev": round(_mean(all_ev), 1) if all_ev else None,
            "hr_log": hr_log,
            "backfilled": True,
        }
    return out


def _name_map(batter_ids):
    """batter MLBAM id -> 'First Last', via pybaseball's reverse lookup (one call for all ids)."""
    ids = sorted({int(b) for b in batter_ids})
    if not ids:
        return {}
    try:
        from pybaseball import playerid_reverse_lookup
        df = playerid_reverse_lookup(ids, key_type="mlbam")
        out = {}
        for _, r in df.iterrows():
            first = str(r.get("name_first", "") or "").title()
            last = str(r.get("name_last", "") or "").title()
            out[int(r["key_mlbam"])] = (first + " " + last).strip()
        return out
    except Exception as e:
        print(f"[backfill] name lookup failed ({e}); falling back to ids", file=sys.stderr)
        return {}


def _starters_map(sc):
    """{(game_pk, inning_topbot): starting_pitcher_id} — the pitcher who threw the game's first
    pitch for each side — so HRs can be split SP vs BP. Mirrors track.py's approach."""
    import pandas as pd
    starters = {}
    try:
        df = sc.dropna(subset=["game_pk", "pitcher", "inning", "inning_topbot"]).copy()
        df = df.sort_values(["game_pk", "inning", "inning_topbot"])
        first = df.groupby(["game_pk", "inning_topbot"], as_index=False).first()
        for _, r in first.iterrows():
            starters[(int(r["game_pk"]), r["inning_topbot"])] = int(r["pitcher"])
    except Exception as e:
        print(f"[backfill] starters map skipped ({e})", file=sys.stderr)
    return starters


def _month_chunks(start, end):
    """Yield (chunk_start, chunk_end) date strings, one per calendar month, so a big range is
    fetched in digestible pieces."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    cur = s
    while cur <= e:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        chunk_end = min(e, nxt - timedelta(days=1))
        yield cur.isoformat(), chunk_end.isoformat()
        cur = nxt


def fetch_days(start, end):
    """Fetch Statcast for [start, end] (chunked by month) and return {date: day_record}."""
    from pybaseball import statcast
    all_days = {}
    for cs, ce in _month_chunks(start, end):
        print(f"[backfill] fetching Statcast {cs} → {ce} ...")
        try:
            sc = statcast(start_dt=cs, end_dt=ce)
        except Exception as e:
            print(f"[backfill] statcast {cs}-{ce} failed: {e}", file=sys.stderr)
            continue
        if sc is None or len(sc) == 0:
            print(f"[backfill]   no data for {cs}-{ce}")
            continue
        hrs = sc[sc["events"].astype(str) == "home_run"]
        if len(hrs) == 0:
            continue
        names = _name_map(hrs["batter"].dropna().tolist())
        starters = _starters_map(sc)
        chunk = _days_from_hrs(hrs, names, starters)
        all_days.update(chunk)
        print(f"[backfill]   {len(chunk)} day(s), {sum(d['total_hr'] for d in chunk.values())} HR")
    return all_days


def merge_into_history(new_days, path=HISTORY_PATH):
    """Merge backfilled days into history.json. Never clobbers a real (non-backfilled) tracked day."""
    try:
        with open(path) as f:
            hist = json.load(f)
    except Exception:
        hist = {"updated": None, "days": []}
    days = hist.setdefault("days", [])
    by_date = {d["date"]: d for d in days}
    added = replaced = skipped = 0
    for dt, rec in new_days.items():
        existing = by_date.get(dt)
        if existing is None:
            days.append(rec); by_date[dt] = rec; added += 1
        elif existing.get("backfilled"):
            by_date[dt].update(rec); replaced += 1     # refresh a prior backfill
        else:
            skipped += 1                               # real tracked day — leave it alone
    days.sort(key=lambda d: d["date"])
    hist["days"] = days
    hist["updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    with open(path, "w") as f:
        json.dump(hist, f, indent=2, default=str)
    print(f"[backfill] merged: +{added} new, {replaced} refreshed, {skipped} tracked-days preserved "
          f"→ {path} ({len(days)} total days)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default=HISTORY_PATH)
    args = ap.parse_args()
    days = fetch_days(args.start, args.end)
    if not days:
        print("[backfill] nothing to merge."); return
    merge_into_history(days, args.out)


if __name__ == "__main__":
    main()
