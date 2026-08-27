"""
backfill_hr_heat.py — reconstruct heat/badges/heat_rank for hr_log entries that are missing
them, using the exact same no-leakage methodology etl/backtest.py's replay() already uses
and has proven correct (heat_calibration's real Spearman=1.0 result depends on this same
reconstruction being right).

Deliberately a SEPARATE script from backfill_calendar.py, not an extension of it -- that
script's own pull only fetches home_run rows per month-chunk, which is enough for HR counts/
distance/EV but NOT enough to reconstruct heat, which needs a batter's FULL plate-appearance
history (not just his homers) going back to season start. Keeping this separate means there's
zero risk to that script's proven, working behavior.

What this CAN reconstruct (real, permanent Statcast, hitter-side only):
    heat (the frozen 0-100 model, computed from data strictly BEFORE that date -- no leakage)
    badges: POWER, DUE, HOT, MAY COOL, WARMING (the hitter-only badges replay() already limits
        itself to, for the same reason -- the opponent-context badges (WEAK ARM, PLATOON,
        PITCH EDGE, WEAK PEN) need that day's live lineup/pen data, which isn't reconstructed
        here, so they're left out rather than approximated)
    heat_rank: this hitter's rank among every real batter who had a plate appearance that
        specific day -- computed by reconstructing heat for the WHOLE day's batter pool, not
        just the target hitter, then finding his position
    team and SP/BP role: both permanently derivable from the same day's Statcast rows already
        being pulled for the heat reconstruction -- no extra data source needed

What this does NOT touch (already established as genuinely unreconstructable, not just
unbuilt -- see etl/backtest.py's replay_longest_hr() docstring and the conversation that led
to this script): Long Ball board membership, ownership_tier, is_gold_pick, jackpot_ev. Those
depend on that specific day's live betting odds, which were never archived anywhere. This
script does not attempt them and will not write fake values for them.

Only touches hr_log entries that are ALREADY present in docs/history.json and are currently
missing a real "heat" value -- it never invents a day, never overwrites a real live-graded
entry, and never touches the Long Ball fields at all.

Run on GitHub Actions (Statcast isn't reachable from every environment, and this needs a full
season-to-date pull, not a small chunk -- expect this to be slow, budget real time for it):

    python etl/backfill_hr_heat.py --start 2026-03-27 --end 2026-07-31

`--start` should be SEASON_START (or earlier) even if the dates you actually need heat for are
later -- the model needs real history before each target date to compute anything, and using
the true season start keeps every target date's reconstruction on the same footing the live
board itself uses. `--end` should cover the LATEST date you need backfilled; earlier target
dates are computed using only data strictly before them, same as always.
"""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime

HISTORY_PATH = os.environ.get("HISTORY_OUT", "docs/history.json")


def _dates_needing_heat(history_path):
    """Which (date, hitter name) pairs in the existing hr_log are missing a real heat value.
    Only ever reads -- the actual write-back happens in merge_heat(), and only for entries
    this function identifies."""
    try:
        with open(history_path) as f:
            hist = json.load(f)
    except Exception as e:
        print(f"[backfill-heat] could not read {history_path}: {e}", file=sys.stderr)
        return {}
    out = {}
    for d in hist.get("days", []):
        date = d.get("date")
        for entry in (d.get("hr_log") or []):
            if entry.get("heat") is None and entry.get("name"):
                out.setdefault(date, []).append(entry["name"])
    return out


def reconstruct_day(df, D, target_names, statcast_data, compute, props):
    """For one date D, reconstruct heat/badges for every real batter with a plate appearance
    that day (needed for the rank, not just the target names), then return just the subset
    the caller actually asked for. Mirrors backtest.py's replay() loop directly -- same
    past/day split, same batter_profiles() call, same heat_score()/player_badges() calls --
    rather than a re-derived approximation of that logic.
    """
    day = df[df["_gd"] == D]
    past = df[df["_gd"] < D]
    if day.empty or len(past) < 500:
        return {}

    batters = sorted({int(b) for b in day["batter"].dropna().unique()})
    if not batters:
        return {}

    try:
        name_map = statcast_data.player_names(batters)
    except Exception as e:
        print(f"[backfill-heat] {D}: name lookup failed ({e}), skipping", file=sys.stderr)
        return {}

    # team + SP/BP role -- both permanently derivable from this same day's rows, no extra pull.
    # inning_topbot tells us which side is batting: Top -> away team's batter, Bot -> home.
    team_of = {}
    try:
        for _, r in day.dropna(subset=["batter"]).iterrows():
            bid = int(r["batter"])
            if bid in team_of:
                continue
            side = r.get("inning_topbot")
            team_of[bid] = r.get("away_team") if side == "Top" else r.get("home_team")
    except Exception as e:
        print(f"[backfill-heat] {D}: team lookup failed ({e})", file=sys.stderr)

    starters = {}
    try:
        srt = day.dropna(subset=["game_pk", "pitcher", "inning", "inning_topbot"]).copy()
        srt = srt.sort_values(["game_pk", "inning", "inning_topbot"])
        first = srt.groupby(["game_pk", "inning_topbot"], as_index=False).first()
        for _, r in first.iterrows():
            starters[(int(r["game_pk"]), r["inning_topbot"])] = int(r["pitcher"])
    except Exception as e:
        print(f"[backfill-heat] {D}: starters lookup failed ({e})", file=sys.stderr)

    def _off_role(bid):
        rows = day[day["batter"] == bid]
        rows = rows[rows["events"].astype(str) == "home_run"]
        if rows.empty:
            return None
        r0 = rows.iloc[0]
        gp, half, pit = r0.get("game_pk"), r0.get("inning_topbot"), r0.get("pitcher")
        if gp != gp or pit != pit:
            return None
        return "SP" if starters.get((int(gp), half)) == int(pit) else "BP"

    bprof = statcast_data.batter_profiles(past, batters, asof=D)

    scored = []
    for bid in batters:
        prof = bprof.get(bid)
        if not prof:
            continue
        recent = prof.get("recent") or {}
        if not (recent.get("bb_count") or 0):
            continue
        try:
            heat, _ = compute.heat_score(recent, None)   # no opponent-pitcher nudge -- that
        except Exception:                                # context isn't reconstructed here,
            continue                                      # same limitation replay() documents
        try:
            windows = prof.get("windows") or {}
            tr = compute.trend(windows.get("L5") or {}, windows.get("L30") or {},
                               mid_w=windows.get("L15") or {})
            badges = compute.player_badges(
                luck_gap=recent.get("luck_gap"), trend=tr,
                max_ev=recent.get("max_ev"), xwobacon=recent.get("xwobacon"),
            )
            badge_keys = [b["k"] for b in badges]
        except Exception:
            badge_keys = []
        scored.append({"id": bid, "name": name_map.get(bid, str(bid)),
                       "heat": round(heat, 1), "badges": badge_keys,
                       "team": team_of.get(bid), "off": _off_role(bid)})

    scored.sort(key=lambda r: -r["heat"])
    by_name = {}
    for i, r in enumerate(scored):
        if r["name"] in target_names:
            by_name[r["name"]] = {"heat": r["heat"], "badges": r["badges"],
                                  "heat_rank": i + 1, "n_that_day": len(scored),
                                  "team": r.get("team"), "off": r.get("off")}
    return by_name


def merge_heat(results_by_date, history_path=HISTORY_PATH):
    """Write reconstructed heat/badges/heat_rank into the matching hr_log entries. Only ever
    fills a currently-missing "heat" field -- never touches Long Ball fields, never overwrites
    a real live-graded value (there wouldn't be one to overwrite, since this only targets
    entries that were already confirmed missing "heat" before this ran)."""
    with open(history_path) as f:
        hist = json.load(f)
    n_filled = 0
    for d in hist.get("days", []):
        date = d.get("date")
        result = results_by_date.get(date)
        if not result:
            continue
        for entry in (d.get("hr_log") or []):
            r = result.get(entry.get("name"))
            if r and entry.get("heat") is None:
                entry["heat"] = r["heat"]
                entry["badges"] = r["badges"]
                entry["heat_rank"] = r["heat_rank"]
                entry["n_that_day"] = r["n_that_day"]
                if entry.get("team") is None and r.get("team"):
                    entry["team"] = r["team"]
                if entry.get("off") is None and r.get("off"):
                    entry["off"] = r["off"]
                entry["heat_reconstructed"] = True   # honest marker -- this is a retroactive
                n_filled += 1                        # reconstruction, not a live-graded value
    hist["updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    with open(history_path, "w") as f:
        json.dump(hist, f, indent=2, default=str)
    print(f"[backfill-heat] filled heat/badges/rank for {n_filled} hr_log entries "
          f"→ {history_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, should be season start")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD, latest date needing backfill")
    ap.add_argument("--history", default=HISTORY_PATH)
    args = ap.parse_args()

    needed = _dates_needing_heat(args.history)
    needed = {d: names for d, names in needed.items() if args.start <= d <= args.end}
    if not needed:
        print("[backfill-heat] nothing in range is missing heat -- nothing to do.")
        return
    print(f"[backfill-heat] {sum(len(v) for v in needed.values())} entries across "
          f"{len(needed)} date(s) need heat reconstruction")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from etl import statcast_data, compute, props

    print(f"[backfill-heat] pulling full Statcast {args.start} → {args.end} "
          f"(this is a large, slow pull -- budget real time)")
    df = statcast_data.pull_season(args.start, args.end)
    if df is None or df.empty:
        print("[backfill-heat] Statcast pull came back empty -- nothing to reconstruct from.",
              file=sys.stderr)
        return
    df = df.copy()
    df["_gd"] = df["game_date"].astype(str).str[:10]

    results = {}
    for D, names in sorted(needed.items()):
        print(f"[backfill-heat] reconstructing {D} ({len(names)} target hitter(s))...")
        results[D] = reconstruct_day(df, D, set(names), statcast_data, compute, props)

    # ADDED after a real race: this whole run is slow (30+ min for a wide date range), and
    # docs/history.json is also written by the daily board-update workflow -- if that runs
    # during this window, the on-disk copy loaded implicitly by merge_heat() could be stale.
    # The reconstruction results themselves are already safely in memory at this point, so
    # re-pulling just before the write/merge step costs nothing and closes the race.
    import subprocess
    try:
        subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                       check=True, capture_output=True, text=True, timeout=60)
        print("[backfill-heat] pulled latest history.json before merging")
    except Exception as e:
        print(f"[backfill-heat] git pull before merge failed (non-fatal, proceeding with "
              f"on-disk copy): {e}", file=sys.stderr)

    merge_heat(results, args.history)


if __name__ == "__main__":
    main()
