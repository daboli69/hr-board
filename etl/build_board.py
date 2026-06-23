"""
build_board.py — the one script the cron runs.

  python -m etl.build_board

It pulls the slate (StatsAPI) + Statcast season data (Savant), computes every
hitter in today's posted lineups, scores them, and writes docs/board.json.

Designed to fail soft: any single data source hiccup degrades that column to
null rather than crashing the whole run, so an unattended cron keeps producing
a board.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from etl import statsapi, statcast_data, parks, compute

try:                       # cache Savant pulls to disk so repeat runs are fast
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()
except Exception:
    pass

ET = ZoneInfo("America/New_York")
SEASON_START = os.environ.get("SEASON_START", "2026-03-26")
RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "45"))  # window for L5/L15/L30
OUT_PATH = os.environ.get("BOARD_OUT", "docs/board.json")
MIN_STATCAST_ROWS = int(os.environ.get("MIN_STATCAST_ROWS", "5000"))
PULL_RETRIES = int(os.environ.get("PULL_RETRIES", "3"))


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ").strip()


def build(date_str: str | None = None) -> dict:
    now = datetime.now(ET)
    date_str = date_str or now.strftime("%Y-%m-%d")
    print(f"[build] slate {date_str}")

    slate = statsapi.get_slate(date_str)
    games = slate["games"]
    print(f"[build] {len(games)} games, lineups posted for {len(slate['lineups'])}")

    # collect batter ids from posted lineups
    batter_ids, game_of_batter, side_of_batter = [], {}, {}
    for pk, lu in slate["lineups"].items():
        gmeta = next((g for g in games if g["game_pk"] == pk), None)
        if not gmeta:
            continue
        for bid in lu["away"]:
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "away"
        for bid in lu["home"]:
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "home"
    batter_ids = list(dict.fromkeys(batter_ids))

    pitcher_ids = [p for g in games for p in (g["away_pitcher_id"], g["home_pitcher_id"])]

    # handedness for everyone
    hands = statsapi.get_handedness(batter_ids + [p for p in pitcher_ids if p])

    # one big Statcast pull -> recent windows + season + pitcher allowed
    end = date_str
    print(f"[build] pulling Statcast {SEASON_START}..{end}")
    df = statcast_data.pd.DataFrame()
    for attempt in range(1, PULL_RETRIES + 1):
        try:
            df = statcast_data.pull_season(SEASON_START, end)
        except Exception as e:
            print(f"[build] statcast pull attempt {attempt} failed: {e}")
            df = statcast_data.pd.DataFrame()
        if len(df) >= MIN_STATCAST_ROWS:
            break
        if attempt < PULL_RETRIES:
            wait = 30 * attempt
            print(f"[build] got {len(df)} rows (<{MIN_STATCAST_ROWS}); retrying in {wait}s")
            time.sleep(wait)

    # GUARD: if Savant came back empty/short, do NOT zero out a good board.
    if len(df) < MIN_STATCAST_ROWS:
        print(f"[build] Statcast insufficient ({len(df)} rows). Keeping last good board.")
        raise statcast_data.StatcastUnavailable(len(df))

    profiles = statcast_data.batter_profiles(df, batter_ids, date_str)
    pitch_profiles = statcast_data.pitcher_profiles(df, pitcher_ids, date_str)
    career = statcast_data.career_table(2015, now.year)

    # score every probable pitcher's HR vulnerability once
    pitcher_hr = {}
    for pid, prof_p in pitch_profiles.items():
        pitcher_hr[pid] = compute.pitcher_hr_score(prof_p.get("recent", {}), prof_p.get("season", {}))

    # opposing pitcher lookup per batter
    def opp_pitcher(pk, side):
        g = next((x for x in games if x["game_pk"] == pk), None)
        if not g:
            return None, None
        pid = g["home_pitcher_id"] if side == "away" else g["away_pitcher_id"]
        return pid, g

    players = []
    for bid in batter_ids:
        prof = profiles.get(bid, {})
        recent = prof.get("recent", {})
        season = prof.get("season", {})
        hand = hands.get(bid, {})
        bats = hand.get("bats", "R")
        name = hand.get("name", str(bid))
        car = career.get(_norm(name), {})

        pk = game_of_batter[bid]
        side = side_of_batter[bid]
        pid, g = opp_pitcher(pk, side)
        pprof = pitch_profiles.get(pid, {}) if pid else {}
        phr = pitcher_hr.get(pid, {}) if pid else {}
        meta = slate["pitchers"].get(pid, {}) if pid else {}
        throws = hands.get(pid, {}).get("throws", "") if pid else ""

        # switch hitters bat opposite the pitcher's hand — use that side for park factor
        eff_side = bats
        if bats == "S":
            eff_side = "L" if throws == "R" else "R"
        pf = parks.park_factor(g["park"], eff_side) if g else 1.0

        # hitter ranking = four signals, modulated by opposing-arm vulnerability
        score, breakdown = compute.heat_score(recent, phr.get("score"))

        pr = pprof.get("recent", {})
        ps = pprof.get("season", {})
        opp_pitcher_obj = {
            "name": meta.get("name", ""),
            "throws": throws,
            "hr_score": phr.get("score"),
            "recent_score": phr.get("recent_score"),
            "season_score": phr.get("season_score"),
            "form": phr.get("form"),
            "flags": phr.get("flags", []),
            "recent": {
                "barrel_pct_allowed": pr.get("barrel_pct_allowed"),
                "hardhit_pct_allowed": pr.get("hardhit_pct_allowed"),
                "avg_ev_allowed": pr.get("avg_ev_allowed"),
                "hr_per_pa": pr.get("hr_per_pa"),
                "ideal_aa_allowed": pr.get("ideal_aa_allowed"),
                "pull_air_allowed": pr.get("pull_air_allowed"),
                "swstr_pct_allowed": pr.get("swstr_pct_allowed"),
                "fb_velo": pr.get("fb_velo"),
                "velo_trend": pr.get("velo_trend"),
                "bbe": pr.get("bbe"),
            },
            "season": {
                "barrel_pct_allowed": ps.get("barrel_pct_allowed"),
                "hardhit_pct_allowed": ps.get("hardhit_pct_allowed"),
                "avg_ev_allowed": ps.get("avg_ev_allowed"),
                "hr_per_pa": ps.get("hr_per_pa"),
                "ideal_aa_allowed": ps.get("ideal_aa_allowed"),
                "pull_air_allowed": ps.get("pull_air_allowed"),
                "swstr_pct_allowed": ps.get("swstr_pct_allowed"),
                "fb_velo": ps.get("fb_velo"),
            },
        }

        metrics = {}
        # the four headline signals first (in your order), then context metrics
        for key in ("pull_air_pct", "avg_ev", "barrel_pct", "ideal_aa_pct",
                    "bat_speed", "hardhit_pct", "iso", "launch_angle",
                    "fb_pct", "pull_pct", "swstr_pct", "k_pct"):
            metrics[key] = {
                "recent": recent.get(key),
                "season": season.get(key),
                "career": car.get(key),
            }

        players.append({
            "id": bid,
            "name": name,
            "bats": bats,
            "team": g["away"] if side == "away" else g["home"],
            "opp_team": g["home"] if side == "away" else g["away"],
            "game_pk": pk,
            "time": g["time"],
            "park": g["park"],
            "park_hr_factor": round(pf, 2),
            "sample": {                       # batted-ball counts so tiny windows are obvious
                "L5": (prof.get("windows", {}).get("L5", {}) or {}).get("bb_count"),
                "L15": (prof.get("windows", {}).get("L15", {}) or {}).get("bb_count"),
                "L30": (prof.get("windows", {}).get("L30", {}) or {}).get("bb_count"),
                "season": season.get("bb_count"),
            },
            "opp_pitcher": opp_pitcher_obj,
            "heat": score,
            "score_breakdown": breakdown,
            "metrics": metrics,
            "windows": prof.get("windows", {}),
            "hr_recent": {w: prof.get("windows", {}).get(w, {}).get("hr") for w in ("L5", "L15", "L30")},
        })

    players.sort(key=lambda p: p["heat"], reverse=True)

    board = {
        "generated_at": now.isoformat(timespec="seconds"),
        "slate_date": date_str,
        "league_avg": compute.LEAGUE_AVG,
        "games": [{
            "game_pk": g["game_pk"], "away": g["away"], "home": g["home"],
            "park": g["park"], "time": g["time"],
        } for g in games],
        "lineups_pending": [g["game_pk"] for g in games if g["game_pk"] not in slate["lineups"]],
        "recent_window": {
            "days": 14,
            "start": (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d"),
            "end": date_str,
        },
        "players": players,
        "arms": sorted([
            {
                "name": slate["pitchers"].get(pid, {}).get("name", str(pid)),
                "throws": hands.get(pid, {}).get("throws", ""),
                "team": next((g["home"] if g["home_pitcher_id"] == pid else g["away"]
                              for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "opp": next((g["away"] if g["home_pitcher_id"] == pid else g["home"]
                             for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "park": next((g["park"] for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "hr_score": phr.get("score"),
                "recent_score": phr.get("recent_score"),
                "season_score": phr.get("season_score"),
                "form": phr.get("form"),
                "flags": phr.get("flags", []),
            }
            for pid, phr in pitcher_hr.items()
        ], key=lambda a: (a["hr_score"] is not None, a["hr_score"] or 0), reverse=True),
    }
    return board


def main():
    try:
        board = build()
    except statcast_data.StatcastUnavailable as e:
        # Savant was unavailable/throttled. Leave the existing board.json in place
        # (no write) so the page keeps serving the last good data instead of zeros.
        print(f"[build] SKIPPED write — Statcast unavailable ({e}). Last good board preserved.")
        return
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(board, f, indent=2, default=str)
    print(f"[build] wrote {OUT_PATH}: {len(board['players'])} hitters, "
          f"{len(board['games'])} games")

    # ---- first-run smell test: do the right names surface? ----
    ps = board["players"]
    def topby(key, label):
        ranked = sorted(
            [p for p in ps if (p["metrics"].get(key) or {}).get("recent") is not None],
            key=lambda p: p["metrics"][key]["recent"], reverse=True)[:5]
        print(f"  top {label}:")
        for p in ranked:
            print(f"    {p['metrics'][key]['recent']:>6}  {p['name']}")
    print("[sanity] eyeball these — known pull sluggers should be high on pull-air:")
    topby("pull_air_pct", "pull-air%")
    topby("ideal_aa_pct", "ideal AA%")
    thin = [p["name"] for p in ps
            if any(str(f).startswith("thin") for f in p.get("score_breakdown", {}).get("flags", []))]
    if thin:
        print(f"[sanity] {len(thin)} thin-sample hitters flagged (dimmed on board): {', '.join(thin[:8])}")


if __name__ == "__main__":
    main()
