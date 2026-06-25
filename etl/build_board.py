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

    # Fall back to each team's last batting order (projected) where today's
    # lineup isn't posted yet, so the board isn't blank in the morning.
    yest = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    _recent_cache = {}

    def _recent(team_id):
        if team_id not in _recent_cache:
            _recent_cache[team_id] = statsapi.get_recent_lineup(team_id, yest)
        return _recent_cache[team_id]

    projected_sides = set()
    for g in games:
        pk = g["game_pk"]
        lu = slate["lineups"].get(pk) or {}
        away = lu.get("away") or None
        home = lu.get("home") or None
        # fill EACH side independently — a posted away lineup must not block a
        # projected home lineup (partial postings otherwise erase a whole team)
        if not away:
            away = _recent(g["away_id"])
            if away:
                projected_sides.add((pk, "away"))
        if not home:
            home = _recent(g["home_id"])
            if home:
                projected_sides.add((pk, "home"))
        if away or home:
            slate["lineups"][pk] = {"away": away or [], "home": home or []}
    proj_game_pks = {pk for (pk, _s) in projected_sides}
    print(f"[build] projected {len(projected_sides)} lineup side(s) across {len(proj_game_pks)} game(s)")

    # collect batter ids from posted lineups
    batter_ids, game_of_batter, side_of_batter, spot_of_batter, status_of_batter = [], {}, {}, {}, {}
    for pk, lu in slate["lineups"].items():
        gmeta = next((g for g in games if g["game_pk"] == pk), None)
        if not gmeta:
            continue
        for i, bid in enumerate(lu.get("away", [])):
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "away"; spot_of_batter[bid] = i + 1
            status_of_batter[bid] = "projected" if (pk, "away") in projected_sides else "confirmed"
        for i, bid in enumerate(lu.get("home", [])):
            batter_ids.append(bid); game_of_batter[bid] = pk; side_of_batter[bid] = "home"; spot_of_batter[bid] = i + 1
            status_of_batter[bid] = "projected" if (pk, "home") in projected_sides else "confirmed"
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
    bullpens = statcast_data.bullpen_profiles(df, date_str)
    career = statcast_data.career_table(2015, now.year)

    # statsapi and Statcast mostly share team abbreviations; a few differ.
    _TEAM_ALIAS = {"AZ": "ARI", "ARI": "AZ", "CWS": "CHW", "CHW": "CWS",
                   "WSH": "WSN", "WSN": "WSH", "KC": "KCR", "KCR": "KC",
                   "SD": "SDP", "SDP": "SD", "SF": "SFG", "SFG": "SF",
                   "TB": "TBR", "TBR": "TB"}

    def _bullpen_for(abbr):
        if abbr in bullpens:
            return bullpens[abbr]
        return bullpens.get(_TEAM_ALIAS.get(abbr))

    # score every probable pitcher's HR vulnerability once
    pitcher_hr = {}
    for pid, prof_p in pitch_profiles.items():
        pitcher_hr[pid] = compute.pitcher_hr_score(prof_p.get("recent", {}), prof_p.get("season", {}))

    # 2-year HR-by-hand per starter (cached in a repo file so we don't re-pull hourly)
    _HAND2YR_PATH = os.path.join(os.path.dirname(OUT_PATH) or ".", "hand2yr.json")
    try:
        with open(_HAND2YR_PATH) as _f:
            hand2yr_cache = json.load(_f)
    except Exception:
        hand2yr_cache = {}
    hand2yr = {}
    for pid in {p for p in pitcher_ids if p}:
        key = str(pid)
        ent = hand2yr_cache.get(key)
        fresh = False
        if ent and ent.get("asof"):
            try:
                fresh = 0 <= (datetime.strptime(date_str, "%Y-%m-%d") -
                              datetime.strptime(ent["asof"], "%Y-%m-%d")).days <= 10
            except Exception:
                fresh = False
        if fresh:
            hand2yr[pid] = ent.get("data")
        else:
            data = statcast_data.pitcher_hand_hr_2yr(pid, date_str)
            if data is not None:
                hand2yr_cache[key] = {"asof": date_str, "data": data}
                hand2yr[pid] = data
            elif ent:                       # pull failed but we have an older value — keep it
                hand2yr[pid] = ent.get("data")

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

        # vs-pitch-mix variant: re-weight EV/LA/whiff by THIS arm's pitch mix (last 2wk),
        # then recompute Heat with the mix-adjusted EV. Display-toggle only.
        mix_prof = compute.pitch_mix_profile(prof.get("pitch_splits_recent"), pprof.get("usage"))
        pmatch = compute.pitch_matchup(prof.get("pitch_splits"), pprof.get("usage"), season.get("barrel_pct"))
        heat_mix = score
        if mix_prof and mix_prof.get("avg_ev") is not None:
            recent_mix = {**recent, "avg_ev": mix_prof["avg_ev"]}
            heat_mix, _ = compute.heat_score(recent_mix, phr.get("score"))
            # barrel-mix layer: if he barrels THIS arm's mix harder than his overall,
            # nudge up slightly (capped) — rewards a strong barrel-vs-mix matchup
            if pmatch and pmatch.get("edge"):
                heat_mix = min(100, heat_mix + max(0, min(4, round(pmatch["edge"]))))

        # trend (contact-quality) + the synthesized one-line read
        tr = compute.trend(prof.get("windows", {}).get("L5", {}),
                           prof.get("windows", {}).get("L30", {}))
        eff_hand = eff_side if bats == "S" else bats
        angle = compute.read_angle(
            hand=bats, trend=tr, pitch_matchup=pmatch,
            luck_gap=recent.get("luck_gap"),
            opp_form=(phr.get("form") or {}).get("label"),
            hand_hr=(hand2yr.get(pid) or {}).get("two_yr"), eff_hand=eff_hand)

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

        # pitcher platoon splits — what he allows vs this hitter's hand
        eff_hand = eff_side if bats == "S" else bats
        psplits = pprof.get("splits") or {}
        opp_pitcher_obj["platoon"] = compute.platoon_note(psplits)
        opp_pitcher_obj["hr_by_hand"] = {
            "R_hr": (psplits.get("R") or {}).get("season", {}).get("hr_allowed"),
            "R_pa": (psplits.get("R") or {}).get("season", {}).get("pa"),
            "L_hr": (psplits.get("L") or {}).get("season", {}).get("hr_allowed"),
            "L_pa": (psplits.get("L") or {}).get("season", {}).get("pa"),
        }
        opp_pitcher_obj["hr_by_hand_2yr"] = hand2yr.get(pid)
        vh = compute.hand_vuln(psplits.get(eff_hand)) if eff_hand in ("R", "L") else None
        opp_pitcher_obj["vs_hand"] = eff_hand
        opp_pitcher_obj["vs_hand_score"] = vh["score"] if vh else None
        _vhs = (psplits.get(eff_hand) or {})
        opp_pitcher_obj["vs_hand_metrics"] = {
            "barrel_pct_allowed": (_vhs.get("season") or {}).get("barrel_pct_allowed"),
            "hr_per_pa": (_vhs.get("season") or {}).get("hr_per_pa"),
            "bbe": (_vhs.get("season") or {}).get("bbe"),
        }

        # opposing BULLPEN vulnerability (overall + vs this hitter's hand)
        opp_abbr = g["home"] if side == "away" else g["away"]
        opp_bullpen = compute.bullpen_vuln(_bullpen_for(opp_abbr), eff_hand) if g else None

        metrics = {}
        # the four headline signals first (in your order), then context metrics
        for key in ("pull_air_pct", "avg_ev", "barrel_pct", "ideal_aa_pct",
                    "bat_speed", "hardhit_pct", "iso", "slg", "launch_angle",
                    "fb_pct", "pull_pct", "swstr_pct", "k_pct"):
            metrics[key] = {
                "recent": recent.get(key),
                "season": season.get(key),
                "career": car.get(key),
            }

        # auto "why" line — the cleared signals + arm read, for instant scanning
        why_bits = []
        if recent.get("pull_air_pct") is not None and recent["pull_air_pct"] >= 40:
            why_bits.append(f"{recent['pull_air_pct']:.0f}% air-pull")
        if recent.get("barrel_pct") is not None and recent["barrel_pct"] >= 11:
            why_bits.append(f"{recent['barrel_pct']:.0f}% brl")
        if recent.get("avg_ev") is not None and recent["avg_ev"] >= 88.5:
            why_bits.append(f"{recent['avg_ev']:.1f} EV")
        if recent.get("ideal_aa_pct") is not None and recent["ideal_aa_pct"] >= 58:
            why_bits.append(f"{recent['ideal_aa_pct']:.0f}% IAA")
        if recent.get("iso") is not None and recent["iso"] >= 0.200:
            why_bits.append(f".{int(round(recent['iso']*1000)):03d} ISO")
        oform = (opp_pitcher_obj.get("form") or {}).get("label", "")
        why = " · ".join(why_bits[:3])
        if oform in ("SHELLABLE", "STEADY-BAD", "SLIPPING", "HITTABLE"):
            why = (why + " · " if why else "") + f"vs {oform} arm"

        players.append({
            "id": bid,
            "name": name,
            "bats": bats,
            "lineup_spot": spot_of_batter.get(bid),
            "lineup_status": status_of_batter.get(bid, "confirmed"),
            "trend": tr,
            "angle": angle,
            "team": g["away"] if side == "away" else g["home"],
            "opp_team": g["home"] if side == "away" else g["away"],
            "game_pk": pk,
            "time": g["time"],
            "park": g["park"],
            "park_hr_factor": round(pf, 2),
            "why": why,
            "tier": breakdown.get("tier"),
            "cleared": breakdown.get("cleared"),
            "sample": {                       # batted-ball counts so tiny windows are obvious
                "L5": (prof.get("windows", {}).get("L5", {}) or {}).get("bb_count"),
                "L15": (prof.get("windows", {}).get("L15", {}) or {}).get("bb_count"),
                "L30": (prof.get("windows", {}).get("L30", {}) or {}).get("bb_count"),
                "season": season.get("bb_count"),
            },
            "opp_pitcher": opp_pitcher_obj,
            "opp_bullpen": opp_bullpen,
            "pitch_splits": prof.get("pitch_splits"),
            "pitch_usage": pprof.get("usage"),
            "pitch_matchup": pmatch,
            "heat_mix": heat_mix,
            "mix": mix_prof,
            "ev_overall": recent.get("avg_ev"),
            "luck": {
                "recent": {k: recent.get(k) for k in ("xwobacon", "wobacon", "luck_gap", "barrel_pct", "hr", "bb_count")},
                "season": {k: season.get(k) for k in ("xwobacon", "wobacon", "luck_gap", "barrel_pct", "hr", "bb_count")},
            },
            "max_ev": {"recent": recent.get("max_ev"), "season": season.get("max_ev")},
            "heat": score,
            "score_breakdown": breakdown,
            "metrics": metrics,
            "windows": prof.get("windows", {}),
            "hr_recent": {w: prof.get("windows", {}).get(w, {}).get("hr") for w in ("L5", "L15", "L30")},
        })

    players.sort(key=lambda p: p["heat"], reverse=True)

    # ---- decision helpers ----
    def _thin(p):
        return any(str(f).startswith("small sample") for f in p["score_breakdown"].get("flags", []))

    # Top Plays: strongest, non-thin hitters not facing a DEALING arm
    top_plays = []
    for p in players:
        if _thin(p) or p["heat"] < 60:
            continue
        if (p["opp_pitcher"].get("form") or {}).get("label") == "DEALING":
            continue
        top_plays.append({
            "name": p["name"], "team": p["team"], "opp_team": p["opp_team"],
            "heat": p["heat"], "tier": p["tier"], "why": p["why"],
            "spot": p["lineup_spot"], "time": p["time"],
            "arm": p["opp_pitcher"].get("name"),
            "arm_form": (p["opp_pitcher"].get("form") or {}).get("label"),
            "arm_score": p["opp_pitcher"].get("hr_score"),
        })
        if len(top_plays) >= 12:
            break

    # Stacks (pairing): a vulnerable arm with 2+ strong hitters facing him
    from collections import defaultdict
    groups = defaultdict(list)
    for p in players:
        groups[(p["game_pk"], p["opp_pitcher"].get("name"))].append(p)
    stacks = []
    for (gpk, arm), ps in groups.items():
        if not arm:
            continue
        form = (ps[0]["opp_pitcher"].get("form") or {}).get("label", "")
        if form not in ("SHELLABLE", "STEADY-BAD", "SLIPPING", "HITTABLE"):
            continue
        strong = sorted([x for x in ps if x["heat"] >= 55], key=lambda x: x["heat"], reverse=True)
        if len(strong) < 2:
            continue
        stacks.append({
            "arm": arm,
            "form": ps[0]["opp_pitcher"].get("form"),
            "arm_score": ps[0]["opp_pitcher"].get("hr_score"),
            "game_pk": gpk,
            "team": strong[0]["team"], "opp_team": strong[0]["opp_team"],
            "time": strong[0]["time"],
            "hitters": [{
                "name": x["name"], "heat": x["heat"], "tier": x["tier"],
                "spot": x["lineup_spot"], "bats": x["bats"],
            } for x in strong[:6]],
        })
    stacks.sort(key=lambda s: (s["arm_score"] or 0, len(s["hitters"])), reverse=True)

    board = {
        "generated_at": now.isoformat(timespec="seconds"),
        "slate_date": date_str,
        "league_avg": compute.LEAGUE_AVG,
        "games": [{
            "game_pk": g["game_pk"], "away": g["away"], "home": g["home"],
            "park": g["park"], "time": g["time"],
        } for g in games],
        "lineups_pending": [g["game_pk"] for g in games if g["game_pk"] not in slate["lineups"]],
        "projected_games": [
            {"game_pk": g["game_pk"], "away": g["away"], "home": g["home"]}
            for g in games if g["game_pk"] in proj_game_pks
        ],
        "recent_window": {
            "days": 14,
            "start": (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d"),
            "end": date_str,
        },
        "players": players,
        "top_plays": top_plays,
        "stacks": stacks,
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
                "platoon": compute.platoon_note(pitch_profiles.get(pid, {}).get("splits")),
            }
            for pid, phr in pitcher_hr.items()
        ], key=lambda a: (a["hr_score"] is not None, a["hr_score"] or 0), reverse=True),
    }

    # persist the 2-year HR-by-hand cache so future builds reuse it (avoids hourly re-pulls)
    try:
        with open(_HAND2YR_PATH, "w") as _f:
            json.dump(hand2yr_cache, _f)
    except Exception as _e:
        print(f"[build] hand2yr cache write failed (non-fatal): {_e}")

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

    # slim daily snapshot so the grader can grade this day even after the live
    # board rolls over to tomorrow's slate
    try:
        snap_dir = os.path.join(os.path.dirname(OUT_PATH) or ".", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        snap = {
            "date": board["slate_date"],
            "players": [{
                "id": p["id"], "name": p["name"], "team": p["team"],
                "heat": p["heat"], "tier": p.get("tier"), "cleared": p.get("cleared"),
                "signals": p["score_breakdown"].get("signals", {}),
                "opp_form": (p["opp_pitcher"].get("form") or {}).get("label"),
                "iso": (p.get("windows", {}).get("L14d", {}) or {}).get("iso"),
                "barrel_pct": (p.get("windows", {}).get("L14d", {}) or {}).get("barrel_pct"),
            } for p in board["players"]],
        }
        with open(os.path.join(snap_dir, f"{board['slate_date']}.json"), "w") as f:
            json.dump(snap, f, default=str)
        import glob
        for old in sorted(glob.glob(os.path.join(snap_dir, "20*.json")))[:-16]:
            os.remove(old)
    except Exception as e:
        print(f"[build] snapshot write failed (non-fatal): {e}")

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
            if any(str(f).startswith("small sample") for f in p.get("score_breakdown", {}).get("flags", []))]
    if thin:
        print(f"[sanity] {len(thin)} thin-sample hitters flagged (dimmed on board): {', '.join(thin[:8])}")


if __name__ == "__main__":
    main()
