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
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from etl import statsapi, statcast_data, parks, compute, park_model

try:                       # cache Savant pulls to disk so repeat runs are fast
    from pybaseball import cache as pyb_cache
    pyb_cache.enable()
except Exception:
    pass

ET = ZoneInfo("America/New_York")
SEASON_START = os.environ.get("SEASON_START", "2026-03-26")
BUILD_HEALTH = []          # subsystem skip notes; shipped in board.json as build_health


def _hnote(sub, err):
    BUILD_HEALTH.append({"sub": sub, "issue": f"{type(err).__name__}: {err}"[:160]})


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
    bb_samples = statcast_data.batted_ball_sample(df, batter_ids)
    pitch_profiles = statcast_data.pitcher_profiles(df, pitcher_ids, date_str)
    bullpens = statcast_data.bullpen_profiles(df, date_str)
    career = statcast_data.career_table(2015, now.year)

    # season batter-vs-pitcher (for the Matchup tab): has this hitter homered off today's
    # starter, or off any active arm in the opponent's pen? Computed from the slate frame.
    bvp = {}
    pen_arms = {}
    pen_names = {}
    try:
        bvp = statcast_data.bvp_table(df)
        pen_arms = statcast_data.bullpen_arms(df, date_str)
        all_arms = sorted({pid for arms in pen_arms.values() for pid in arms})
        if all_arms:
            try:
                pen_names = {int(k): v.get("name", "") for k, v in
                             statsapi.get_handedness(all_arms).items()}
            except Exception as e:
                _hnote("bullpen name lookup", e); print(f"[build] bullpen name lookup skipped: {e}")
        print(f"[build] BvP: {len(bvp)} matchups, {sum(len(a) for a in pen_arms.values())} active arms")
    except Exception as e:
        _hnote("BvP", e); print(f"[build] BvP skipped: {e}")

    # career BvP vs the starter (MLB Stats API), cached so the day's builds share one fetch
    import time as _t
    bvp_cache_path = os.path.join(os.path.dirname(__file__), "..", "docs", "bvp_career.json")
    try:
        bvp_career_cache = json.load(open(bvp_cache_path)).get("pairs", {})
    except Exception:
        bvp_career_cache = {}
    _bvp_now = _t.time(); _bvp_ttl = 64800; _bvp_fetched = [0]; _BVP_MAX = 340

    try:                                           # season HR distribution by batting-order slot
        hr_spot = statcast_data.hr_by_lineup_spot(df)
    except Exception as e:
        hr_spot = {}; _hnote("hr_by_spot", e); print(f"[build] hr_by_spot skipped: {e}")

    try:                                           # opener detection: how deep starters really go
        start_lens = statcast_data.starter_lengths(df)
        p_apps = statcast_data.pitcher_appearances(df)
    except Exception as e:
        start_lens, p_apps = {}, {}; _hnote("starter lengths", e); print(f"[build] starter lengths skipped: {e}")

    try:                                           # B2B: homered in his most recent game
        b2b_set = statcast_data.hr_last_game(df)
    except Exception as e:
        b2b_set = set(); _hnote("hr_last_game", e); print(f"[build] hr_last_game skipped: {e}")

    try:                                           # per-park wind sensitivity (weekly, archived wx)
        from etl import wind_sens as WS
        ws_cache = os.path.join(os.path.dirname(__file__), "..", "docs", "wind_sens.json")
        park_model.set_wind_sens(WS.load_wind_sensitivity(ws_cache, df=df))
    except Exception as e:
        _hnote("wind sensitivity", e); print(f"[build] wind sensitivity skipped: {e}")

    try:                                           # pitcher batted-ball mix allowed (FB% = target)
        p_batted = statcast_data.pitcher_batted_profile(df)
    except Exception as e:
        p_batted = {}; _hnote("pitcher batted profile", e); print(f"[build] pitcher batted profile skipped: {e}")

    try:                                           # PF-style profile labels (trailing 14d)
        _lab_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
        hit_labels = statcast_data.hitter_labels(df, _lab_start)   # same 2wk window as the model
        print(f"[build] labels: {sum(1 for v in hit_labels.values() if v=='elite')} elite, "
              f"{sum(1 for v in hit_labels.values() if v=='fb')} fb, "
              f"{sum(1 for v in hit_labels.values() if v=='ld')} ld")
    except Exception as e:
        hit_labels = {}; _hnote("labels", e); print(f"[build] labels skipped: {e}")

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

        # vs-pitch-mix variant: re-weight the two pitch-dependent power signals —
        # avg EV and barrel% — to THIS arm's pitch mix (last 2wk), then recompute Heat.
        # Barrel varies most by pitch type and is the heaviest signal, so this is what
        # actually moves the ranking. Bidirectional: weak-vs-mix hitters drop too.
        mix_prof = compute.pitch_mix_profile(prof.get("pitch_splits_recent"), pprof.get("usage"))
        pmatch = compute.pitch_matchup(prof.get("pitch_splits"), pprof.get("usage"), season.get("barrel_pct"))
        heat_mix = score
        if mix_prof and mix_prof.get("avg_ev") is not None:
            recent_mix = dict(recent)
            recent_mix["avg_ev"] = mix_prof["avg_ev"]
            if mix_prof.get("barrel_pct") is not None:
                recent_mix["barrel_pct"] = mix_prof["barrel_pct"]
            heat_mix, _ = compute.heat_score(recent_mix, phr.get("score"))

        # trend (contact-quality) + the synthesized one-line read
        tr = compute.trend(prof.get("windows", {}).get("L5", {}),
                           prof.get("windows", {}).get("L30", {}),
                           mid_w=prof.get("windows", {}).get("L15", {}))
        eff_hand = eff_side if bats == "S" else bats
        angle = compute.read_angle(
            hand=bats, trend=tr, pitch_matchup=pmatch,
            luck_gap=recent.get("luck_gap"), xwobacon=recent.get("xwobacon"),
            opp_form=(phr.get("form") or {}).get("label"),
            hand_hr=(hand2yr.get(pid) or {}).get("two_yr"), eff_hand=eff_hand)
        badges = compute.player_badges(
            opp_form=(phr.get("form") or {}).get("label"),
            hand_hr=(hand2yr.get(pid) or {}).get("two_yr"), eff_hand=eff_hand,
            pitch_matchup=pmatch, luck_gap=recent.get("luck_gap"), trend=tr,
            xwobacon=recent.get("xwobacon"),
            max_ev=(prof.get("season", {}) or {}).get("max_ev"))

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

        # opener detection: listed SP whose real starts run 1-2 innings, or a pure
        # reliever getting the "start". Downstream, BvP-vs-SP matters less (one look)
        # and the bullpen matters much more.
        _bat = p_batted.get(pid)
        if _bat:
            opp_pitcher_obj["fb_pct"] = _bat["fb_pct"]
            opp_pitcher_obj["gb_pct"] = _bat["gb_pct"]
        _sl = start_lens.get(pid)
        if _sl and _sl["starts"] >= 2 and _sl["med_len"] <= 2.0:
            opp_pitcher_obj["opener"] = True
            opp_pitcher_obj["start_len"] = round(_sl["med_len"], 1)
        elif _sl is None and p_apps.get(pid, 0) >= 5:
            opp_pitcher_obj["opener"] = True          # relieves all year, "starting" today
            opp_pitcher_obj["start_len"] = None
        else:
            opp_pitcher_obj["opener"] = False
            opp_pitcher_obj["start_len"] = round(_sl["med_len"], 1) if _sl else None

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

        opp_abbr = g["home"] if side == "away" else g["away"]
        sp_bvp = bvp.get((bid, pid)) if pid else None
        bp_list = []
        for apid in pen_arms.get(opp_abbr, []):
            rec = bvp.get((bid, apid))
            if rec and rec[0] > 0:
                bp_list.append({"name": pen_names.get(apid, ""), "pa": rec[0], "hr": rec[1]})
        bp_list.sort(key=lambda x: (x["hr"], x["pa"]), reverse=True)
        player_bvp = {
            "sp": {"name": opp_pitcher_obj.get("name", ""),
                   "pa": sp_bvp[0] if sp_bvp else 0, "hr": sp_bvp[1] if sp_bvp else 0},
            "bp": [a for a in bp_list if a["name"]][:12],
            "bp_hr": any(a["hr"] > 0 for a in bp_list),
        }
        if pid:                                   # career vs today's starter (cached)
            _k = f"{bid}-{pid}"
            _c = bvp_career_cache.get(_k)
            _sc = None
            if _c and (_bvp_now - _c.get("ts", 0) < _bvp_ttl):
                _sc = {"pa": _c["pa"], "hr": _c["hr"]}
            elif _bvp_fetched[0] < _BVP_MAX:
                _r = statsapi.bvp_career(bid, pid)
                _bvp_fetched[0] += 1
                if _r is not None:
                    bvp_career_cache[_k] = {"pa": _r["pa"], "hr": _r["hr"], "ts": _bvp_now}
                    _sc = {"pa": _r["pa"], "hr": _r["hr"]}
                _t.sleep(0.02)
            if _sc is not None:
                player_bvp["sp_career"] = {"name": opp_pitcher_obj.get("name", ""), **_sc}

        # tags for having gone deep off today's arms (flow into tracker + parlay weighting)
        _bvp_badges = []
        if player_bvp.get("sp_career", {}).get("hr", 0) or player_bvp["sp"]["hr"]:
            _bvp_badges.append({"t": "HR vs SP", "k": "hrsp"})
        if player_bvp["bp_hr"]:
            _bvp_badges.append({"t": "HR vs PEN", "k": "hrbp"})
        badges = _bvp_badges + badges

        players.append({
            "id": bid,
            "name": name,
            "bats": bats,
            "bvp": player_bvp,
            "hr_by_spot": hr_spot.get(bid, {}),
            "hr_last_game": bid in b2b_set,
            "hit_label": hit_labels.get(bid),
            "lineup_spot": spot_of_batter.get(bid),
            "lineup_status": status_of_batter.get(bid, "confirmed"),
            "trend": tr,
            "angle": angle,
            "badges": badges,
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

    try:                                           # persist career-BvP cache for the next build
        json.dump({"pairs": bvp_career_cache, "updated": _bvp_now}, open(bvp_cache_path, "w"))
        print(f"[build] career BvP: {_bvp_fetched[0]} fetched this run, {len(bvp_career_cache)} cached")
    except Exception as e:
        print(f"[build] bvp cache write failed: {e}")

    # ---- park + weather HR model: a separate lens, computed per game in one vectorized
    # pass, attached as p["park_hr"]. Never feeds the heat score or the grader. The park's
    # overall HR level comes from Baseball Savant (auto-pulled, rolling, handedness-split);
    # the physics only adds per-hitter spray + live weather on top, anchored to Savant so a
    # dimension/orientation that's slightly off can't make the park factor wrong. Wrapped so
    # a Savant/weather outage degrades gracefully instead of breaking the board.
    try:
        from etl import park_factors
        pf_cache = os.path.join(os.path.dirname(__file__), "..", "docs", "park_factors.json")
        savant = park_factors.load_park_factors(pf_cache, df=df, year=now.year)
        gpk_home = {g["game_pk"]: g["home"] for g in games}     # durable key for the factor lookup

        try:                                       # recent deep contact for spray chart + robbed scan
            _drv_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
            drives_map = statcast_data.recent_drives(df, _drv_start)
        except Exception as e:
            drives_map = {}; _hnote("recent drives", e); print(f"[build] recent drives skipped: {e}")

        by_game = {}
        for p in players:
            by_game.setdefault((p["park"], p["time"], p["game_pk"]), []).append(p)
        for (venue, gtime, pk), ps in by_game.items():
            # tonight's conditions once per game, reused for every hitter's robbed check
            try:
                rho_g, wind_g = park_model.game_conditions(venue, gtime)
            except Exception:
                rho_g, wind_g = None, None
            for p in ps:
                drv = drives_map.get(p["id"]) or []
                if not drv:
                    continue
                flags = [0] * len(drv)
                if rho_g is not None:
                    deep = [(j, d) for j, d in enumerate(drv)
                            if (not d["hr"]) and d["dist"] >= 330 and 15 <= d["la"] <= 45]
                    if deep:
                        cl = park_model.clears_here([d["ev"] for _, d in deep],
                                                    [d["la"] for _, d in deep],
                                                    [d["spray"] for _, d in deep],
                                                    venue, rho_g, wind_g)
                        for (j, d), c in zip(deep, cl):
                            if bool(c):
                                flags[j] = 2                       # robbed: out here tonight
                for j, d in enumerate(drv):
                    if d["hr"]:
                        flags[j] = 1
                p["drives"] = [[d["spray"], d["dist"], flags[j]] for j, d in enumerate(drv)]
                robbed = [(d, ) for j, d in enumerate(drv) if flags[j] == 2]
                if robbed:
                    best = max((d for j, d in enumerate(drv) if flags[j] == 2), key=lambda x: x["dist"])
                    p["robbed"] = {"n": sum(1 for f in flags if f == 2),
                                   "best_ft": best["dist"], "best_date": best["date"]}
            evs, las, sprays, spans = [], [], [], []
            for p in ps:
                s = bb_samples.get(p["id"])
                n = 0 if (s is None) else len(s["ev"])
                spans.append((p, n))
                if n:
                    evs.append(s["ev"]); las.append(s["la"]); sprays.append(s["spray"])
            if not evs:
                continue
            ev_all = np.concatenate(evs); la_all = np.concatenate(las); sp_all = np.concatenate(sprays)
            hr_park, hr_neut, meta = park_model.evaluate_game(ev_all, la_all, sp_all, venue, gtime)
            i = 0
            for p, n in spans:
                if not n:
                    continue
                hand = p.get("bats", "R")
                sav = park_factors.factor_for(savant, venue, hand, team=gpk_home.get(pk))
                anchor = park_model.savant_anchor(venue, hand, sav)
                agg = park_model.aggregate_hitter(hr_park[i:i+n], hr_neut[i:i+n], meta,
                                                  anchor=anchor, savant_factor=sav)
                if agg:
                    p["park_hr"] = agg
                i += n
    except Exception as e:
        _hnote("park model", e); print(f"[build] park model skipped: {e}")

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

    # Stacks (pairing): a target for the whole lineup — either a vulnerable arm with
    # 2+ strong hitters facing him, OR a bullpen game (opener "start" + weak pen),
    # which the old form filter wrongly excluded. Sorted by a blended stack score so
    # a slightly-less-bad arm facing five monsters can outrank a worse arm facing two.
    from collections import defaultdict
    groups = defaultdict(list)
    for p in players:
        groups[(p["game_pk"], p["opp_pitcher"].get("name"))].append(p)
    stacks = []
    for (gpk, arm), ps in groups.items():
        if not arm:
            continue
        op = ps[0]["opp_pitcher"]
        form = (op.get("form") or {}).get("label", "")
        pen_sc = (ps[0].get("opp_bullpen") or {}).get("score")
        opener = bool(op.get("opener"))
        targetable_arm = form in ("SHELLABLE", "STEADY-BAD", "SLIPPING", "HITTABLE")
        pen_game = opener and (pen_sc or 0) >= 55
        if not (targetable_arm or pen_game):
            continue
        strong = sorted([x for x in ps if x["heat"] >= 55], key=lambda x: x["heat"], reverse=True)
        if len(strong) < 2:
            continue
        vuln = max(op.get("hr_score") or 0, (pen_sc or 0) if opener else 0)
        top3 = sum(x["heat"] for x in strong[:3]) / min(3, len(strong))
        stacks.append({
            "arm": arm,
            "form": op.get("form"),
            "arm_score": op.get("hr_score"),
            "pen_score": pen_sc,
            "opener": opener,
            "pen_game": pen_game,
            "stack_score": int(round(0.55 * vuln + 0.45 * top3)),
            "park": strong[0].get("park"),
            "park_factor": strong[0].get("park_hr_factor"),
            "game_pk": gpk,
            "team": strong[0]["team"], "opp_team": strong[0]["opp_team"],
            "time": strong[0]["time"],
            "hitters": [{
                "id": x["id"], "name": x["name"], "heat": x["heat"], "tier": x["tier"],
                "spot": x["lineup_spot"], "bats": x["bats"],
                "b2b": bool(x.get("hr_last_game")),
                "owns": any(b.get("k") in ("hrsp", "hrbp") for b in (x.get("badges") or [])),
            } for x in strong[:6]],
        })
    stacks.sort(key=lambda s: (s["stack_score"], len(s["hitters"])), reverse=True)

    try:                                       # career HR milestone watch
        mstones = statcast_data.career_hr_milestones([p["id"] for p in players])
        for p in players:
            if p["id"] in mstones:
                p["milestone"] = mstones[p["id"]]
    except Exception as e:
        _hnote("milestones", e); print(f"[build] milestones skipped: {e}")

    # ---- fences for the spray chart + the morning briefing ----
    fences = {}
    try:
        for g in games:
            if g["park"] not in fences:
                fences[g["park"]] = park_model.fence_polyline(g["park"])
    except Exception as e:
        _hnote("fences", e); print(f"[build] fences skipped: {e}")

    briefing = []
    try:
        boosts = [(p["park_hr"]["boost"], p) for p in players
                  if p.get("park_hr") and p["park_hr"].get("boost") is not None]
        if boosts:
            b, p = max(boosts, key=lambda x: x[0])
            if b >= 8:
                w = (p["park_hr"].get("wind_mph"))
                briefing.append(f"Best environment: {p['park']} at +{b}%"
                                + (f" with wind {w} mph" if w else "") + ".")
        opener_teams = sorted({p["opp_team"] for p in players
                               if (p.get("opp_pitcher") or {}).get("opener")})
        if opener_teams:
            briefing.append(("Bullpen game" if len(opener_teams) == 1 else "Bullpen games")
                            + f" vs {', '.join(opener_teams)} — weigh the pen, not the listed arm.")
        rb = sorted([p for p in players if p.get("robbed")],
                    key=lambda p: (-p["robbed"]["n"], -p["robbed"]["best_ft"]))[:2]
        for p in rb:
            r = p["robbed"]
            briefing.append(f"Robbed watch: {p['name']} — {r['best_ft']}ft out on {r['best_date'][5:]} "
                            f"clears here tonight" + (f" ({r['n']} such balls)." if r["n"] > 1 else "."))
        b2b = [p for p in players if p.get("hr_last_game") and p["heat"] >= 70]
        if b2b:
            names = ", ".join(p["name"] for p in b2b[:2])
            briefing.append(f"B2B fade: {names} homered last night — bases over HR by your rules.")
        ms1 = [p for p in players if (p.get("milestone") or {}).get("away") == 1]
        for p in ms1[:2]:
            m = p["milestone"]
            briefing.append(f"Milestone watch: {p['name']} sits at {m['career_hr']} career HR — "
                            f"one swing from {m['next']}.")
        nlab = {"elite": 0, "fb": 0, "ld": 0}
        for p in players:
            if p.get("hit_label"): nlab[p["hit_label"]] += 1
        if sum(nlab.values()):
            briefing.append(f"Profiles on slate: {nlab['elite']} ELITE · {nlab['fb']} FB · {nlab['ld']} LD.")
    except Exception as e:
        _hnote("briefing", e); print(f"[build] briefing skipped: {e}")

    # per-game weather summaries for the Weather dashboard (roof call, disruption status,
    # wind rendered relative to each park's actual orientation)
    wx_list = []
    try:
        from etl import weather as W, park_geometry as PG
        for g in games:
            lat, lon, _ = PG.park_coords(g["park"])
            wx = W.get_weather(lat, lon, g["time"], venue=g["park"])
            roof = W.roof_call(g["park"], wx)
            pp = (wx or {}).get("precip_prob")
            if roof in ("dome", "closed", "canopy") or pp is None:
                status = "clear" if roof else "unknown"
            elif pp < 20:
                status = "clear"
            elif pp < 45:
                status = "chance"
            elif pp < 70:
                status = "likely"
            else:
                status = "postpone"
            cf = PG.cf_bearing(g["park"])
            frm = (wx or {}).get("wind_from_deg")
            rel = round(((frm + 180.0) - cf) % 360.0) if (frm is not None and cf is not None) else None
            wx_list.append({
                "game_pk": g["game_pk"], "away": g["away"], "home": g["home"],
                "park": g["park"], "time": g["time"],
                "temp_f": round((wx or {}).get("temp_f")) if (wx or {}).get("temp_f") is not None else None,
                "rh_pct": round((wx or {}).get("rh_pct")) if (wx or {}).get("rh_pct") is not None else None,
                "precip_prob": round(pp) if pp is not None else None,
                "wind_mph": round((wx or {}).get("wind_mph")) if (wx or {}).get("wind_mph") is not None else None,
                "wind_rel_deg": rel, "roof": roof, "status": status,
            })
    except Exception as e:
        _hnote("weather summaries", e); print(f"[build] weather summaries skipped: {e}")

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
        "wx": wx_list,
        "fences": fences,
        "briefing": briefing,
        "label_diag": getattr(statcast_data, "LAST_LABEL_DIAG", {}),
        "build_health": {
            "df_rows": int(len(df)) if df is not None else 0,
            "players": len(players), "arms_ok": True,
            "labeled": sum(1 for p in players if p.get("hit_label")),
            "b2b": sum(1 for p in players if p.get("hr_last_game")),
            "openers": sum(1 for p in players if (p.get("opp_pitcher") or {}).get("opener")),
            "stacks": len(stacks), "wx": len(wx_list),
            "issues": BUILD_HEALTH,
        },
        "arms": sorted([
            {
                "name": slate["pitchers"].get(pid, {}).get("name", str(pid)),
                "throws": hands.get(pid, {}).get("throws", ""),
                "team": next((g["home"] if g["home_pitcher_id"] == pid else g["away"]
                              for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "opp": next((g["away"] if g["home_pitcher_id"] == pid else g["home"]
                             for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "park": next((g["park"] for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "time": next((g["time"] for g in games if pid in (g["home_pitcher_id"], g["away_pitcher_id"])), ""),
                "hr_score": phr.get("score"),
                "recent_score": phr.get("recent_score"),
                "season_score": phr.get("season_score"),
                "delta": phr.get("delta"),
                "form": phr.get("form"),
                "flags": phr.get("flags", []),
                "opener": bool(
                    (start_lens.get(pid) and start_lens[pid]["starts"] >= 2
                     and start_lens[pid]["med_len"] <= 2.0)
                    or (start_lens.get(pid) is None and p_apps.get(pid, 0) >= 5)),
                "fb_pct": (p_batted.get(pid) or {}).get("fb_pct"),
                "gb_pct": (p_batted.get(pid) or {}).get("gb_pct"),
                "start_len": (round(start_lens[pid]["med_len"], 1)
                              if start_lens.get(pid) else None),
                # heaviest 2yr HR-by-hand side, raw numbers for the strip
                "hand_hr": (lambda ty: (max(
                    ({"side": h, "hr": s["hr"], "pa": s["pa"]}
                     for h, s in (ty or {}).items() if s and s.get("pa", 0) >= 100),
                    key=lambda x: x["hr"] / max(1, x["pa"]), default=None)))(
                        (hand2yr.get(pid) or {}).get("two_yr")),
                "badges": compute.pitcher_badges(
                    recent=pitch_profiles.get(pid, {}).get("recent", {}),
                    score=phr.get("score"), recent_score=phr.get("recent_score"),
                    season_score=phr.get("season_score"),
                    two_yr=(hand2yr.get(pid) or {}).get("two_yr")),
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

    # slate-level SMASH selection, mirrored from the UI's convergence scorer so the
    # grader can measure the flag's real conversion rate (the whole point of the flag).
    # Uses standard heat (the UI's default view).
    def _smash_score(p):
        H = p.get("heat") or 0
        s = max(0.0, min(3.0, (H - 45) / 10.0)); r = 0
        ks = {b["k"] for b in (p.get("badges") or [])}
        tr = p.get("trend") or {}
        if "lock" in ks: s += 1.5; r += 1
        elif "hot" in ks: s += 0.75; r += 1
        if "due" in ks: s += 1.0; r += 1
        if tr.get("dir") == "up": s += 1.0; r += 1
        pb = (p.get("park_hr") or {}).get("boost") or 0
        if pb >= 12: s += 1.5; r += 1
        elif pb >= 6: s += 0.75; r += 1
        opnr = bool((p.get("opp_pitcher") or {}).get("opener"))
        if "hrsp" in ks: s += (0.75 if opnr else 1.5); r += 1
        if "hrbp" in ks: s += (1.5 if opnr else 1.0); r += 1
        spot_hr = (p.get("hr_by_spot") or {}).get(p.get("lineup_spot") or 0, 0)
        if spot_hr >= 3: s += 1.5; r += 1
        elif spot_hr >= 2: s += 0.75; r += 1
        mixd = (p.get("heat_mix") - p["heat"]) if (p.get("heat_mix") is not None and p.get("heat") is not None) else 0
        if mixd >= 6: s += 1.0; r += 1
        elif mixd >= 4: s += 0.5; r += 1
        arm = (p.get("opp_pitcher") or {}).get("hr_score") or 0
        if arm >= 65 or "arm" in ks: s += 1.0; r += 1
        if "mix" in ks: s += 0.75; r += 1
        if "pow" in ks: s += 0.5; r += 1
        if "plat" in ks: s += 0.5; r += 1
        return s, r
    smash_ids = set()
    try:
        cand = []
        for p in board["players"]:
            if p.get("lineup_status") == "out":
                continue
            sc, nr = _smash_score(p)
            if sc >= 6.5 and nr >= 3 and (p.get("heat") or 0) >= 55:
                cand.append((sc, p["id"]))
        cand.sort(reverse=True)
        smash_ids = {pid for _, pid in cand[:3]}
        print(f"[build] SMASH: {len(smash_ids)} flagged")
        if smash_ids:
            _nm = [p["name"] for p in board["players"] if p["id"] in smash_ids]
            board.setdefault("briefing", []).insert(0, "SMASH today: " + " · ".join(_nm) + ".")
    except Exception as e:
        _hnote("smash calc", e); print(f"[build] smash calc skipped: {e}")

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
                # ---- enrichment: context that can't be backfilled later ----
                "badges": [b["k"] for b in (p.get("badges") or [])],
                "bp_score": (p.get("opp_bullpen") or {}).get("score"),
                "sp_vuln": (p["opp_pitcher"].get("hr_score")),
                "luck_gap": (((p.get("luck") or {}).get("recent")) or {}).get("luck_gap"),
                "heat_mix": p.get("heat_mix"),
                "spot": p.get("lineup_spot"),
                "park_boost": (p.get("park_hr") or {}).get("boost"),
                "trend": (p.get("trend") or {}).get("dir"),
                "b2b": p.get("hr_last_game"),
                "smash": p["id"] in smash_ids,
                "opener": bool((p.get("opp_pitcher") or {}).get("opener")),
                "hlabel": p.get("hit_label"),
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
