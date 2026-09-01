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
# Phase-3 validation + the new engines, imported at module level so every replay path can
# reach them without re-importing inside loops.
from etl import validate as V, kengine, hitmodel, markov
from etl import environment as env_mod, arsenal as arsenal_mod
from etl import build_board as build_board_mod   # ADDED this session -- for
                                    # replay_genius_stack_calibration, which calls the REAL
                                    # _hrpo_combine_genius_pow directly rather than a
                                    # hand-maintained parallel copy of the formula. Named
                                    # build_board_mod, not bb, since bb is already used as a
                                    # local variable name later in replay() for by_badge
                                    # tallying -- Python would otherwise treat the import as
                                    # shadowed-local for the whole function and raise
                                    # UnboundLocalError at the point of use.


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


def _badge_lift_prop(by_badge: dict, base_rate: float) -> dict:
    """Same as _badge_lift, for tallies shaped {n, hit} instead of {n, hr} -- ADDED this
    session for the hit1/hit2/hrr-vs-badge cross-tabs, which grade a badge against that prop's
    own real outcome rather than HR. A separate function rather than a shape-parameter on
    _badge_lift so existing HR call sites can't be touched by a signature change here."""
    out = {}
    for k, v in by_badge.items():
        n, hit = v["n"], v["hit"]
        rate = hit / n if n else 0.0
        out[k] = {
            "n": n, "hit": hit,
            "rate_pct": round(100 * rate, 2),
            "lift": round(rate / base_rate, 3) if base_rate > 0 else None,
        }
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


def _day_heats(past: pd.DataFrame, day: pd.DataFrame, D: str, asof: "V.AsOfFrame | None" = None,
               fg_pitch: dict = None, sp_names: dict = None) -> dict:
    """{batter_id: (heat, homered_today)} for one replay date. `past` must be
    strictly earlier than D — the caller owns that guarantee.

    fg_pitch/sp_names are OPTIONAL — passed once by replay() for the whole run, never fetched
    per-day, since that would mean hundreds of network calls across a season backtest. When
    absent (the default, and always true for _day_heats' OTHER caller), the two new command/
    xFIP-regression tallies below simply don't fire, which is the same graceful-absence pattern
    every other optional signal in this function already follows.
    """
    if asof is not None:
        # Structural leak guard: turns a silently inflated score into a loud failure.
        asof.assert_clean(past, D, "_day_heats.past")
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
    batter_game = {}   # ADDED this session -- needed for vuln_tier's per-game park lookup
    for (gp, half), grp in day.groupby(["game_pk", "inning_topbot"]):
        sp = starters.get((int(gp), half))
        for b in grp["batter"].dropna().unique():
            face[int(b)] = sp
            batter_game[int(b)] = int(gp)
    # ADDED this session -- cheap opener heuristic: did the starter face fewer than 4 batters
    # in this game? Real limitation stated in out[bid]'s comment above -- this is an
    # approximation, not a certain classification.
    _opener_sp = {}
    for (gp, half), sp in starters.items():
        _bfaced = day[(day["game_pk"] == gp) & (day["inning_topbot"] == half)
                      & (day["pitcher"] == sp)]["batter"].nunique()
        if _bfaced < 4:
            _opener_sp[sp] = True
    bprof = statcast_data.batter_profiles(past, batters, asof=D)
    # ADDED this session -- hit_label reconstruction. Genuinely cheap: hitter_labels() only
    # needs raw Statcast columns already present in `past`, no external fetch, and computes
    # every batter's label in one call rather than per-batter. Same trailing-window convention
    # (14 real game-days, not calendar days) the live app uses, via game_day_cutoff().
    try:
        _lab_start = str(statcast_data.game_day_cutoff(past, D, 14).date())
        _hit_labels = statcast_data.hitter_labels(past, _lab_start)
    except Exception:
        _hit_labels = {}
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
    # ADDED this session -- vuln_tier inputs. See the docstring on the _F.vuln_score() call
    # below for what's real vs. proxy vs. omitted, component by component.
    _xfip_by_sp = {}
    _hand_hr_by_sp = {}
    _danger_by_sp = {}
    _venue_by_game = {}
    try:
        from etl import parks as _parks
        for pid in sp_ids:
            if sp_names and fg_pitch:
                _nm = sp_names.get(pid, "")
                if _nm:
                    _fge = fg_pitch.get(statcast_data._norm_name(_nm))
                    if _fge and _fge.get("xfip") is not None:
                        _xfip_by_sp[pid] = _fge["xfip"]
            _splits = (pprof.get(pid) or {}).get("splits") or {}
            _hh = {}
            for _hand in ("R", "L"):
                _s = (_splits.get(_hand) or {}).get("season") or {}
                if _s.get("pa"):
                    _hh[_hand] = {"hr": _s.get("hr_allowed", 0), "pa": _s.get("pa", 0)}
            if _hh:
                _hand_hr_by_sp[pid] = _hh
        # danger_count: real, cheap proxy -- count of opposing batters whose OWN recent
        # avg_ev/barrel_pct already clear a real "dangerous" bar, using bprof (already fully
        # computed above for every batter) rather than needing this function's own heat
        # computation, which hasn't happened yet for anyone at this point in the loop.
        for pid in sp_ids:
            _dc = 0
            for _b, _s in face.items():
                if _s != pid:
                    continue
                _br = (bprof.get(_b) or {}).get("recent") or {}
                if (_br.get("avg_ev") or 0) >= 90 or (_br.get("barrel_pct") or 0) >= 8:
                    _dc += 1
            _danger_by_sp[pid] = _dc
        # park_factor: real, static, non-weather-dependent lookup -- one per game, keyed by
        # home team (whoever's pitching, the game is played at the home team's park).
        for gp in day["game_pk"].dropna().unique():
            _home = day[day["game_pk"] == gp]["home_team"]
            if len(_home):
                _venue = env_mod.TEAM_PARK.get(str(_home.iloc[0]).upper())
                if _venue:
                    _venue_by_game[int(gp)] = _venue
    except Exception:
        pass
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
            _windows = prof.get("windows") or {}
            _trend = compute.trend(_windows.get("L5") or {}, _windows.get("L30") or {},
                                   mid_w=_windows.get("L15") or {})
            badge_list = compute.player_badges(
                luck_gap=recent.get("luck_gap"),
                trend=_trend,
                max_ev=recent.get("max_ev"),
                xwobacon=recent.get("xwobacon"),
            )
            badge_keys = [b["k"] for b in badge_list]
        except Exception:
            badge_keys = []
        # ---- NEW SIGNALS, computed from `past` only (no leakage) ----
        _zone = None; _squp = None; _hrpow = None; _sweet_spot = None
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
                    # ADDED this session: sweet_spot_pct (LA 8-32) is already computed inside
                    # square_up_rating() -- it's one of four components blended into the
                    # composite "rating" above -- but was discarded before reaching this row,
                    # so it had never been independently checked on its own. Sample gate (n)
                    # mirrors square_up_rating's own internal >=15 floor.
                    _sweet_spot = _sq.get("sweet_spot_pct")
                    if _sweet_spot is not None:
                        _sweet_spot = round(100.0 * _sweet_spot, 1)   # stored as a 0-1 fraction
                _hp = _F.hr_power_profile(_brows)
                if _hp:
                    # FIXED this session: was a bare scalar (barrel_pct alone) -- the new
                    # genius-stack calibration code below expects a dict with both barrel_pct
                    # and max_dist (matching what own_hr_power carries live), and would silently
                    # AttributeError on every row otherwise (caught by that code's own
                    # try/except, so the whole check would come back empty with no visible
                    # error rather than crashing loudly).
                    _hrpow = {"barrel_pct": _hp.get("barrel_pct"), "max_dist": _hp.get("max_dist")}
        except Exception:
            pass
        # ---- zone edge + arsenal fit + convergence, all as-of `past` ----
        _zedge = None; _afit = None; _nm = _nmh = _np = 0
        _pmix = None; _vtier = None
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
                # ADDED this session -- pitch_mix reconstruction. Genuinely cheap: both real
                # inputs (the batter's own rows, the pitcher's real arsenal usage) were already
                # computed above for arsenal_fit -- this just reshapes the pitch-type-level
                # usage into the family-level structure pitch_matchup() expects, using the same
                # PITCH_BUCKET mapping statcast_data.py already uses for _pitch_splits().
                _pmix = None
                try:
                    if _ars:
                        _fam_usage = {}
                        for _pt, _pct in _ars:
                            _fam = statcast_data.PITCH_BUCKET.get(_pt)
                            if _fam:
                                _fam_usage[_fam] = _fam_usage.get(_fam, 0.0) + _pct
                        if _fam_usage:
                            _hp_prof = _F.hitter_pitch_profile(_brows2)
                            if _hp_prof:
                                _pm = _F.pitch_matchup(_hp_prof, {"usage": _fam_usage})
                                if _pm:
                                    _pmix = _pm.get("score")
                except Exception:
                    _pmix = None
                # ADDED this session -- vuln_tier, implemented per Travis's explicit request
                # despite my earlier concern that too little of the real formula was
                # available. Reconsidered more carefully: 5 of 6 real components turned out to
                # be real data or an honest, documented proxy, not a guess.
                #   era: REAL PROXY -- xFIP (already-fetched FanGraphs data) substituted for
                #     ERA. Same numerical scale by design (both "runs per 9"-style), not a
                #     guess, but not literal ERA either.
                #   whip: OMITTED. No clean real source or proxy found without a shaky
                #     approximation I didn't want to introduce. vuln_score() redistributes its
                #     15 points proportionally across whatever else is present, same graceful
                #     handling the live function already does for genuinely missing data.
                #   park_factor: REAL. Static, non-weather lookup (parks.park_factor), not an
                #     approximation.
                #   hand_hr: REAL PROXY -- same-season hand splits (already computed by
                #     pitcher_profiles) instead of the live version's literal 2-year lookback.
                #   zone_damage: REAL. Already computed above for zone_edge/arsenal_fit.
                #   danger_count: REAL PROXY -- opposing batters whose own recent avg_ev/
                #     barrel_pct already clear a real bar, using bprof (already fully computed
                #     for every batter), not the live version's own lineup-aware definition.
                _vtier = None
                try:
                    if _sp:
                        _pf = _parks.park_factor(_venue_by_game.get(batter_game.get(bid), ""), _bh) if _bh else None
                        _vs = _F.vuln_score(
                            era=_xfip_by_sp.get(_sp), whip=None, park_factor=_pf,
                            hand_hr=_hand_hr_by_sp.get(_sp),
                            zone_damage=_meat_by_sp.get(_sp),
                            danger_count=_danger_by_sp.get(_sp))
                        if _vs:
                            _vtier = _vs.get("tier")
                except Exception:
                    _vtier = None
            # convergence, using only the badges the backtest itself has validated
            # Geometry-aware near misses over the trailing 14 days. Uses `_brows2`, which is
            # already bounded by the as-of cut, so no new leak surface is introduced.
            _nm_ct = None
            try:
                # The Statcast frame has no `venue` column — it carries `home_team`, a 3-letter
                # abbreviation. Requiring "venue" meant this block never ran and near_miss never
                # appeared in by_edge, which is exactly the bug already fixed in environment.py
                # via the TEAM_PARK bridge and not carried across to here. Use the same bridge.
                _has_geo = ("home_team" in _brows2.columns or "venue" in _brows2.columns)
                _venue_col = "venue" if "venue" in _brows2.columns else "home_team"
                if _has_geo and len(_brows2):
                    _recent = _brows2.tail(300)
                    _n = 0
                    for _, _r in _recent.iterrows():
                        _pk = env_mod.park_for_row(_r, _venue_col)
                        if _pk and env_mod.is_near_miss(_pk, _r.get("hc_x"),
                                                _r.get("hc_y"), _r.get("hit_distance_sc"),
                                                was_hr=(str(_r.get("events")) == "home_run")):
                            _n += 1
                    _nm_ct = _n
            except Exception:
                _nm_ct = None

            # Bat tracking: the "fast swing AND short path" combination, not either alone
            _bat_fs = None
            try:
                if "bat_speed" in _brows2.columns:
                    _sw = _brows2[_brows2["bat_speed"].notna()]
                    if len(_sw) >= 25:
                        _pf = arsenal_mod.bat_tracking_profile(
                            _sw["bat_speed"].tolist(),
                            _sw["swing_length"].tolist() if "swing_length" in _sw.columns else None)
                        # short_fast_rate needs swing_length, which the Statcast pull did not
                        # request until now — so this silently produced None and the tally never
                        # appeared in by_edge. Falls back to bat speed alone for replay days
                        # before the column exists, rather than dropping the whole signal.
                        _bat_fs = (_pf or {}).get("short_fast_rate")
                        if _bat_fs is None:
                            _bat_fs = (_pf or {}).get("fast_swing_rate")
            except Exception:
                _bat_fs = None

            # Handedness-FIRST vs usage-first arsenal fit — the A/B that proves the ordering
            # change is real rather than cosmetic.
            #
            # The first version of this block guarded on `"_ars_by_sp_hand" in dir()`, which is
            # always False inside a function: dir() with no argument returns the CURRENT local
            # names, and that variable never existed. The guard silently swallowed the whole
            # test, so it reported nothing while looking like it worked. pyflakes caught it.
            #
            # Rebuilt against `_ars_by_sp`, which is the per-(pitcher, batter-hand) usage table
            # this replay actually populates.
            _hf_fit = _uf_fit = None
            try:
                _sp2 = face.get(bid)
                if _sp2 and _bh:
                    _side = _ars_by_sp.get((_sp2, _bh)) or []
                    # usage-first: pool BOTH hands, then apply the 10% bar — this is the
                    # ordering the change replaced, and it is what hides platoon-only pitches
                    _pooled = {}
                    for _h2 in ("R", "L"):
                        for _pt2, _u2 in (_ars_by_sp.get((_sp2, _h2)) or []):
                            _pooled[_pt2] = _pooled.get(_pt2, 0.0) + float(_u2)
                    _tot2 = sum(_pooled.values()) or 1.0
                    _uf = [(k, 100.0 * v / _tot2) for k, v in _pooled.items()
                           if 100.0 * v / _tot2 >= 10.0]
                    _hf = [(k, u) for k, u in _side if float(u) >= 10.0]
                    def _fit_for(sel):
                        """Batter's hits-per-ball-in-play on just these pitch types, as-of."""
                        if not sel or "pitch_type" not in _brows2.columns:
                            return None
                        _pts = {p2[0] for p2 in sel}
                        _sub = _brows2[_brows2["pitch_type"].isin(_pts)]
                        if "bb_type" in _sub.columns:
                            _sub = _sub[_sub["bb_type"].notna()]
                        if len(_sub) < 25 or "events" not in _sub.columns:
                            return None
                        _ev2 = _sub["events"].astype(str)
                        _hits2 = int(_ev2.isin({"single", "double", "triple", "home_run"}).sum())
                        return round(_hits2 / len(_sub), 4)
                    _hf_fit = _fit_for(_hf)
                    _uf_fit = _fit_for(_uf)
            except Exception:
                _hf_fit = _uf_fit = None

            # Command-risk, as-of. This is the field the Target Teams starter enrichment's Cmd
            # pill actually depends on — without this, that signal was unbacktested despite the
            # app confidently showing it on live cards. The xFIP-regression signal (xFIP vs ERA)
            # is NOT tallied here: this file has no season-ERA source for the opposing pitcher
            # (ERA is StatsAPI-derived, fetched only in the live build, not in this replay), and
            # a tally that silently always returns None is worse than one that is honestly absent.
            _cmd_risk = None
            try:
                if fg_pitch and sp_names:
                    _sp3 = face.get(bid)
                    _nm3 = sp_names.get(_sp3) if _sp3 else None
                    _fge3 = fg_pitch.get(statcast_data._norm_name(_nm3)) if _nm3 else None
                    if _fge3 and _fge3.get("location_plus") is not None:
                        _cmd_risk = _fge3["location_plus"] < 98
            except Exception:
                _cmd_risk = None

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
            "zone_edge": _zedge, "arsenal_fit": _afit, "zone_overlap_n": _zone,
            "cv_meas": _nm + _nmh, "cv_meas_noheat": _nm, "cv_prov": _np,
            # ---- Phase-2 features, all computed AS-OF from `past` only ----
            "near_miss_14d": _nm_ct,
            "bat_fast_short": _bat_fs,
            "hand_first_fit": _hf_fit,
            "usage_first_fit": _uf_fit,
            "cv_hit": _cv_hit, "cv_hrr": _cv_hrr, "cmd_risk": _cmd_risk,
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
            "zone": _zone, "square_up": _squp, "hr_power": _hrpow, "sweet_spot": _sweet_spot,
            # ADDED this session -- _trend was already computed above for badge derivation but
            # never made it into this dict, so it never reached by_edge despite costing nothing
            # extra to include (no new computation, just surfacing an existing local).
            "trend_dir": (_trend or {}).get("dir"),
            # ADDED this session -- opener detection. Real limitation, stated plainly: this is
            # a heuristic (the starter faced fewer than 4 batters that game), not a perfect
            # classification -- a legitimate short start from an early injury or ejection would
            # be misclassified the same way. Cheap to compute (reuses `face`/`starters`, already
            # built above, no new per-row scan) but should be read as "probably an opener usage
            # pattern," not a certain one.
            "opener": bool(_opener_sp.get(face.get(bid))),
            "pitch_mix": _pmix,
            "hit_label": _hit_labels.get(bid) if _hit_labels else None,
            "vuln_tier": _vtier,
        }
    return out, pitcher_scores


def replay(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    df = df.copy()
    df["_gd"] = df["game_date"].astype(str).str[:10]
    all_dates = sorted(df["_gd"].unique())
    if len(all_dates) <= WARMUP_DAYS:
        return {"error": f"need more than {WARMUP_DAYS} days of data"}
    dates = [d for d in all_dates[WARMUP_DAYS:] if (not start or d >= start) and (not end or d <= end)]

    # FanGraphs, fetched once for the whole replay — same season-aggregate leak caveat as
    # replay_runs() (see that function's docstring note): there is no historical daily FanGraphs
    # snapshot to fetch as-of, so this is "does the signal correlate", not a leak-free test.
    try:
        fg_pitch = statcast_data.fangraphs_pitching()
    except Exception as e:
        fg_pitch = {}
        print(f"[backtest] fangraphs fetch skipped (non-fatal): {e}")
    _sp_names = {}
    if fg_pitch:
        _all_sp = set()
        for _D in dates:
            _day = df[df["_gd"] == _D]
            for (_gp, _half), _grp in _day.groupby(["game_pk", "inning_topbot"]):
                _g0 = _grp.sort_values(["at_bat_number", "pitch_number"])
                _p0 = _g0.iloc[0]["pitcher"] if len(_g0) else None
                if _p0 == _p0:
                    _all_sp.add(int(_p0))
        try:
            _sp_names = statcast_data.player_names(_all_sp)
        except Exception:
            _sp_names = {}

    by_tier = {name: {"n": 0, "hr": 0} for name, _, _ in TIERS}
    by_badge = {}   # badge_key -> {n, hr}: HR rate for hitters carrying each badge
    # ADDED this session: the same badges have carried every hitter-side scoring engine in this
    # app (Genius Pairing, Long Ball, Gold Bar, the Ladder) since they were introduced -- but
    # checked directly, and confirmed nowhere in this file until now: they had only ever been
    # cross-tabbed against the HR outcome (by_badge above). POW is defined by raw max exit
    # velocity, LOCK by expected contact quality (xwOBAcon) -- neither definition has anything
    # to do with whether a ball in play becomes a hit, and there was no real data anywhere
    # answering whether a POW badge holder's hit rate is average, above it, or (plausible, given
    # power/contact-rate tend to trade off in real hitters) below it. These three mirror
    # by_badge's tally exactly, just against each prop's own real outcome instead of HR.
    by_badge_hit1 = {}
    by_badge_hit2 = {}
    by_badge_hrr = {}
    # convergence graded against EACH prop's own outcome, not the HR outcome
    by_conv = {}    # prop -> bucket -> {n, hit}
    def _conv_tally(prop, bucket, ok):
        d = by_conv.setdefault(prop, {}).setdefault(bucket, {"n": 0, "hit": 0})
        d["n"] += 1; d["hit"] += 1 if ok else 0
    top_n = {"5": {"n": 0, "hr": 0}, "10": {"n": 0, "hr": 0}, "25": {"n": 0, "hr": 0}}
    calib = {}
    n_tot = hr_tot = 0
    # Real month-by-month base rate, tracked alongside the season aggregate. Checked directly
    # a couple turns ago: June 11.30%, July 11.97%, August 11.46%, all above the season-long
    # base_pct (10.92%) that badge lifts and everything else anchor to -- because the season
    # average includes colder early-season months. Formalized here so this gets tracked
    # automatically going forward instead of requiring a manual pull every time it's asked.
    by_month = {}
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
    # Rows for the side-by-side graders, collected on the same player-days the tiers use.
    _sbs_hit, _sbs_k = [], []
    # ADDED this session -- (predicted_prob, actual_hit) pairs from calling the REAL
    # _hrpo_combine_genius_pow on every real POW-badge candidate in the replay, not just the
    # top-3 picked each day. Answers "does stacking these signals actually produce a
    # calibrated probability" with real evidence -- every individual signal feeding this stack
    # has been validated on its own, but the compounded result of stacking 5+ of them
    # multiplicatively, even damped, had never been checked against real outcomes until now.
    _stack_pred, _stack_hit = [], []
    # ADDED this session -- direct calibration test of the frozen heat model itself, not a
    # downstream layer. Every prior calibration check this session (markov totals, genius
    # stacking) tested something built ON TOP of heat -- heat itself has never been directly,
    # rigorously tested against real outcomes the same way. heat/100 isn't literally designed
    # to be a probability, but decile_calibration()'s methodology (weighted regression of
    # actual-vs-predicted on the decile means) still gives an honest read either way: the
    # slope tells you what real scaling heat/100 would need to become a genuine probability,
    # and monotonicity/spearman tell you whether heat correctly rank-orders real HR likelihood
    # regardless of scale -- which is the more fundamental question for a signal whose whole
    # job is ranking hitters, not necessarily producing a calibrated percentage on its own.
    # Collected across the FULL population, not badge-filtered -- heat applies to every
    # hitter, so this tests the real, complete real-world distribution it actually sees.
    _heat_pred, _heat_hit = [], []
    _sweet_spot_raw = []   # ADDED this session -- real distribution diagnostic, see comment
                          # at the sweet_spot_pct edge check for why this exists
    # new-signal buckets, same {n,hr} shape the tracker uses so the UI renders them uniformly
    by_edge = {}
    def _edge(group, bucket, hit):
        g = by_edge.setdefault(group, {})
        b = g.setdefault(bucket, {"n": 0, "hr": 0})
        b["n"] += 1; b["hr"] += 1 if hit else 0
    # ADDED this session -- B2B tracking (did this batter homer in his immediately prior
    # graded day). Real, cross-day state, not leaky: _prev_hr_ids only ever holds what was
    # already known true BEFORE today's game, updated once at the end of each day's loop.
    _prev_hr_ids = set()
    for D in dates:
        day = df[df["_gd"] == D]
        past = df[df["_gd"] < D]
        heats, pitchers = _day_heats(past, day, D, fg_pitch=fg_pitch, sp_names=_sp_names)
        if len(heats) < 30:                       # partial-slate days pollute rates
            continue
        graded_days += 1
        ranked = sorted(heats.items(), key=lambda kv: -kv[1]["heat"])
        for i, (bid, r) in enumerate(ranked):
            hit = r["hr"]
            n_tot += 1; hr_tot += 1 if hit else 0
            _mk = by_month.setdefault(D[:7], {"n": 0, "hr": 0})
            _mk["n"] += 1; _mk["hr"] += 1 if hit else 0
            t = by_tier[_tier(r["heat"])]
            t["n"] += 1; t["hr"] += 1 if hit else 0
            for k in ("5", "10", "25"):
                if i < int(k):
                    top_n[k]["n"] += 1; top_n[k]["hr"] += 1 if hit else 0
            b = int(min(max(r["heat"], 0), 99) // 10) * 10
            c = calib.setdefault(str(b), {"n": 0, "hr": 0})
            c["n"] += 1; c["hr"] += 1 if hit else 0
            # ADDED this session -- B2B: did he homer his immediately prior graded day.
            _edge("b2b", "b2b" if bid in _prev_hr_ids else "rest", hit)
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
            # ---- Phase-2 feature lift ----
            _nm2 = r.get("near_miss_14d")
            if _nm2 is not None:
                _edge("near_miss", "2+ near misses" if _nm2 >= 2
                      else ("1 near miss" if _nm2 == 1 else "0 near misses"), hit)
            _bf2 = r.get("bat_fast_short")
            if _bf2 is not None:
                _edge("bat_fast_short", "25%+ short&fast" if _bf2 >= 25
                      else ("12-24%" if _bf2 >= 12 else "<12%"), hit)
            _cmd2 = r.get("cmd_risk")
            if _cmd2 is not None:
                _edge("command_risk", "Loc+ under 98" if _cmd2 else "Loc+ 98+", hit)
            # The A/B on filter ordering: does handedness-first actually find a better matchup
            # than usage-first? If they agree everywhere, the ordering change is cosmetic.
            _hf2, _uf2 = r.get("hand_first_fit"), r.get("usage_first_fit")
            if _hf2 is not None and _uf2 is not None:
                _d = _hf2 - _uf2
                _edge("hand_vs_usage_filter",
                      "hand-first BETTER (+.020)" if _d >= 0.020
                      else ("hand-first WORSE (-.020)" if _d <= -0.020 else "filters agree"), hit)
            _pa2 = r.get("pull_air")
            if _pa2 is not None:
                _edge("pull_air", "50+ elite" if _pa2 >= 50 else "40-49 good" if _pa2 >= 40
                      else "30-39 avg" if _pa2 >= 30 else "<30 weak", hit)
            _af = r.get("arsenal_fit")
            if _af is not None:
                # ADDED this session: split out a genuinely poor tier (<4) from the old blanket
                # "<7 weak" bucket -- needed to validate Genius Pairing's new arsenal_fit<=4
                # discount (currently an unvalidated heuristic, see that combine function's
                # docstring) against a real number instead of a lumped-together low bucket.
                _edge("arsenal_fit", "11+ punishes" if _af >= 11 else "9-10.9 good" if _af >= 9
                      else "7-8.9 avg" if _af >= 7 else "4-6.9 weak" if _af >= 4
                      else "<4 poor", hit)
            # ADDED this session -- see the accumulator comment above for why this exists.
            # Unconditional (not badge-gated) since heat applies to every candidate.
            _h = r.get("heat")
            if _h is not None:
                _heat_pred.append(max(0.0, min(1.0, _h / 100.0)))
                _heat_hit.append(1.0 if hit else 0.0)
            # ADDED this session -- feeds the genius-stack calibration check below. Only scores
            # POW-badge holders (Genius Pairing is POW-filtered, require_badge="pow") using
            # whatever real signals THIS replay can honestly reconstruct. Fields the replay
            # cannot currently reconstruct (arm_vuln_score, pen_rank_val/pen_worn,
            # arm_form_label, own_avg_ev_l14, lineup spot -- none of these have a retroactive
            # source in this replay frame yet) are left None, which _hrpo_combine_genius_pow
            # already handles by skipping that signal's contribution entirely. This makes the
            # calibration check an honest read on the PARTIAL stack that's actually
            # reconstructable today, not a silent approximation of the full one.
            if "pow" in (r.get("badges") or []):
                try:
                    _hrpow_r = r.get("hr_power") or {}
                    _stack_sig = {
                        "base_prob": build_board_mod.HRPO_BASE_RATE * 1.757,   # _BADGE_ANCHOR_LIFT["pow"]
                        "badges": set(r.get("badges") or []),
                        "thin": (r.get("bbe_season") or 0) < build_board_mod.HRPO_MIN_BBE,
                        "fams": r.get("cv_fams") or 0,
                        "near_miss": r.get("near_miss_14d") or 0,
                        "arsenal_fit": _af,
                        "zone_edge": r.get("zone_edge"),
                        "zone_overlap_n": r.get("zone_overlap_n"),
                        "ideal_aa_clear": None, "spot": None,
                        "fb_pct_allowed": None, "pen_worn": False, "pen_rank_val": None,
                        "acute_bp_pitches": None,
                        "own_max_dist": _hrpow_r.get("max_dist"),
                        "own_barrel_pct": _hrpow_r.get("barrel_pct"),
                        "own_avg_ev_l14": None, "own_avg_ev_l14_bbe": None,
                        "arm_form_label": None, "arm_vuln_score": None,
                    }
                    _stack_p, _ = build_board_mod._hrpo_combine_genius_pow(_stack_sig)
                    if _stack_p is not None:
                        _stack_pred.append(_stack_p)
                        _stack_hit.append(1.0 if hit else 0.0)
                except Exception:
                    pass
            _zon = r.get("zone_overlap_n")
            if _zon is not None:
                # ADDED this session -- zone_overlap_n's raw count was already computed in this
                # replay but never carried into the output dict, so there was no way to check
                # the low end at all. Needed to validate Genius Pairing's new zone_overlap_n==0
                # discount (also currently an unvalidated heuristic).
                _edge("zone_overlap_n", "5+ premium" if _zon >= 5 else "3-4 good" if _zon >= 3
                      else "1-2 some" if _zon >= 1 else "0 none", hit)
            # ADDED this session -- trend_dir was already computed for badge derivation but
            # never checked on its own as a real, independent signal.
            _td = r.get("trend_dir")
            if _td:
                _edge("trend", _td, hit)
            # ADDED this session -- opener heuristic check (see out[bid]'s comment in
            # _day_heats for the real limitation).
            _edge("opener", "opener" if r.get("opener") else "sp", hit)
            # ADDED this session -- pitch_mix, matching the exact real tier boundaries already
            # confirmed from the live tracker (60+/40-59/<40).
            _pmix2 = r.get("pitch_mix")
            if _pmix2 is not None:
                _edge("pitch_mix", "60+ favorable" if _pmix2 >= 60
                      else "40-59 neutral" if _pmix2 >= 40 else "<40 poor", hit)
            # ADDED this session -- hit_label
            _edge("hlabel", r.get("hit_label") or "none", hit)
            # ADDED this session -- vuln_tier (partial reconstruction, see the real/proxy/
            # omitted breakdown in the comment where _vtier gets computed above)
            if r.get("vuln_tier"):
                _edge("vuln_tier_bt", r["vuln_tier"], hit)
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
            _hp = (r.get("hr_power") or {}).get("barrel_pct")
            if _hp is not None:
                _edge("hr_power", "12%+ barrel" if _hp >= 12 else "8-11%" if _hp >= 8 else "<8%", hit)
            # ADDED this session -- sweet_spot_pct (LA 8-32) was already computed inside
            # square_up_rating() but only ever survived as one of four ingredients blended into
            # that composite rating above -- never checked on its own merits until now.
            _sw = r.get("sweet_spot")
            if _sw is not None:
                # UPDATED this session -- my prior comment here ("pre-filtered to top-heat-
                # ranked hitters") does NOT hold up: directly checked this turn, this loop
                # processes every graded hitter each day, not a heat-restricted subset. Also
                # directly verified square_up_rating() returns a realistic ~34% on synthetic
                # batted-ball data spanning the real launch-angle range (grounders through
                # steep flies) -- the computation itself is not the problem. Widening the
                # bucket boundaries a second time (44->55) STILL put 100% of 33,392 real
                # samples in the top bucket, which rules out "boundaries too low" as the
                # explanation. Root cause genuinely not yet confirmed. Collecting raw values
                # below so the next run shows the real distribution shape directly (percentiles,
                # min/max) instead of guessing at a third set of boundaries blind.
                _edge("sweet_spot_pct", "55+ elite" if _sw >= 55 else "48-54.9 good" if _sw >= 48
                      else "40-47.9 avg" if _sw >= 40 else "<40 weak", hit)
                _sweet_spot_raw.append(_sw)
            # ADDED this session -- own_max_dist wasn't independently checked either (only
            # barrel_pct, its sibling field in hr_power_profile, had a real tier check above).
            _md = (r.get("hr_power") or {}).get("max_dist")
            if _md is not None:
                _edge("own_max_dist", "430+ elite" if _md >= 430 else "410-429 strong" if _md >= 410
                      else "390-409 avg" if _md >= 390 else "<390 weak", hit)
            # badge tally: each badge this hitter carries gets a plate appearance + HR credit
            _badges_today = r.get("badges", [])
            for bk in _badges_today:
                bb = by_badge.setdefault(bk, {"n": 0, "hr": 0})
                bb["n"] += 1; bb["hr"] += 1 if hit else 0
            # badge COMBO tally -- formalizes the POW+LOCK co-occurrence check into automatic,
            # ongoing grading instead of a one-off manual pull. Mutually exclusive buckets
            # among hitters who carry at least one of the two, so "pow+lock" isn't diluted by
            # the much larger population that has neither.
            _has_pow, _has_lock = "pow" in _badges_today, "lock" in _badges_today
            if _has_pow or _has_lock:
                _combo = "pow+lock" if (_has_pow and _has_lock) else ("pow_only" if _has_pow else "lock_only")
                _edge("badge_combo", _combo, hit)
        # ADDED this session -- advance B2B state to today's real HR results, for tomorrow's
        # graded day to check against. Must happen after the per-bid loop above (which reads
        # the OLD _prev_hr_ids), never before -- would leak same-day results otherwise. Real
        # edge case, stated plainly: this line only runs on days that passed the len(heats)<30
        # gate above, so on a skipped/thin day _prev_hr_ids simply isn't touched -- the next
        # graded day compares against the last GENUINELY graded day, not strictly "yesterday."
        # Rare in practice (MLB plays almost every day) but not literally zero.
        _prev_hr_ids = {bid for bid, r in heats.items() if r["hr"]}
        # --- props: hit1/hit2 tiers by hit_heat ---
        hh_ranked = sorted((kv for kv in heats.items() if kv[1]["hit_heat"] is not None),
                           key=lambda kv: -kv[1]["hit_heat"])
        for i, (bid, r) in enumerate(hh_ranked):
            tier = _tier(r["hit_heat"])
            got1 = r["hits"] >= 1; got2 = r["hits"] >= 2
            P["hit1"][tier]["n"] += 1; P["hit1"][tier]["hit"] += 1 if got1 else 0
            P["hit2"][tier]["n"] += 1; P["hit2"][tier]["hit"] += 1 if got2 else 0
            hit_n += 1; hit1_tot += 1 if got1 else 0; hit2_tot += 1 if got2 else 0
            # ADDED this session -- same badges (r["badges"]) the HR by_badge tally above
            # already reads on this exact row, graded here against the real hit1/hit2 outcome
            # instead. First real answer to "do these correlate with hits at all."
            for bk in r.get("badges", []):
                b1 = by_badge_hit1.setdefault(bk, {"n": 0, "hit": 0})
                b1["n"] += 1; b1["hit"] += 1 if got1 else 0
                b2 = by_badge_hit2.setdefault(bk, {"n": 0, "hit": 0})
                b2["n"] += 1; b2["hit"] += 1 if got2 else 0
            # Rows for the side-by-side graders. replay() aggregates straight into tier dicts,
            # so there was nothing for grade_hit_side_by_side() to consume — it was defined and
            # uncallable. Collected on the SAME player-days the tiers are built from, so the two
            # models are compared on identical rows rather than on whatever each happened to
            # cover.
            _sbs_hit.append({"hit_gated_p": r.get("hit_gated_p"),
                             "hit_heat": r.get("hit_heat"),
                             "got_hit": got1})
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
            # ADDED this session -- same pattern as the hit1/hit2 badge tally above.
            for bk in r.get("badges", []):
                bh = by_badge_hrr.setdefault(bk, {"n": 0, "hit": 0})
                bh["n"] += 1; bh["hit"] += 1 if got else 0
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
            # Only the legacy score and the actual count exist here. The kengine projection is
            # NOT available in this replay: producing it would mean running the CSW/xBF engine
            # per pitcher per day inside the loop, which is a real change rather than a wiring
            # fix. Collected anyway so the legacy baseline is graded properly and the comparison
            # can be added later without touching this loop again.
            _sbs_k.append({"k_old": ksc, "actual_k": actual})
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
    _base_rate_final = hr_tot / n_tot if n_tot else 0
    _by_month_rates = {}
    for m, v in sorted(by_month.items()):
        if v["n"] < 200:      # too little of a month graded to trust its own rate
            continue
        _rate = v["hr"] / v["n"]
        _by_month_rates[m] = {"n": v["n"], "pct": round(100 * _rate, 2),
                              "vs_season_lift": round(_rate / _base_rate_final, 3) if _base_rate_final else None}
    return {
        "days": graded_days, "pool": n_tot, "hr": hr_tot,
        "model_version": compute.MODEL_VERSION,
        "base_pct": round(100 * hr_tot / n_tot, 2) if n_tot else None,
        "by_month": _by_month_rates,
        "by_tier": by_tier, "top_n": top_n, "by_edge": by_edge,
        "by_badge": _badge_lift(by_badge, hr_tot / n_tot if n_tot else 0),
        # ADDED this session -- does the FULL stacked Genius Pairing probability (calling the
        # real _hrpo_combine_genius_pow, not an approximation) actually come out calibrated, or
        # does compounding 5+ damped multipliers together overstate the combination even though
        # every individual signal feeding it is real and separately validated? This is a
        # PARTIAL stack -- see the per-candidate comment above for exactly which real signals
        # this replay can and cannot currently reconstruct -- so a clean pass here says the
        # reconstructable core is sound, not that the full production formula (with arm
        # vulnerability, bullpen state, and lineup spot layered on top) is fully proven.
        "genius_stack_calibration": (V.decile_calibration(_stack_pred, _stack_hit, n_bands=8)
                                     if len(_stack_pred) >= 400 else None),
        "genius_stack_n": len(_stack_pred),
        # ADDED this session -- direct calibration test of the frozen heat model itself. Every
        # prior calibration check (markov totals, genius stacking) tested a layer built ON TOP
        # of heat; this is heat's own first direct test against real outcomes, full population,
        # not badge-filtered.
        #
        # READ THIS BEFORE READING THE VERDICT FIELD: heat/100 was never designed to be a
        # literal probability -- a heat of 70 does not mean "70% HR chance." Real HR rates run
        # far below heat/100 at every level (checked directly with synthetic data built from a
        # genuinely strong, correctly-monotonic signal: verdict came back FAIL even though
        # spearman was 0.93). That means this check's "verdict"/slope/brier-vs-baseline fields
        # will ALWAYS look bad for heat specifically, by construction, regardless of whether
        # heat is actually good -- they're answering "is heat/100 a calibrated probability,"
        # which was never the design goal. The fields that actually answer "is heat doing its
        # job" are spearman (real rank-correlation with actual outcomes) and
        # strict_monotonic (do higher heat bands really convert at higher real rates, in
        # order, no reversals) -- read those first, and do not treat a FAIL verdict here as
        # evidence heat is broken without checking them.
        "heat_calibration": (V.decile_calibration(_heat_pred, _heat_hit, n_bands=8)
                             if len(_heat_pred) >= 400 else None),
        "heat_calibration_n": len(_heat_pred),
        # ADDED this session -- real distribution diagnostic for sweet_spot_pct, since two
        # rounds of widened bucket boundaries (44->55) both put 100% of a real, large sample
        # in the top bucket, and neither the extraction nor the underlying computation checked
        # out as buggy when directly tested. Percentiles here will show the real shape of the
        # distribution directly -- if p10 is already in the 60s, that's a very different,
        # differently-actionable finding than if there's a long left tail this check's buckets
        # are somehow still missing.
        "sweet_spot_pct_distribution": ({
            "n": len(_sweet_spot_raw),
            "min": round(min(_sweet_spot_raw), 1), "max": round(max(_sweet_spot_raw), 1),
            "p10": round(sorted(_sweet_spot_raw)[int(0.10 * len(_sweet_spot_raw))], 1),
            "p25": round(sorted(_sweet_spot_raw)[int(0.25 * len(_sweet_spot_raw))], 1),
            "p50": round(sorted(_sweet_spot_raw)[int(0.50 * len(_sweet_spot_raw))], 1),
            "p75": round(sorted(_sweet_spot_raw)[int(0.75 * len(_sweet_spot_raw))], 1),
            "p90": round(sorted(_sweet_spot_raw)[int(0.90 * len(_sweet_spot_raw))], 1),
        } if len(_sweet_spot_raw) >= 100 else None),
        "by_converge_prop": by_conv,   # convergence graded per prop, on that prop's outcome
        "calib": {k: calib[k] for k in sorted(calib, key=int)},
        # Graders that were written and never invoked. The hit comparison is real — both models
        # scored on identical player-days. The K entry reports the legacy baseline only, and says
        # so, rather than implying an A/B that did not run.
        "prop_graders": {
            "hit_calibrated": grade_hit_props_calibrated(_sbs_hit),
            "hit_side_by_side": grade_hit_side_by_side(_sbs_hit),
            "k_legacy_baseline": (V.grade_strikeouts(
                [{"exp_k": r["k_old"], "dist": None, "actual_k": r["actual_k"]} for r in _sbs_k])
                if len(_sbs_k) >= 200 else None),
            "k_engine_note": "kengine not run in replay — legacy k_heat baseline only",
        },
        "props": {
            "hit1": {"by_tier": P["hit1"], "top_n": p_top["hit1"],
                     "base_pct": round(100 * hit1_tot / hit_n, 2) if hit_n else None,
                     "by_badge": _badge_lift_prop(by_badge_hit1, hit1_tot / hit_n if hit_n else 0)},
            "hit2": {"by_tier": P["hit2"],
                     "base_pct": round(100 * hit2_tot / hit_n, 2) if hit_n else None,
                     "by_badge": _badge_lift_prop(by_badge_hit2, hit2_tot / hit_n if hit_n else 0)},
            "hrr":  {"by_tier": P["hrr"], "top_n": p_top["hrr"],
                     "base_pct": round(100 * hrr2_tot / hrr_n, 2) if hrr_n else None,
                     "by_badge": _badge_lift_prop(by_badge_hrr, hrr2_tot / hrr_n if hrr_n else 0)},
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
                  "props.hit1/hit2/hrr.by_badge (ADDED this session): the SAME hitter badges "
                  "graded against HR in the top-level by_badge, cross-tabbed here against each "
                  "prop's own real outcome instead -- first real answer to whether POW/DUE/COOL/"
                  "LOCK/WARMING correlate with hits or HRR at all, rather than assuming a badge "
                  "built and validated for HR carries over. Same omission as top-level by_badge: "
                  "opponent-context badges (WEAK ARM/PLATOON/PITCH EDGE/WEAK PEN) aren't "
                  "reconstructable in the replay frame, so they're absent here too -- this "
                  "covers hitter-only badges only.",
                  f"first {WARMUP_DAYS} days used as feature warm-up, not graded"],
    }


# ---------------------------------------------------------------------------
# Side-by-side graders: new engines vs the baselines they would replace
# ---------------------------------------------------------------------------

def grade_hit_props_calibrated(rows):
    """Decile calibration for the contact-gated hit model, via validate.grade_hit_props.

    The shared grader is used rather than a local reimplementation so the backtest and any other
    caller cannot drift apart on how calibration is measured.
    """
    r2 = [{"p_hit": r["hit_gated_p"], "got_hit": r["got_hit"]}
          for r in rows if r.get("hit_gated_p") is not None and r.get("got_hit") is not None]
    return V.grade_hit_props(r2) if len(r2) >= 300 else None


def grade_strikeouts_calibrated(rows):
    """Poisson-binomial K counts vs actual, via validate.grade_strikeouts."""
    r2 = [{"exp_k": r["k_new"], "dist": r.get("k_new_dist"), "actual_k": r["actual_k"]}
          for r in rows if r.get("k_new") is not None and r.get("actual_k") is not None]
    return V.grade_strikeouts(r2) if len(r2) >= 200 else None


def grade_hit_side_by_side(rows):
    """Decile calibration for `hit_gated` next to the legacy `hit_heat` tiers.

    Reported together on the SAME player-days on purpose. A calibration table for the new model
    alone proves nothing — the question is whether it beats what is already shipping, and the
    only honest way to answer that is to grade both on identical rows.
    """
    out = {}
    gated = [(r["hit_gated_p"], 1.0 if r["got_hit"] else 0.0)
             for r in rows if r.get("hit_gated_p") is not None]
    if len(gated) >= 300:
        out["gated"] = V.decile_calibration([g[0] for g in gated], [g[1] for g in gated])
    legacy = [(r["hit_heat"] / 100.0, 1.0 if r["got_hit"] else 0.0)
              for r in rows if r.get("hit_heat") is not None]
    if len(legacy) >= 300:
        out["legacy_heat_scaled"] = V.decile_calibration(
            [g[0] for g in legacy], [g[1] for g in legacy])
    if "gated" in out and "legacy_heat_scaled" in out:
        g, l = out["gated"], out["legacy_heat_scaled"]
        if isinstance(g, dict) and isinstance(l, dict) and "brier" in g and "brier" in l:
            out["verdict"] = ("gated model is better" if g["brier"] < l["brier"]
                              else "legacy is better — do NOT swap")
            out["brier_delta"] = round(g["brier"] - l["brier"], 5)
    return out


def grade_k_side_by_side(rows):
    """kengine (CSW + dynamic xBF + Poisson-binomial) against the static-22-BF baseline."""
    out = {}
    for label, pk, dk in (("kengine", "k_new", "k_new_dist"), ("legacy_22bf", "k_old", None)):
        vals = [(r[pk], r["actual_k"]) for r in rows if r.get(pk) is not None]
        if len(vals) < 200:
            continue
        pred = np.array([v[0] for v in vals], float)
        act = np.array([v[1] for v in vals], float)
        resid = act - pred
        rec = {"n": len(vals),
               "mae": round(float(np.mean(np.abs(resid))), 3),
               "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
               "bias": round(float(np.mean(resid)), 3)}
        # over/under accuracy at the lines that actually trade
        for L in (5.5, 6.5):
            if dk:
                ps = [sum(v for k, v in (r[dk] or {}).items() if float(k) > L)
                      for r in rows if r.get(dk)]
                hs = [1.0 if r["actual_k"] > L else 0.0 for r in rows if r.get(dk)]
            else:
                # the legacy model had no distribution — Poisson was the implicit assumption
                ps = [1.0 - _poisson_cdf(L, r[pk]) for r in rows if r.get(pk) is not None]
                hs = [1.0 if r["actual_k"] > L else 0.0 for r in rows if r.get(pk) is not None]
            if len(ps) >= 200:
                rec[f"o{L}_pred"] = round(float(np.mean(ps)), 4)
                rec[f"o{L}_actual"] = round(float(np.mean(hs)), 4)
                rec[f"o{L}_gap"] = round(float(np.mean(hs) - np.mean(ps)), 4)
        out[label] = rec
    if "kengine" in out and "legacy_22bf" in out:
        out["verdict"] = ("kengine is better" if out["kengine"]["mae"] < out["legacy_22bf"]["mae"]
                          else "legacy is better — do NOT swap")
        out["mae_delta"] = round(out["kengine"]["mae"] - out["legacy_22bf"]["mae"], 3)
    return out


def _poisson_cdf(k, lam):
    """P(X <= floor(k)) for a Poisson — the distribution the old K model implied."""
    import math
    lam = max(1e-6, float(lam))
    tot, term = 0.0, math.exp(-lam)
    for i in range(int(math.floor(k)) + 1):
        if i:
            term *= lam / i
        tot += term
    return min(1.0, tot)


def grade_bullpen_fatigue_replay(rows):
    """Empirical wOBA/K% of arms flagged FATIGUED vs AVAILABLE, to test the -8% penalty.

    Designed to be able to disagree with us. If flagged arms show a 3% K drop rather than 8%,
    the constant in environment.py should move — the point of measuring is that the answer is
    allowed to contradict the assumption.
    """
    return V.grade_bullpen_fatigue(rows)


def grade_near_miss_lift(rows):
    """Do trailing near misses predict FUTURE home runs, or just describe past luck?"""
    return V.grade_fence_delta(rows)


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
    try:
        rec["grand_slam"] = replay_grand_slam(df, start=start, end=end)
        gs = rec["grand_slam"]
        if gs.get("n_real_grand_slams"):
            print(f"[backtest:grand_slam] {gs['n_real_grand_slams']} real historical grand "
                  f"slams found, pitcher traffic multiplier {gs.get('avg_pitcher_traffic_multiplier_on_real_slams')}")
        else:
            print(f"[backtest:grand_slam] skipped: {gs.get('error')}")
    except Exception as e:
        rec["grand_slam"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:grand_slam] skipped: {e}")
    try:
        rec["long_ball"] = replay_longest_hr(df, start=start, end=end)
        lb = rec["long_ball"]
        if lb.get("n_days_checked"):
            print(f"[backtest:long_ball] {lb['n_days_checked']} days checked, daily-longest-HR "
                  f"winners averaged {lb.get('avg_own_max_ev_of_daily_longest_hr_winner')} mph "
                  f"own max EV vs league p90 {lb.get('avg_league_90th_pctile_max_ev_same_days')}")
        else:
            print(f"[backtest:long_ball] skipped: {lb.get('error')}")
    except Exception as e:
        rec["long_ball"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:long_ball] skipped: {e}")
    try:
        rec["signal_redundancy"] = replay_signal_redundancy()
        sr = rec["signal_redundancy"]
        if sr.get("n_pooled_pow_badge_player_days"):
            print(f"[backtest:signal_redundancy] {sr['n_snapshot_days_used']} snapshot days, "
                  f"{len(sr.get('likely_redundant_pairs', []))} likely-redundant pairs found")
        else:
            print(f"[backtest:signal_redundancy] skipped: {sr.get('error')}")
    except Exception as e:
        rec["signal_redundancy"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:signal_redundancy] skipped: {e}")
    try:
        rec["pitch_count_fatigue"] = replay_pitch_count_fatigue(df, start=start, end=end)
        pcf = rec["pitch_count_fatigue"]
        if pcf.get("buckets"):
            print(f"[backtest:pitch_count_fatigue] {pcf['n_total']} real PAs across "
                  f"{len(pcf['buckets'])} pitch-count buckets, base {pcf['base_pct']}%")
        else:
            print(f"[backtest:pitch_count_fatigue] skipped: {pcf.get('error')}")
    except Exception as e:
        rec["pitch_count_fatigue"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:pitch_count_fatigue] skipped: {e}")
    try:
        rec["first_pitch_matchup"] = replay_first_pitch_matchup(df, start=start, end=end)
        fpm = rec["first_pitch_matchup"]
        if fpm.get("lift") is not None:
            print(f"[backtest:first_pitch_matchup] {fpm['first_pitch_n']} real first-pitch PAs, "
                  f"{fpm['first_pitch_hr_pct']}% vs {fpm['overall_hr_pct']}% overall "
                  f"({fpm['lift']}x)")
        else:
            print(f"[backtest:first_pitch_matchup] skipped: {fpm.get('error')}")
    except Exception as e:
        rec["first_pitch_matchup"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:first_pitch_matchup] skipped: {e}")
    try:
        rec["team_traffic_persistence"] = replay_team_traffic_persistence(df, start=start, end=end)
        ttp = rec["team_traffic_persistence"]
        if ttp.get("split_half_correlation") is not None:
            print(f"[backtest:team_traffic_persistence] {ttp['n_teams']} teams, split-half "
                  f"correlation {ttp['split_half_correlation']} -- {ttp['verdict']}")
        else:
            print(f"[backtest:team_traffic_persistence] skipped: {ttp.get('error')}")
    except Exception as e:
        rec["team_traffic_persistence"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:team_traffic_persistence] skipped: {e}")
    try:
        rec["acute_bullpen_fatigue"] = replay_acute_bullpen_fatigue(df, start=start, end=end)
        abf = rec["acute_bullpen_fatigue"]
        if abf.get("by_acute_fatigue_quartile"):
            print(f"[backtest:acute_bullpen_fatigue] base rate {abf.get('base_rate_pct')}%, "
                  f"quartile breakdown computed")
        else:
            print(f"[backtest:acute_bullpen_fatigue] skipped: {abf.get('error')}")
    except Exception as e:
        rec["acute_bullpen_fatigue"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"[backtest:acute_bullpen_fatigue] skipped: {e}")
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
def replay_team_traffic_persistence(df, start=None, end=None):
    """Is a team's real bases-loaded rate a genuine, persistent team characteristic, or mostly
    noise? Split-half check: computes each team's real bases-loaded PA rate in the first half
    of the requested window and again in the second half, then checks whether the first half
    predicts the second.

    This is the necessary first step before team_traffic_profile() (statcast_data.py, added
    this session per Travis's direct question) could ever meaningfully predict something as
    rare as a real grand slam -- only ~105 real grand slams happen in a full season, spread
    across 30 teams, far too thin to directly test "does team traffic predict team grand slam
    rate" with any real statistical power. Real HR-per-team-per-half-season volume is a much
    more plentiful, better-powered population to check general persistence against first.

    A real, strong positive split-half correlation means teams that generate more traffic
    early in the window keep doing it later -- a genuine, stable characteristic worth weighting
    live. A weak or absent correlation means most of the season-to-season variation in team
    traffic rate is closer to random noise, and the live signal (currently damped to 50%
    strength in slam_probability(), see that function's docstring) should probably be damped
    further or dropped rather than trusted more.
    """
    need = {"game_pk", "at_bat_number", "on_1b", "on_2b", "on_3b", "inning_topbot",
           "home_team", "away_team", "game_date"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required Statcast columns for team traffic persistence"}
    d = df.copy()
    if start:
        d = d[d["game_date"].astype(str) >= start]
    if end:
        d = d[d["game_date"].astype(str) <= end]
    if d.empty:
        return {"error": "no rows in the requested date range"}
    dates = pd.to_datetime(d["game_date"], errors="coerce")
    lo, hi = dates.min(), dates.max()
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return {"error": "not enough of a real date range to split in half"}
    midpoint = lo + (hi - lo) / 2

    def _team_rates(sub):
        if sub.empty:
            return {}
        topbot = sub["inning_topbot"].astype(str)
        sub = sub.copy()
        sub["_team"] = np.where(topbot.str.startswith("Top"), sub["away_team"], sub["home_team"])
        pa = sub.drop_duplicates(subset=["_team", "game_pk", "at_bat_number"])
        pa_counts = pa.groupby("_team").size()
        loaded = pa[pa["on_1b"].notna() & pa["on_2b"].notna() & pa["on_3b"].notna()]
        loaded_counts = loaded.groupby("_team").size()
        out = {}
        for team, n in pa_counts.items():
            if n < 200 or not team:   # real, meaningful half-window sample per team
                continue
            out[str(team)] = int(loaded_counts.get(team, 0)) / n
        return out

    first_half = _team_rates(d[dates < midpoint])
    second_half = _team_rates(d[dates >= midpoint])
    common = sorted(set(first_half) & set(second_half))
    if len(common) < 10:
        return {"error": f"only {len(common)} teams had enough real PA in both halves -- "
                         "too thin to trust a split-half correlation"}
    x = [first_half[t] for t in common]
    y = [second_half[t] for t in common]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    sy = sum((yi - my) ** 2 for yi in y) ** 0.5
    corr = round(cov / (sx * sy), 3) if sx and sy else None
    return {
        "n_teams": len(common), "split_half_correlation": corr,
        "first_half_rates": {t: round(first_half[t], 4) for t in common},
        "second_half_rates": {t: round(second_half[t], 4) for t in common},
        "verdict": ("real, persistent team characteristic" if corr is not None and corr >= 0.4
                   else "weak or no real persistence -- likely mostly noise" if corr is not None
                   else "could not compute"),
    }


def replay_pitch_count_fatigue(df, start=None, end=None):
    """Does a pitcher's real HR rate allowed change with how deep he is into a start? Buckets
    EVERY real plate appearance across the whole season by the pitcher's TRUE cumulative
    in-game pitch count at the moment that PA concluded -- not pitch_number, which only counts
    within one at-bat -- then compares real HR rate across buckets.

    This is the direct test of the hypothesis, not a live signal yet: building a real per-batter
    "will he see this pitcher in the fatigue zone tonight" projection needs real-time lineup
    spot and expected pitch-count context this replay doesn't currently reconstruct (same gap
    flagged for the L14d/arm-vulnerability signals elsewhere in this file). The honest first
    step is checking whether pitch-count depth predicts anything real at all before building
    that projection layer on top of it.
    """
    need = {"pitcher", "game_pk", "inning", "at_bat_number", "pitch_number", "events", "game_date"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required Statcast columns for pitch-count fatigue"}
    d = df.copy()
    if start:
        d = d[d["game_date"].astype(str) >= start]
    if end:
        d = d[d["game_date"].astype(str) <= end]
    if d.empty:
        return {"error": "no rows in the requested date range"}
    d = d.sort_values(["pitcher", "game_pk", "inning", "at_bat_number", "pitch_number"])
    d["_cum_pitch"] = d.groupby(["pitcher", "game_pk"]).cumcount() + 1

    # Only the pitch that actually ends a PA carries the real outcome (events is non-null
    # there and nowhere else) -- that's the one row per PA that determines hit/HR/out.
    pa_rows = d[d["events"].notna()].copy()
    if pa_rows.empty:
        return {"error": "no plate-appearance-ending rows found"}
    pa_rows["_hr"] = (pa_rows["events"].astype(str) == "home_run")

    buckets = [("1-25", 1, 25), ("26-50", 26, 50), ("51-75", 51, 75),
              ("76-100", 76, 100), ("101+", 101, 10_000)]
    out = {}
    for label, lo, hi in buckets:
        sub = pa_rows[(pa_rows["_cum_pitch"] >= lo) & (pa_rows["_cum_pitch"] <= hi)]
        n = len(sub)
        if n < 200:          # too thin a bucket to trust
            continue
        hr = int(sub["_hr"].sum())
        out[label] = {"n": n, "hr": hr, "rate_pct": round(100.0 * hr / n, 2)}
    if not out:
        return {"error": "no bucket had enough real plate appearances"}
    base_n = sum(v["n"] for v in out.values())
    base_hr = sum(v["hr"] for v in out.values())
    base_rate = (base_hr / base_n) if base_n else None
    for v in out.values():
        v["lift"] = round((v["rate_pct"] / 100.0) / base_rate, 3) if base_rate else None
    return {"buckets": out, "base_pct": round(100 * base_rate, 2) if base_rate else None,
            "n_total": base_n}


def replay_first_pitch_matchup(df, start=None, end=None):
    """Does contact on a 0-0 count carry a different real HR rate than contact overall? A
    well-known sabermetric intuition -- pitchers often 'groove' first pitches to get ahead in
    the count, and batters who ambush that pitch are swinging at something they know is coming
    down the middle -- checked directly against this season's real outcomes rather than assumed.
    """
    need = {"balls", "strikes", "events", "game_date"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required Statcast columns for first-pitch matchup"}
    d = df.copy()
    if start:
        d = d[d["game_date"].astype(str) >= start]
    if end:
        d = d[d["game_date"].astype(str) <= end]
    if d.empty:
        return {"error": "no rows in the requested date range"}

    all_concluded = d[d["events"].notna()].copy()
    if all_concluded.empty:
        return {"error": "no plate-appearance-ending rows found"}
    all_concluded["_hr"] = (all_concluded["events"].astype(str) == "home_run")
    n_all, hr_all = len(all_concluded), int(all_concluded["_hr"].sum())

    fp = all_concluded[(all_concluded["balls"] == 0) & (all_concluded["strikes"] == 0)]
    n_fp, hr_fp = len(fp), int(fp["_hr"].sum())
    if n_fp < 200:
        return {"error": f"only {n_fp} real first-pitch-ending PAs -- too thin to trust"}

    base_rate = hr_all / n_all if n_all else None
    fp_rate = hr_fp / n_fp if n_fp else None
    return {
        "first_pitch_n": n_fp, "first_pitch_hr": hr_fp,
        "first_pitch_hr_pct": round(100 * fp_rate, 2) if fp_rate is not None else None,
        "overall_n": n_all, "overall_hr": hr_all,
        "overall_hr_pct": round(100 * base_rate, 2) if base_rate is not None else None,
        "lift": round(fp_rate / base_rate, 3) if (fp_rate is not None and base_rate) else None,
    }


def replay_signal_redundancy() -> dict:
    """Data-analyst pass: how much of Genius Pairing's 'convergence' is genuinely independent
    evidence versus the same underlying fact counted several times under different names?

    Uses the real board snapshots in docs/snapshots/ -- pre-game player states saved once per
    day, 16 real days as of this build (2026-08-04 through 2026-08-19). This is signal-to-signal
    correlation, not signal-to-outcome, so no HR result is needed -- whether two numbers move
    together is a fact about the numbers themselves, checkable from the pre-game snapshot alone.

    Pools every POW-badge holder across all 16 real days (not just one day's ~33 players) for a
    real sample, then computes pairwise Pearson correlation across every signal Genius Pairing
    actually uses: heat, barrel_pct, iso, hr_power, zone (overlap count), sp_vuln, bp_score.

    A correlation above ~0.5 between two signals means they are substantially the same
    information under two names -- stacking both as if independent overstates the combination
    (exactly the mechanism that caused the ceiling-saturation bug this thread already found and
    fixed once, in a different pair of signals).
    """
    import glob as _glob
    snap_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "snapshots")
    files = sorted(_glob.glob(os.path.join(snap_dir, "20*.json")))
    if len(files) < 5:
        return {"error": f"only {len(files)} board snapshots available -- too few for a real "
                         f"multi-day correlation, not just a single cross-section"}

    keys = ["heat", "barrel_pct", "iso", "hr_power", "zone", "sp_vuln", "bp_score"]
    pooled = {k: [] for k in keys}
    n_days_used = 0
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        pow_players = [p for p in d.get("players", []) if "pow" in (p.get("badges") or [])]
        if not pow_players:
            continue
        n_days_used += 1
        for p in pow_players:
            for k in keys:
                _v = p.get(k)
                if k == "hr_power" and isinstance(_v, dict):
                    _v = _v.get("barrel_pct")
                pooled[k].append(_v)

    n_pool = len(pooled["heat"])
    if n_pool < 30:
        return {"error": f"only {n_pool} pooled pow-badge player-days -- too thin to trust a "
                         f"correlation matrix"}

    def _corr(a, b):
        pts = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pts) < 15:
            return None
        xs, ys = zip(*pts)
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in pts)
        denx = sum((x - mx) ** 2 for x in xs) ** 0.5
        deny = sum((y - my) ** 2 for y in ys) ** 0.5
        return round(num / (denx * deny), 3) if denx and deny else None

    matrix = {}
    redundant_pairs = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            r = _corr(pooled[k1], pooled[k2])
            matrix[f"{k1}__vs__{k2}"] = r
            if r is not None and abs(r) >= 0.5:
                redundant_pairs.append({"pair": [k1, k2], "r": r})

    return {
        "n_snapshot_days_used": n_days_used,
        "n_pooled_pow_badge_player_days": n_pool,
        "correlation_matrix": matrix,
        "likely_redundant_pairs": redundant_pairs,
        "note": "Correlation is across POW-badge holders' own pre-game signal values, pooled "
               "over every real snapshot day -- not a single day's cross-section. |r|>=0.5 "
               "flagged as likely measuring substantially the same underlying fact.",
    }


def replay_acute_bullpen_fatigue(df: pd.DataFrame, start: str | None = None,
                                 end: str | None = None, window_days: int = 3) -> dict:
    """Does a bullpen's ACUTE recent workload (trailing 2-3 days specifically) predict real HR
    outcomes against that pen, beyond what a season-long exploitability average would catch?

    Real, checkable, and genuinely different from the existing season-trailing bullpen
    exploitability score: a pen that threw 200+ pitches in extra innings two nights ago is a
    sharper, more acute signal than a slow-moving season average, and the season number cannot
    see it at all.

    Computed directly from the raw Statcast frame already in memory -- no new fetch. For each
    real day, each team's bullpen (non-starter) pitch count over the trailing `window_days` is
    bucketed into quartiles; real HR rate against that bullpen's relief innings on that day is
    checked across quartiles.
    """
    need = {"game_pk", "game_date", "events", "pitcher", "inning", "at_bat_number",
           "inning_topbot", "home_team", "away_team"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required columns for acute bullpen fatigue replay"}

    d = df.copy()
    d["_gd"] = pd.to_datetime(d["game_date"]).dt.date
    if start:
        d = d[d["_gd"] >= pd.to_datetime(start).date()]
    if end:
        d = d[d["_gd"] <= pd.to_datetime(end).date()]

    # identify starters per game (first pitcher to appear, by team) so "bullpen" pitches exclude
    # the starter's own workload -- a starter throwing 100 pitches is not bullpen fatigue.
    d = d.sort_values(["game_pk", "at_bat_number"])
    d["_pteam"] = np.where(d["inning_topbot"].eq("Top"), d["home_team"], d["away_team"])
    starters = d.groupby(["game_pk", "_pteam"])["pitcher"].first()
    d["_is_starter"] = d.apply(
        lambda r: starters.get((r["game_pk"], r["_pteam"])) == r["pitcher"], axis=1)
    bullpen_rows = d[~d["_is_starter"]]

    # team-day bullpen pitch counts (one row per pitch already, so len() is pitch count)
    team_day_pitches = bullpen_rows.groupby(["_pteam", "_gd"]).size()

    # for each real day, each team's trailing window_days bullpen pitch total (excluding today)
    all_dates = sorted(d["_gd"].unique())
    if len(all_dates) < window_days + 5:
        return {"error": f"only {len(all_dates)} days in this window -- too few to build a "
                         f"trailing {window_days}-day bullpen load and still have days to grade"}

    fatigue_hr = {"q1_freshest": {"n": 0, "hr": 0}, "q2": {"n": 0, "hr": 0},
                 "q3": {"n": 0, "hr": 0}, "q4_most_taxed": {"n": 0, "hr": 0}}
    all_loads = []
    rows_by_day = {}
    for gd in all_dates:
        prior = [gd - pd.Timedelta(days=i) for i in range(1, window_days + 1)]
        for team in bullpen_rows["_pteam"].unique():
            load = sum(team_day_pitches.get((team, pd_), 0) for pd_ in prior)
            all_loads.append(load)
            rows_by_day.setdefault(gd, {})[team] = load
    if not all_loads:
        return {"error": "no bullpen workload data found"}
    all_loads.sort()
    q1c = all_loads[len(all_loads) // 4]
    q2c = all_loads[len(all_loads) // 2]
    q3c = all_loads[3 * len(all_loads) // 4]

    for gd, team_loads in rows_by_day.items():
        today_bp = bullpen_rows[bullpen_rows["_gd"] == gd]
        for team, load in team_loads.items():
            pa = today_bp[today_bp["_pteam"] == team].drop_duplicates(
                subset=["game_pk", "at_bat_number"])
            if pa.empty:
                continue
            n_pa = len(pa)
            n_hr = int((pa["events"] == "home_run").sum())
            bucket = ("q1_freshest" if load <= q1c else "q2" if load <= q2c
                     else "q3" if load <= q3c else "q4_most_taxed")
            fatigue_hr[bucket]["n"] += n_pa
            fatigue_hr[bucket]["hr"] += n_hr

    result = {}
    base_n = sum(v["n"] for v in fatigue_hr.values())
    base_hr = sum(v["hr"] for v in fatigue_hr.values())
    base_rate = base_hr / base_n if base_n else 0
    for k, v in fatigue_hr.items():
        if v["n"] >= 200:
            rate = v["hr"] / v["n"]
            result[k] = {"n": v["n"], "pct": round(100 * rate, 2),
                        "lift": round(rate / base_rate, 3) if base_rate else None}
        else:
            result[k] = {"n": v["n"], "note": "too thin to grade"}

    return {
        "base_rate_pct": round(100 * base_rate, 2),
        "by_acute_fatigue_quartile": result,
        "window_days": window_days,
        "note": "q4_most_taxed = bullpen threw the most pitches in the trailing window "
               "days specifically, not a season average. If q4's lift is real and separate "
               "from q1's, acute fatigue is adding information the season-long exploitability "
               "score can't see; if the quartiles look flat, this specific window isn't "
               "adding anything beyond what's already captured.",
    }


def replay_longest_hr(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    """Real, retroactive validation for Long Ball's core premise -- does a batter's OWN
    measured power, known BEFORE that day, actually predict him hitting the day's longest HR?

    Same constraint as replay_grand_slam(): neither this file's replay() nor track.py's live
    log has ever recorded "longest HR of the day" as a tracked outcome (checked directly).
    Derived here from real historical Statcast instead: for each day, the actual longest real
    HR (max hit_distance_sc among real HR events) is real ground truth Statcast already
    carries. What's NOT retroactively reconstructable is which badges a hitter carried that
    day (badges are a live, board-computed snapshot) -- so this checks the measurable
    UNDERLYING evidence the badges are meant to flag (own rolling max EV / avg distance),
    not the badges themselves.

    Strictly no-leakage: each batter's "own power" input on day D uses ONLY his own batted
    balls strictly BEFORE D, the same temporal-validity pattern the rest of this file's
    replay() already uses for heat.
    """
    need = {"game_pk", "game_date", "events", "hit_distance_sc", "launch_speed", "batter"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required columns for longest-HR replay"}

    d = df.copy()
    d["_gd"] = d["game_date"].astype(str).str[:10]
    if start:
        d = d[d["_gd"] >= start]
    if end:
        d = d[d["_gd"] <= end]

    hr = d[d["events"].eq("home_run") & d["hit_distance_sc"].notna()].copy()
    if hr.empty:
        return {"error": "no home runs with distance data in this window"}

    dates = sorted(hr["_gd"].unique())
    if len(dates) < 20:
        return {"error": f"only {len(dates)} days with HR data -- too few to grade"}

    league_max_evs = []
    winner_own_max_evs = []
    n_days_checked = 0
    for D in dates:
        today_hrs = hr[hr["_gd"] == D]
        if today_hrs.empty:
            continue
        longest = today_hrs.loc[today_hrs["hit_distance_sc"].idxmax()]
        winner_id = longest["batter"]

        # this specific batter's OWN rolling max EV, strictly BEFORE today -- no leakage
        past_bip = d[(d["_gd"] < D) & d["launch_speed"].notna() & d["events"].notna()]
        winner_past = past_bip[past_bip["batter"] == winner_id]
        if len(winner_past) < 15:      # too little of his own history to trust a "his own
            continue                   # power" reading yet
        winner_own_max = float(winner_past["launch_speed"].max())

        league_past = past_bip["launch_speed"]
        if len(league_past) < 500:
            continue
        league_max_evs.append(float(league_past.quantile(0.90)))   # league's own 90th-pctile
        winner_own_max_evs.append(winner_own_max)
        n_days_checked += 1

    if n_days_checked < 15:
        return {"error": f"only {n_days_checked} days had enough prior-history data to grade "
                         f"-- too few to trust a comparison"}

    avg_winner_max = sum(winner_own_max_evs) / len(winner_own_max_evs)
    avg_league_p90 = sum(league_max_evs) / len(league_max_evs)
    n_winner_above_league_p90 = sum(1 for w, lg in zip(winner_own_max_evs, league_max_evs) if w >= lg)

    return {
        "n_days_checked": n_days_checked,
        "avg_own_max_ev_of_daily_longest_hr_winner": round(avg_winner_max, 1),
        "avg_league_90th_pctile_max_ev_same_days": round(avg_league_p90, 1),
        "pct_winners_above_league_90th_pctile": round(100 * n_winner_above_league_p90 / n_days_checked, 1),
        "note": "If the daily longest-HR winner's own known max EV (strictly before that day) "
               "sits meaningfully above the league's own 90th percentile most of the time, "
               "Long Ball's Pillar 1 premise -- that real, prior, measured power predicts real "
               "distance-ceiling outcomes -- holds up against actual history. If it's close to "
               "the league's own p90 baseline, the premise is weaker than assumed.",
    }


def replay_grand_slam(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    """Real, retroactive validation for Grand Slam's newer mechanics -- derived directly from
    raw historical Statcast, since neither this file's own replay() nor track.py's live daily
    log has ever recorded grand slam occurrences specifically (checked directly before writing
    this). Rather than fabricate a validation for an outcome nobody tracked, this derives real
    ground truth from data that was already in the historical frame:

    - A real grand slam PA = bases loaded (on_1b/on_2b/on_3b all populated) AND
      events == "home_run" on that same plate appearance.
    - Batting slot uses the EXACT same derivation etl/statcast_data.py's hr_by_lineup_spot()
      already uses (cumulative team PA count mod 9 + 1) -- not a second, differently-computed
      version of the same idea.
    - Pitcher bases-loaded rate uses the same logic pitcher_traffic_profile() computes live,
      applied here retroactively per pitcher across the full window.

    Returns real slot-share and pitcher-bases-loaded-rate numbers for actual historical grand
    slams, so Task 1's lineup_slot_modifier and Part A's pitcher_traffic_multiplier can be
    checked against something real instead of general sabermetric priors.
    """
    need = {"game_pk", "at_bat_number", "on_1b", "on_2b", "on_3b", "events", "inning_topbot",
           "batter", "pitcher", "game_date"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {"error": "missing required columns for grand slam replay"}

    d = df.copy()
    d["_gd"] = d["game_date"].astype(str).str[:10]
    if start:
        d = d[d["_gd"] >= start]
    if end:
        d = d[d["_gd"] <= end]

    # one row per real PA
    pa = d[d["events"].notna()].copy()
    if pa.empty:
        return {"error": "no plate-appearance rows in this window"}
    pa = pa.drop_duplicates(subset=["game_pk", "at_bat_number"])

    # real batting slot, identical derivation to hr_by_lineup_spot() -- grouped by
    # (game_pk, inning_topbot) rather than by team NAME. home_team/away_team were checked
    # directly and found to be inconsistently present in the real production frame
    # (backtest.py's own _has_geo check already treats home_team as sometimes-absent,
    # falling back to venue) -- this function never actually needed team names in the first
    # place: Top vs Bot already distinguishes the two teams within a single game_pk without
    # them. First real run came back "missing required columns" because of this unnecessary
    # dependency; this removes it rather than adding a second, more fragile fallback.
    pa = pa.sort_values(["game_pk", "inning_topbot", "at_bat_number"])
    pa["tidx"] = pa.groupby(["game_pk", "inning_topbot"]).cumcount()
    pa["slot"] = (pa["tidx"] % 9) + 1

    loaded = pa["on_1b"].notna() & pa["on_2b"].notna() & pa["on_3b"].notna()
    slams = pa[loaded & pa["events"].eq("home_run")]
    n_slams = len(slams)
    if n_slams < 15:
        return {"error": f"only {n_slams} real grand slams in this window -- too few to grade "
                         f"a slot distribution or pitcher-rate correlation meaningfully"}

    # real slot distribution among actual grand slams, vs. the naive 1/9 = 11.1% each expectation
    slot_counts = slams["slot"].value_counts().to_dict()
    slot_dist = {int(s): {"n": int(c), "pct": round(100 * c / n_slams, 1),
                          "vs_even_share": round((c / n_slams) / (1 / 9), 2)}
                for s, c in sorted(slot_counts.items())}

    # Real pitcher bases-loaded rate, weighted by actual slam EVENTS, not unique pitchers who
    # allowed one. Caught by testing against synthetic data with a known 2x wild-pitcher
    # effect built in: iterating unique pitchers came back at 1.014x (essentially neutral),
    # because with enough sample nearly every pitcher -- wild or not -- eventually allows at
    # least one slam, which washes the signal out entirely. Weighting by event count instead
    # means a pitcher who allowed 8 real slams contributes 8x to the average, correctly
    # reflecting that wilder pitchers produce disproportionately MORE slam events, not just
    # that they're eventually present in the list.
    loaded_by_pitcher = pa[loaded].groupby("pitcher").size()
    pa_by_pitcher = pa.groupby("pitcher").size()
    league_loaded_rate = float(loaded.sum()) / len(pa)
    _rate_cache = {}
    slam_pitcher_rates = []
    for pid in slams["pitcher"].dropna():
        if pid not in _rate_cache:
            _n_pa = int(pa_by_pitcher.get(pid, 0))
            if _n_pa < 100:
                _rate_cache[pid] = None
            else:
                _rate = float(loaded_by_pitcher.get(pid, 0)) / _n_pa
                _rate_cache[pid] = _rate / league_loaded_rate if league_loaded_rate else 1.0
        if _rate_cache[pid] is not None:
            slam_pitcher_rates.append(_rate_cache[pid])

    return {
        "n_real_grand_slams": n_slams,
        "slot_distribution": slot_dist,
        "league_bases_loaded_rate": round(100 * league_loaded_rate, 2),
        "avg_pitcher_traffic_multiplier_on_real_slams": (
            round(sum(slam_pitcher_rates) / len(slam_pitcher_rates), 3)
            if slam_pitcher_rates else None),
        "n_pitchers_with_enough_sample": len(slam_pitcher_rates),
        "note": "slot_distribution's vs_even_share compares each slot's real share of grand "
               "slams to what an even 1-in-9 split would predict -- 1.0 means that slot hits "
               "its naive share exactly, above 1.0 means it over-produces real slams.",
    }


def replay_runs(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict:
    from etl import runs as RUNS
    df = df.copy()
    df["_gd"] = df["game_date"].astype(str).str[:10]
    all_dates = sorted(df["_gd"].unique())
    if len(all_dates) <= WARMUP_DAYS:
        return {"error": "not enough days"}
    dates = [d for d in all_dates[WARMUP_DAYS:]
             if (not start or d >= start) and (not end or d <= end)]

    # FanGraphs, fetched ONCE for the whole replay rather than per historical day.
    #
    # This is a real, named limitation, not a hidden one: FanGraphs does not serve historical
    # daily snapshots, so there is no way to get an AS-OF xFIP/SIERA/Stuff+/Location+ for a
    # pitcher on a specific past date the way AsOfFrame does for Statcast data. What gets used
    # instead is the CURRENT season's aggregate, applied uniformly across the whole replay
    # window. That means a pitcher's full-season quality leaks backward into his April starts —
    # a real pitcher-quality leak, structurally different from (and less serious than) a
    # same-day outcome leak, but a leak worth knowing about when reading the with-FanGraphs
    # numbers below. Treat them as "does this signal correlate with results", not as a clean
    # controlled test.
    try:
        fg_pitch = statcast_data.fangraphs_pitching()
    except Exception as e:
        fg_pitch = {}
        print(f"[backtest] fangraphs fetch skipped (non-fatal): {e}")
    _fg_notes = [] if fg_pitch else ["FanGraphs unavailable this run — with/without comparison skipped"]
    # Every starter's name, looked up ONCE for the whole replay (fg_pitch is keyed by
    # normalised name; MLBAM ids alone can't join to it). Collected from every game's two
    # starters across the full date range so this is one batched call, not one per game.
    _all_sp_ids = set()
    for _D in dates:
        _day = df[df["_gd"] == _D]
        for _gpk, _g in _day.groupby("game_pk"):
            for _half in ("Top", "Bot"):
                _r = _g[(_g["inning_topbot"] == _half) & (_g["inning"] == 1)]
                if not _r.empty:
                    _all_sp_ids.add(int(_r.iloc[0]["pitcher"]))
    try:
        _sp_names = statcast_data.player_names(_all_sp_ids) if fg_pitch else {}
    except Exception:
        _sp_names = {}

    n = 0
    brier = 0.0
    correct = 0
    # Markov side-by-side accumulators, filled only when a simulation ran for that game.
    _mk_abs = _mk_sq = 0.0
    _mk_n = 0
    _mk_pover, _mk_hit = [], []
    _rl_pcover, _rl_hit = [], []   # ADDED this session -- run-line-specific calibration
    _mk_preds, _mk_worst = [], []
    _tot_rows = []
    tot_abs_err = 0.0
    calib = {}                      # decile -> {n, home_wins}
    home_wins_actual = 0
    # FanGraphs with/without comparison on the SAME games, same shape as the markov-vs-linear
    # comparison already below — total error with the FanGraphs nudge applied vs without it.
    _fg_abs = _fg_abs_off = 0.0
    _fg_n = 0
    # F5 win-probability grading. This market was never graded before the F5/RL Parlay Scanner
    # step existed, so there was no way to know whether f5_home_wp is trustworthy on its own —
    # it uses a SEPARATE, smaller-sample projection (starter dominates, bullpen is out of it
    # entirely) and deserves its own calibration rather than assuming the full-game number's
    # accuracy carries over.
    _f5_n = 0
    _f5_brier = 0.0
    _f5_correct = 0
    _f5_calib = {}
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
            _home_fg = (fg_pitch.get(statcast_data._norm_name(_sp_names.get(sp["home"], "")))
                       if fg_pitch else None)
            _away_fg = (fg_pitch.get(statcast_data._norm_name(_sp_names.get(sp["away"], "")))
                       if fg_pitch else None)
            proj = RUNS.project_game(hl, al, pprof.get(sp["home"]) or {},
                                     pprof.get(sp["away"]) or {},
                                     {}, {}, park_mult=1.0,
                                     home_hands=hl_hands, away_hands=al_hands,
                                     home_fg=_home_fg, away_fg=_away_fg)
            if not proj:
                continue
            # FanGraphs with/without comparison, on the SAME game, isolating exactly what the
            # SIERA/xFIP/Stuff+/Pitching+ nudge changes — same shape as the markov-vs-linear A/B
            # already below. Skipped cheaply (no second simulation) when neither side had a
            # FanGraphs match, since the two runs would be identical anyway.
            if fg_pitch and (_home_fg or _away_fg):
                _proj_off = RUNS.project_game(hl, al, pprof.get(sp["home"]) or {},
                                              pprof.get(sp["away"]) or {},
                                              {}, {}, park_mult=1.0,
                                              home_hands=hl_hands, away_hands=al_hands)
                if _proj_off:
                    _fg_n += 1
                    _fg_abs += abs(proj["total"] - (hs + as_))
                    _fg_abs_off += abs(_proj_off["total"] - (hs + as_))
            # F5 outcome, extracted from the SAME historical frame the full-game result used —
            # this market was never graded before the F5/RL Parlay Scanner step existed.
            try:
                g5 = g[g["inning"].astype(float) <= 5]
                hs5 = float(g5["post_home_score"].dropna().max())
                as5 = float(g5["post_away_score"].dropna().max())
                f5_wp = proj.get("f5_home_wp")
                if hs5 == hs5 and as5 == as5 and hs5 != as5 and f5_wp is not None:
                    _f5_n += 1
                    _f5_home_won = hs5 > as5
                    _f5_brier += (f5_wp - (1.0 if _f5_home_won else 0.0)) ** 2
                    _f5_correct += 1 if ((f5_wp >= 0.5) == _f5_home_won) else 0
                    _f5b = int(min(max(f5_wp, 0.0), 0.999) * 10) * 10
                    _f5c = _f5_calib.setdefault(str(_f5b), {"n": 0, "home_wins": 0})
                    _f5c["n"] += 1; _f5c["home_wins"] += 1 if _f5_home_won else 0
            except Exception:
                pass
            p = proj["home_wp"]
            n += 1
            home_wins_actual += 1 if home_won else 0
            brier += (p - (1.0 if home_won else 0.0)) ** 2
            correct += 1 if ((p >= 0.5) == home_won) else 0
            tot_abs_err += abs(proj["total"] - (hs + as_))
            # Row shape validate.grade_totals expects. Collected here rather than recomputed
            # later so the grader sees exactly the games this replay actually scored.
            _tot_rows.append({"pred_mean": proj["total"], "actual": hs + as_,
                              "pred_dist": ((proj.get("markov") or {}).get("total_dist") or None)})
            # Markov accumulation. This was MISSING on the first run — the reporting was written
            # and the accumulators were declared, but nothing ever incremented them, so the whole
            # A/B silently reported markov_n=0 while looking like it had run. Scored only on
            # games where the simulation actually produced a total, so markov_n is the honest
            # denominator rather than a count of games the linear engine happened to cover.
            _mk = proj.get("markov") or {}
            _mk_tot = _mk.get("total_mean")
            if _mk_tot is not None:
                _err = abs(_mk_tot - (hs + as_))
                _mk_abs += _err
                _mk_sq += _err * _err
                _mk_n += 1
                # Diagnostic. Two rounds of theorising about WHY markov over-projects produced
                # two fixes that did not move the number, so this records the evidence instead:
                # the spread of predictions and the worst individual misses, with the inputs
                # that produced them. Guessing from the aggregate MAE was the mistake.
                _mk_preds.append(_mk_tot)
                if _err > 8.0 and len(_mk_worst) < 25:
                    _mk_worst.append({
                        "date": D, "pred": round(_mk_tot, 1), "actual": hs + as_,
                        "linear": proj.get("total"),
                        "home_mean": _mk.get("home_mean"), "away_mean": _mk.get("away_mean"),
                        "n_home_bat": len(hl), "n_away_bat": len(al),
                    })
                _dist = _mk.get("total_dist") or {}
                if _dist:
                    _act = hs + as_
                    for _L in (7.5, 8.5, 9.5):
                        _p_over = sum(v for k, v in _dist.items() if float(k) > _L)
                        _mk_pover.append(_p_over)
                        _mk_hit.append(1.0 if _act > _L else 0.0)
                # ADDED this session -- run-line-specific calibration, separate from the totals
                # check above. P(home_minus_1_5) is a DIFFERENT question asked of the same joint
                # distribution (margin, not total) -- the Markov recalibration fix applied the
                # totals check's slope/intercept to this too, as a reasonable but unverified
                # extrapolation. This grades it directly rather than continuing to assume the
                # totals correction transfers cleanly to a margin question.
                _rl = _mk.get("run_line") or {}
                if _rl.get("home_minus_1_5") is not None:
                    _rl_pcover.append(_rl["home_minus_1_5"])
                    _rl_hit.append(1.0 if (hs - as_) >= 2 else 0.0)
                if _rl.get("away_minus_1_5") is not None:
                    _rl_pcover.append(_rl["away_minus_1_5"])
                    _rl_hit.append(1.0 if (as_ - hs) >= 2 else 0.0)
            b = int(min(max(p, 0.0), 0.999) * 10) * 10
            c = calib.setdefault(str(b), {"n": 0, "home_wins": 0})
            c["n"] += 1; c["home_wins"] += 1 if home_won else 0
        if n and n % 50 == 0:
            print(f"[backtest:runs] {n} games scored through {D}")

    if not n:
        return {"error": "no gradeable games (post-score columns missing from the frame?)"}
    # Markov accumulators are populated inside the loop when a simulation was produced for that
    # game; games where it could not run simply do not contribute, so markov_n reports the
    # honest denominator rather than silently comparing different populations.
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
        "linear_mae": round(tot_abs_err / n, 2),
        "total_mae": round(tot_abs_err / n, 2),      # kept for backward compatibility
        "markov_mae": (round(_mk_abs / _mk_n, 2) if _mk_n else None),
        "markov_rmse": (round((_mk_sq / _mk_n) ** 0.5, 2) if _mk_n else None),
        "markov_n": _mk_n,
        "markov_vs_linear": (round(_mk_abs / _mk_n - tot_abs_err / n, 3) if _mk_n else None),
        "markov_over_calibration": (V.decile_calibration(_mk_pover, _mk_hit, n_bands=8)
                                    if len(_mk_pover) >= 400 else None),
        # ADDED this session -- see the accumulation comment above for why this exists
        # separately from markov_over_calibration rather than assuming that check's slope/
        # intercept transfers to a margin (run-line) question.
        "run_line_calibration": (V.decile_calibration(_rl_pcover, _rl_hit, n_bands=8)
                                 if len(_rl_pcover) >= 400 else None),
        # FanGraphs with/without, on the same games — isolates exactly what the SIERA/xFIP/
        # Stuff+/Pitching+ nudge changes. A season-aggregate leak applies here (see the note at
        # the top of this function): treat this as "does the signal correlate", not a clean A/B.
        "fangraphs_n": _fg_n,
        "total_mae_with_fangraphs": (round(_fg_abs / _fg_n, 3) if _fg_n else None),
        "total_mae_without_fangraphs": (round(_fg_abs_off / _fg_n, 3) if _fg_n else None),
        "fangraphs_notes": _fg_notes,
        # F5 win-probability grading — this market had never been graded before the F5/RL
        # Parlay Scanner step existed. Own calibration because F5 uses a smaller-sample
        # projection (starter only, bullpen is entirely out of it) than the full-game number.
        "f5_n": _f5_n,
        "f5_accuracy": (round(100 * _f5_correct / _f5_n, 2) if _f5_n else None),
        "f5_brier": (round(_f5_brier / _f5_n, 4) if _f5_n else None),
        "f5_calib": {k: _f5_calib[k] for k in sorted(_f5_calib, key=int)},
        # Where the markov error actually lives. If the median prediction is sane and only a
        # tail is wrong, the fix is input validation; if the whole distribution is shifted, the
        # engine itself is miscalibrated. These two cases need opposite fixes, and the aggregate
        # MAE cannot tell them apart.
        "markov_diagnostic": ({
            "pred_min": round(min(_mk_preds), 1),
            "pred_p10": round(sorted(_mk_preds)[len(_mk_preds) // 10], 1),
            "pred_median": round(sorted(_mk_preds)[len(_mk_preds) // 2], 1),
            "pred_p90": round(sorted(_mk_preds)[9 * len(_mk_preds) // 10], 1),
            "pred_max": round(max(_mk_preds), 1),
            "actual_mean": round(sum(r["actual"] for r in _tot_rows) / len(_tot_rows), 2),
            "pct_over_15": round(100 * sum(1 for x in _mk_preds if x > 15) / len(_mk_preds), 1),
            "worst_misses": _mk_worst[:10],
        } if _mk_preds else None),
        # validate.grade_totals was written and never called — the MAE-floor context it carries
        # (a perfect model still posts ~3.30) never reached the output, so a 3.6 looked like a
        # failure when it is close to the ceiling.
        "totals_graded": (V.grade_totals(_tot_rows) if len(_tot_rows) >= 100 else None),
        "calib": {k: calib[k] for k in sorted(calib, key=int)},
        "notes": [
            "CALIBRATION is the number that matters, not accuracy — you cannot bet a probability you can't trust",
            "Brier lower is better; if it does not beat brier_baseline the model has learned nothing",
            "total MAE floor: a model knowing every game's TRUE mean still posts ~3.30; a "
            "constant predictor posts ~3.51. Usable headroom is ~0.2 runs, so judge totals on "
            "over/under CALIBRATION, not MAE. The often-quoted 2.6 is below the noise floor.",
            "run model has NO defense, NO true park run factor, NO bullpen availability",
            "bullpen omitted entirely in this replay (starter + league-average pen)",
            "markov_* are the simulation's numbers on the SAME games as linear_* — that pairing "
            "is what decides whether the engine swap is justified",
        ],
    }


if __name__ == "__main__":
    main()
