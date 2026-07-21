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

import numpy as np
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
            d = json.load(f)
            return d.get("players", []), d.get("parlay_picks", []), d.get("pitcher_props", [])
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
                "hit_heat": p.get("hit_heat"),
                "hrr_heat": p.get("hrr_heat"),
                "k_heat_bat": p.get("k_heat_bat"),
            } for p in b.get("players", [])], b.get("parlay_picks", []), b.get("pitcher_props", [])
    return None, [], []


def _normalize_sc(sc):
    """Modern pandas hands back nullable/Arrow dtypes (Int64, string) whose masks and
    scalars raise on NA. Coerce once so the grading math runs on plain numpy types."""
    import pandas as pd
    sc = sc.copy()
    for c in ("inning", "at_bat_number", "pitch_number", "batter", "pitcher", "game_pk"):
        if c in sc.columns:
            sc[c] = pd.to_numeric(sc[c], errors="coerce")
    for c in ("events", "inning_topbot"):
        if c in sc.columns:
            sc[c] = np.asarray(sc[c].astype(object).where(sc[c].notna(), ""), dtype=object)
    return sc


def _starters(sc):
    starters = {}
    inn1 = sc[sc["inning"].to_numpy() == 1]
    for (gp, half), grp in inn1.groupby(["game_pk", "inning_topbot"]):
        g = grp.sort_values(["at_bat_number", "pitch_number"])
        p0 = g.iloc[0]["pitcher"]
        if p0 == p0:                                   # NaN-safe
            starters[(int(gp), half)] = int(p0)
    return starters


def _hr_map(sc):
    starters = _starters(sc)
    out = {}
    hrs = sc[sc["events"].to_numpy() == "home_run"]
    for _, row in hrs.iterrows():
        if row["batter"] != row["batter"] or row["game_pk"] != row["game_pk"]:
            continue
        bid = int(row["batter"]); gp = int(row["game_pk"]); half = row["inning_topbot"]
        pit = row["pitcher"]
        is_sp = (pit == pit) and starters.get((gp, half)) == int(pit)
        rec = out.setdefault(bid, {"hr": 0, "sp": 0, "bp": 0})
        rec["hr"] += 1
        rec["sp" if is_sp else "bp"] += 1
    return out


HIT_EVENTS = {"single", "double", "triple", "home_run"}
K_EVENTS = {"strikeout", "strikeout_double_play"}


def _truthy(v):
    """Coerce a possibly-stringified boolean. json.dump(default=str) turns numpy
    bools into "True"/"False" strings, and the string "False" is truthy in Python —
    so any bare truth-test on snapshot data silently counts misses as hits."""
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)
# Runs/RBI attribution is not in the raw Statcast frame, so we approximate:
#   * Runs: whenever a hitter is credited with reaching base AND a later event in
#     the same half-inning is any hit/HR (they score). Approximation only —
#     doesn't catch every path home. For prop-tracking purposes it's a floor.
#   * RBIs: HRs = auto RBI. Other hits w/ runners on = approximated as +1 RBI
#     when a previous batter in the same half-inning reached and hasn't scored.
# We use a lightweight rebuild: process each half-inning's PA events in order,
# tracking who's on base, and credit R/RBI on that basis.


def _hrr_map(sc):
    """{bid: {hits,runs,rbis}} — delegates to the shared hrr_recon module so live grading uses
    IDENTICAL logic to the backtest calibration (authoritative rbi + score-delta runs). The old
    per-file base-runner sim undercounted runs/RBIs and biased HRR low."""
    from etl import hrr_recon
    return hrr_recon.hrr_map(sc)


# PA_EVENTS defined in statcast_data — mirror it here to avoid an extra import at load time
PA_EVENTS_TRACK = {
    "single", "double", "triple", "home_run", "field_out", "strikeout",
    "strikeout_double_play", "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "field_error", "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice", "fielders_choice_out", "catcher_interf", "intent_walk",
    "triple_play", "sac_fly_double_play",
}


def _k_map(sc):
    """{bid: n_strikeouts} — hitter K counts (kept for internal use)"""
    out = {}
    ks = sc[sc["events"].isin(list(K_EVENTS))]
    for _, row in ks.iterrows():
        if row["batter"] != row["batter"]:
            continue
        out[int(row["batter"])] = out.get(int(row["batter"]), 0) + 1
    return out


def _pitcher_k_map(sc):
    """{pitcher_id: n_strikeouts_thrown} — the number that pitcher K props settle on."""
    out = {}
    ks = sc[sc["events"].isin(list(K_EVENTS))]
    for _, row in ks.iterrows():
        pit = row.get("pitcher")
        if pit != pit:
            continue
        out[int(pit)] = out.get(int(pit), 0) + 1
    return out


def _tier(h):
    if h is None: return "n/a"
    return "70+" if h >= 70 else "55-69" if h >= 55 else "40-54" if h >= 40 else "<40"


def grade_date(date):
    """Grade a single date -> record dict, or None if it can't be graded yet (no snapshot,
    or results not posted). Pure: does not read or write history.json."""
    players, parlay_picks, pitcher_props = _load_day(date)
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

    sc = _normalize_sc(sc)
    hrmap = _hr_map(sc)
    hrrmap = _hrr_map(sc)
    kmap = _k_map(sc)
    pkmap = _pitcher_k_map(sc)
    def homered(p): return hrmap.get(p["id"], {}).get("hr", 0) > 0
    def got_hits(p, n=1): return hrrmap.get(p["id"], {}).get("hits", 0) >= n
    def hrr_val(p):
        h = hrrmap.get(p["id"], {})
        return h.get("hits", 0) + h.get("runs", 0) + h.get("rbis", 0)
    def struck_out(p, n=1): return kmap.get(p["id"], 0) >= n

    tiers, forms = {}, {}
    by_signal = {k: {"cleared": {"n": 0, "hr": 0}, "not": {"n": 0, "hr": 0}} for k in SIGNALS}
    by_badge = {}
    by_park, by_trend = {}, {}      # does park+weather boost / trend direction actually convert?
    by_smash, by_opener, by_spot, by_b2b, by_hlabel = {}, {}, {}, {}, {}
    def _park_bucket(b):
        if b is None: return "n/a"
        if b >= 12: return "strong+"
        if b >= 6: return "lean+"
        if b > -6: return "neutral"
        return "against"
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
            # CRITICAL: numpy bools serialize to the STRINGS "True"/"False" via
            # json.dump(default=str), and "False" is TRUTHY in Python. A bare
            # `if sig.get(k)` therefore buckets every missed signal as "cleared"
            # and destroys the lift numbers. Coerce explicitly.
            b = "cleared" if _truthy(sig.get(k)) else "not"
            by_signal[k][b]["n"] += 1
            by_signal[k][b]["hr"] += 1 if hit else 0
        badges = p.get("badges") or []
        for k in badges:
            bb = by_badge.setdefault(k, {"n": 0, "hr": 0})
            bb["n"] += 1; bb["hr"] += 1 if hit else 0
        pk = by_park.setdefault(_park_bucket(p.get("park_boost")), {"n": 0, "hr": 0})
        pk["n"] += 1; pk["hr"] += 1 if hit else 0
        tdir = p.get("trend") or "n/a"
        td = by_trend.setdefault(tdir, {"n": 0, "hr": 0})
        td["n"] += 1; td["hr"] += 1 if hit else 0
        sm = by_smash.setdefault("smash" if p.get("smash") else "rest", {"n": 0, "hr": 0})
        sm["n"] += 1; sm["hr"] += 1 if hit else 0
        oo = by_opener.setdefault("opener" if p.get("opener") else "sp", {"n": 0, "hr": 0})
        oo["n"] += 1; oo["hr"] += 1 if hit else 0
        bb2 = by_b2b.setdefault("b2b" if p.get("b2b") else "rest", {"n": 0, "hr": 0})
        bb2["n"] += 1; bb2["hr"] += 1 if hit else 0
        if p.get("spot"):
            sp = by_spot.setdefault(str(p["spot"]), {"n": 0, "hr": 0})
            sp["n"] += 1; sp["hr"] += 1 if hit else 0
        hl = by_hlabel.setdefault(p.get("hlabel") or "none", {"n": 0, "hr": 0})
        hl["n"] += 1; hl["hr"] += 1 if hit else 0
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

    # ---- parlay grading: how did the auto-generated picks actually do? ----
    # For each strategy captured in the snapshot, record: legs_n, legs_hr,
    # all_hit (whole parlay), any_hit. Also expected combo count for RR.
    by_parlay = {}
    for pk in parlay_picks or []:
        kind = pk.get("kind"); legs = pk.get("legs") or []
        if not kind or not legs:
            continue
        hits = [1 if hrmap.get(l.get("id"), {}).get("hr", 0) > 0 else 0 for l in legs]
        entry = by_parlay.setdefault(kind, {
            "n": 0, "leg_n": 0, "leg_hr": 0, "all_hit": 0, "any_hit": 0, "pair_hits": 0,
        })
        entry["n"] += 1
        entry["leg_n"] += len(legs)
        entry["leg_hr"] += sum(hits)
        if hits and all(hits): entry["all_hit"] += 1
        if any(hits): entry["any_hit"] += 1
        # round-robin: count pair combos that hit (any 2 legs both homered)
        if kind == "rr5" and len(legs) >= 2:
            pair_hits = sum(1 for i in range(len(hits)) for j in range(i+1, len(hits))
                            if hits[i] and hits[j])
            entry["pair_hits"] += pair_hits

    # ---- Props grading: tier conversion for hit_heat, hrr_heat, k_heat ----
    # Same tier framework as HR heat, applied to each prop's independent score.
    def _prop_tier(v):
        if v is None: return "n/a"
        return "70+" if v >= 70 else "55-69" if v >= 55 else "40-54" if v >= 40 else "<40"

    by_hit_tier, by_hit2_tier = {}, {}
    by_hrr_tier = {}
    for p in players:
        # 1+ hit and 2+ hit tiers off hit_heat
        ht = _prop_tier(p.get("hit_heat"))
        e = by_hit_tier.setdefault(ht, {"n": 0, "hit": 0}); e["n"] += 1
        if got_hits(p, 1): e["hit"] += 1
        e2 = by_hit2_tier.setdefault(ht, {"n": 0, "hit": 0}); e2["n"] += 1
        if got_hits(p, 2): e2["hit"] += 1
        # HRR tiers off hrr_heat — bucket by hrr value (approximate 2+ / 3+ line)
        hrt = _prop_tier(p.get("hrr_heat"))
        eh = by_hrr_tier.setdefault(hrt, {"n": 0, "hit2": 0, "hit3": 0, "sum": 0, "sum_ct": 0})
        eh["n"] += 1
        v = hrr_val(p)
        eh["sum"] += v; eh["sum_ct"] += 1
        if v >= 2: eh["hit2"] += 1
        if v >= 3: eh["hit3"] += 1

    # Pitcher K props: track by pitcher_k_heat tier. For each tier, record
    # n_pitchers, actual K totals thrown, and O5.5/O6.5/O7.5 hit rates.
    by_pk_tier = {}
    for pp in pitcher_props or []:
        kh = pp.get("k_heat")
        if kh is None:
            continue
        tier = _prop_tier(kh)
        actual_ks = pkmap.get(pp.get("id"), 0)
        e = by_pk_tier.setdefault(tier, {"n": 0, "total_ks": 0, "o5": 0, "o6": 0, "o7": 0})
        e["n"] += 1
        e["total_ks"] += actual_ks
        if actual_ks >= 6: e["o5"] += 1  # over 5.5
        if actual_ks >= 7: e["o6"] += 1  # over 6.5
        if actual_ks >= 8: e["o7"] += 1  # over 7.5

    # Top-N by each prop score, same idea as HR top_n
    def _prop_topN(key, hit_fn, ns=(5, 10, 25)):
        rk = sorted((p for p in players if p.get(key) is not None),
                    key=lambda p: p.get(key) or 0, reverse=True)
        return {str(n): {"n": min(n, len(rk)), "hit": sum(1 for p in rk[:n] if hit_fn(p))}
                for n in ns}

    # Top-N for pitcher K props — different sizes because there are fewer starters
    def _pk_topN(ns=(3, 5, 10)):
        rk = sorted((pp for pp in (pitcher_props or []) if pp.get("k_heat") is not None),
                    key=lambda pp: pp.get("k_heat") or 0, reverse=True)
        out = {}
        for n in ns:
            top = rk[:n]
            ks_thrown = [pkmap.get(pp.get("id"), 0) for pp in top]
            out[str(n)] = {
                "n": len(top),
                "total_ks": sum(ks_thrown),
                "o5": sum(1 for k in ks_thrown if k >= 6),
                "o6": sum(1 for k in ks_thrown if k >= 7),
            }
        return out

    # Batter K UNDER: tiers by k_heat_bat but INVERTED intent — LOW score is the
    # bet (contact hitter vs low-K arm → UNDER 0.5 Ks). "hit" = did NOT strike out.
    by_bku_tier = {}
    def _bku_tier(v):
        if v is None: return "n/a"
        return "<30" if v < 30 else "30-44" if v < 45 else "45-59" if v < 60 else "60+"
    for p in players:
        kb = p.get("k_heat_bat")
        if kb is None:
            continue
        tier = _bku_tier(kb)
        e = by_bku_tier.setdefault(tier, {"n": 0, "hit": 0})
        e["n"] += 1
        if not struck_out(p, 1): e["hit"] += 1

    by_props = {
        "hit1": {"by_tier": by_hit_tier, "top_n": _prop_topN("hit_heat", lambda p: got_hits(p, 1))},
        "hit2": {"by_tier": by_hit2_tier, "top_n": _prop_topN("hit_heat", lambda p: got_hits(p, 2))},
        "hrr":  {"by_tier": by_hrr_tier, "top_n": _prop_topN("hrr_heat", lambda p: hrr_val(p) >= 2)},
        "pk":   {"by_tier": by_pk_tier, "top_n": _pk_topN()},
        "bku":  {"by_tier": by_bku_tier},
    }

    record = {
        "date": date, "players": len(players),
        "hitters_homered": n_hit,
        "total_hr": total_hr, "sp_hr": sp_hr, "bp_hr": bp_hr,
        "by_tier": tiers, "by_form": forms, "by_signal": by_signal, "by_badge": by_badge,
        "by_park": by_park, "by_trend": by_trend,
        "by_smash": by_smash, "by_opener": by_opener, "by_spot": by_spot, "by_b2b": by_b2b,
        "by_hlabel": by_hlabel,
        "by_parlay": by_parlay,
        "by_props": by_props,
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
        try:
            rec = grade_date(date)              # only returns a record if it CAN be graded
        except Exception as e:
            print(f"[track] {date} grade failed ({type(e).__name__}: {e}) — skipping, will retry next run.")
            rec = None
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
    import sys
    if "--regrade" in sys.argv:
        # Re-grade every day that still has a snapshot on disk, discarding the old
        # record. Needed after a grading-logic fix: the snapshots hold the raw
        # inputs, so corrected logic recovers the true numbers.
        import glob
        snaps = sorted(glob.glob(os.path.join(SNAP_DIR, "20*.json")))
        dates = [os.path.basename(s)[:-5] for s in snaps]
        print(f"[track] REGRADE: {len(dates)} snapshot(s) on disk: {dates[0] if dates else '-'} … {dates[-1] if dates else '-'}")
        try:
            with open(HISTORY_PATH) as f:
                old = json.load(f).get("days", [])
        except Exception:
            old = []
        old_by_date = {d["date"]: d for d in old}
        rebuilt, failed, kept = [], [], []
        for date in dates:
            try:
                rec = grade_date(date)
            except Exception as e:
                print(f"[track] regrade {date} failed ({type(e).__name__}: {e})")
                rec = None
            if rec:
                rebuilt.append(rec)
            elif date in old_by_date:
                kept.append(date)             # results not retrievable; keep old record
                rebuilt.append(old_by_date[date])
            else:
                failed.append(date)
        # keep any historical days whose snapshots have since been pruned
        have = {d["date"] for d in rebuilt}
        for d in old:
            if d["date"] not in have:
                rebuilt.append(d)
        rebuilt.sort(key=lambda d: d["date"])
        with open(HISTORY_PATH, "w") as f:
            json.dump({"updated": max(have) if have else None, "days": rebuilt},
                      f, indent=2, default=str)
        print(f"[track] REGRADE done: {len(rebuilt)} day(s) in history "
              f"({len(have)} re-graded from snapshots, {len(kept)} kept as-is, "
              f"{len(failed)} unavailable).")
        print("[track] by_signal buckets are now computed with corrected boolean coercion.")
    else:
        grade()
