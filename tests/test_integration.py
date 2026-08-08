"""Integration verification — proves the new engines are WIRED, not merely importable.

Written because "it imports" is a much weaker claim than it sounds. Every one of these six
modules imported cleanly while being completely dead code: nothing called them, and the live
board produced byte-identical output with or without them present. An import check would have
passed the entire time.

So this script asserts three separate things, in increasing order of strength:

  1. IMPORTABLE  — the module loads
  2. REACHABLE   — a live file actually references it (dead-code detection)
  3. LIVE        — calling the production entry point produces the new field with a real value

Only (3) proves integration. Run it before every deploy:

    python tests/test_integration.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILS: list[str] = []
WARNS: list[str] = []


def check(label, fn, warn_only=False):
    try:
        detail = fn()
        print(f"  PASS  {label}" + (f"  [{detail}]" if detail else ""))
        return True
    except Exception as e:
        msg = f"{label}: {type(e).__name__}: {e}"
        (WARNS if warn_only else FAILS).append(msg)
        print(f"  {'WARN' if warn_only else 'FAIL'}  {msg}")
        return False


# ---------------------------------------------------------------------------
# 1. Importable
# ---------------------------------------------------------------------------

NEW_MODULES = ["markov", "hitmodel", "kengine", "arsenal", "environment", "validate"]
LIVE_FILES = ["build_board", "runs", "props", "backtest", "fetch_odds", "statsapi"]


def test_imports():
    import importlib
    for m in NEW_MODULES + LIVE_FILES:
        check(f"import etl.{m}", lambda m=m: importlib.import_module(f"etl.{m}") and None)


# ---------------------------------------------------------------------------
# 2. Reachable — the dead-code check that would have caught the earlier gap
# ---------------------------------------------------------------------------

# module -> (consumer files, the names it is actually bound to at the call site).
# Aliases matter: build_board imports `environment as env_mod` and backtest imports
# `validate as V`, so a check that only looked for the module's own name would report both as
# dead code when they are fully wired. Getting this wrong in either direction is bad — a false
# pass hides real dead code, a false fail sends you chasing a bug that does not exist.
WIRING = {
    "markov":      (["runs.py"], ["markov"]),
    "hitmodel":    (["props.py", "build_board.py"], ["hitmodel"]),
    "kengine":     (["props.py", "build_board.py"], ["kengine"]),
    "arsenal":     (["build_board.py"], ["arsenal_mod"]),
    "environment": (["build_board.py"], ["env_mod"]),
    "validate":    (["backtest.py"], ["V"]),
}


def test_reachable():
    for mod, (consumers, aliases) in WIRING.items():
        for f in consumers:
            path = os.path.join(ROOT, "etl", f)

            def run(path=path, mod=mod, f=f, aliases=aliases):
                src = open(path, encoding="utf-8").read()
                if not re.search(rf"\b{mod}\b", src):
                    raise AssertionError(f"{f} never references {mod}")
                # an import alone is not enough — something must actually CALL into it
                pat = "|".join(re.escape(a) for a in aliases)
                calls = re.findall(rf"\b(?:{pat})\s*\.\s*\w+\s*\(", src)
                if not calls:
                    raise AssertionError(f"{f} imports {mod} but never calls it (DEAD CODE)")
                return f"{len(calls)} call site(s)"
            check(f"{mod} is called by {f}", run)


# ---------------------------------------------------------------------------
# 3. Live — production entry points emit the new fields with real values
# ---------------------------------------------------------------------------

def test_runs_markov_live():
    from etl import runs

    def run():
        def bat(x, k, bb, pa, ab, hr, slg, xba):
            return {"xwobacon": x, "k_pct": k, "bb_pct": bb, "pa": pa,
                    "ab": ab, "hr": hr, "slg": slg, "xba": xba}
        avg = bat(.340, 22, 8, 150, 133, 5, .410, .255)
        sp = {"season": {"xwobacon_allowed": .360, "k_pct_allowed": 22,
                         "bb_pct_allowed": 8, "pa": 600, "hr_per_pa": 3.2},
              "recent": {"xwobacon_allowed": .360, "k_pct_allowed": 22,
                         "bb_pct_allowed": 8, "pa": 120}}
        out = runs.project_game([avg] * 9, [avg] * 9, sp, sp, sp, sp,
                                park_mult=1.0, home_bf=22, away_bf=22)
        mk = out.get("markov")
        assert mk is not None, "project_game returned no markov block — engine not wired"
        assert mk.get("total_dist"), "markov block has no run distribution"
        assert 5.0 < mk["total_mean"] < 14.0, f"implausible total {mk['total_mean']}"
        return f"total {mk['total_mean']}, {len(mk['total_dist'])} buckets, delta {mk['markov_delta']:+.2f}"
    check("runs.project_game emits a live markov distribution", run)


def test_props_live():
    from etl import props

    def hit():
        bat = {"whiff_pct": 18.0, "k_pct": 20.0, "xba": .290, "bb_pct": 8.5}
        arm = {"season": {"swstr_pct_allowed": 11.0, "k_pct_allowed": 22.0,
                          "xba_allowed": .285, "bb_pct_allowed": 8.0}}
        p2, bd2 = props.hit_prob_gated(bat, arm, lineup_spot=2, implied_team_total=5.4)
        p9, _ = props.hit_prob_gated(bat, arm, lineup_spot=9, implied_team_total=3.6)
        assert p2 is not None, "gated hit model fell back to the anchor model"
        assert p2 > p9, "lineup spot / implied total not affecting the projection"
        return f"spot2 {100*p2:.1f}% vs spot9 {100*p9:.1f}%"
    check("props.hit_prob_gated responds to lineup spot & implied total", hit)

    def ks():
        r_long = props.pitcher_k_projection(
            {"csw": .315, "p_per_pa": 3.8}, [.25] * 9,
            arsenal=[("FF", 33, 250), ("SL", 22, 170), ("CH", 18, 140)], pitch_limit=100)
        r_short = props.pitcher_k_projection(
            {"csw": .315, "p_per_pa": 3.8}, [.25] * 9,
            arsenal=[("FF", 33, 250), ("SL", 22, 170), ("CH", 18, 140)], pitch_limit=75)
        assert r_long["xbf"] != 22, "xBF is still the static 22 — engine not wired"
        assert r_long["exp_k"] > r_short["exp_k"], "pitch limit not affecting the projection"
        assert r_long.get("dist"), "no Poisson-binomial distribution returned"
        return f"100p -> {r_long['exp_k']}K (xBF {r_long['xbf']}), 75p -> {r_short['exp_k']}K"
    check("props.pitcher_k_projection uses dynamic xBF", ks)

    def hrr():
        bat = {"whiff_pct": 18.0, "k_pct": 20.0, "xba": .290, "bb_pct": 8.5}
        arm = {"season": {"swstr_pct_allowed": 11.0, "k_pct_allowed": 22.0,
                          "xba_allowed": .285, "bb_pct_allowed": 8.0}}
        _, bd = props.hit_prob_gated(bat, arm, lineup_spot=2, implied_team_total=5.4)
        h = props.hrr_projection(bd, 2, implied_team_total=5.4,
                                 batters_ahead=[{"obp_30": .370}],
                                 batters_behind=[{"woba_30": .360, "slg_30": .500}])
        assert h["hrr"] > 0 and h["xpa"] > 4.0
        return f"HRR {h['hrr']} on {h['xpa']} xPA"
    check("props.hrr_projection produces volume-first output", hrr)


def test_environment_live():
    from etl import environment as env

    def fence():
        import numpy as np
        hx = 125.42 + np.tan(np.radians(-38)) * 80
        hy = 198.27 - 80
        d = env.fence_delta("Fenway Park", hx, hy, 355)
        assert d is not None, "fence geometry not reachable"
        return f"355ft down the LF line at Fenway = {d:+.1f}ft vs the wall"
    check("environment.fence_delta uses real stadium geometry", fence)

    def carry():
        c = env.carry_multiplier(75, 5200, 40)
        assert 1.04 < c < 1.09, f"Coors carry {c} outside the published range"
        return f"Coors carry x{c:.4f} (400ft -> {env.carry_feet(400,75,5200,40):.0f}ft)"
    check("environment.carry_multiplier matches published Coors effect", carry)

    def fatigue():
        s, _ = env.reliever_status([{"days_ago": 1, "pitches": 40}])
        assert s == env.UNAVAILABLE
        s2, _ = env.reliever_status([{"days_ago": 1, "pitches": 28}])
        assert s2 == env.FATIGUED
        return "40p->UNAVAILABLE, 28p->FATIGUED"
    check("environment.reliever_status classifies correctly", fatigue)


def test_statsapi_bullpen():
    from etl import statsapi

    def run():
        assert hasattr(statsapi, "get_reliever_pitch_logs"), "bullpen fetcher missing"
        return "get_reliever_pitch_logs present"
    check("statsapi bullpen fetcher exists", run)


def test_odds_archiver():
    from etl import fetch_odds

    def run():
        assert hasattr(fetch_odds, "_archive_odds_snapshot"), "archiver missing"
        src = open(os.path.join(ROOT, "etl", "fetch_odds.py"), encoding="utf-8").read()
        assert "_archive_odds_snapshot(payload)" in src, "archiver defined but never called"
        return "defined AND called from _write()"
    check("fetch_odds archives closing lines", run)


def test_backtest_wiring():
    def run():
        src = open(os.path.join(ROOT, "etl", "backtest.py"), encoding="utf-8").read()
        for needle, why in (
            ("asof.assert_clean", "AsOfFrame leak guard not integrated into _day_heats"),
            ('_edge("near_miss"', "near-miss lift not tallied"),
            ('_edge("bat_fast_short"', "bat-tracking lift not tallied"),
            ('_edge("hand_vs_usage_filter"', "handedness-first A/B not tallied"),
            ("markov_mae", "markov vs linear MAE not reported"),
            ("def grade_hit_side_by_side", "hit side-by-side grader missing"),
            ("def grade_k_side_by_side", "K side-by-side grader missing"),
            ("def poison_check", "GUARDRAIL VIOLATION: poison_check removed"),
        ):
            assert needle in src, why
        return "all graders + guardrail intact"
    check("backtest.py wiring complete", run)


def test_frozen_heat_model():
    """The guardrail. compute.py must be byte-identical to HEAD."""
    def run():
        import subprocess
        r = subprocess.run(["git", "diff", "--stat", "etl/compute.py"],
                           cwd=ROOT, capture_output=True, text=True)
        assert not r.stdout.strip(), f"FROZEN HEAT MODEL MODIFIED:\n{r.stdout}"
        return "compute.py byte-identical to HEAD"
    check("frozen HR heat model untouched", run)


if __name__ == "__main__":
    print("INTEGRATION VERIFICATION\n")
    print("1. importable:")
    test_imports()
    print("\n2. reachable (dead-code detection):")
    test_reachable()
    print("\n3. live (production entry points emit real values):")
    test_runs_markov_live()
    test_props_live()
    test_environment_live()
    test_statsapi_bullpen()
    test_odds_archiver()
    test_backtest_wiring()
    print("\n4. guardrail:")
    test_frozen_heat_model()

    print()
    if FAILS:
        print(f"INTEGRATION FAIL — {len(FAILS)} problem(s):")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    if WARNS:
        print(f"({len(WARNS)} warning(s))")
    print("INTEGRATION PASS — all engines wired and producing live values")
