"""Offline smoke test for the ETL.

Why this exists: the grader silently stopped adding days for three days because a variable
was used before it was defined. `py_compile` passes on that — Python only raises
UnboundLocalError at runtime — and the grader's own try/except turned a hard failure into a
"will retry next run" message that repeated forever. Nothing was watching.

This runs the real code paths against a stubbed Statcast frame, so an exception surfaces
immediately instead of after days of empty commits. It needs no network and no live data.

    python tests/smoke.py
"""
from __future__ import annotations
import glob
import json
import os
import random
import sys
import types

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILS: list[str] = []


def _fake_statcast_frame(batter_ids, date, n=2500):
    """A Statcast-shaped frame with every column the ETL touches."""
    random.seed(7)
    ev = ["single", "double", "triple", "home_run", "field_out", "strikeout", "walk"]
    rows = []
    for i in range(n):
        rows.append(dict(
            batter=random.choice(batter_ids), pitcher=600000 + i % 40,
            events=random.choice(ev), game_date=date, game_pk=800000 + i % 15,
            inning=random.randint(1, 9), at_bat_number=i % 80,
            pitch_number=random.randint(1, 6),
            inning_topbot=random.choice(["Top", "Bot"]),
            home_team="NYM", away_team="ATL",
            launch_speed=random.uniform(60, 115), launch_angle=random.uniform(-30, 50),
            hit_distance_sc=random.uniform(50, 450),
            bb_type=random.choice(["fly_ball", "line_drive", "ground_ball", None]),
            p_throws=random.choice(["R", "L"]), stand=random.choice(["R", "L"]),
            estimated_ba_using_speedangle=random.uniform(0, 1),
            estimated_woba_using_speedangle=random.uniform(0, 1.5),
            launch_speed_angle=random.choice([1, 2, 3, 4, 5, 6]),
            description="hit_into_play", zone=random.randint(1, 14),
            pitch_type=random.choice(["FF", "SL", "CH", "SI", "FC"]),
            release_speed=random.uniform(75, 100), bat_speed=random.uniform(60, 85),
            balls=random.randint(0, 3), strikes=random.randint(0, 2),
            woba_value=random.choice([0, 0.9, 1.25, 2.0]), woba_denom=1,
            hc_x=random.uniform(50, 200), hc_y=random.uniform(50, 200),
        ))
    return pd.DataFrame(rows)


def _install_pybaseball_stub(df):
    fake = types.ModuleType("pybaseball")
    fake.statcast = lambda *a, **k: df
    fake.batting_stats = lambda *a, **k: pd.DataFrame()
    fake.playerid_reverse_lookup = lambda *a, **k: pd.DataFrame()

    class _Cache:
        @staticmethod
        def enable():
            pass
    fake.cache = _Cache()
    sys.modules["pybaseball"] = fake


def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
    except Exception as e:
        FAILS.append(f"{label}: {type(e).__name__}: {e}")
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")


def test_grader():
    """The grader must return a record, not None and not raise."""
    snaps = sorted(glob.glob(os.path.join(ROOT, "docs", "snapshots", "20*.json")))
    if not snaps:
        print("  SKIP  grader (no snapshots on disk)")
        return
    snap_path = snaps[-1]
    date = os.path.basename(snap_path)[:-5]
    with open(snap_path) as f:
        players = (json.load(f).get("players") or [])
    if not players:
        print("  SKIP  grader (snapshot has no players)")
        return
    frame = _fake_statcast_frame([p["id"] for p in players], date)
    _install_pybaseball_stub(frame)
    from etl import track
    # track.py binds `statcast` at import time, so patch the bound name directly — otherwise
    # a stub installed later is ignored and the test silently tests nothing.
    track.statcast = lambda *a, **k: frame

    def run():
        rec = track.grade_date(date)
        assert rec is not None, "grade_date returned None with a valid snapshot + full frame"
        assert rec.get("date") == date
        assert rec.get("players"), "no player count in record"
        assert isinstance(rec.get("hr_log"), list), "hr_log missing"
        # the exact bug this test was written for: ranks referenced before assignment
        for entry in (rec.get("hr_log") or [])[:1]:
            assert "heat_rank" in entry, "heat_rank not captured in hr_log"
            assert "cv_rank" in entry, "cv_rank not captured in hr_log"
    check(f"grader runs end-to-end on {date}", run)


def test_feature_modules():
    """Feature helpers must survive a realistic frame without raising."""
    df = _fake_statcast_frame([111, 222, 333], "2026-08-02", n=800)
    _install_pybaseball_stub(df)
    from etl import features as F

    def zones():
        bz = F.batter_zone_damage(df[df["batter"] == 111])
        grid = F.pitcher_zone_grid(df, hand="R")
        F.zone_matchup_edges(bz, grid)
    check("features: zone damage / grid / matchup edges", zones)

    from etl import statcast_data as S

    def sc():
        S.batted_ball_log(df, [111], n=5)
        S.pitcher_arsenal(df, [600000])
        S.batter_vs_pitch(df, [111], min_pitches=1)
        S.team_defense(df)
        S.team_k_splits(df)
    check("statcast_data: logs / arsenal / vs-pitch / defense / K splits", sc)


def test_degenerate_inputs():
    """Empty / partial frames must not raise.

    This exists because a build died in Actions on `ValueError: not enough values to unpack`:
    a helper's happy path was updated to return four values while its two early-exit guards
    still returned two. Every pre-deploy check passed, because the guards only fire when a
    hitter has no batted balls with coordinates — a case the normal test data never produced.
    Feed each helper the degenerate inputs deliberately."""
    import pandas as _pd
    from etl import statcast_data as S
    from etl import features as F

    empties = [
        ("empty frame", _pd.DataFrame()),
        ("missing columns", _pd.DataFrame({"stand": ["R"]})),
        ("all-NaN coords", _pd.DataFrame({"hc_x": [None], "hc_y": [None], "stand": ["R"],
                                          "bb_type": ["fly_ball"], "launch_angle": [20.0],
                                          "launch_speed": [95.0], "zone": [5]})),
    ]
    for label, frame in empties:
        def run(frame=frame):
            a, b, c, d = S._pull_metrics(frame)      # must always unpack to four
            F.batter_zone_damage(frame)
            F.pitcher_zone_grid(frame)
            S.batted_ball_log(frame, [1])
            S.batter_vs_pitch(frame, [1])
            S.team_defense(frame)
            S.first_inning_splits(frame, [1])
        check(f"degenerate input: {label}", run)


def test_imports():
    """Every ETL module must import cleanly."""
    import importlib
    mods = ["etl.build_board", "etl.features", "etl.statcast_data", "etl.runs",
            "etl.props", "etl.park_model", "etl.track", "etl.backtest", "etl.compute"]
    for m in mods:
        check(f"import {m}", lambda m=m: importlib.import_module(m))


if __name__ == "__main__":
    print("SMOKE TEST — offline, no network required\n")
    print("imports:")
    _install_pybaseball_stub(_fake_statcast_frame([1], "2026-08-02", n=10))
    test_imports()
    print("\nfeature modules:")
    test_feature_modules()

    print("\ndegenerate inputs:")
    test_degenerate_inputs()
    print("\ngrader:")
    test_grader()

    print()
    if FAILS:
        print(f"SMOKE FAIL — {len(FAILS)} problem(s):")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE PASS — ETL paths execute cleanly")
