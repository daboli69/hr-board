"""Pre-build canary: run the pipeline's fragile paths against nullable/Arrow-style
frames BEFORE the real build. If the runner's pandas ever changes semantics again
(the bug family that broke labels and the grader), the workflow fails HERE with a
clear message instead of shipping a silently degraded board."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from etl import statcast_data as S
from etl import track as T


def frame():
    rows = []
    for i in range(60):
        rows.append(dict(
            game_pk=1001, inning=1 + i % 9, inning_topbot="Top" if i % 2 else "Bot",
            at_bat_number=i + 1, pitch_number=1, pitcher=111 if i % 2 else 222,
            batter=9000 + i % 5, stand="R", p_throws="R", game_date="2026-06-20",
            events="home_run" if i in (7, 30) else ("field_out" if i % 3 else None),
            description="hit_into_play", launch_speed=88.0 + i % 15, launch_angle=5.0 + i % 35,
            launch_speed_angle=6 if i % 7 == 0 else 3, hc_x=80.0 + i, hc_y=100.0,
            hit_distance_sc=250.0 + (i % 16) * 12, bat_speed=70.0 + i % 8, release_speed=91.0,
            attack_angle=10.0, woba_value=0.0, estimated_woba_using_speedangle=0.3,
            bb_type="fly_ball", pitch_type="FF", home_team="BOS", away_team="NYY",
            type="X", woba_denom=1))
    df = pd.DataFrame(rows)
    for c in ("events", "stand", "inning_topbot", "home_team", "away_team", "description",
              "bb_type", "pitch_type", "p_throws", "game_date", "type"):
        df[c] = df[c].astype("string")
    for c in ("launch_speed", "hit_distance_sc", "bat_speed", "release_speed", "hc_x",
              "hc_y", "launch_angle", "attack_angle", "woba_value",
              "estimated_woba_using_speedangle"):
        df[c] = df[c].astype("Float64")
    for c in ("inning", "at_bat_number", "pitch_number", "batter", "pitcher", "game_pk",
              "launch_speed_angle", "woba_denom"):
        df[c] = df[c].astype("Int64")
    df.loc[df.index[3], "events"] = pd.NA
    df.loc[df.index[5], "bat_speed"] = pd.NA
    return df


def main():
    df = frame()
    norm = S.normalize_frame(df.copy())
    S.hitter_labels(norm, "2026-06-01", min_bbe=3)          # must not raise
    prof = S.batter_profiles(norm, [9001], asof="2026-06-21")
    assert prof and prof[9001]["recent"].get("bb_count"), "profiles broke on nullable frame"
    hm = T._hr_map(T._normalize_sc(df.copy()))
    assert sum(v["hr"] for v in hm.values()) == 2, "grader HR extraction broke"
    print("CANARY PASS — nullable-dtype semantics intact")


if __name__ == "__main__":
    main()
