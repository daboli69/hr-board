"""
update_live_hrs.py -- lightweight, standalone update of "who's homered tonight so far."

Deliberately kept SEPARATE from the main build (build_board.py). The main build calls
statcast_data.pull_season(), a full-season pitch-level pull that's already been slow enough to
occasionally time out as the season grows (real incident, fixed earlier this session by adding
retry margin). Running that full build every 15-20 minutes through an entire evening would be
wasteful at best and a real repeat-timeout risk at worst.

This script does the opposite: it reads the EXISTING board.json (already built by the normal
schedule), pulls just the real game_pks from board["games"], calls MLB's live game feed (fast,
cheap, no Statcast involved) via statsapi.get_live_hrs_today(), and patches only the
"live_hrs_tonight" field into the existing file. Every other field is left completely alone.

Run frequently during the evening game window (see the matching workflow, live-hrs.yml) --
this is cheap enough to run every 15 minutes without the load or risk the full build carries.
"""
from __future__ import annotations
import json, os, sys

BOARD_PATH = os.environ.get("BOARD_PATH", "docs/board.json")


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from etl import statsapi

    try:
        with open(BOARD_PATH) as f:
            board = json.load(f)
    except Exception as e:
        print(f"[live-hrs] could not read {BOARD_PATH}: {e}", file=sys.stderr)
        sys.exit(1)

    game_pks = [g["game_pk"] for g in (board.get("games") or []) if g.get("game_pk")]
    if not game_pks:
        print("[live-hrs] no games found in board.json -- nothing to check.")
        return

    try:
        live_hrs = statsapi.get_live_hrs_today(game_pks)
    except Exception as e:
        print(f"[live-hrs] live fetch failed, leaving existing field untouched: {e}",
              file=sys.stderr)
        return   # deliberately don't overwrite a good list with an empty one on a transient
                 # fetch failure -- same "never zero out real data on a bad fetch" principle
                 # the main build already follows elsewhere

    prev = board.get("live_hrs_tonight") or []
    board["live_hrs_tonight"] = live_hrs

    with open(BOARD_PATH, "w") as f:
        json.dump(board, f, separators=(",", ":"), default=str)

    print(f"[live-hrs] {len(live_hrs)} real home runs found across {len(game_pks)} games "
          f"checked (was {len(prev)} before this run) → {BOARD_PATH}")


if __name__ == "__main__":
    main()
