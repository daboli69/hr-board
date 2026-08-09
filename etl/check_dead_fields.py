"""Dead-field check: does the frontend actually READ what the ETL writes?

This exists because seven fields shipped in board.json for weeks while nothing displayed them —
markov run distributions, bat tracking, near misses, pull-barrel, first-inning splits. Each was
computed correctly, verified in the ETL, and invisible in the app. Data landing in the JSON felt
like completion; it isn't, because nothing renders until something reads it.

The check is deliberately one-directional and forgiving:
  - flags a field ONLY when it has real data AND no frontend reference exists
  - ignores fields with no data (that is an upstream problem, not a wiring one)
  - recognises dynamic access like g[side + "_converge"], which a naive search misses and
    would otherwise report as a false alarm

A false alarm here is expensive — it sends you hunting a bug that does not exist — so when in
doubt this stays quiet rather than shouting.

    python tests/check_dead_fields.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD = os.path.join(ROOT, "docs", "board.json")
HTML = os.path.join(ROOT, "docs", "index.html")

# Fields that legitimately never reach the UI: internal bookkeeping, or inputs consumed by
# other ETL stages rather than displayed.
IGNORE = {
    "id", "name", "team", "opp_team", "game_pk", "generated_at", "slate_date",
    "model_version", "recent_window", "lineups_pending", "projected_games",
    "_processed_games", "_balls", "_complete", "updated",
    # Identifiers and intermediate inputs: consumed by other ETL stages or used for lookups,
    # never rendered directly. Flagging them would be noise, and a check that cries wolf gets
    # ignored — which defeats the point.
    "home_sp_id", "away_sp_id", "home_id", "away_id", "pitcher_id", "batter_id",
    "opp_lineup_n", "recent_weight", "k_signals", "est_line_over",
}

# Minimum share of records that must carry a field before absence in the UI counts as a
# problem. A field present on three of 270 players is probably a rare edge case, not a
# forgotten wire-up.
MIN_COVERAGE = 0.25


def frontend_reads(html, field):
    """Does the frontend reference this field, including via dynamic key construction?"""
    if re.search(rf'\b{re.escape(field)}\b', html):
        return True
    # dynamic access: g[side + "_converge"], p["hit_" + k], etc.
    for suffix in ("_converge", "_wp", "_runs", "_dist", "_mean", "_heat", "_rank"):
        if field.endswith(suffix) and re.search(rf'\+\s*"{re.escape(suffix)}"', html):
            return True
    return False


def scan(records, label, html, out):
    """Coverage of every key across a list of records, vs frontend references."""
    if not records:
        return
    counts = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            if v is None or v == [] or v == {}:
                continue
            counts[k] = counts.get(k, 0) + 1
    total = len(records)
    for k, n in sorted(counts.items()):
        if k in IGNORE:
            continue
        if n / total < MIN_COVERAGE:
            continue
        if not frontend_reads(html, k):
            out.append((label, k, n, total))


def main():
    if not os.path.exists(BOARD) or not os.path.exists(HTML):
        print("SKIP — board.json or index.html not found")
        return 0
    with open(BOARD) as f:
        board = json.load(f)
    html = open(HTML, encoding="utf-8").read()

    dead = []
    scan(board.get("players") or [], "player", html, dead)
    scan(board.get("pitcher_props") or [], "arm", html, dead)
    scan(board.get("pitcher_edges") or [], "arm_edge", html, dead)
    scan(board.get("game_projections") or [], "game", html, dead)

    # top-level payloads
    for k, v in board.items():
        if k in IGNORE or v is None or v == [] or v == {}:
            continue
        if isinstance(v, (dict, list)) and not frontend_reads(html, k):
            dead.append(("board", k, 1, 1))

    print(f"scanned board.json against index.html "
          f"({len(board.get('players') or [])} players, "
          f"{len(board.get('pitcher_props') or [])} arms, "
          f"{len(board.get('game_projections') or [])} games)")
    print()
    if not dead:
        print("PASS — every populated field is read somewhere in the frontend")
        return 0

    print(f"{len(dead)} field(s) have data but are never read by the UI:")
    for label, k, n, total in dead:
        print(f"  [{label}] {k}  — present on {n}/{total} records, no frontend reference")
    print()
    print("Each of these is computed, shipped, and invisible. Either wire it into a view or")
    print("stop emitting it — carrying it costs payload for nothing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
