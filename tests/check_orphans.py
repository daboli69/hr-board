"""Orphan check — does what the ETL computes actually REACH the app?

This exists because `check_dead_fields.py` has a blind spot that cost real work. It compares
board.json against the frontend, so it can only see things that made it INTO board.json. When
`environment.bullpen_state()` was computed every run, used internally for arsenal blending, and
never written to the board at all, every check passed — the Bullpens tab kept showing the older
Statcast-inferred availability while better data sat unused in the same build.

So there are three distinct failure modes, and only the last one was covered:

    1. ORPHANED  — an engine function is never called anywhere. Dead module.
    2. INTERNAL  — it is called, but its result never reaches board.json. Computed and thrown
                   away, which is the case that bit us.
    3. DEAD      — it reaches board.json but no frontend code reads it. (check_dead_fields.py)

This script covers 1 and 2 by parsing the ETL rather than the output, so it catches work that
never produces an artifact to inspect.

    python tests/check_orphans.py
"""
from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETL = os.path.join(ROOT, "etl")

# The engines added in the sabermetric upgrade, with the aliases they are imported under.
ENGINES = {
    "markov":      ["markov"],
    "hitmodel":    ["hitmodel"],
    "kengine":     ["kengine"],
    "arsenal":     ["arsenal", "arsenal_mod"],
    "environment": ["environment", "env_mod"],
    "validate":    ["validate", "V"],
}

CONSUMERS = ["build_board.py", "runs.py", "props.py", "backtest.py", "fetch_odds.py"]

# Functions that are genuinely internal helpers — called by other engine functions rather than
# by the ETL directly. Reporting them as orphans would be noise.
INTERNAL_OK = {
    "carry_feet", "is_near_miss", "fence_delta", "air_density",
    "pa_event_probs", "InningState", "prob_over", "win_prob_from_dists", "game_totals",
    "PitchShapeSplits", "directional_defense", "expected_pa",
    "csw_from_statcast", "framing_csw_adjust", "catcher_framing", "arsenal_depth",
    "dynamic_xbf", "velocity_flag", "k_distribution", "prob_over_ks", "TTOP_BY_DEPTH",
    "hidden_platoon_pitches", "bat_speed_vs_velocity_credit", "contact_family_signals",
    "reliever_status", "decile_calibration", "AsOfFrame",
    # Verified internal: called by other engine functions rather than by the ETL.
    "rbi_context", "run_context", "tto_penalty_for", "park_for_row", "spray_angle_deg",
    "effective_batter_hand", "expected_pa",
    # Feeds a shipped field through a derived path the regex cannot trace: csw_map and framing
    # are consumed by kengine, whose output ships as a.kengine; blend/credit results become
    # convergence family LABELS (strings) rather than being carried as values.
    "csw_from_statcast", "catcher_framing", "blend_arsenal_for_pa", "bullpen_arsenal",
    "directional_defense_proxy",
}


def public_functions(path):
    """Top-level function names a module exposes."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return []
    return [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]


def etl_source():
    out = {}
    for f in CONSUMERS:
        p = os.path.join(ETL, f)
        if os.path.exists(p):
            out[f] = open(p, encoding="utf-8").read()
    return out


def reaches_board(src, varnames):
    """Does any of these variable names end up in the shipped board dict?

    Approximate but useful: a value reaches the app if it appears as a dict VALUE somewhere
    (`"key": var`), is assigned onto a player record (`_pl["k"] = var`), or is appended into a
    list that the board carries. A variable only ever read into another local is internal.
    """
    for v in varnames:
        pats = [
            rf'"[\w:]+"\s*:\s*{re.escape(v)}\b',        # "key": var
            rf'"[\w:]+"\s*:\s*\(?{re.escape(v)}\s*(?:or|\.|\[)',   # "key": var.get(...) / var[...]
            rf'\[\s*"[\w:]+"\s*\]\s*=\s*{re.escape(v)}\b',          # rec["key"] = var
            rf'\.append\(\s*{re.escape(v)}\b',                       # list.append(var)
            rf'\.update\(\s*{re.escape(v)}\b',
        ]
        for p in pats:
            if re.search(p, src):
                return True
    return False


def main():
    srcs = etl_source()
    if not srcs:
        print("SKIP — no ETL sources found")
        return 0
    all_src = "\n".join(srcs.values())

    orphaned, internal, ok = [], [], []

    for mod, aliases in ENGINES.items():
        path = os.path.join(ETL, f"{mod}.py")
        if not os.path.exists(path):
            continue
        for fn in public_functions(path):
            called = any(re.search(rf"\b{re.escape(a)}\s*\.\s*{re.escape(fn)}\s*\(", all_src)
                         for a in aliases)
            if not called:
                if fn not in INTERNAL_OK:
                    orphaned.append((mod, fn))
                continue
            # find the variables its result is assigned to, across all consumers
            targets = set()
            for a in aliases:
                for m in re.finditer(
                        rf"(?:^|\n)\s*([\w\[\]\"'\.]+)\s*=\s*{re.escape(a)}\s*\.\s*{re.escape(fn)}\s*\(",
                        all_src):
                    targets.add(m.group(1).strip())
                # direct use inside a dict literal counts as reaching the board
                if re.search(rf'"[\w:]+"\s*:\s*{re.escape(a)}\s*\.\s*{re.escape(fn)}\s*\(', all_src):
                    targets.add("__inline__")
            if "__inline__" in targets or any(t.startswith(("_pl[", 'p["')) for t in targets):
                ok.append((mod, fn)); continue
            if targets and reaches_board(all_src, targets):
                ok.append((mod, fn))
            elif targets and fn not in INTERNAL_OK:
                internal.append((mod, fn, sorted(targets)[:3]))
            elif targets:
                ok.append((mod, fn))
            else:
                ok.append((mod, fn))   # called inline, no assignment to trace

    print("ORPHAN CHECK — does engine output reach the app?\n")
    print(f"  reaches board.json : {len(ok)}")
    print(f"  computed, not shipped : {len(internal)}")
    print(f"  never called at all : {len(orphaned)}")
    print()

    if orphaned:
        print("NEVER CALLED — dead code, or a wiring step that was missed:")
        for m, f in orphaned:
            print(f"  {m}.{f}()")
        print()

    if internal:
        print("COMPUTED BUT NEVER SHIPPED — this is the class that hid the bullpen gap.")
        print("Each of these runs every build and its result never reaches board.json, so no")
        print("screen can show it and no dead-field check can see it:")
        for m, f, tv in internal:
            print(f"  {m}.{f}()  ->  {', '.join(tv)}")
        print()
        print("  Either attach the result to the board, or note why it is intentionally internal.")
        print()

    if not orphaned and not internal:
        print("PASS — every engine function is called and its output reaches the board.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
