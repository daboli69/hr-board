"""
hrr_recon.py — the single, authoritative HRR (Hits + Runs + RBIs) reconstruction.

Both backtest.py and track.py import this so calibration and live grading use IDENTICAL logic.
Previously each had its own base-runner simulation that only advanced runners on hits/walks
and only credited RBIs on hits — missing sac flies, groundout/fielders-choice RBIs, and runners
scoring on outs. Every omission biased HRR DOWNWARD, so the model's calibrated P(HRR>=2) read
low and Unders looked +EV almost everywhere. That was a calibration artifact, not an edge.

The fix uses AUTHORITATIVE Statcast fields where they exist:
  * HITS  — exact, from `events`.
  * RBIs  — exact, from the `rbi` field (Statcast credits the batter directly). No simulation.
  * RUNS  — attributed to the player who CROSSED THE PLATE. Statcast has no per-runner run flag,
            so runs still need a base-state walk, but we anchor it to the authoritative per-PA
            run total (post_bat_score - bat_score) so the COUNT of runs each PA is exact even
            when our guess of WHICH runner scored is approximate. This removes the systematic
            undercount; any residual error is in attribution, not totals, and is unbiased.

If the authoritative fields are absent (older pulls), we fall back to an improved simulation
that at least credits sac flies and out-based RBIs.
"""

from __future__ import annotations

HIT_EVENTS = {"single", "double", "triple", "home_run"}
# events that can score a runner / credit an RBI without being a hit
RBI_OUT_EVENTS = {"sac_fly", "sac_fly_double_play", "field_out", "fielders_choice",
                  "fielders_choice_out", "grounded_into_double_play", "force_out",
                  "double_play", "sac_bunt"}
PA_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "strikeout",
    "strikeout_double_play", "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "field_error", "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice", "fielders_choice_out", "catcher_interf", "intent_walk",
    "triple_play", "sac_fly_double_play",
}


def _num(v):
    try:
        f = float(v)
        return f if f == f else None      # NaN guard
    except Exception:
        return None


def hrr_map(sc):
    """{bid: {"hits","runs","rbis"}} for a Statcast frame. Uses authoritative rbi + score-delta
    when present; otherwise an improved simulation. Runs are attributed to the runner who scored
    via a base-state walk, but the per-PA run COUNT is pinned to the authoritative score delta
    so totals are exact.
    """
    out = {}
    if sc is None or getattr(sc, "empty", True):
        return out
    have_rbi = "rbi" in sc.columns
    have_score = "bat_score" in sc.columns and "post_bat_score" in sc.columns

    need = ["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number",
            "events", "batter"]
    if not all(c in sc.columns for c in need):
        return out
    df = sc.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"])
    df = df[df["events"].isin(list(PA_EVENTS))]
    if df.empty:
        return out

    for (gp, inn, half), grp in df.groupby(["game_pk", "inning", "inning_topbot"], sort=False):
        seen = set()
        base = {}      # runner_id -> base (1/2/3)
        for _, row in grp.iterrows():
            ab = row.get("at_bat_number")
            if ab in seen or row["batter"] != row["batter"]:
                continue
            seen.add(ab)
            bid = int(row["batter"])
            ev = row["events"]
            rec = out.setdefault(bid, {"hits": 0, "runs": 0, "rbis": 0})

            # HITS — exact
            if ev in HIT_EVENTS:
                rec["hits"] += 1

            # RBIs — authoritative when available
            if have_rbi:
                r = _num(row.get("rbi"))
                if r:
                    rec["rbis"] += int(r)
            else:
                # fallback: credit RBIs from the improved sim (hits + run-scoring outs)
                if ev == "home_run":
                    rec["rbis"] += 1 + len(base)
                elif ev in ("single", "double", "triple"):
                    scored = len(base) if ev != "single" else sum(1 for b in base.values() if b == 3)
                    rec["rbis"] += scored
                elif ev in RBI_OUT_EVENTS and any(b == 3 for b in base.values()):
                    rec["rbis"] += 1

            # RUNS — how many scored THIS PA (authoritative count via score delta)
            runs_this_pa = None
            if have_score:
                bs = _num(row.get("bat_score")); pbs = _num(row.get("post_bat_score"))
                if bs is not None and pbs is not None:
                    runs_this_pa = max(0, int(round(pbs - bs)))

            # advance base state + attribute runs to whoever scored
            scored_runners = []
            if ev == "home_run":
                scored_runners = list(base) + [bid]        # everyone on + the batter
                base = {}
            elif ev in ("single", "double", "triple"):
                adv = {"single": 1, "double": 2, "triple": 3}[ev]
                for runner, b in list(base.items()):
                    if b + adv >= 4:
                        scored_runners.append(runner); del base[runner]
                    else:
                        base[runner] = b + adv
                base[bid] = adv
            elif ev in ("walk", "intent_walk", "hit_by_pitch", "catcher_interf"):
                # force only advances runners when forced
                if base.get(_first_base_runner(base)) is not None:
                    pass
                # simple force chain: bump anyone forced
                occupied = set(base.values())
                if 1 in occupied:
                    for runner in sorted(list(base), key=lambda r: -base[r]):
                        if base[runner] == 3 and 1 in occupied and 2 in occupied:
                            scored_runners.append(runner); del base[runner]
                        elif base[runner] == 2 and 1 in occupied:
                            base[runner] = 3
                        elif base[runner] == 1:
                            base[runner] = 2
                base[bid] = 1
            elif ev in RBI_OUT_EVENTS:
                # sac fly / groundout: a runner on 3rd typically scores
                for runner, b in list(base.items()):
                    if b == 3:
                        scored_runners.append(runner); del base[runner]
                        break

            # Reconcile with authoritative run count: if the score delta says N runs scored but
            # our base walk found a different number, trust the authoritative COUNT. Credit the
            # runs to the runners we think scored; if we're short, credit the extra to the most
            # advanced remaining runners (closest to home = likeliest to have scored).
            if runs_this_pa is not None:
                if len(scored_runners) < runs_this_pa:
                    remaining = sorted(base.items(), key=lambda kv: -kv[1])
                    for runner, _b in remaining:
                        if len(scored_runners) >= runs_this_pa:
                            break
                        scored_runners.append(runner); base.pop(runner, None)
                elif len(scored_runners) > runs_this_pa:
                    scored_runners = scored_runners[:runs_this_pa]

            for runner in scored_runners:
                out.setdefault(int(runner), {"hits": 0, "runs": 0, "rbis": 0})["runs"] += 1

    return out


def _first_base_runner(base):
    for r, b in base.items():
        if b == 1:
            return r
    return None


def hrr_total(sc):
    """{bid: hits+runs+rbis} convenience wrapper."""
    m = hrr_map(sc)
    return {bid: v["hits"] + v["runs"] + v["rbis"] for bid, v in m.items()}
