"""Arsenal matchups — handedness-first filtering, late-inning blending, bat tracking.

THE ORDERING BUG THIS FIXES. Filtering an arsenal by usage BEFORE splitting by handedness
hides exactly the pitches that decide platoon at-bats. A right-hander who throws his changeup
6% overall looks like he does not have one — but that 6% is 22% of what he throws to lefties,
because he almost never uses it against righties. Filter first by hand, THEN by usage, and the
putaway pitch reappears. Overall usage is a blend of two different pitchers.
"""
from __future__ import annotations

MIN_USAGE_PCT = 10.0
FAST_SWING_MPH = 75.0
SHORT_SWING_FT = 7.0
HIGH_VELO_MPH = 94.0


def effective_batter_hand(bats, pitcher_throws):
    """Which side the hitter actually bats from tonight. Switch hitters take the platoon edge."""
    if bats == "S":
        return "L" if pitcher_throws == "R" else "R"
    return bats or "R"


def handedness_first_arsenal(arsenals_by_hand, batter_hand, min_usage=MIN_USAGE_PCT):
    """[(pitch_type, usage_pct_vs_this_hand, n)] — split by hand FIRST, then thresholded.

    `arsenals_by_hand` is {"R": [(pt, usage, n), ...], "L": [...]} where usage is already
    computed WITHIN that handedness split, so the percentages sum to ~100 per side.
    """
    side = (arsenals_by_hand or {}).get(batter_hand) or []
    return [tuple(p) for p in side if float(p[1]) >= min_usage]


def hidden_platoon_pitches(arsenals_by_hand, batter_hand, overall_arsenal,
                           min_usage=MIN_USAGE_PCT):
    """Pitches that clear the usage bar vs THIS hand but are invisible in the overall mix.

    Reporting these is the point of the whole ordering change — these are the offerings a
    generic arsenal view would have dropped, and they are disproportionately putaway pitches.
    """
    side = handedness_first_arsenal(arsenals_by_hand, batter_hand, min_usage)
    overall = {p[0]: float(p[1]) for p in (overall_arsenal or [])}
    return [{"pt": pt, "usage_vs_hand": round(float(u), 1),
             "usage_overall": round(overall.get(pt, 0.0), 1)}
            for pt, u, *_ in side if overall.get(pt, 0.0) < min_usage]


def blend_arsenal_for_pa(starter_arsenal, pen_arsenal, pa_index, xbf):
    """Pitch mix a hitter's Nth plate appearance will actually face.

    PAs before the starter's xBF see the starter; PAs after it see the bullpen. The boundary PA
    is blended in proportion, because a projected xBF of 22.4 means the 23rd hitter faces the
    starter about 40% of the time — treating that as a hard cutoff throws away real information
    on precisely the plate appearance most likely to decide a prop.
    """
    if not pen_arsenal:
        return list(starter_arsenal or [])
    if not starter_arsenal:
        return list(pen_arsenal)
    lo, hi = int(xbf), int(xbf) + 1
    if pa_index < lo:
        w_sp = 1.0
    elif pa_index >= hi:
        w_sp = 0.0
    else:
        w_sp = float(xbf) - lo                 # fractional PA at the boundary

    merged = {}
    for pt, u, *rest in starter_arsenal:
        merged[pt] = merged.get(pt, 0.0) + float(u) * w_sp
    for pt, u, *rest in pen_arsenal:
        merged[pt] = merged.get(pt, 0.0) + float(u) * (1.0 - w_sp)
    tot = sum(merged.values()) or 1.0
    return sorted(((pt, round(100.0 * v / tot, 1), None) for pt, v in merged.items()),
                  key=lambda x: -x[1])


def bullpen_arsenal(relievers_available):
    """Aggregate the AVAILABLE bullpen into one pitch mix, leverage-weighted.

    Only available arms are included — an unavailable reliever's pitches are not pitches this
    hitter can face tonight, so averaging them in would describe a bullpen that will not appear.
    """
    merged, den = {}, 0.0
    for r in (relievers_available or []):
        w = float(r.get("leverage") or 1.0)
        for pt, u, *rest in (r.get("arsenal") or []):
            merged[pt] = merged.get(pt, 0.0) + w * float(u)
        den += w
    if not merged or den <= 0:
        return []
    tot = sum(merged.values()) or 1.0
    return sorted(((pt, round(100.0 * v / tot, 1), None) for pt, v in merged.items()),
                  key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Bat tracking -> Contact Quality family
# ---------------------------------------------------------------------------

def bat_tracking_profile(bat_speeds, swing_lengths=None):
    """{avg_bat_speed, fast_swing_rate, avg_swing_length, short_fast_rate}

    "Short and fast" is the combination worth crediting. Bat speed alone rewards big loopy
    swings that cannot catch up to velocity; swing length alone rewards slap hitters with no
    power. A hitter who pairs 75+ mph with a compact path is the one who handles a good
    fastball, which is exactly the matchup the arsenal model cares about.
    """
    speeds = [float(s) for s in (bat_speeds or []) if s is not None]
    if not speeds:
        return None
    lens = [float(x) for x in (swing_lengths or []) if x is not None]
    fast = sum(1 for s in speeds if s >= FAST_SWING_MPH)
    prof = {
        "avg_bat_speed": round(sum(speeds) / len(speeds), 1),
        "fast_swing_rate": round(100.0 * fast / len(speeds), 1),
        "n_swings": len(speeds),
    }
    if lens:
        prof["avg_swing_length"] = round(sum(lens) / len(lens), 2)
        if len(lens) == len(speeds):
            short_fast = sum(1 for s, l in zip(speeds, lens)
                             if s >= FAST_SWING_MPH and l <= SHORT_SWING_FT)
            prof["short_fast_rate"] = round(100.0 * short_fast / len(speeds), 1)
    return prof


def bat_speed_vs_velocity_credit(profile, opp_fb_velo):
    """Contact-quality credit (0-1) for a hitter who can handle THIS arm's velocity.

    The credit is conditional on the matchup on purpose: a compact 78-mph swing is a real
    advantage against 97, and close to irrelevant against a soft-tossing 89-mph starter. A
    context-free "fast swing" bonus would fire in both spots and mean nothing in either.
    """
    if not profile:
        return None
    velo = float(opp_fb_velo or 92.0)
    if velo < HIGH_VELO_MPH:
        return None                    # not a velocity matchup; no credit either way
    bs = profile.get("avg_bat_speed")
    if bs is None:
        return None
    speed_score = max(0.0, min(1.0, (bs - 68.0) / 12.0))
    sf = profile.get("short_fast_rate")
    if sf is not None:
        short_score = max(0.0, min(1.0, sf / 35.0))
        base = 0.6 * speed_score + 0.4 * short_score
    else:
        base = speed_score
    # scale by how extreme the velocity actually is
    velo_weight = min(1.0, (velo - HIGH_VELO_MPH) / 4.0 + 0.5)
    return round(base * velo_weight, 3)


def contact_family_signals(percentiles, square_up, hr_power, bat_profile, opp_fb_velo):
    """Signals for the Contact Quality family, bat tracking included.

    Bat tracking joins Contact Quality rather than forming its own family because it is
    measuring the same underlying thing as exit velocity and barrel rate — how well this hitter
    strikes a baseball. Giving it a separate family would let one piece of evidence count twice,
    which is the exact collinearity the family structure exists to prevent.
    """
    sigs = []
    hi = [f"{v.get('label') or k} P{v['pctl']}" for k, v in (percentiles or {}).items()
          if isinstance(v, dict) and (v.get("pctl") or 0) >= 80]
    if len(hi) >= 3:
        sigs.append("MLB pctl: " + ", ".join(sorted(hi)[:3]))
    if (square_up or {}).get("rating", 0) >= 30:
        sigs.append(f"Square Up {square_up['rating']}")
    if (hr_power or {}).get("barrel_pct", 0) >= 10:
        sigs.append(f"Barrel {hr_power['barrel_pct']}%")
    credit = bat_speed_vs_velocity_credit(bat_profile, opp_fb_velo)
    if credit is not None and credit >= 0.60:
        bs = bat_profile.get("avg_bat_speed")
        sf = bat_profile.get("short_fast_rate")
        label = f"Bat speed {bs} vs {opp_fb_velo:.0f} mph"
        if sf is not None:
            label += f" ({sf}% short+fast)"
        sigs.append(label)
    return sigs
