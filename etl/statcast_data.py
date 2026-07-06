"""
Statcast data engine (Baseball Savant, free, keyless via pybaseball).

Strategy:
  * Pull raw pitch-level Statcast ONCE for the season window. From that single
    pull we compute BOTH recent-form windows (L5/L15/L30 games) AND season-to-date,
    keyed by mlbam id, with identical metric definitions.
  * Career comes from FanGraphs (batting_stats over a multi-year range), joined
    by normalized name. Statcast-era only (2015+). Wrapped so a schema hiccup
    degrades to null instead of crashing the cron.

Metric definitions (computed from batted balls / pitches):
  barrel_pct  = barrels / batted_balls         (launch_speed_angle == 6)
  hardhit_pct = (launch_speed >= 95) / batted_balls
  avg_ev      = mean launch_speed over batted balls
  launch_angle= mean launch_angle over batted balls
  fb_pct      = fly_balls / batted_balls
  iso         = (2B + 2*3B + 3*HR) / AB
  swstr_pct   = swinging strikes / pitches seen
  k_pct       = strikeouts / PA
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pybaseball import statcast, batting_stats


class StatcastUnavailable(Exception):
    """Raised when the Statcast pull comes back empty/short so callers can
    preserve the last good board instead of publishing blanks."""

# events that end a plate appearance (used for PA / K / ISO accounting)
PA_EVENTS = {
    "single", "double", "triple", "home_run", "field_out", "strikeout",
    "strikeout_double_play", "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "field_error", "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice", "fielders_choice_out", "catcher_interf", "intent_walk",
    "triple_play", "sac_fly_double_play",
}
SWING_STRIKE = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}


def pull_season(start: str, end: str) -> pd.DataFrame:
    """One ranged Statcast pull. Heavy on first run, cached after by pybaseball."""
    df = statcast(start_dt=start, end_dt=end)
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [
        "game_date", "game_pk", "batter", "pitcher", "events", "description",
        "launch_speed", "launch_angle", "launch_speed_angle", "bb_type", "hit_distance_sc",
        "stand", "p_throws", "type", "hc_x", "hc_y",
        "attack_angle", "bat_speed", "release_speed", "pitch_type",
        "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "home_team", "away_team",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def batted_ball_sample(df: pd.DataFrame, batter_ids) -> dict:
    """
    Per-batter raw batted-ball arrays for the park/weather trajectory model:
      {batter_id: {"ev": [...], "la": [...], "spray": [...]}}
    Spray is the field spray angle in degrees (negative = LF / 3B side, positive =
    RF / 1B side), same convention the trajectory engine and pull metrics use.
    Only batted balls with exit velocity, launch angle, and hit coordinates are kept.
    """
    out = {}
    need = {"launch_speed", "launch_angle", "hc_x", "hc_y", "batter"}
    if df.empty or not need.issubset(df.columns):
        return out
    d = df.dropna(subset=["launch_speed", "launch_angle", "hc_x", "hc_y"])
    if d.empty:
        return out
    spray = np.degrees(np.arctan2(d["hc_x"].to_numpy() - 125.42, 198.27 - d["hc_y"].to_numpy()))
    bat = d["batter"].to_numpy()
    ev = d["launch_speed"].to_numpy(dtype=float)
    la = d["launch_angle"].to_numpy(dtype=float)
    wanted = set(batter_ids)
    for bid in wanted:
        m = bat == bid
        if not m.any():
            continue
        out[bid] = {"ev": ev[m], "la": la[m], "spray": spray[m]}
    return out


def _pull_metrics(bb: pd.DataFrame) -> tuple:
    """
    Returns (pull_pct, pull_air_pct):
      pull_pct     = pulled / all batted balls (context)
      pull_air_pct = pulled / (fly balls + line drives) = THE pull metric you read,
                     where 40%+ is the "good" mark for HR creation.
    spray angle = atan2(hc_x-125.42, 198.27-hc_y): negative=LF (3B side), positive=RF.
    RHB pulls to LF (negative), LHB to RF (positive). Pull threshold = beyond +/-15 deg.
    """
    if bb.empty or "hc_x" not in bb or "hc_y" not in bb:
        return None, None
    d = bb.dropna(subset=["hc_x", "hc_y"])
    n = len(d)
    if n == 0:
        return None, None
    angle = np.degrees(np.arctan2(d["hc_x"] - 125.42, 198.27 - d["hc_y"]))
    is_r = d["stand"].values == "R"
    pulled = np.where(is_r, angle.values < -15, angle.values > 15)
    air = d["bb_type"].isin(["fly_ball", "line_drive"]).values
    n_air = int(air.sum())
    pull_pct = round(100.0 * pulled.sum() / n, 1)
    pull_air_pct = round(100.0 * (pulled & air).sum() / n_air, 1) if n_air else None
    return pull_pct, pull_air_pct


def _ideal_aa(rows: pd.DataFrame) -> tuple:
    """
    Ideal Attack Angle rate + avg bat speed over competitive swings.
    Competitive swing proxy: a tracked swing (bat_speed present) with bat_speed >= 60
    (approximates Statcast's 'fastest 90% of swings' rule, stable on small windows).
    ideal_aa_pct = share of competitive swings with attack_angle in [5,20].
    """
    if "bat_speed" not in rows or "attack_angle" not in rows:
        return None, None
    sw = rows[rows["bat_speed"].notna()]
    comp = sw[sw["bat_speed"] >= 60]
    nc = len(comp)
    if nc < 1:
        return None, None
    bs = round(comp["bat_speed"].mean(), 1)
    if nc < 12:                       # too few competitive swings for a reliable IAA
        return None, bs
    aa = comp["attack_angle"]
    ideal = ((aa >= 5) & (aa <= 20)).sum()
    return round(100.0 * ideal / nc, 1), bs


PITCH_BUCKET = {
    # fastballs
    "FF": "FB", "FA": "FB", "SI": "FB", "FT": "FB", "FC": "FB",
    # breaking
    "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR", "CS": "BR", "SV": "BR", "SC": "BR", "KN": "BR",
    # offspeed
    "CH": "OFF", "FS": "OFF", "FO": "OFF",
}


def _pitch_splits(rows: pd.DataFrame) -> dict:
    """Per pitch-family (FB/BR/OFF) batted-ball damage for a hitter."""
    if rows.empty or "pitch_type" not in rows.columns:
        return {}
    work = rows.copy()
    work["fam"] = work["pitch_type"].map(PITCH_BUCKET)
    bb_all = work[work["launch_speed"].notna()]
    SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
    out = {}
    for fam in ("FB", "BR", "OFF"):
        fam_rows = work[work["fam"] == fam]            # all pitches of this family
        sub = bb_all[bb_all["fam"] == fam]             # batted balls of this family
        n = len(sub)
        if n < 5:
            continue
        swings = fam_rows["description"].isin(SWINGS).sum()
        whiffs = fam_rows["description"].isin(SWING_STRIKE).sum()
        out[fam] = {
            "barrel_pct": round(100.0 * (sub["launch_speed_angle"] == 6).sum() / n, 1) if "launch_speed_angle" in sub else None,
            "avg_ev": round(sub["launch_speed"].mean(), 1),
            "la": round(sub["launch_angle"].mean(), 1) if sub["launch_angle"].notna().any() else None,
            "whiff_pct": round(100.0 * whiffs / swings, 1) if swings else None,
            "hr": int((sub["events"] == "home_run").sum()),
            "bbe": n,
        }
    return out


def _pitch_usage(rows: pd.DataFrame) -> dict:
    """A pitcher's pitch-family usage % (of all pitches thrown)."""
    if rows.empty or "pitch_type" not in rows.columns:
        return {}
    fam = rows["pitch_type"].map(PITCH_BUCKET).dropna()
    total = len(fam)
    if total < 30:
        return {}
    out = {}
    for f in ("FB", "BR", "OFF"):
        out[f] = round(100.0 * (fam == f).sum() / total, 1)
    return out


def pitcher_hand_hr_2yr(pid: int, end_date: str) -> dict | None:
    """
    Actual HRs (and PA) a pitcher has allowed to RHB vs LHB over the trailing ~2 years,
    plus the this-season subset. Uses a per-pitcher Statcast pull (cached upstream so
    this isn't run every hourly build). Returns None on any failure (degrades cleanly).
    """
    from datetime import datetime, timedelta
    try:
        from pybaseball import statcast_pitcher
        end = datetime.strptime(end_date, "%Y-%m-%d")
        start2 = (end - timedelta(days=730)).strftime("%Y-%m-%d")
        df = statcast_pitcher(start2, end_date, int(pid))
    except Exception:
        return None
    if df is None or df.empty or "stand" not in df.columns or "events" not in df.columns:
        return None
    pa_rows = df[df["events"].isin(PA_EVENTS)]
    if pa_rows.empty:
        return None
    this_year = str(end.year)

    def _split(rows):
        o = {}
        for hand in ("R", "L"):
            h = rows[rows["stand"] == hand]
            o[hand] = {"hr": int((h["events"] == "home_run").sum()), "pa": int(len(h))}
        return o

    ty_rows = pa_rows[pa_rows["game_date"].astype(str).str.startswith(this_year)]
    return {"two_yr": _split(pa_rows), "this_yr": _split(ty_rows)}


def _agg_metrics(rows: pd.DataFrame) -> dict:
    """Compute the metric dict for an arbitrary subset of pitch-level rows."""
    if rows.empty:
        return {}
    bb = rows[rows["launch_speed"].notna()]            # batted balls
    n_bb = len(bb)
    pa_rows = rows[rows["events"].isin(PA_EVENTS)]
    pa = len(pa_rows)
    pitches = len(rows)

    def pct(numer, denom):
        return round(100.0 * numer / denom, 1) if denom else None

    ev = rows["events"]
    singles = (ev == "single").sum()
    doubles = (ev == "double").sum()
    triples = (ev == "triple").sum()
    hr = (ev == "home_run").sum()
    walks = ev.isin(["walk", "intent_walk"]).sum()
    hbp = (ev == "hit_by_pitch").sum()
    sacs = ev.isin(["sac_fly", "sac_bunt", "sac_fly_double_play"]).sum()
    ci = (ev == "catcher_interf").sum()
    ab = pa - walks - hbp - sacs - ci
    ks = ev.isin(["strikeout", "strikeout_double_play"]).sum()

    iso = round((doubles + 2 * triples + 3 * hr) / ab, 3) if ab > 0 else None
    slg = round((singles + 2 * doubles + 3 * triples + 4 * hr) / ab, 3) if ab > 0 else None

    pull_pct, pull_air_pct = _pull_metrics(bb)
    ideal_aa_pct, bat_speed = _ideal_aa(rows)

    # luck gap: expected vs actual on contact (Savant xwOBAcon vs wOBAcon)
    xwobacon = wobacon = luck_gap = None
    if n_bb and "estimated_woba_using_speedangle" in bb:
        xser = bb["estimated_woba_using_speedangle"].dropna()
        if len(xser) >= 8:
            xwobacon = round(xser.mean(), 3)
            if "woba_value" in bb and "woba_denom" in bb:
                wv = bb.loc[xser.index, "woba_value"]
                wd = bb.loc[xser.index, "woba_denom"]
                denom = wd.sum()
                if denom > 0:
                    wobacon = round(wv.sum() / denom, 3)
                    luck_gap = round(xwobacon - wobacon, 3)   # + = under-rewarded (due)

    out = {
        "barrel_pct": pct((bb["launch_speed_angle"] == 6).sum(), n_bb) if "launch_speed_angle" in bb else None,
        "hardhit_pct": pct((bb["launch_speed"] >= 95).sum(), n_bb),
        "avg_ev": round(bb["launch_speed"].mean(), 1) if n_bb else None,
        "max_ev": round(bb["launch_speed"].max(), 1) if n_bb else None,
        "launch_angle": round(bb["launch_angle"].mean(), 1) if n_bb else None,
        "fb_pct": pct((bb["bb_type"] == "fly_ball").sum(), n_bb) if "bb_type" in bb else None,
        "pull_pct": pull_pct,
        "pull_air_pct": pull_air_pct,
        "ideal_aa_pct": ideal_aa_pct,
        "bat_speed": bat_speed,
        "iso": iso,
        "slg": slg,
        "xwobacon": xwobacon,
        "wobacon": wobacon,
        "luck_gap": luck_gap,
        "swstr_pct": pct(rows["description"].isin(SWING_STRIKE).sum(), pitches),
        "k_pct": pct(ks, pa),
        "hr": int(hr),
        "pa": int(pa),
        "bb_count": int(n_bb),
    }
    return out


def batter_profiles(df: pd.DataFrame, batter_ids: list[int], asof: str,
                    recent_days: int = 14) -> dict:
    """
    For each batter: the headline RECENT line is the trailing `recent_days`
    calendar days (your "last 2 weeks of play"). Also keep L5/L15/L30 game
    windows + season for context in the expanded view.
    """
    from datetime import timedelta
    out = {}
    sub = df[df["batter"].isin(batter_ids)].copy()
    sub["_gd"] = pd.to_datetime(sub["game_date"], errors="coerce")
    cutoff = pd.to_datetime(asof) - timedelta(days=recent_days)
    for bid, g in sub.groupby("batter"):
        season = _agg_metrics(g)
        recent = _agg_metrics(g[g["_gd"] >= cutoff])   # trailing 2 weeks
        game_days = sorted(g["game_date"].unique(), reverse=True)
        windows = {"L14d": recent}
        for n in (5, 15, 30):
            days = set(game_days[:n])
            windows[f"L{n}"] = _agg_metrics(g[g["game_date"].isin(days)])
        out[int(bid)] = {
            "season": season,
            "windows": windows,
            "recent": recent,          # headline = last 2 weeks
            "pitch_splits": _pitch_splits(g),                       # season-long by pitch family
            "pitch_splits_recent": _pitch_splits(g[g["_gd"] >= cutoff]),  # last 2 weeks by family
        }
    return out


def _pitcher_metrics(rows: pd.DataFrame) -> dict:
    """HR-vulnerability metrics allowed by a pitcher over an arbitrary row subset."""
    if rows.empty:
        return {}
    bb = rows[rows["launch_speed"].notna()]
    n_bb = len(bb)
    pa = rows["events"].isin(PA_EVENTS).sum()
    pitches = len(rows)
    hr = int((rows["events"] == "home_run").sum())

    def pct(num, den):
        return round(100.0 * num / den, 1) if den else None

    pull_pct, pull_air_pct = _pull_metrics(bb)
    ideal_aa_pct, _ = _ideal_aa(rows)

    # fastball velo (four-seam / sinker) for the fatigue/decline signal
    fb_velo = None
    if "release_speed" in rows and "pitch_type" in rows:
        fb = rows[rows["pitch_type"].isin(["FF", "SI", "FC"])]
        if len(fb):
            fb_velo = round(fb["release_speed"].mean(), 1)

    return {
        "barrel_pct_allowed": pct((bb["launch_speed_angle"] == 6).sum(), n_bb) if "launch_speed_angle" in bb else None,
        "hardhit_pct_allowed": pct((bb["launch_speed"] >= 95).sum(), n_bb),
        "avg_ev_allowed": round(bb["launch_speed"].mean(), 1) if n_bb else None,
        "fb_pct_allowed": pct((bb["bb_type"] == "fly_ball").sum(), n_bb) if "bb_type" in bb else None,
        "pull_air_allowed": pull_air_pct,
        "ideal_aa_allowed": ideal_aa_pct,
        "hr_per_pa": pct(hr, pa),
        "hr_allowed": hr,
        "swstr_pct_allowed": pct(rows["description"].isin(SWING_STRIKE).sum(), pitches),
        "fb_velo": fb_velo,
        "bbe": int(n_bb),
        "pa": int(pa),
    }


def pitcher_profiles(df: pd.DataFrame, pitcher_ids: list[int], asof: str,
                     recent_days: int = 14) -> dict:
    """Per pitcher: season + trailing-2-week HR-vulnerability metrics, incl. velo trend."""
    from datetime import timedelta
    out = {}
    ids = [p for p in pitcher_ids if p]
    sub = df[df["pitcher"].isin(ids)].copy()
    sub["_gd"] = pd.to_datetime(sub["game_date"], errors="coerce")
    cutoff = pd.to_datetime(asof) - timedelta(days=recent_days)
    for pid, g in sub.groupby("pitcher"):
        season = _pitcher_metrics(g)
        recent = _pitcher_metrics(g[g["_gd"] >= cutoff])
        # velocity trend: recent fastball velo vs season
        vt = None
        if recent.get("fb_velo") is not None and season.get("fb_velo") is not None:
            vt = round(recent["fb_velo"] - season["fb_velo"], 1)
        recent["velo_trend"] = vt
        # platoon splits — what the pitcher allows vs RHB vs LHB
        g_recent = g[g["_gd"] >= cutoff]
        splits = {}
        for hand in ("R", "L"):
            splits[hand] = {
                "season": _pitcher_metrics(g[g["stand"] == hand]),
                "recent": _pitcher_metrics(g_recent[g_recent["stand"] == hand]),
            }
        out[int(pid)] = {"season": season, "recent": recent, "splits": splits,
                         "usage": _pitch_usage(g)}
    return out


def bullpen_profiles(df: pd.DataFrame, asof: str, recent_days: int = 14) -> dict:
    """
    Per team: HR-vulnerability of the BULLPEN (all non-starter pitchers), season +
    trailing 2 weeks, with RHB/LHB platoon splits. Starters are identified as the
    pitcher who threw the first pitch of each half-inning's game; everyone else that
    appeared for that team is a reliever.
    """
    from datetime import timedelta
    if df.empty or "inning" not in df.columns:
        return {}
    work = df.copy()
    work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
    cutoff = pd.to_datetime(asof) - timedelta(days=recent_days)

    # starter per (game_pk, half) = first pitcher of inning 1 that half
    inn1 = work[work["inning"] == 1].sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    starter_df = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                  [["game_pk", "inning_topbot", "pitcher"]].rename(columns={"pitcher": "_starter"}))

    # pitching team for each row: Top = home pitches, Bot = away pitches
    work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"), work["home_team"], work["away_team"])
    work = work.merge(starter_df, on=["game_pk", "inning_topbot"], how="left")
    pen = work[work["pitcher"] != work["_starter"]]

    out = {}
    for team, g in pen.groupby("pitch_team"):
        if not isinstance(team, str):
            continue
        g_recent = g[g["_gd"] >= cutoff]
        splits = {}
        for hand in ("R", "L"):
            splits[hand] = {
                "season": _pitcher_metrics(g[g["stand"] == hand]),
                "recent": _pitcher_metrics(g_recent[g_recent["stand"] == hand]),
            }
        out[team] = {
            "season": _pitcher_metrics(g),
            "recent": _pitcher_metrics(g_recent),
            "splits": splits,
            "arms": int(g["pitcher"].nunique()),
        }
    return out


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ").strip()


def career_table(start_season: int, end_season: int) -> dict:
    """
    Career-ish rates from FanGraphs over a season range, keyed by normalized name.
    Defensive: returns {} on any failure so the board still renders.
    """
    try:
        fg = batting_stats(start_season, end_season, qual=0, ind=0)
    except Exception:
        return {}
    if fg is None or fg.empty:
        return {}

    colmap = {
        "Barrel%": "barrel_pct", "HardHit%": "hardhit_pct", "ISO": "iso",
        "EV": "avg_ev", "maxEV": "max_ev", "FB%": "fb_pct",
        "SwStr%": "swstr_pct", "K%": "k_pct", "Pull%": "pull_pct",
    }
    out = {}
    for _, row in fg.iterrows():
        name = _norm_name(row.get("Name", ""))
        if not name:
            continue
        rec = {}
        for src, dst in colmap.items():
            if src in fg.columns and pd.notna(row.get(src)):
                val = float(row[src])
                # FanGraphs gives rate stats as fractions (0.095) for some, % for others.
                if dst in ("iso",):
                    rec[dst] = round(val, 3)
                elif dst in ("avg_ev", "max_ev", "launch_angle"):
                    rec[dst] = round(val, 1)
                else:
                    # normalize fractions like 0.095 -> 9.5
                    rec[dst] = round(val * 100, 1) if val < 1.5 else round(val, 1)
        out[name] = rec
    return out


def bvp_table(df: pd.DataFrame) -> dict:
    """Season batter-vs-pitcher from the slate frame -> {(batter,pitcher): [pa, hr]}.
    PAs are rows where `events` is set (the last pitch of a plate appearance)."""
    if df is None or df.empty or "events" not in df.columns:
        return {}
    pa = df[df["events"].notna()]
    if pa.empty:
        return {}
    grp = pa.groupby(["batter", "pitcher"])["events"]
    agg = grp.agg(pa="size", hr=lambda s: int((s == "home_run").sum()))
    out = {}
    for (b, p), row in agg.iterrows():
        try:
            out[(int(b), int(p))] = [int(row["pa"]), int(row["hr"])]
        except Exception:
            continue
    return out


def bullpen_arms(df: pd.DataFrame, asof: str, recent_days: int = 21, min_pitches: int = 8) -> dict:
    """Active relievers per team in the trailing window -> {team_abbr: [pitcher_id, ...]}.
    Same starter-exclusion logic as bullpen_profiles."""
    from datetime import timedelta
    if df is None or df.empty or "inning" not in df.columns:
        return {}
    work = df.copy()
    work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
    cutoff = pd.to_datetime(asof) - timedelta(days=recent_days)
    inn1 = work[work["inning"] == 1].sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    starter_df = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                  [["game_pk", "inning_topbot", "pitcher"]].rename(columns={"pitcher": "_starter"}))
    work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"), work["home_team"], work["away_team"])
    work = work.merge(starter_df, on=["game_pk", "inning_topbot"], how="left")
    pen = work[(work["pitcher"] != work["_starter"]) & (work["_gd"] >= cutoff)]
    out = {}
    for team, g in pen.groupby("pitch_team"):
        if not isinstance(team, str):
            continue
        counts = g.groupby("pitcher").size()
        out[team] = [int(pid) for pid, n in counts.items() if n >= min_pitches]
    return out


def hr_by_lineup_spot(df: pd.DataFrame) -> dict:
    """{batter_id: {slot: hr_count}} for the season. The batting order slot cycles strictly,
    so the k-th plate appearance by a team is in slot ((k-1) mod 9)+1 — exact even with subs."""
    need = {"events", "at_bat_number", "inning_topbot", "home_team", "away_team", "batter"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return {}
    pa["bteam"] = np.where(pa["inning_topbot"].eq("Top"), pa["away_team"], pa["home_team"])
    pa = pa.sort_values(["game_pk", "bteam", "at_bat_number"])
    pa["tidx"] = pa.groupby(["game_pk", "bteam"]).cumcount()
    pa["slot"] = (pa["tidx"] % 9) + 1
    hr = pa[pa["events"].eq("home_run")]
    out = {}
    for (bid, slot), n in hr.groupby(["batter", "slot"]).size().items():
        out.setdefault(int(bid), {})[int(slot)] = int(n)
    return out


def starter_lengths(df: pd.DataFrame) -> dict:
    """How deep each starter actually goes: {pitcher_id: {"starts": n, "med_len": innings}}.
    Starter = first pitcher for the defense in inning 1; length = deepest inning reached
    that game. Median across starts is robust to one blowup. Also lets the board spot
    openers: listed 'SP' whose starts are 1-2 innings, or who has only relieved."""
    need = {"game_pk", "inning", "inning_topbot", "at_bat_number", "pitcher"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[["game_pk", "inning", "inning_topbot", "at_bat_number", "pitcher"]].copy()
    # defense half: pitcher in "Top" belongs to the HOME defense, "Bot" to the AWAY defense
    d["half"] = d["inning_topbot"].astype(str)
    first = (d[d["inning"] == 1]
             .sort_values("at_bat_number")
             .groupby(["game_pk", "half"], as_index=False)
             .first()[["game_pk", "half", "pitcher"]]
             .rename(columns={"pitcher": "starter"}))
    depth = (d.groupby(["game_pk", "half", "pitcher"], as_index=False)["inning"].max()
             .rename(columns={"inning": "deep"}))
    m = first.merge(depth, left_on=["game_pk", "half", "starter"],
                    right_on=["game_pk", "half", "pitcher"], how="left")
    out = {}
    for pid, grp in m.groupby("starter"):
        lens = grp["deep"].dropna().astype(float)
        if len(lens):
            out[int(pid)] = {"starts": int(len(lens)), "med_len": float(lens.median())}
    return out


def pitcher_appearances(df: pd.DataFrame) -> dict:
    """{pitcher_id: total games appeared} — with starter_lengths, spots the pure reliever
    listed as today's opener (appears often, zero traditional starts)."""
    if df is None or df.empty or "pitcher" not in df.columns or "game_pk" not in df.columns:
        return {}
    g = df.groupby("pitcher")["game_pk"].nunique()
    return {int(k): int(v) for k, v in g.items()}


def hr_last_game(df: pd.DataFrame) -> set:
    """Batter ids who homered in their most recent game — the 'back-to-back' fade signal
    (public loads up on last night's HR hitters; the repeat is priced badly)."""
    need = {"batter", "game_date", "events"}
    if df is None or df.empty or not need.issubset(df.columns):
        return set()
    d = df[["batter", "game_date", "events"]].copy()
    last = d.groupby("batter")["game_date"].max().rename("last_g")
    d = d.merge(last, on="batter")
    recent = d[d["game_date"] == d["last_g"]]
    hr = recent[recent["events"].eq("home_run")]
    return set(int(b) for b in hr["batter"].unique())


def pitcher_batted_profile(df: pd.DataFrame) -> dict:
    """{pitcher_id: {fb_pct, gb_pct, n}} — season batted-ball mix allowed.
    GB = launch angle < 10 deg, FB = >= 25 deg (fly balls + popups). A fly-ball-heavy
    arm puts more balls in HR territory — the target profile."""
    need = {"pitcher", "launch_angle", "launch_speed"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[df["launch_speed"].notna() & df["launch_angle"].notna() & df["events"].notna()]
    if d.empty:
        return {}
    g = d.groupby("pitcher")["launch_angle"]
    n = g.size()
    fb = d[d["launch_angle"] >= 25].groupby("pitcher").size().reindex(n.index, fill_value=0)
    gb = d[d["launch_angle"] < 10].groupby("pitcher").size().reindex(n.index, fill_value=0)
    out = {}
    for pid in n.index:
        if n[pid] >= 30:
            out[int(pid)] = {"fb_pct": round(float(fb[pid]) * 100.0 / float(n[pid]), 1),
                             "gb_pct": round(float(gb[pid]) * 100.0 / float(n[pid]), 1),
                             "n": int(n[pid])}
    return out


LAST_LABEL_DIAG = {}   # written by hitter_labels; the build ships it in board.json so
                       # label problems are diagnosable from the public artifact, not logs

_AB_EVENTS = {"single", "double", "triple", "home_run", "field_out", "strikeout",
              "grounded_into_double_play", "force_out", "double_play", "field_error",
              "fielders_choice", "fielders_choice_out", "strikeout_double_play",
              "triple_play", "other_out"}


def hitter_labels(df: pd.DataFrame, start_date: str | None = None, min_bbe: int = 15) -> dict:
    """One profile label per hitter over the trailing window (same 2-week sample as the
    model), mirroring the PF highlight rules: 'elite' (all 14 thresholds), else 'fb'
    (all 10, FB%-based, priority), else 'ld' (all 10, LD%-based), else no label.

    Approximations, stated plainly: Near HR = warning-track balls — non-HR drives
    carrying 325+ ft with real loft (LA >= 15); Blast = squared-up contact (EV >= 80% of the bat+pitch
    speed ceiling) with a 75+ mph swing, per Savant's public definitions, rated per
    batted ball. Everything else is exact from Statcast fields.
    """
    need = {"batter", "game_date", "events", "launch_speed", "launch_angle",
            "launch_speed_angle", "hc_x", "hc_y", "stand", "hit_distance_sc"}
    global LAST_LABEL_DIAG
    if df is None or df.empty or not need.issubset(df.columns):
        missing = sorted(need - set(df.columns)) if df is not None and not df.empty else ["<empty>"]
        print(f"[labels] skipped — missing columns: {missing}")
        LAST_LABEL_DIAG = {"skip": f"missing columns: {missing}"}
        return {}
    w = df[df["game_date"].astype(str).str[:10] >= start_date] if start_date else df
    pa = w[w["events"].notna()]
    # in-play only: Statcast tracks EV on some FOULS too, which would pollute every
    # rate denominator (HH%, Brl%, LD%...) — an in-play ball always has an event
    bb = w[w["launch_speed"].notna() & w["launch_angle"].notna() & w["events"].notna()].copy()
    for _c in ("launch_speed", "launch_angle", "launch_speed_angle", "hc_x", "hc_y",
               "hit_distance_sc", "bat_speed", "release_speed"):
        if _c in bb.columns:
            bb[_c] = pd.to_numeric(bb[_c], errors="coerce")
    for _c in ("stand", "events"):     # nullable/Arrow string dtypes -> plain objects
        bb[_c] = np.asarray(bb[_c].astype(object).where(bb[_c].notna(), ""), dtype=object)
    if bb.empty:
        LAST_LABEL_DIAG = {"skip": f"no batted balls in window >= {start_date} "
                                   f"(df rows {len(df)}, window rows {len(w)})"}
        return {}
    try:
        return _labels_core(bb, pa, min_bbe)
    except Exception as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        ours = [f for f in tb if "statcast_data" in (f.filename or "")]
        last = (ours[-1] if ours else (tb[-1] if tb else None))
        LAST_LABEL_DIAG = {"skip": f"exception {type(e).__name__}: {e}"
                                   + (f" @ line {last.lineno}: {last.line}" if last else "")}
        globals()["LAST_LABEL_DIAG"] = LAST_LABEL_DIAG
        print(f"[labels] EXCEPTION: {LAST_LABEL_DIAG['skip']}")
        return {}


def _labels_core(bb, pa, min_bbe):
    global LAST_LABEL_DIAG
    spray = np.degrees(np.arctan2(bb["hc_x"].to_numpy(float) - 125.42,
                                  198.27 - bb["hc_y"].to_numpy(float)))
    bb["spray"] = spray
    stand_r = (bb["stand"].to_numpy() == "R")
    bb["pull"] = np.where(stand_r, spray <= -15.0, spray >= 15.0)
    bb["brl"] = bb["launch_speed_angle"].eq(6)
    la = bb["launch_angle"]
    bb["air"] = la >= 10; bb["fb"] = la >= 25; bb["ld"] = (la >= 10) & (la < 25); bb["gb"] = la < 10
    bb["hh"] = bb["launch_speed"] >= 95
    dist = bb["hit_distance_sc"]
    bb["d300"] = dist >= 300; bb["d350"] = dist >= 350
    bb["near"] = (dist >= 325).to_numpy() & (la >= 15).to_numpy() & (bb["events"].to_numpy() != "home_run")
    have_bt = "bat_speed" in bb.columns and "release_speed" in bb.columns \
        and bb["bat_speed"].notna().mean() > 0.3
    if have_bt:
        # official Savant definition: squared-up% x 100 + bat speed >= 164 (sliding
        # scale — a 90% squared 75mph swing and an 80% squared 85mph swing both blast)
        ceil = 1.23 * bb["bat_speed"] + 0.198 * bb["release_speed"]
        squp = (bb["launch_speed"] / ceil).clip(upper=1.0)
        bb["blast"] = ((squp * 100.0 + bb["bat_speed"]) >= 164.0).fillna(False)
        bb["tracked"] = bb["bat_speed"].notna()
    else:
        bb["blast"] = False
        bb["tracked"] = False
        print("[labels] bat tracking unavailable — Blast% criterion waived this build")
    out = {}
    iso_num = {"single": 0, "double": 1, "triple": 2, "home_run": 3}
    n_pool = n_base = 0
    gate_pass = {k: 0 for k in ("near2", "iso20", "d300", "d350", "hh30", "blast5",
                                "air50", "gb45", "brl10", "fb30", "ld30")}
    near_misses = []
    for bid, g in bb.groupby("batter"):
        n = len(g)
        if n < min_bbe:
            continue
        n_pool += 1
        gpa = pa[pa["batter"] == bid]
        ab = int(gpa["events"].isin(_AB_EVENTS).sum())
        iso = (sum(iso_num.get(e, 0) for e in gpa["events"]) / ab) if ab else 0.0
        hr = int((g["events"].to_numpy() == "home_run").sum())
        pct = lambda col: 100.0 * float(g[col].sum()) / n
        n_trk = int(g["tracked"].sum())
        # Blast% over TRACKED batted balls (untracked swings shouldn't deflate the
        # rate); criterion waived when tracking is unavailable for this build/hitter
        blast = (100.0 * float(g["blast"].sum()) / n_trk) if n_trk >= 8 else None
        blast_ok = lambda thr: (blast is None) or (blast >= thr)
        m = {"hr": hr, "near": int(g["near"].sum()), "iso": iso,
             "ev": float(g["launch_speed"].mean()),
             "d300": int(g["d300"].sum()), "d350": int(g["d350"].sum()),
             "brl": pct("brl"), "pullbrl": 100.0 * float((g["brl"] & g["pull"]).sum()) / n,
             "hh": pct("hh"), "air": pct("air"),
             "fb": pct("fb"), "ld": pct("ld"), "gb": pct("gb"), "pull": pct("pull")}
        gates_now = (("near2", m["near"] >= 2), ("iso20", m["iso"] >= 0.2),
                     ("d300", m["d300"] >= 2), ("d350", m["d350"] >= 2),
                     ("hh30", m["hh"] >= 30), ("blast5", blast_ok(5)),
                     ("air50", m["air"] >= 50), ("gb45", m["gb"] < 45),
                     ("brl10", m["brl"] >= 10), ("fb30", m["fb"] >= 30),
                     ("ld30", m["ld"] >= 30))
        fails = [k for k, ok in gates_now if not ok]
        near_misses.append((len([k for k in fails if k not in ("fb30", "ld30")])
                            + (0 if ("fb30" not in fails or "ld30" not in fails) else 1),
                            int(bid), fails))
        for k, ok in gates_now:
            if ok:
                gate_pass[k] += 1
        base = (m["near"] >= 2 and m["iso"] >= 0.2 and m["d300"] >= 2 and m["d350"] >= 2
                and m["hh"] >= 30 and blast_ok(5) and m["air"] >= 50 and m["gb"] < 45)
        if base:
            n_base += 1
        if (m["hr"] >= 1 and m["near"] >= 2 and m["iso"] >= 0.2 and m["ev"] >= 89
                and m["d300"] >= 2 and m["d350"] >= 2 and m["brl"] >= 18 and m["pullbrl"] >= 6
                and m["hh"] >= 35 and blast_ok(10) and m["air"] >= 50 and m["fb"] >= 30
                and m["gb"] < 45 and m["pull"] >= 25):
            out[int(bid)] = "elite"
        elif base and m["brl"] >= 10 and m["fb"] >= 30:
            out[int(bid)] = "fb"
        elif base and m["brl"] >= 10 and m["ld"] >= 30:
            out[int(bid)] = "ld"
    near_misses.sort()
    LAST_LABEL_DIAG = {
        "pool": n_pool, "base": n_base,
        "labels": {k: sum(1 for v in out.values() if v == k) for k in ("elite", "fb", "ld")},
        "gates_pct": {k: (100 * v // n_pool if n_pool else 0) for k, v in gate_pass.items()},
        "near_misses": [{"id": bid, "failed": fl} for _, bid, fl in near_misses[:3]],
    }
    if n_pool:
        print("[labels] gates: " + " ".join(f"{k}={100*v//n_pool}%" for k, v in gate_pass.items()))
    print(f"[labels] funnel: {n_pool} hitters with {min_bbe}+ BBE -> {n_base} pass base -> "
          f"{sum(1 for v in out.values() if v=='elite')}/{sum(1 for v in out.values() if v=='fb')}/"
          f"{sum(1 for v in out.values() if v=='ld')} elite/fb/ld")
    return out
