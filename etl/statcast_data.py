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
        "launch_speed", "launch_angle", "launch_speed_angle", "bb_type",
        "stand", "p_throws", "type", "hc_x", "hc_y",
        "attack_angle", "bat_speed", "release_speed", "pitch_type",
        "inning", "inning_topbot", "at_bat_number", "pitch_number",
        "home_team", "away_team",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


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

    out = {
        "barrel_pct": pct((bb["launch_speed_angle"] == 6).sum(), n_bb) if "launch_speed_angle" in bb else None,
        "hardhit_pct": pct((bb["launch_speed"] >= 95).sum(), n_bb),
        "avg_ev": round(bb["launch_speed"].mean(), 1) if n_bb else None,
        "launch_angle": round(bb["launch_angle"].mean(), 1) if n_bb else None,
        "fb_pct": pct((bb["bb_type"] == "fly_ball").sum(), n_bb) if "bb_type" in bb else None,
        "pull_pct": pull_pct,
        "pull_air_pct": pull_air_pct,
        "ideal_aa_pct": ideal_aa_pct,
        "bat_speed": bat_speed,
        "iso": iso,
        "slg": slg,
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
        out[int(pid)] = {"season": season, "recent": recent, "splits": splits}
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
