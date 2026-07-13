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
        "post_home_score", "post_away_score",
        "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
        "woba_value", "woba_denom",
    ]
    keep = [c for c in keep if c in df.columns]
    return normalize_frame(df[keep].copy())


_NUMERIC_COLS = ("launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc",
                 "hc_x", "hc_y", "bat_speed", "release_speed", "attack_angle",
                 "inning", "at_bat_number", "pitch_number", "batter", "pitcher", "game_pk",
                 "woba_value", "estimated_woba_using_speedangle",
                 "estimated_ba_using_speedangle", "release_spin_rate",
                 "post_home_score", "post_away_score")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Single choke point that immunizes every downstream function against pandas
    nullable/Arrow dtypes (the runner-vs-dev skew that broke labels and the grader).
    Values are untouched: numerics become plain float64 (NaN preserved), text becomes
    plain objects with np.nan for missing — so .notna(), .eq(), == and boolean masks
    all behave classically. Provably semantics-preserving; see the equivalence test."""
    if df is None or df.empty:
        return df
    for c in df.columns:
        if c in _NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        else:
            col = df[c]
            if not (col.dtype == object):
                df[c] = np.asarray(col.astype(object).where(col.notna(), np.nan), dtype=object)
    return df


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
    Actual HRs (and PA) a pitcher has allowed to RHB vs LHB, two windows:
      two_yr  — current season + previous season (calendar convention, matching
                PropFinder-style reference tools; NOT trailing-730-days)
      this_yr — current season only

    REGULAR SEASON ONLY: statcast_pitcher returns spring training ('S'),
    exhibition ('E'), and postseason ('F','D','L','W') rows too. Those inflate
    HR/PA counts vs any sportsbook reference, so we filter to game_type == 'R'.

    Uses a per-pitcher Statcast pull (cached upstream so this isn't run every
    hourly build). Returns None on any failure (degrades cleanly).
    """
    from datetime import datetime
    try:
        from pybaseball import statcast_pitcher
        end = datetime.strptime(end_date, "%Y-%m-%d")
        # two-season window: Jan 1 of PREVIOUS year → today (regular-season filter
        # below trims spring noise; Jan 1 start guarantees we catch opening day)
        start2 = f"{end.year - 1}-01-01"
        df = statcast_pitcher(start2, end_date, int(pid))
    except Exception:
        return None
    if df is None or df.empty or "stand" not in df.columns or "events" not in df.columns:
        return None
    # regular season only — this is the fix that aligns totals with reference tools
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
        if df.empty:
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

    # Hits-props signals (parallel track to the HR heat model — used only for the
    # Other Props tab, never fed back into heat_score):
    hits = int(singles + doubles + triples + hr)
    ba = round(hits / ab, 3) if ab > 0 else None
    bb_pct = pct(walks, pa)                                    # walk rate (plate discipline)
    xba = None                                                  # expected BA on contact
    if n_bb and "estimated_ba_using_speedangle" in bb:
        xba_ser = bb["estimated_ba_using_speedangle"].dropna()
        if len(xba_ser) >= 8:
            xba = round(float(xba_ser.mean()), 3)
    # line-drive % — highest BABIP bucket by launch angle
    ld_pct_hit = pct(((bb["launch_angle"] >= 10) & (bb["launch_angle"] < 25)).sum(), n_bb) if n_bb else None
    # contact % = 100 - swinging strike %  (pitches where the batter didn't whiff)
    contact_pct = None
    if pitches:
        swstr = rows["description"].isin(SWING_STRIKE).sum()
        contact_pct = round(100.0 * (1 - swstr / pitches), 1)

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
        # Hits-props signals (used ONLY by the Other Props tab, not the HR heat model):
        "ba": ba,
        "xba": xba,
        "bb_pct": bb_pct,
        "ld_pct_hit": ld_pct_hit,
        "contact_pct": contact_pct,
        "hits": hits,
        "ab": int(ab) if ab is not None else 0,
        "hr": int(hr),
        "pa": int(pa),
        "bb_count": int(n_bb),
    }
    return out


def game_day_cutoff(df: pd.DataFrame, asof: str, n_days: int = 14) -> "pd.Timestamp":
    """Resolve "the last N GAME-days" to a real cutoff date.

    Why this exists: the window used to be `asof - timedelta(days=14)` — 14 CALENDAR
    days. That was identical to 14 game-days only because MLB plays nearly every day.
    The All-Star break puts a ~4-day hole in the schedule, so a 14-calendar-day window
    silently holds ~10 game-days: every hitter's batted-ball count drops ~30%, the
    confidence discount pulls him toward the median, and every heat score compresses
    for two weeks. Same failure (smaller) around any multi-day league-wide gap.

    Counting actual dates on which games were played fixes it. Falls back to the
    calendar cutoff if the frame is too short to supply N game-days.
    """
    from datetime import timedelta
    asof_ts = pd.to_datetime(asof)
    if df is None or df.empty or "game_date" not in df.columns:
        return asof_ts - timedelta(days=n_days)
    dates = pd.to_datetime(pd.Series(df["game_date"].unique()), errors="coerce").dropna()
    dates = sorted(d for d in dates if d <= asof_ts)
    if len(dates) < n_days:
        return asof_ts - timedelta(days=n_days)      # not enough history; behave as before
    return dates[-n_days]


def batter_profiles(df: pd.DataFrame, batter_ids: list[int], asof: str,
                    recent_days: int = 14) -> dict:
    """
    For each batter: the headline RECENT line is the trailing `recent_days`
    GAME-days (dates on which MLB actually played) — not calendar days. See
    game_day_cutoff() for why that distinction is load-bearing across the
    All-Star break. Also keep L5/L15/L30 game windows + season for context.
    """
    out = {}
    sub = df[df["batter"].isin(batter_ids)].copy()
    sub["_gd"] = pd.to_datetime(sub["game_date"], errors="coerce")
    cutoff = game_day_cutoff(df, asof, recent_days)
    for bid, g in sub.groupby("batter"):
        season = _agg_metrics(g)
        recent = _agg_metrics(g[g["_gd"] >= cutoff])   # trailing 2 weeks of PLAY
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
    ev = rows["events"]
    ks_allowed = int(ev.isin(["strikeout", "strikeout_double_play"]).sum())
    walks_allowed = int(ev.isin(["walk", "intent_walk"]).sum())
    hbp_allowed = int((ev == "hit_by_pitch").sum())
    sacs_allowed = int(ev.isin(["sac_fly", "sac_bunt", "sac_fly_double_play"]).sum())
    ci_allowed = int((ev == "catcher_interf").sum())
    ab_allowed = pa - walks_allowed - hbp_allowed - sacs_allowed - ci_allowed
    hits_allowed = int(ev.isin(["single", "double", "triple", "home_run"]).sum())

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

    # xBA allowed (opponent expected BA on contact) — needed for hit props
    xba_allowed = None
    if n_bb and "estimated_ba_using_speedangle" in bb:
        xba_ser = bb["estimated_ba_using_speedangle"].dropna()
        if len(xba_ser) >= 12:
            xba_allowed = round(float(xba_ser.mean()), 3)

    # xwOBA on contact allowed — the core run-prevention input for the run model
    xwobacon_allowed = None
    if n_bb and "estimated_woba_using_speedangle" in bb:
        xw = bb["estimated_woba_using_speedangle"].dropna()
        if len(xw) >= 12:
            xwobacon_allowed = round(float(xw.mean()), 3)

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
        # Hit-props signals (used by props tab only):
        "k_pct_allowed": pct(ks_allowed, pa),
        "bb_pct_allowed": pct(walks_allowed, pa),
        "ba_allowed": round(hits_allowed / ab_allowed, 3) if ab_allowed > 0 else None,
        "xba_allowed": xba_allowed,
        "xwobacon_allowed": xwobacon_allowed,
        "ld_pct_allowed": pct(((bb["launch_angle"] >= 10) & (bb["launch_angle"] < 25)).sum(), n_bb) if n_bb else None,
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
    cutoff = game_day_cutoff(df, asof, recent_days)   # GAME-days, not calendar (break-proof)
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


def bullpen_profiles(df: pd.DataFrame, asof: str, recent_days: int = 14,
                     roles: dict | None = None) -> dict:
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
    cutoff = game_day_cutoff(df, asof, recent_days)   # GAME-days, not calendar (break-proof)

    # starter per (game_pk, half) = first pitcher of inning 1 that half
    inn1 = work[work["inning"] == 1].sort_values(["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    starter_df = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                  [["game_pk", "inning_topbot", "pitcher"]].rename(columns={"pitcher": "_starter"}))

    # pitching team for each row: Top = home pitches, Bot = away pitches
    work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"), work["home_team"], work["away_team"])
    work = work.merge(starter_df, on=["game_pk", "inning_topbot"], how="left")
    pen = work[work["pitcher"] != work["_starter"]]
    # Exclude TRUE starters. A bulk starter following an opener throws "relief" innings by
    # the per-game rule, and folding his quality into the pen score flatters teams that
    # open — the hitter will not face him in the 7th-9th.
    if roles is None:
        roles = pitcher_roles(df)
    if roles:
        pen = pen[~pen["pitcher"].map(
            lambda p: (roles.get(int(p)) or {}).get("role") == "SP" if p == p else False)]

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


def _tag_appearance_role(df: pd.DataFrame) -> pd.DataFrame:
    """Add `_is_relief` to every row: was this pitch thrown in a RELIEF appearance?

    The starter of each game/half is whoever threw the first pitch of the 1st inning.
    Every other pitcher in that game/half is relieving *in that game*. This is a
    per-APPEARANCE fact, not a per-pitcher fact — which is exactly the distinction
    the old code lost.
    """
    work = df.copy()
    inn1 = work[work["inning"] == 1].sort_values(
        ["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    starters = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                [["game_pk", "inning_topbot", "pitcher"]]
                .rename(columns={"pitcher": "_starter"}))
    work = work.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    work["_is_relief"] = work["pitcher"] != work["_starter"]
    return work


def pitcher_roles(df: pd.DataFrame) -> dict:
    """{pid: {"starts": n, "relief": n, "role": "SP"|"RP"|"SWING"}}

    Why this exists: role must be decided from a pitcher's SEASON of appearances, not
    from a single game. The old logic ("not this game's starter -> reliever") put real
    starters into the bullpen pool two ways:
      * a starter making one spot relief appearance
      * OPENERS — when a team opens, the bulk starter enters in the 2nd inning and is
        classified as a reliever for that game

    Once a starter was in the pen ID list, his HRs allowed AS A STARTER got counted as
    "HR vs bullpen." This function is what stops that.
    """
    need = {"pitcher", "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    w = _tag_appearance_role(df)
    # one row per (pitcher, game) — role is constant within an appearance
    app = (w.groupby(["pitcher", "game_pk"], as_index=False)["_is_relief"]
           .first())
    out = {}
    for pid, g in app.groupby("pitcher"):
        if pid != pid:
            continue
        relief = int(g["_is_relief"].sum())
        total = int(len(g))
        starts = total - relief
        share = relief / total if total else 0.0
        # A guy with 18 starts and 1 relief outing is a STARTER. A swingman who mostly
        # comes out of the pen is a reliever. Threshold is deliberately strict on the
        # starter side — false "reliever" labels are what caused the bug.
        if starts >= 3 and share < 0.75:
            role = "SP"
        elif share >= 0.75:
            role = "RP"
        else:
            role = "SWING"
        out[int(pid)] = {"starts": starts, "relief": relief,
                         "apps": total, "relief_share": round(share, 3), "role": role}
    return out


def bvp_table(df: pd.DataFrame, relief_only: bool = False) -> dict:
    """Season batter-vs-pitcher from the slate frame -> {(batter,pitcher): [pa, hr]}.
    PAs are rows where `events` is set (the last pitch of a plate appearance).

    relief_only=True restricts to PAs that happened while the pitcher was RELIEVING.
    That's what the "HR vs PEN" badge actually means — a homer off the bullpen, not a
    homer off a guy who happens to appear in the pen list. Without this filter, a HR
    hit off a starter in his start counts as a bullpen HR the moment that starter shows
    up in the pen pool for any reason.
    """
    if df is None or df.empty or "events" not in df.columns:
        return {}
    src = df
    if relief_only:
        need = {"game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"}
        if not need.issubset(df.columns):
            return {}
        src = _tag_appearance_role(df)
        src = src[src["_is_relief"]]
    pa = src[src["events"].notna()]
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


def bullpen_arms(df: pd.DataFrame, asof: str, recent_days: int = 21, min_pitches: int = 8,
                 roles: dict | None = None) -> dict:
    """Active relievers per team in the trailing window -> {team_abbr: [pitcher_id, ...]}.

    Filters to TRUE relievers using season-long role (pitcher_roles), not per-game role.
    A starter who made one spot relief appearance — or a bulk starter who followed an
    opener — is NOT a bullpen arm, and must not be, because this ID list is crossed with
    a batter-vs-pitcher table downstream to produce the "HR vs PEN" badge.
    """
    if df is None or df.empty or "inning" not in df.columns:
        return {}
    if roles is None:
        roles = pitcher_roles(df)
    work = df.copy()
    work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
    cutoff = game_day_cutoff(df, asof, recent_days)   # GAME-days, not calendar (break-proof)
    work = _tag_appearance_role(work)
    work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"),
                                  work["home_team"], work["away_team"])
    pen = work[work["_is_relief"] & (work["_gd"] >= cutoff)]
    out = {}
    for team, g in pen.groupby("pitch_team"):
        if not isinstance(team, str):
            continue
        counts = g.groupby("pitcher").size()
        arms = []
        for pid, n in counts.items():
            if n < min_pitches:
                continue
            r = (roles.get(int(pid)) or {}).get("role")
            if r == "SP":                       # real starter — not a bullpen arm
                continue
            arms.append(int(pid))
        out[team] = arms
    return out


def bullpen_availability(df: pd.DataFrame, asof: str, roles: dict | None = None) -> dict:
    """Per team: which relievers are likely UNAVAILABLE tonight, and how gassed the
    pen is overall.

    The problem this solves: bullpen_profiles() averages every reliever on the roster.
    But you don't face the roster — you face whoever isn't burnt. A pen that threw four
    innings last night has its closer and setup men down, so the arms you actually see
    are middle relief and mop-up guys, who are meaningfully more HR-prone. The model
    was showing the same pen number either way.

    Availability rules (standard MLB usage patterns, not fitted):
      * threw on BOTH of the last two days      -> out
      * threw 35+ pitches yesterday             -> out
      * threw 3 of the last 4 days              -> out
      * threw 25+ pitches on each of last 2 days-> out

    Returns {team: {"unavailable": [ids], "available": [ids], "pen_pitches_l1": n,
                    "pen_pitches_l2": n, "fatigue": 0-100, "label": str}}
    Degrades to empty on any failure — callers must treat it as optional.
    """
    need = {"game_date", "pitcher", "inning", "inning_topbot",
            "home_team", "away_team", "at_bat_number", "pitch_number"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    work = df.copy()
    work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
    asof_ts = pd.to_datetime(asof)
    # only the last 4 days of play matter for availability
    played = sorted({d for d in work["_gd"].dropna().unique() if d < asof_ts})
    if not played:
        return {}
    last4 = played[-4:]
    day_rank = {d: i for i, d in enumerate(reversed(last4))}   # 0 = most recent

    inn1 = work[work["inning"] == 1].sort_values(
        ["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
    starters = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                [["game_pk", "inning_topbot", "pitcher"]].rename(columns={"pitcher": "_starter"}))
    work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"),
                                  work["home_team"], work["away_team"])
    work = work.merge(starters, on=["game_pk", "inning_topbot"], how="left")
    pen = work[(work["pitcher"] != work["_starter"]) & (work["_gd"].isin(last4))]
    if pen.empty:
        return {}
    # true relievers only — a starter's spot relief outing is bullpen USAGE but he is
    # not a bullpen ARM, and counting him distorts both the roster count and fatigue
    if roles is None:
        roles = pitcher_roles(df)
    if roles:
        pen = pen[~pen["pitcher"].map(
            lambda p: (roles.get(int(p)) or {}).get("role") == "SP"
            if p == p else False)]
        if pen.empty:
            return {}

    out = {}
    for team, g in pen.groupby("pitch_team"):
        if not isinstance(team, str):
            continue
        # pitches thrown per reliever per day
        per = g.groupby(["pitcher", "_gd"]).size().reset_index(name="pitches")
        per["rank"] = per["_gd"].map(day_rank)
        unavailable, available = [], []
        for pid, pg in per.groupby("pitcher"):
            ranks = set(pg["rank"].tolist())
            by_rank = dict(zip(pg["rank"], pg["pitches"]))
            y = by_rank.get(0, 0)              # yesterday
            d2 = by_rank.get(1, 0)             # two days ago
            back2back = 0 in ranks and 1 in ranks
            three_of_four = len(ranks & {0, 1, 2, 3}) >= 3
            out_flag = (
                back2back
                or y >= 35
                or three_of_four
                or (y >= 25 and d2 >= 25)
            )
            (unavailable if out_flag else available).append(int(pid))
        l1 = int(per[per["rank"] == 0]["pitches"].sum())
        l2 = int(per[per["rank"].isin([0, 1])]["pitches"].sum())
        total_arms = len(unavailable) + len(available)
        # fatigue: share of the pen that's down, weighted by recent workload
        share_out = (len(unavailable) / total_arms) if total_arms else 0.0
        load = min(1.0, l2 / 120.0)            # 120 pen pitches over 2 days = heavy
        fatigue = round(100 * (0.65 * share_out + 0.35 * load), 1)
        label = ("GASSED" if fatigue >= 55 else
                 "WORN" if fatigue >= 35 else
                 "RESTED" if fatigue <= 15 else "NORMAL")
        out[team] = {
            "unavailable": unavailable,
            "available": available,
            "n_out": len(unavailable),
            "n_arms": total_arms,
            "pen_pitches_l1": l1,
            "pen_pitches_l2": l2,
            "fatigue": fatigue,
            "label": label,
        }
    return out


def bullpen_profiles_available(df: pd.DataFrame, asof: str, avail: dict,
                               recent_days: int = 14, roles: dict | None = None) -> dict:
    """Same as bullpen_profiles(), but each team's pen is rebuilt from ONLY the arms
    likely available tonight (per bullpen_availability). This is the number that
    actually describes what a hitter will face in the 7th-9th — the full-roster
    average silently includes a closer who is unavailable.

    Returns {} on failure; the caller keeps the full-pen number as the fallback."""
    if not avail or df is None or df.empty:
        return {}
    try:
        work = df.copy()
        work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
        cutoff = game_day_cutoff(df, asof, recent_days)
        inn1 = work[work["inning"] == 1].sort_values(
            ["game_pk", "inning_topbot", "at_bat_number", "pitch_number"])
        starters = (inn1.groupby(["game_pk", "inning_topbot"], as_index=False).first()
                    [["game_pk", "inning_topbot", "pitcher"]].rename(columns={"pitcher": "_starter"}))
        work["pitch_team"] = np.where(work["inning_topbot"].eq("Top"),
                                      work["home_team"], work["away_team"])
        work = work.merge(starters, on=["game_pk", "inning_topbot"], how="left")
        pen = work[work["pitcher"] != work["_starter"]]
        if roles is None:
            roles = pitcher_roles(df)
        if roles:
            pen = pen[~pen["pitcher"].map(
                lambda p: (roles.get(int(p)) or {}).get("role") == "SP" if p == p else False)]
        out = {}
        for team, info in avail.items():
            ok = set(info.get("available") or [])
            if len(ok) < 3:                     # too few arms to be meaningful
                continue
            g = pen[(pen["pitch_team"] == team) & (pen["pitcher"].isin(ok))]
            if g.empty:
                continue
            g_recent = g[g["_gd"] >= cutoff]
            entry = {"season": _pitcher_metrics(g), "recent": _pitcher_metrics(g_recent)}
            for hand in ("R", "L"):
                entry[f"vs_{hand}"] = {
                    "season": _pitcher_metrics(g[g["stand"] == hand]),
                    "recent": _pitcher_metrics(g_recent[g_recent["stand"] == hand]),
                }
            entry["n_arms"] = len(ok)
            out[team] = entry
        return out
    except Exception:
        return {}


def team_changes(df: pd.DataFrame, asof: str, lookback_days: int = 45) -> dict:
    """Detect players who changed teams (trade / waiver / callup to a new org).

    Why it matters: a traded hitter's trailing profile is PARK-CONTAMINATED. A guy
    who spent three months in Coors and got dealt to Miami carries inflated power
    numbers into a park that suppresses them. The model reads his hot 14-day line and
    has no idea half of it happened at altitude. Same in reverse for pitchers.

    Detection: the batting team on each of a player's PAs. If the most recent team
    differs from the team he played for earlier in the window, he moved.

    Returns {player_id: {"from": "COL", "to": "MIA", "games_with_new": n,
                         "first_game_new": "2026-08-02"}}
    """
    need = {"batter", "game_date", "inning_topbot", "home_team", "away_team"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    from datetime import timedelta
    work = df.copy()
    work["_gd"] = pd.to_datetime(work["game_date"], errors="coerce")
    cut = pd.to_datetime(asof) - timedelta(days=lookback_days)
    work = work[(work["_gd"] >= cut) & work["batter"].notna()]
    if work.empty:
        return {}
    work["bteam"] = np.where(work["inning_topbot"].eq("Top"),
                             work["away_team"], work["home_team"])
    out = {}
    for bid, g in work.groupby("batter"):
        by_day = (g.groupby("_gd")["bteam"]
                  .agg(lambda s: s.value_counts().index[0])
                  .sort_index())
        teams = list(by_day.values)
        if len(set(teams)) < 2:
            continue
        newest = teams[-1]
        # walk back to the first day he appeared for the new team, contiguously
        i = len(teams) - 1
        while i >= 0 and teams[i] == newest:
            i -= 1
        if i < 0:
            continue                                   # never played for anyone else
        old = teams[i]
        if old == newest:
            continue
        first_new = by_day.index[i + 1]
        out[int(bid)] = {
            "from": str(old),
            "to": str(newest),
            "games_with_new": int(len(teams) - 1 - i),
            "first_game_new": str(first_new.date()),
        }
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
    """{pitcher_id: {fb_pct, ld_pct, gb_pct, n}} — season batted-ball mix allowed.
    GB = launch angle < 10 deg, LD = 10-24 deg, FB = >= 25 deg. The three sum to
    ~100 (rounding). A fly-ball-heavy arm puts more balls in HR territory — the
    target profile. A ground-ball-heavy arm suppresses HRs."""
    need = {"pitcher", "launch_angle", "launch_speed"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    d = df[df["launch_speed"].notna() & df["launch_angle"].notna() & df["events"].notna()]
    if d.empty:
        return {}
    g = d.groupby("pitcher")["launch_angle"]
    n = g.size()
    la = d["launch_angle"]
    fb = d[la >= 25].groupby("pitcher").size().reindex(n.index, fill_value=0)
    ld = d[(la >= 10) & (la < 25)].groupby("pitcher").size().reindex(n.index, fill_value=0)
    gb = d[la < 10].groupby("pitcher").size().reindex(n.index, fill_value=0)
    out = {}
    for pid in n.index:
        if n[pid] >= 30:
            out[int(pid)] = {
                "fb_pct": round(float(fb[pid]) * 100.0 / float(n[pid]), 1),
                "ld_pct": round(float(ld[pid]) * 100.0 / float(n[pid]), 1),
                "gb_pct": round(float(gb[pid]) * 100.0 / float(n[pid]), 1),
                "n": int(n[pid]),
            }
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


def recent_drives(df: pd.DataFrame, start_date: str, min_dist: int = 240, cap: int = 18) -> dict:
    """Recent air balls per batter for the spray chart + robbed scan:
    {batter: [{ev, la, spray, dist, hr, date}]} — in-play, la >= 10, tracked distance,
    most recent `cap` per batter. Distances are the REAL observed carries."""
    need = {"batter", "game_date", "events", "launch_speed", "launch_angle",
            "hc_x", "hc_y", "hit_distance_sc"}
    if df is None or df.empty or not need.issubset(df.columns):
        return {}
    w = df[df["game_date"].astype(str).str[:10] >= start_date]
    d = w[w["launch_speed"].notna() & w["launch_angle"].notna() & w["events"].notna()
          & w["hc_x"].notna() & w["hc_y"].notna() & w["hit_distance_sc"].notna()].copy()
    d = d[(d["launch_angle"] >= 10) & (d["hit_distance_sc"] >= min_dist)]
    if d.empty:
        return {}
    d["spray"] = np.degrees(np.arctan2(d["hc_x"].to_numpy(float) - 125.42,
                                       198.27 - d["hc_y"].to_numpy(float)))
    d = d.sort_values("game_date")
    out = {}
    for bid, g in d.groupby("batter"):
        g = g.tail(cap)
        out[int(bid)] = [{
            "ev": round(float(r.launch_speed), 1), "la": round(float(r.launch_angle), 1),
            "spray": round(float(r.spray), 1), "dist": int(r.hit_distance_sc),
            "hr": (r.events == "home_run"), "date": str(r.game_date)[:10],
        } for r in g.itertuples()]
    return out


# Franchise HR leader benchmarks (top ~5 all-time per team). These are static
# reference points to flag when a hitter is within reach of a team-history number.
# Names + totals per team (approximate, curated). If a hitter passes one, it's a
# real narrative moment worth watching.
TEAM_HR_LEADERS = {
    "ARI": [("Luis Gonzalez", 224), ("Paul Goldschmidt", 209), ("Steve Finley", 153)],
    "ATL": [("Hank Aaron", 733), ("Eddie Mathews", 493), ("Chipper Jones", 468), ("Dale Murphy", 371)],
    "BAL": [("Cal Ripken Jr.", 431), ("Eddie Murray", 343), ("Boog Powell", 303), ("Brooks Robinson", 268)],
    "BOS": [("Ted Williams", 521), ("David Ortiz", 483), ("Carl Yastrzemski", 452), ("Jim Rice", 382)],
    "CHC": [("Sammy Sosa", 545), ("Ernie Banks", 512), ("Billy Williams", 392), ("Ron Santo", 337)],
    "CWS": [("Frank Thomas", 448), ("Paul Konerko", 432), ("Harold Baines", 221), ("Carlton Fisk", 214)],
    "CHW": [("Frank Thomas", 448), ("Paul Konerko", 432), ("Harold Baines", 221)],
    "CIN": [("Johnny Bench", 389), ("Frank Robinson", 324), ("Adam Dunn", 270), ("Tony Perez", 287)],
    "CLE": [("Jim Thome", 337), ("Albert Belle", 242), ("Manny Ramirez", 236), ("Earl Averill", 226)],
    "COL": [("Todd Helton", 369), ("Larry Walker", 258), ("Nolan Arenado", 235), ("Charlie Blackmon", 227)],
    "DET": [("Al Kaline", 399), ("Norm Cash", 373), ("Hank Greenberg", 306), ("Miguel Cabrera", 373)],
    "HOU": [("Jeff Bagwell", 449), ("Lance Berkman", 326), ("Craig Biggio", 291), ("Jose Altuve", 232)],
    "KC": [("George Brett", 317), ("Mike Sweeney", 197), ("Amos Otis", 193), ("Salvador Perez", 250)],
    "KCR": [("George Brett", 317), ("Mike Sweeney", 197), ("Salvador Perez", 250)],
    "LAA": [("Mike Trout", 378), ("Tim Salmon", 299), ("Garret Anderson", 272), ("Brian Downing", 222)],
    "LAD": [("Duke Snider", 389), ("Eric Karros", 270), ("Ron Cey", 228), ("Steve Garvey", 211)],
    "MIA": [("Giancarlo Stanton", 267), ("Dan Uggla", 154), ("Miguel Cabrera", 138)],
    "MIL": [("Ryan Braun", 352), ("Robin Yount", 251), ("Prince Fielder", 230), ("Cecil Cooper", 201)],
    "MIN": [("Harmon Killebrew", 559), ("Kirby Puckett", 207), ("Kent Hrbek", 293), ("Joe Mauer", 143)],
    "NYM": [("Darryl Strawberry", 252), ("David Wright", 242), ("Mike Piazza", 220), ("Howard Johnson", 192)],
    "NYY": [("Babe Ruth", 659), ("Mickey Mantle", 536), ("Lou Gehrig", 493), ("Joe DiMaggio", 361), ("Yogi Berra", 358), ("Aaron Judge", 315)],
    "OAK": [("Mark McGwire", 363), ("Reggie Jackson", 269), ("Jose Canseco", 254), ("Sal Bando", 192)],
    "ATH": [("Mark McGwire", 363), ("Reggie Jackson", 269), ("Jose Canseco", 254)],
    "PHI": [("Mike Schmidt", 548), ("Ryan Howard", 382), ("Del Ennis", 259), ("Chase Utley", 233)],
    "PIT": [("Willie Stargell", 475), ("Ralph Kiner", 301), ("Barry Bonds", 176), ("Andrew McCutchen", 203)],
    "SD": [("Nate Colbert", 163), ("Adrian Gonzalez", 161), ("Manny Machado", 197), ("Fernando Tatis Jr.", 155)],
    "SDP": [("Nate Colbert", 163), ("Adrian Gonzalez", 161), ("Manny Machado", 197)],
    "SF": [("Willie Mays", 646), ("Barry Bonds", 586), ("Willie McCovey", 469), ("Mel Ott", 511)],
    "SFG": [("Willie Mays", 646), ("Barry Bonds", 586), ("Willie McCovey", 469)],
    "SEA": [("Ken Griffey Jr.", 417), ("Edgar Martinez", 309), ("Nelson Cruz", 163), ("Kyle Seager", 242)],
    "STL": [("Stan Musial", 475), ("Albert Pujols", 469), ("Mark McGwire", 220), ("Ken Boyer", 255)],
    "TB": [("Evan Longoria", 261), ("Carlos Pena", 163), ("Aubrey Huff", 128), ("Jose Bautista", 59)],
    "TBR": [("Evan Longoria", 261), ("Carlos Pena", 163)],
    "TEX": [("Juan Gonzalez", 372), ("Rafael Palmeiro", 321), ("Iván Rodríguez", 217), ("Adrián Beltré", 199)],
    "TOR": [("Carlos Delgado", 336), ("José Bautista", 288), ("Vernon Wells", 223), ("Vladimir Guerrero Jr.", 195)],
    "WSH": [("Ryan Zimmerman", 284), ("Alfonso Soriano", 46), ("Bryce Harper", 184), ("Juan Soto", 119)],
    "WSN": [("Ryan Zimmerman", 284), ("Bryce Harper", 184), ("Juan Soto", 119)],
}


def career_hr_milestones(batter_ids, id_to_team=None, within: int = 5, timeout: int = 12) -> dict:
    """Career HR totals for slate hitters via statsapi (batched), flagging anyone
    within `within` of a round-number milestone (every 50 from 50 up) OR within
    `within` of passing a franchise HR leader on their current team.
    {batter_id: {"career_hr": n, "next": m, "away": k, "kind": "career"|"team",
                 "target": name-if-team}} — only near-milestone hitters returned.
    id_to_team: optional {batter_id: team_abbr} so team-specific comparisons work.
    Fails soft to {} on any network/shape issue."""
    import json as _json
    import urllib.request
    ids = [int(b) for b in batter_ids]
    totals = {}
    for i in range(0, len(ids), 80):
        chunk = ids[i:i + 80]
        url = ("https://statsapi.mlb.com/api/v1/people?personIds="
               + ",".join(map(str, chunk))
               + "&hydrate=stats(group=[hitting],type=[career])")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = _json.loads(r.read().decode())
        except Exception as e:
            print(f"[milestones] batch fetch failed: {e}")
            continue
        for person in data.get("people", []):
            try:
                for s in person.get("stats", []):
                    for sp in s.get("splits", []):
                        hr = sp.get("stat", {}).get("homeRuns")
                        if hr is not None:
                            totals[int(person["id"])] = int(hr)
            except Exception:
                continue
    out = {}
    # round-50 milestones from 50 upward
    marks = list(range(50, 900, 50))
    id_to_team = id_to_team or {}
    for bid, hr in totals.items():
        best = None  # best candidate = smallest "away" value
        # career round number
        nxt = next((m for m in marks if m > hr), None)
        if nxt is not None and (nxt - hr) <= within:
            best = {"career_hr": hr, "next": nxt, "away": nxt - hr, "kind": "career"}
        # franchise leader (only if we know the team)
        team = id_to_team.get(bid)
        if team and team in TEAM_HR_LEADERS:
            for name, mark in TEAM_HR_LEADERS[team]:
                if mark > hr and (mark + 1 - hr) <= within:  # passing them = mark+1
                    away = mark + 1 - hr
                    cand = {"career_hr": hr, "next": mark + 1, "away": away,
                            "kind": "team", "target": name, "target_hr": mark, "team": team}
                    if best is None or away < best["away"]:
                        best = cand
        if best is not None:
            out[bid] = best
    print(f"[milestones] {len(totals)} careers fetched, {len(out)} near a milestone")
    return out
