"""
MLB StatsAPI puller — the official, free, keyless source.

Gives us the daily slate: games, venues, probable pitchers, posted lineups,
and batter/pitcher handedness. Nothing here needs an API key and the endpoint
is rock solid.

Docs base: https://statsapi.mlb.com/api/v1
"""
from __future__ import annotations
import requests
from datetime import datetime, timedelta

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 20


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _playable(g: dict) -> bool:
    """True only if this game is actually going to be (or is being) played on the queried date.
    MLB's schedule endpoint returns postponed/cancelled/suspended games too — and when a game is
    postponed and made up later, a ghost entry can appear under the original date. We must drop
    those, or the board shows a game that isn't happening (the 'thinks today's game was yesterday'
    glitch). We key off the documented status fields and are conservative: anything clearly not a
    live/scheduled/final game is excluded.
    """
    st = g.get("status", {}) or {}
    detailed = (st.get("detailedState") or "").lower()
    coded = (st.get("codedGameState") or "").upper()
    abstract = (st.get("abstractGameState") or "").lower()
    # explicit non-playable detailed states
    bad_words = ("postponed", "cancelled", "canceled", "suspended", "forfeit")
    if any(w in detailed for w in bad_words):
        return False
    # coded/abstract fallbacks: 'D' = Postponed, 'C' = Cancelled, 'U'/'T' = Suspended (MLB codes)
    if coded in ("D", "C", "U", "T"):
        return False
    # a game with a rescheduleDate set (and not yet resumed) has been moved off this date
    if g.get("rescheduleDate") and abstract not in ("live", "final"):
        return False
    return True


def get_slate(date_str: str) -> dict:
    """
    Return the full slate for a given YYYY-MM-DD.

    Output:
      {
        "games": [ {game_pk, away, home, away_id, home_id, park, time,
                    away_pitcher, home_pitcher} ... ],
        "lineups": { game_pk: {"away": [batter_id...], "home": [batter_id...]} },
        "pitchers": { pitcher_id: {"name","throws"} },
      }
    Lineups are only populated once teams post them (usually a few hours before
    first pitch). Re-running through the afternoon fills them in.

    Postponement handling: the schedule endpoint can return postponed/cancelled games and,
    around makeup dates, ghost entries whose date doesn't match what we asked for. We (1) skip
    non-playable games via _playable(), and (2) only accept games whose own date matches the
    queried slate date, so a game postponed to/from another day never bleeds in.
    """
    hydrate = "probablePitcher(note),lineups,team,venue"
    data = _get(
        f"{BASE}/schedule",
        {"sportId": 1, "date": date_str, "hydrate": hydrate},
    )

    games, lineups, pitchers = [], {}, {}
    seen_pks = set()
    for d in data.get("dates", []):
        # the schedule groups games under a "date"; only trust the block that matches our query
        block_date = d.get("date")
        for g in d.get("games", []):
            pk = g["gamePk"]
            if pk in seen_pks:            # de-dupe: a makeup can appear under two date blocks
                continue
            if not _playable(g):          # drop postponed / cancelled / suspended / moved
                continue
            # date consistency: the game's OWN date must be the slate date. officialDate is the
            # authoritative calendar day a game counts for; fall back to the block date, then to
            # the gameDate's date portion. If none match date_str, this game isn't today's slate.
            official = g.get("officialDate") or block_date
            game_day = official or (g.get("gameDate", "")[:10])
            if game_day and game_day != date_str:
                continue
            seen_pks.add(pk)

            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            venue = g.get("venue", {}).get("name", "")

            ap = g["teams"]["away"].get("probablePitcher")
            hp = g["teams"]["home"].get("probablePitcher")
            ap_id = ap["id"] if ap else None
            hp_id = hp["id"] if hp else None

            games.append({
                "game_pk": pk,
                "away": away.get("abbreviation", away.get("name", "")),
                "home": home.get("abbreviation", home.get("name", "")),
                "away_id": away["id"],
                "home_id": home["id"],
                "away_name": away.get("name", ""),
                "home_name": home.get("name", ""),
                "park": venue,
                "time": g.get("gameDate", ""),
                "official_date": g.get("officialDate") or block_date or date_str,
                "game_number": g.get("gameNumber", 1),   # doubleheader game 1 vs 2
                "day_night": (g.get("dayNight") or "").lower(),   # 'day' | 'night' (authoritative)
                "status": (g.get("status", {}) or {}).get("detailedState", ""),
                "away_pitcher_id": ap_id,
                "home_pitcher_id": hp_id,
            })

            # lineups (present only when posted)
            lu = g.get("lineups", {})
            away_lu = [p["id"] for p in lu.get("awayPlayers", [])]
            home_lu = [p["id"] for p in lu.get("homePlayers", [])]
            if away_lu or home_lu:
                lineups[pk] = {"away": away_lu, "home": home_lu}

            for p in (ap, hp):
                if p:
                    pitchers[p["id"]] = {
                        "name": p.get("fullName", ""),
                        "throws": (p.get("pitchHand", {}) or {}).get("code", ""),
                    }

    return {"games": games, "lineups": lineups, "pitchers": pitchers}


def get_reliever_pitch_logs(team_ids, date_str, days=3):
    """{pitcher_id: [{"days_ago": n, "pitches": n, "date": "YYYY-MM-DD"}, ...]}

    Feeds `environment.bullpen_state()`. Pitch counts are the only reliable public signal of
    who a manager will actually use tonight: an arm at 35+ pitches yesterday is unavailable
    regardless of how he feels, and modelling him as available means modelling a pitcher who
    will not appear.

    Walks each team's completed games over the trailing window and reads pitch counts from the
    boxscore. Non-fatal per game, so one unavailable feed cannot take the board down.
    """
    import datetime as _dt
    out = {}
    try:
        base = _dt.date.fromisoformat(date_str)
    except Exception:
        return out
    for team_id in {int(t) for t in (team_ids or []) if t}:
        for back in range(1, int(days) + 1):
            day = (base - _dt.timedelta(days=back)).isoformat()
            try:
                sched = _get("https://statsapi.mlb.com/api/v1/schedule",
                             {"sportId": 1, "teamId": team_id, "date": day})
            except Exception:
                continue
            for d in (sched.get("dates") or []):
                for g in (d.get("games") or []):
                    if str((g.get("status") or {}).get("abstractGameState")) != "Final":
                        continue
                    gid = g.get("gamePk")
                    if not gid:
                        continue
                    try:
                        box = _get(f"https://statsapi.mlb.com/api/v1/game/{gid}/boxscore", {})
                    except Exception:
                        continue
                    for side in ("home", "away"):
                        tm = ((box.get("teams") or {}).get(side) or {})
                        if int(((tm.get("team") or {}).get("id") or 0)) != team_id:
                            continue
                        players = tm.get("players") or {}
                        # `pitchers` lists them in order used; the first is the starter, and a
                        # starter's rest pattern is nothing like a reliever's, so he is skipped
                        order = tm.get("pitchers") or []
                        for idx, pid in enumerate(order):
                            if idx == 0:
                                continue
                            rec = players.get(f"ID{pid}") or {}
                            stats = ((rec.get("stats") or {}).get("pitching") or {})
                            pitches = stats.get("numberOfPitches") or stats.get("pitchesThrown")
                            if not pitches:
                                continue
                            # Record the TEAM alongside each appearance. Without it the caller
                            # has no way to group these arms into bullpens: pitcher stats are
                            # fetched for starters only, so a reliever appears in no other
                            # table. An earlier version matched against a "team" key that did
                            # not exist, silently produced zero arms per pen, and left the
                            # Bullpens tab on the older Statcast-inferred numbers.
                            out.setdefault(int(pid), []).append(
                                {"days_ago": back, "pitches": int(pitches), "date": day,
                                 "team_id": team_id})
    return out


def get_team_records(season: int | None = None) -> dict:
    """{TEAM_ABBR: {w, l, rs, ra, pyth}} from the MLB standings endpoint.

    Why Pythagorean rather than win-loss: a team's run differential predicts its FUTURE record
    better than its own record does, because W-L bakes in bullpen sequencing and one-run-game
    luck that don't repeat. Pythagorean expectation is RS^2 / (RS^2 + RA^2).

    This is a team-season prior — it captures the things a lineup-plus-starter model can't see
    (bench depth, baserunning, defense beyond the OAA term, manager), which is exactly why it's
    worth blending in at modest weight rather than trusting the bottom-up projection alone.
    Non-fatal: returns {} on any failure and the run model proceeds unchanged."""
    import datetime as _dt
    yr = season or _dt.date.today().year
    try:
        data = _get("https://statsapi.mlb.com/api/v1/standings",
                    {"leagueId": "103,104", "season": yr, "standingsTypes": "regularSeason"})
        out = {}
        for rec in (data.get("records") or []):
            for tr in (rec.get("teamRecords") or []):
                team = ((tr.get("team") or {}).get("abbreviation")
                        or (tr.get("team") or {}).get("teamName"))
                if not team:
                    continue
                rs = tr.get("runsScored")
                ra = tr.get("runsAllowed")
                w = (tr.get("leagueRecord") or {}).get("wins") or tr.get("wins")
                l = (tr.get("leagueRecord") or {}).get("losses") or tr.get("losses")
                pyth = None
                try:
                    rs_f, ra_f = float(rs), float(ra)
                    if rs_f > 0 and ra_f > 0:
                        pyth = round(rs_f ** 2 / (rs_f ** 2 + ra_f ** 2), 4)
                except Exception:
                    pyth = None
                out[str(team)] = {"w": w, "l": l, "rs": rs, "ra": ra, "pyth": pyth}
        return out
    except Exception as e:
        print(f"[statsapi] team records unavailable (non-fatal): {e}")
        return {}


def get_pitcher_stats(pitcher_ids: list[int], season: int | None = None) -> dict:
    """Season ERA / WHIP / IP / HR-allowed for a list of pitchers, from the official API.
    Feeds the HR Vulnerability score and Bomb Score, which weight ERA and WHIP directly.
    Returns { pitcher_id: {"era": float, "whip": float, "ip": float, "hr": int,
                           "so": int, "bb": int, "h": int} }.
    Missing/unavailable pitchers are simply absent from the dict (callers must handle None).
    """
    out = {}
    if not pitcher_ids:
        return out
    if season is None:
        season = datetime.now().year
    for pid in pitcher_ids:
        if not pid:
            continue
        try:
            data = _get(f"{BASE}/people/{int(pid)}/stats",
                        {"stats": "season", "group": "pitching",
                         "season": season, "sportId": 1})
            splits = (data.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                continue
            st = splits[-1].get("stat", {}) or {}
            def _f(key):
                v = st.get(key)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            def _i(key):
                v = st.get(key)
                try:
                    return int(v)
                except (TypeError, ValueError):
                    return None
            ip_raw = st.get("inningsPitched")
            # MLB reports IP as "123.1" meaning 123 and 1/3 innings — convert properly
            ip = None
            if ip_raw is not None:
                try:
                    whole, _, frac = str(ip_raw).partition(".")
                    ip = float(whole) + ({"1": 1/3, "2": 2/3}.get(frac, 0.0) if frac else 0.0)
                except Exception:
                    ip = _f("inningsPitched")
            rec = {"era": _f("era"), "whip": _f("whip"), "ip": ip,
                   "hr": _i("homeRuns"), "so": _i("strikeOuts"),
                   "bb": _i("baseOnBalls"), "h": _i("hits")}
            if rec["era"] is not None or rec["whip"] is not None:
                out[int(pid)] = rec
        except Exception:
            continue
    return out


def recent_hr_hitters_from_boxscore(date_str: str) -> set:
    """Which real MLB player IDs hit a home run on the given date, straight from the OFFICIAL
    box score -- not Statcast.

    Why this exists: hr_last_game() in statcast_data.py derives "did he homer yesterday" from
    the shared Statcast pitch-tracking frame, which has a real, well-known publication lag --
    Baseball Savant often does not have a night's full pitch-by-pitch data processed until well
    into the following day. Checked directly: a live board built at 7:23 AM ET the morning
    after a slate needed that night's games to already be in the Statcast frame -- very likely
    still processing at that hour, especially for games ending after midnight ET. That's the
    real, most likely explanation for a hitter who genuinely homered the night before still
    showing hr_last_game=False.

    MLB's own official box score (this is the same live-scoreboard data ESPN/MLB.com use)
    publishes within minutes of a game ending -- a completely different, faster pipeline than
    Statcast's full tracking data. Player IDs here are the same MLB Advanced Media numeric ID
    space Statcast's own `batter` column already uses, so this is directly compatible with the
    rest of this app without any name-matching.

    Non-fatal by design, matching the rest of this file: any failure (network, schema, a
    postponed game, a field name that's changed) returns an empty set rather than raising, so a
    boxscore hiccup degrades this one signal instead of breaking the whole build.
    """
    out = set()
    try:
        sched = _get(f"{BASE}/schedule", {"sportId": 1, "date": date_str})
        game_pks = [g["gamePk"] for day in sched.get("dates", []) for g in day.get("games", [])
                   if str(g.get("status", {}).get("abstractGameState", "")).lower() == "final"]
    except Exception as e:
        print(f"[statsapi] recent_hr_hitters_from_boxscore schedule fetch failed (non-fatal): {e}")
        return out
    for gpk in game_pks:
        try:
            box = _get(f"{BASE}/game/{gpk}/boxscore")
            for side in ("home", "away"):
                players = ((box.get("teams") or {}).get(side) or {}).get("players") or {}
                for _pkey, pdata in players.items():
                    hr = (((pdata.get("stats") or {}).get("batting") or {}).get("homeRuns"))
                    if hr and hr > 0:
                        _pid = (pdata.get("person") or {}).get("id")
                        if _pid is not None:
                            out.add(int(_pid))
        except Exception as e:
            print(f"[statsapi] boxscore fetch failed for game {gpk} (non-fatal): {e}")
            continue
    return out


def get_recent_lineup(team_id: int, before_date: str) -> list[int]:
    """
    A team's projected batting order (player ids, in order) before today's is confirmed.

    FIXED this session, per a real report: three star players across three different teams
    were all missing from a real day's projected lineups despite genuinely playing the day
    before. Root cause -- the old version used only the single most recent completed game's
    boxscore battingOrder. If a player rested, pinch-hit, or entered as a late substitute in
    that ONE specific game, they were completely excluded from the projection even though
    they're clearly a regular starter -- a single anomalous game fully determined the result.

    Now looks at the last several completed games (up to 5, within the 10-day window) and
    picks the most frequent STARTER (battingOrder code ending in "00", meaning they began
    the game in that spot, not a later substitution) per batting-order position across them,
    with the most recent game breaking ties. Resilient to one rest day or one pinch-hit
    appearance, rather than fully dependent on whichever single game happened to be most
    recent when this runs.
    """
    try:
        start = (datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")
        data = _get(f"{BASE}/schedule",
                    {"sportId": 1, "teamId": team_id, "startDate": start, "endDate": before_date})
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    games.append((g.get("gameDate", ""), g["gamePk"]))
        if not games:
            return []
        games.sort()
        recent_games = games[-5:]   # most recent up to 5, oldest-to-newest order

        # spot -> {player_id: n_times_started_there}, most recent game weighted last so it
        # naturally wins ties via insertion-order-preserving dict updates below
        spot_counts: dict[int, dict[int, int]] = {}
        spot_most_recent: dict[int, int] = {}   # spot -> player_id from the newest game seen
        for _, game_pk in recent_games:
            try:
                box = _get(f"{BASE}/game/{game_pk}/boxscore")
            except Exception:
                continue
            for side in ("away", "home"):
                t = box.get("teams", {}).get(side, {})
                if t.get("team", {}).get("id") != team_id:
                    continue
                order = t.get("battingOrder", []) or []
                if not order:
                    continue
                for pid_key, pdata in (t.get("players") or {}).items():
                    bo = pdata.get("battingOrder")
                    if bo is None or int(bo) % 100 != 0:
                        continue
                    spot = int(bo) // 100
                    pid = int(pdata.get("person", {}).get("id", 0))
                    if not pid:
                        continue
                    spot_counts.setdefault(spot, {})
                    spot_counts[spot][pid] = spot_counts[spot].get(pid, 0) + 1
                    spot_most_recent[spot] = pid   # overwritten each game -> ends up newest

        if not spot_counts:
            return []
        lineup = []
        for spot in sorted(spot_counts.keys()):
            counts = spot_counts[spot]
            best_n = max(counts.values())
            tied = [pid for pid, n in counts.items() if n == best_n]
            # tie-break: whichever of the tied players started there most recently
            pick = spot_most_recent.get(spot) if spot_most_recent.get(spot) in tied else tied[0]
            lineup.append(pick)
        return lineup
    except Exception:
        return []


def get_handedness(player_ids: list[int]) -> dict:
    """
    Batch-fetch batSide / pitchHand for a list of mlbam person ids.
    Returns { id: {"bats": "R/L/S", "throws": "R/L"} }.
    """
    out = {}
    ids = [str(i) for i in player_ids if i]
    if not ids:
        return out
    # the people endpoint accepts a comma-separated personIds list
    for chunk_start in range(0, len(ids), 100):
        chunk = ids[chunk_start:chunk_start + 100]
        try:
            data = _get(f"{BASE}/people", {"personIds": ",".join(chunk)})
        except Exception:
            continue
        for person in data.get("people", []):
            out[person["id"]] = {
                "bats": (person.get("batSide", {}) or {}).get("code", ""),
                "throws": (person.get("pitchHand", {}) or {}).get("code", ""),
                "name": person.get("fullName", ""),
            }
    return out


def get_live_hrs_today(game_pks: list) -> list:
    """Who has homered so far tonight, across the given games -- real, in-progress data, not a
    replay. Uses the live game feed endpoint (v1.1, not the v1 base this file otherwise uses --
    the live feed is the one MLB actually updates during a game; the v1 schedule/boxscore
    endpoints are more oriented around pre/post-game state). Returns a flat list, one entry per
    home run: {id, name, team, game_pk, inning, half}.

    HONEST LIMITATION: this can't be tested against a real in-progress game from this
    environment (no live game to query while building this). Built defensively -- one game's
    fetch failing never breaks the others -- matching this file's established style, but the
    exact JSON shape hasn't been verified against a real live feed. If MLB's response shape
    differs from what's assumed here, this should fail closed (return fewer/no results) rather
    than crash the board build, but that specific behavior hasn't been confirmed end to end.
    Worth a real spot-check against an actual live game before trusting this fully.
    """
    out = []
    for gpk in (game_pks or []):
        try:
            data = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{int(gpk)}/feed/live",
                                timeout=TIMEOUT).json()
        except Exception:
            continue
        try:
            teams = ((data.get("gameData") or {}).get("teams") or {})
            away_abbr = (teams.get("away") or {}).get("abbreviation")
            home_abbr = (teams.get("home") or {}).get("abbreviation")
            plays = ((data.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
        except Exception:
            continue
        for play in plays:
            try:
                result = play.get("result") or {}
                if result.get("eventType") != "home_run":
                    continue
                matchup = play.get("matchup") or {}
                batter = matchup.get("batter") or {}
                about = play.get("about") or {}
                if batter.get("id") is None:
                    continue
                team = away_abbr if about.get("isTopInning") else home_abbr
                out.append({
                    "id": batter["id"], "name": batter.get("fullName", ""),
                    "team": team,
                    "game_pk": int(gpk),
                    "inning": about.get("inning"),
                    "half": about.get("halfInning"),
                })
            except Exception:
                continue   # one malformed play entry never drops the rest of this game's HRs
    return out


def bvp_career(batter_id: int, pitcher_id: int) -> dict | None:
    """Career batter-vs-pitcher totals (the Stathead/BR-style number) via the official API.
    Returns {pa, hr, ab, h} — zeros if they've never faced — or None on a request error."""
    try:
        data = _get(f"{BASE}/people/{int(batter_id)}/stats",
                    {"stats": "vsPlayerTotal", "opposingPlayerId": int(pitcher_id),
                     "group": "hitting", "sportId": 1})
        splits = (data.get("stats") or [{}])[0].get("splits") or []
        if not splits:
            return {"pa": 0, "hr": 0, "ab": 0, "h": 0}
        st = splits[0].get("stat", {}) or {}
        return {"pa": int(st.get("plateAppearances", 0) or 0),
                "hr": int(st.get("homeRuns", 0) or 0),
                "ab": int(st.get("atBats", 0) or 0),
                "h": int(st.get("hits", 0) or 0)}
    except Exception:
        return None
