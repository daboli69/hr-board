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


def get_recent_lineup(team_id: int, before_date: str) -> list[int]:
    """
    A team's most recent posted batting order (player ids, in order), used as a
    PROJECTED lineup before today's is confirmed. Looks back up to 10 days for the
    team's last completed game and reads its boxscore battingOrder.
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
        game_pk = games[-1][1]
        box = _get(f"{BASE}/game/{game_pk}/boxscore")
        for side in ("away", "home"):
            t = box.get("teams", {}).get(side, {})
            if t.get("team", {}).get("id") == team_id:
                order = t.get("battingOrder", []) or []
                return [int(x) for x in order]
        return []
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
