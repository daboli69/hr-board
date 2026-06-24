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
    """
    hydrate = "probablePitcher(note),lineups,team,venue"
    data = _get(
        f"{BASE}/schedule",
        {"sportId": 1, "date": date_str, "hydrate": hydrate},
    )

    games, lineups, pitchers = [], {}, {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g["gamePk"]
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
