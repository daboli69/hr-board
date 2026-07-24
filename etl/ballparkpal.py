"""
ballparkpal.py — Ballpark Pal API client (park factors + per-hitter park factors).

Ballpark Pal models park factors from ~1M batted balls and 20k games since 2016, and — the
part we care about most — assigns factors to INDIVIDUAL HITTERS based on where they actually
hit the ball. That's the thing park_geometry.py hand-rolled; this replaces it with the real
model for live boards.

API shape (v1):
    Base    https://www.ballparkpal.com/api/v1
    Auth    header  X-API-Key: <key>
    GET /parkfactors?date=YYYY-MM-DD
        -> [{gameId, gameTime, teamAway, teamHome,
             runsPercent, homeRunsPercent, doublesTriplesPercent, singlesPercent,
             runsAmount, homeRunsAmount, doublesTriplesAmount, singlesAmount}]
        Percent fields are INTEGERS meaning "% above/below league average" (18 == +18%).
    GET /parkfactors/hitters?date=YYYY-MM-DD
        -> per-hitter park factors for that slate.
    GET /projections/averages?gameId=<id>
        -> simulated player projections for a game.

IMPORTANT CONSTRAINT: park factors are TODAY-AND-FUTURE ONLY. Requests for past dates return
`date_out_of_range`. So this can drive the live board but can NEVER feed the backtest — the
historical replay keeps using etl/park_factors.py. Expect a small divergence between live park
numbers and backtested ones; that's inherent, not a bug.

The API key is read from the BPP_API_KEY environment variable (a GitHub secret). It is never
hardcoded and never written into any output file.

Because the exact JSON envelope isn't documented publicly, every parser here is defensive: it
accepts a bare list, {"data": [...]}, or {"games": [...]}, and tolerates camelCase or
snake_case keys. Run `python -m etl.ballparkpal --probe` to dump the real shape.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

BASE = "https://www.ballparkpal.com/api/v1"
KEY_ENV = "BPP_API_KEY"
TIMEOUT = 25
RETRIES = 3


# ---------------------------------------------------------------- transport

def _key() -> str | None:
    k = os.environ.get(KEY_ENV) or ""
    k = k.strip()
    return k or None


def _get(path: str, params: dict | None = None) -> object | None:
    """GET a Ballpark Pal endpoint. Returns parsed JSON, or None on any failure
    (missing key, network error, non-200, bad JSON). Never raises."""
    key = _key()
    if not key:
        return None
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={
                "X-API-Key": key,
                "Accept": "application/json",
                "User-Agent": "going-yard/1.0",
            })
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read().decode("utf-8", "replace")
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            last = f"HTTP {e.code} {body}"
            # auth / range errors won't fix themselves on retry
            if e.code in (401, 403, 404, 422):
                break
        except Exception as e:                       # noqa: BLE001 - never fail the build
            last = str(e)
        time.sleep(1.2 * (attempt + 1))
    print(f"[bpp] GET {path} failed: {last}")
    return None


def _rows(payload) -> list:
    """Unwrap whatever envelope the API used into a list of row dicts."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("data", "games", "parkFactors", "park_factors", "results", "rows", "hitters"):
            v = payload.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        # a dict keyed by id -> row
        vals = [v for v in payload.values() if isinstance(v, dict)]
        if vals:
            return vals
    return []


def _pick(row: dict, *names, default=None):
    """First present key among `names`, tolerating camelCase/snake_case variants."""
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
        alt = "".join("_" + c.lower() if c.isupper() else c for c in n)
        if alt in row and row[alt] is not None:
            return row[alt]
    return default


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _pct_to_mult(p):
    """Ballpark Pal percents are '% vs league average' integers: 18 -> 1.18, -12 -> 0.88."""
    n = _num(p)
    return None if n is None else round(1.0 + (n / 100.0), 4)


# ---------------------------------------------------------------- endpoints

def park_factors(date_str: str) -> dict:
    """Per-GAME park factors for `date_str` (YYYY-MM-DD, today or later).

    Returns {"by_game": {gameId: {...}}, "by_teams": {"AWAY@HOME": {...}}, "n": int}
    Each entry: {hr_mult, runs_mult, xbh_mult, single_mult,
                 hr_pct, runs_pct, hr_amount, runs_amount, game_time, away, home}
    """
    payload = _get("/parkfactors", {"date": date_str})
    rows = _rows(payload)
    by_game, by_teams = {}, {}
    for r in rows:
        gid = _pick(r, "gameId", "game_id", "gamePk", "id")
        away = _pick(r, "teamAway", "away", "awayTeam")
        home = _pick(r, "teamHome", "home", "homeTeam")
        ent = {
            "hr_mult":     _pct_to_mult(_pick(r, "homeRunsPercent")),
            "runs_mult":   _pct_to_mult(_pick(r, "runsPercent")),
            "xbh_mult":    _pct_to_mult(_pick(r, "doublesTriplesPercent")),
            "single_mult": _pct_to_mult(_pick(r, "singlesPercent")),
            "hr_pct":      _num(_pick(r, "homeRunsPercent")),
            "runs_pct":    _num(_pick(r, "runsPercent")),
            "hr_amount":   _num(_pick(r, "homeRunsAmount")),
            "runs_amount": _num(_pick(r, "runsAmount")),
            "game_time":   _pick(r, "gameTime", "game_time"),
            "away": away, "home": home,
        }
        if gid is not None:
            by_game[str(gid)] = ent
        if away and home:
            by_teams[f"{away}@{home}"] = ent
    return {"by_game": by_game, "by_teams": by_teams, "n": len(by_game) or len(by_teams)}


def hitter_park_factors(date_str: str) -> dict:
    """Per-HITTER park factors for `date_str` — Ballpark Pal assigns a factor to each hitter
    based on his own spray/contact profile, which is exactly what we want instead of a single
    park-wide number.

    Returns {"by_id": {mlbam_id: {hr_mult, hr_pct, name, team}}, "by_name": {...}, "n": int}
    Player-id key naming isn't documented, so we accept several and also index by normalized
    name as a fallback join.
    """
    payload = _get("/parkfactors/hitters", {"date": date_str})
    rows = _rows(payload)
    by_id, by_name = {}, {}
    for r in rows:
        pid = _pick(r, "playerId", "player_id", "mlbamId", "mlbam_id", "batterId", "id")
        name = _pick(r, "playerName", "player_name", "name", "batter")
        hr_pct = _pick(r, "homeRunsPercent", "hrPercent", "homeRunPercent")
        ent = {
            "hr_mult": _pct_to_mult(hr_pct),
            "hr_pct": _num(hr_pct),
            "runs_mult": _pct_to_mult(_pick(r, "runsPercent")),
            "name": name,
            "team": _pick(r, "team", "teamAbbr", "teamAbbrev"),
            "game_id": _pick(r, "gameId", "game_id"),
        }
        if ent["hr_mult"] is None:
            continue
        if pid is not None:
            try:
                by_id[int(pid)] = ent
            except (TypeError, ValueError):
                by_id[str(pid)] = ent
        if name:
            by_name[_norm_name(str(name))] = ent
    return {"by_id": by_id, "by_name": by_name, "n": len(by_id) or len(by_name)}


def projections(game_id) -> list:
    """Simulated player projections for one game. Used ONLY to power the 'sort by park'
    projection option — never to override the core model."""
    payload = _get("/projections/averages", {"gameId": game_id})
    return _rows(payload)


def _norm_name(name: str) -> str:
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z ]", "", n.lower().strip())
    n = re.sub(r"\s+", " ", n).strip()
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip()


# ---------------------------------------------------------------- build entry

def fetch_all(date_str: str) -> dict:
    """Everything the board needs from Ballpark Pal for one slate date.
    Returns a JSON-safe dict; `ok` is False when the key is missing or the API failed, in
    which case callers must fall back to the local park model."""
    if not _key():
        print(f"[bpp] no {KEY_ENV} set — using local park model")
        return {"ok": False, "reason": "no_key", "date": date_str}
    games = park_factors(date_str)
    hitters = hitter_park_factors(date_str)
    ok = bool(games.get("n") or hitters.get("n"))
    print(f"[bpp] park factors: {games.get('n', 0)} games · {hitters.get('n', 0)} hitters")
    return {
        "ok": ok,
        "date": date_str,
        "fetched": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "games": games.get("by_game", {}),
        "by_teams": games.get("by_teams", {}),
        "hitters": hitters.get("by_id", {}),
        "hitters_by_name": hitters.get("by_name", {}),
    }


def resolve_hr_mult(bpp: dict, *, player_id=None, player_name=None,
                    away=None, home=None, game_id=None, fallback=None):
    """The single place park factor gets resolved. Preference order:
        1. Ballpark Pal per-HITTER factor (best — modeled on his own spray profile)
        2. Ballpark Pal per-GAME factor
        3. `fallback` (our local park model)
    Returns (multiplier, source) where source is 'bpp_hitter' | 'bpp_game' | 'local' | None.
    """
    if bpp and bpp.get("ok"):
        H = bpp.get("hitters") or {}
        if player_id is not None:
            ent = H.get(player_id) or H.get(str(player_id))
            if ent and ent.get("hr_mult") is not None:
                return ent["hr_mult"], "bpp_hitter"
        if player_name:
            ent = (bpp.get("hitters_by_name") or {}).get(_norm_name(str(player_name)))
            if ent and ent.get("hr_mult") is not None:
                return ent["hr_mult"], "bpp_hitter"
        G = bpp.get("games") or {}
        if game_id is not None:
            ent = G.get(str(game_id))
            if ent and ent.get("hr_mult") is not None:
                return ent["hr_mult"], "bpp_game"
        if away and home:
            ent = (bpp.get("by_teams") or {}).get(f"{away}@{home}")
            if ent and ent.get("hr_mult") is not None:
                return ent["hr_mult"], "bpp_game"
    if fallback is not None:
        return fallback, "local"
    return None, None


# ---------------------------------------------------------------- probe / CLI

def _probe(date_str: str | None = None):
    """Dump the RAW response shape for each endpoint so we can correct the parsers if the
    real JSON differs from what's assumed above. Run this once in Actions:
        python -m etl.ballparkpal --probe
    """
    d = date_str or datetime.now().strftime("%Y-%m-%d")
    if not _key():
        print(f"probe: {KEY_ENV} is not set in this environment.")
        return
    for path, params in (("/parkfactors", {"date": d}),
                         ("/parkfactors/hitters", {"date": d})):
        print(f"\n=== {path} ?{urllib.parse.urlencode(params)} ===")
        payload = _get(path, params)
        if payload is None:
            print("  (no response)")
            continue
        print(f"  top-level type: {type(payload).__name__}")
        if isinstance(payload, dict):
            print(f"  top-level keys: {list(payload.keys())[:12]}")
        rows = _rows(payload)
        print(f"  parsed rows: {len(rows)}")
        if rows:
            print(f"  row keys: {list(rows[0].keys())}")
            print("  first row:")
            print("   ", json.dumps(rows[0], indent=2)[:700])


if __name__ == "__main__":
    if "--probe" in sys.argv:
        arg = [a for a in sys.argv[1:] if not a.startswith("-")]
        _probe(arg[0] if arg else None)
    else:
        d = datetime.now().strftime("%Y-%m-%d")
        print(json.dumps(fetch_all(d), indent=2)[:2000])
