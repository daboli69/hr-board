"""Debug script -- dumps the RAW /odds response schema so the actual moneyline field names can
be seen directly, instead of guessing at why ml.home/ml.away come out as 1 or 2.

Run this yourself, with your real PARLAY_API_KEY, from the repo root:

    PARLAY_API_KEY=your_real_key python3 etl/debug_odds_schema.py

Hits the EXACT same endpoint/params/auth fetch_odds.py's real fetch_game_lines() uses -- copied
directly from that function rather than reconstructed from memory, so this debug script can't
silently diverge from what production actually calls.

What to look for in the output: build_game_lines() currently assumes each h2h outcome looks
like {"name": <team name>, "price": <American odds int>} -- the standard "TOA-format" (The Odds
API) shape. If parlay-api.com's real schema differs even slightly (a different key than "price",
odds nested one level deeper, decimal instead of American, etc.), that mismatch is the bug --
_safe_int() would silently return None for a missing/wrong-shaped field, and if this script's
raw dump shows a genuinely different structure, that's the fix target for build_game_lines().
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error

API_BASE = "https://parlay-api.com/v1"
SPORT = "baseball_mlb"


def fetch_raw_odds(api_key: str) -> dict | list:
    """Identical request to fetch_odds.py's fetch_game_lines() -- same endpoint, same query
    params, same auth (both the apiKey query param AND the x-api-key header, since the real
    code sends both and I don't know which one this API actually requires)."""
    q = urllib.parse.urlencode({
        "markets": "h2h,totals",
        "regions": "us",
        "apiKey": api_key,
    })
    url = f"{API_BASE}/sports/{SPORT}/odds?{q}"
    print(f"[debug] GET {url.replace(api_key, '***')}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "going-yard-debug/1.0"})
    req.add_header("x-api-key", api_key)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw)


def inspect(data) -> None:
    print("=" * 70)
    print("TOP-LEVEL SHAPE")
    print("=" * 70)
    if isinstance(data, list):
        print(f"Response is a LIST of {len(data)} items")
        if not data:
            print("  -- empty list, nothing to inspect. Is the market/date right?")
            return
        sample_event = data[0]
    elif isinstance(data, dict):
        print(f"Response is a DICT with top-level keys: {sorted(data.keys())}")
        # Try the common wrapper shapes so this still finds the real event list either way
        for candidate_key in ("data", "events", "games", "odds"):
            if candidate_key in data and isinstance(data[candidate_key], list):
                print(f"  -- found a list under data['{candidate_key}'], inspecting that")
                data = data[candidate_key]
                break
        else:
            print("  -- no obvious event-list key found; dumping the whole dict below instead")
            print(json.dumps(data, indent=2)[:3000])
            return
        if not data:
            print("  -- empty list after unwrapping.")
            return
        sample_event = data[0]
    else:
        print(f"Response is neither a list nor a dict: {type(data)}")
        print(repr(data)[:1000])
        return

    print()
    print("=" * 70)
    print("ONE FULL EVENT, RAW (this is the ground truth -- read this, not my guesses)")
    print("=" * 70)
    print(json.dumps(sample_event, indent=2)[:4000])

    print()
    print("=" * 70)
    print("EVENT-LEVEL KEYS")
    print("=" * 70)
    if isinstance(sample_event, dict):
        print(sorted(sample_event.keys()))
        for team_key in ("home_team", "away_team", "home", "away"):
            if team_key in sample_event:
                print(f"  {team_key!r}: {sample_event[team_key]!r}")

    print()
    print("=" * 70)
    print("BOOKMAKER / MARKET / OUTCOME STRUCTURE -- this is what build_game_lines() parses,")
    print("and where the real fix needs to target once you see the actual key names below")
    print("=" * 70)
    books = sample_event.get("bookmakers") if isinstance(sample_event, dict) else None
    if not books:
        # try other plausible wrapper names in case this API doesn't call it "bookmakers"
        for alt in ("books", "sportsbooks", "odds"):
            if isinstance(sample_event, dict) and sample_event.get(alt):
                print(f"  no 'bookmakers' key -- found odds data under '{alt}' instead")
                books = sample_event[alt]
                break
    if not books:
        print("  NO bookmaker/odds data found on this event at all under any key I checked.")
        print(f"  Full event keys again for reference: {sorted(sample_event.keys()) if isinstance(sample_event, dict) else 'n/a'}")
        return

    print(f"  {len(books)} bookmaker(s) on this event")
    bm = books[0]
    print(f"  first bookmaker keys: {sorted(bm.keys()) if isinstance(bm, dict) else type(bm)}")
    markets = bm.get("markets") if isinstance(bm, dict) else None
    if not markets:
        print("  no 'markets' key on the bookmaker -- dumping the whole bookmaker object:")
        print(json.dumps(bm, indent=2)[:2000])
        return
    for mk in markets:
        mkey = mk.get("key") if isinstance(mk, dict) else None
        print(f"    market key={mkey!r}, market-level keys={sorted(mk.keys()) if isinstance(mk, dict) else '?'}")
        if mkey == "h2h":
            print("    *** THIS IS THE MONEYLINE MARKET -- inspect its outcomes below closely ***")
            outcomes = mk.get("outcomes", [])
            print(f"    {len(outcomes)} outcome(s):")
            for o in outcomes:
                print(f"      {json.dumps(o)}")
                if isinstance(o, dict):
                    print(f"      -> keys on this outcome: {sorted(o.keys())}")
                    print("      -> build_game_lines() currently reads o.get('price') and o.get('name').")
                    print(f"         Real value of 'price' here: {o.get('price')!r}  (type: {type(o.get('price')).__name__})")
                    print(f"         Real value of 'name' here:  {o.get('name')!r}")


if __name__ == "__main__":
    key = os.environ.get("PARLAY_API_KEY")
    if not key:
        print("Set PARLAY_API_KEY first: PARLAY_API_KEY=xxx python3 etl/debug_odds_schema.py",
              file=sys.stderr)
        sys.exit(1)
    try:
        raw = fetch_raw_odds(key)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}", file=sys.stderr)
        sys.exit(1)
    inspect(raw)
