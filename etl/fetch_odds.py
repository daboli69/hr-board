"""Fetch MLB home-run prop odds from parlay-api.com and write docs/odds.json.

Zero-backend safe: runs in a GitHub Action where PARLAY_API_KEY is a repo secret,
so the key never touches the client. The frontend reads odds.json like board.json.

We pull the /props endpoint filtered to player_home_runs, keep only the books the
user actually bets (DraftKings, Fanatics), and for each hitter record the BEST
(longest) over price across those two books — that's the line-shopping win. We also
carry the two-book consensus so the Edge Finder can show "DK vs Fanatics".

Player identity: the API returns full names; the board uses MLBAM ids + full names.
We can't join on id (the API has no MLBAM id), so odds.json is keyed by a NORMALIZED
name (accents stripped, lowercased, punctuation/suffix removed). The frontend maps its
board ids to the same normalized name and looks up the price. Unmatched names are just
absent — the Edge Finder falls back to manual entry for them, so a miss is harmless.

Credits: /props is 3 credits/call. One call per run pulls the whole slate's HR props
(all games, both books) — so a single run is 3 credits, and even hourly all day is well
under the 20k/mo tier. We do ONE call per run.
"""
from __future__ import annotations
import json
import os
import sys
import time
import unicodedata
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

API_BASE = "https://parlay-api.com/v1"
SPORT = "baseball_mlb"
MARKET = "player_home_runs"
# PRIMARY books — the ones the user actually bets. A hitter's headline "best" price comes
# only from these, and where both price him it's line-shopped to the better number.
BOOK_ALIASES = {
    "draftkings": ["draftkings", "draft_kings", "dk"],
    "fanatics":   ["fanatics", "fanaticssportsbook", "fanatics_sportsbook", "fanatic"],
}
BOOKS = list(BOOK_ALIASES.keys())

# FALLBACK book — FanDuel prices the clean 0.5 HR line and covers far more hitters than
# DK/Fanatics's milestone-only feed. Used ONLY to fill hitters that NO primary book priced,
# and always flagged (fallback=True) so the app can show it's a reference price from a book
# the user may not bet, not a primary line.
FALLBACK_ALIASES = {
    "fanduel": ["fanduel", "fan_duel", "fd"],
}
FALLBACK_BOOKS = list(FALLBACK_ALIASES.keys())


def _canon_book(raw: str) -> str | None:
    """Map a raw bookmaker string to a PRIMARY book key, or None. Case/punctuation-insensitive."""
    return _match_alias(raw, BOOK_ALIASES)


def _canon_fallback(raw: str) -> str | None:
    """Map a raw bookmaker string to a FALLBACK book key, or None."""
    return _match_alias(raw, FALLBACK_ALIASES)


def _match_alias(raw: str, table: dict) -> str | None:
    if not raw:
        return None
    norm = "".join(ch for ch in raw.lower() if ch.isalnum())
    for canon, aliases in table.items():
        for a in aliases:
            an = "".join(ch for ch in a.lower() if ch.isalnum())
            if an and (an in norm or norm in an):
                return canon
    return None
OUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "odds.json"


def _norm_name(name: str) -> str:
    """Normalize a player name for cross-source matching: strip accents, lowercase,
    drop punctuation and common suffixes. 'Tyler O'Neill' -> 'tyler oneill',
    'Ronald Acuña Jr.' -> 'ronald acuna'."""
    if not name:
        return ""
    # strip accents
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    n = n.lower().strip()
    # keep letters and spaces only (drop . ' - etc) FIRST, so "Jr." -> "jr" is catchable
    n = "".join(ch for ch in n if ch.isalpha() or ch == " ")
    n = " ".join(n.split())
    # then drop suffixes
    for suf in (" jr", " sr", " ii", " iii", " iv"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip()


def _american_to_prob(odds: int) -> float:
    o = float(odds)
    if o >= 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def _better_over(a: int, b: int) -> int:
    """Return the BETTER over price for a bettor = the one implying LOWER probability
    (longer payout). +600 is better than +500; -110 worse than +100."""
    return a if _american_to_prob(a) <= _american_to_prob(b) else b


def fetch_props(api_key: str, retries: int = 2) -> list:
    # NOTE: we deliberately do NOT send a bookmakers= filter. The API's book keys may not
    # match ours, and a wrong filter can silently drop everything. Pull all books for the
    # HR market and filter locally via _canon_book, where we can log what we actually see.
    q = urllib.parse.urlencode({
        "markets": MARKET,
        "apiKey": api_key,
    })
    url = f"{API_BASE}/sports/{SPORT}/props?{q}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "going-yard/1.0"})
            req.add_header("x-api-key", api_key)   # some tiers prefer header auth; harmless if ignored
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
            data = json.loads(raw)   # raises on non-JSON (e.g. an HTML error page)
            return data
        except urllib.error.HTTPError:
            raise   # 4xx/5xx handled by caller; don't retry auth/quota errors blindly
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))   # brief backoff on transient network/parse issues
                continue
            raise
    if last_err:
        raise last_err
    return []


def _is_hr_prop(row: dict) -> bool:
    """True if this row is the standard 'does he hit a home run tonight' bet.

    This market shows up TWO ways depending on the book:
      1. market_key 'player_home_runs' with line 0.5  (most books)
      2. market_key 'player_home_runs_alt' with line 1.0 and a '1 Or More' milestone label
         (DraftKings & Fanatics only publish it this way on this tier)
    Both are the SAME outcome — 'over 0.5 HR' == '1 or more HR'. We accept both and reject
    everything else: the 2+ milestone (line 2.0), null-line 'to hit 2+' rows, and the
    combined/either-batter markets.
    """
    mk = row.get("market_key")
    line = row.get("line")
    mkt = (row.get("market") or "").lower()
    if mk == "player_home_runs" and line == 0.5:
        return True
    if mk == "player_home_runs_alt" and line == 1.0 and "1 or more" in mkt:
        return True
    return False


def build_odds(rows: list) -> dict:
    """Collapse raw prop rows into {normalized_name: {best, books:{...}, ...}}.

    Two tiers: PRIMARY books (DK, Fanatics) set the headline 'best' price. FanDuel is a
    FALLBACK — its price is only used as 'best' for a hitter that no primary book priced,
    and such entries carry fallback=True so the app can flag them as reference prices.
    A single malformed row is skipped, never fatal."""
    by_player: dict[str, dict] = {}
    fallback: dict[str, dict] = {}       # normalized_name -> {book: price} from FanDuel
    books_seen: dict[str, int] = {}
    hr_rows = 0
    for row in rows:
        try:
            if not isinstance(row, dict):
                continue
            if not _is_hr_prop(row):
                continue
            hr_rows += 1
            raw_book = row.get("bookmaker") or ""
            books_seen[raw_book] = books_seen.get(raw_book, 0) + 1

            over = row.get("over_price")
            if over is None:
                continue
            try:
                over = int(over)
            except (TypeError, ValueError):
                continue
            if over == 0 or over < -100000 or over > 100000:
                continue

            name = row.get("player") or ""
            if "(" in name:
                name = name.split("(")[0].strip()
            key = _norm_name(name)
            if not key:
                continue

            book = _canon_book(raw_book)
            if book is not None:
                entry = by_player.setdefault(key, {
                    "name": name,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "line": 0.5,
                    "books": {},
                    "best": None,
                    "best_book": None,
                    "fallback": False,
                    "last_update": row.get("last_update"),
                })
                entry["books"][book] = over
                if entry["best"] is None:
                    entry["best"], entry["best_book"] = over, book
                else:
                    nb = _better_over(entry["best"], over)
                    if nb != entry["best"]:
                        entry["best"], entry["best_book"] = nb, book
                continue

            fb = _canon_fallback(raw_book)
            if fb is not None:
                slot = fallback.setdefault(key, {"name": name,
                                                 "home_team": row.get("home_team"),
                                                 "away_team": row.get("away_team"),
                                                 "books": {},
                                                 "last_update": row.get("last_update")})
                # keep the better FanDuel price if it somehow appears twice
                prev = slot["books"].get(fb)
                slot["books"][fb] = over if prev is None else _better_over(prev, over)
        except Exception:
            continue

    # apply FanDuel fallback ONLY to hitters no primary book priced
    filled = 0
    for key, slot in fallback.items():
        if key in by_player:
            continue   # a primary book already covers him — don't override
        fb_book = next(iter(slot["books"]), None)
        if fb_book is None:
            continue
        px = slot["books"][fb_book]
        by_player[key] = {
            "name": slot["name"],
            "home_team": slot.get("home_team"),
            "away_team": slot.get("away_team"),
            "line": 0.5,
            "books": {fb_book: px},
            "best": px,
            "best_book": fb_book,
            "fallback": True,           # flag: this is a reference price, not a primary book
            "last_update": slot.get("last_update"),
        }
        filled += 1

    if not by_player and books_seen:
        top = sorted(books_seen.items(), key=lambda kv: -kv[1])[:15]
        print(f"[odds] {hr_rows} HR rows across books: {top}", file=sys.stderr)
        print(f"[odds] none matched {BOOKS+FALLBACK_BOOKS}. Add unmatched keys to the alias "
              f"tables.", file=sys.stderr)
    else:
        prim = sum(1 for v in by_player.values() if not v.get("fallback"))
        print(f"[odds] {prim} hitters on primary books, {filled} filled from FanDuel fallback",
              file=sys.stderr)
    return by_player


def _load_existing() -> dict | None:
    """Read the current odds.json so a failed run can preserve the last good prices
    instead of overwriting them with an empty file."""
    try:
        with open(OUT_PATH) as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("prices"), dict):
            return d
    except Exception:
        pass
    return None


def _write_error(reason: str) -> None:
    """On failure, keep the last good prices but mark them stale, rather than blanking
    the file. The frontend can then still auto-fill from the last good pull and show a
    'prices may be stale' note instead of losing everything on one API hiccup."""
    existing = _load_existing()
    if existing and existing.get("prices"):
        existing["error"] = reason
        existing["stale"] = True
        # keep existing 'updated' (when the good data was pulled); note the failed attempt
        existing["last_attempt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write(existing)
        print(f"[odds] {reason}: kept {len(existing['prices'])} prices from last good pull "
              f"(marked stale)", file=sys.stderr)
    else:
        _write({"updated": None, "market": MARKET, "books": BOOKS,
                "count": 0, "prices": {}, "error": reason})
        print(f"[odds] {reason}: no prior good file to preserve; wrote empty", file=sys.stderr)


def main() -> int:
    api_key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not api_key:
        print("[odds] no PARLAY_API_KEY set", file=sys.stderr)
        _write_error("no_api_key")
        return 0
    try:
        rows = fetch_props(api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        print(f"[odds] HTTP {e.code}: {body}", file=sys.stderr)
        _write_error(f"http_{e.code}")
        return 0
    except Exception as e:
        print(f"[odds] fetch failed: {e}", file=sys.stderr)
        _write_error("fetch_failed")
        return 0

    if not isinstance(rows, list):
        print(f"[odds] unexpected response shape: {type(rows)}", file=sys.stderr)
        _write_error("bad_shape")
        return 0

    try:
        prices = build_odds(rows)
    except Exception as e:
        print(f"[odds] parse failed: {e}", file=sys.stderr)
        _write_error("parse_failed")
        return 0

    # Guard: if the API returned rows but we parsed ZERO usable prices, that's suspicious
    # (schema drift, or all books filtered out). Don't overwrite good prior data with an
    # empty set on a normal game day — but DO write empty if there genuinely were no rows
    # (legitimately no games/props posted yet).
    if not prices and rows:
        print(f"[odds] {len(rows)} rows returned but 0 usable prices parsed — "
              f"possible schema change; preserving last good file", file=sys.stderr)
        _write_error("zero_parsed")
        return 0

    # capture the slate date from the rows so the frontend can verify these odds match
    # today's board before auto-filling (stale odds from yesterday must not fill in today)
    slate = None
    for row in rows:
        if isinstance(row, dict) and row.get("game_date"):
            slate = row["game_date"]
            break

    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slate_date": slate,
        "market": MARKET,
        "books": BOOKS,
        "count": len(prices),
        "prices": prices,
    }
    _write(payload)
    both = sum(1 for v in prices.values() if len(v["books"]) == 2)
    print(f"[odds] wrote {len(prices)} hitters with HR prices ({both} priced by both books)")
    return 0


def _write(payload: dict) -> None:
    """Atomic write: serialize to a temp file, then rename. A crash mid-write can never
    leave a half-written odds.json that the frontend would fail to parse — the rename is
    atomic, so readers see either the old file or the complete new one, never a partial."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, OUT_PATH)   # atomic on POSIX
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
