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

# Additional prop markets fetched alongside HR — all in ONE API call (comma-separated markets
# = still 3 credits, no extra cost). Each maps to a props sub-view in the app. These are
# two-sided over/under markets with real lines, unlike HR (which is a milestone yes/no).
PROP_MARKETS = {
    "hits":  "player_hits",              # 1+ hits, line 0.5
    "hrr":   "player_hits_runs_rbis",    # hits+runs+rbis, line 1.5 typically
    "pk":    "player_strikeouts",        # pitcher strikeouts O/U, real lines (4.5, 5.5...)
}
ALL_MARKETS = [MARKET] + list(PROP_MARKETS.values())
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


def _fetch(url: str, api_key: str, retries: int = 2):
    """GET a parlay-api URL with retry/backoff, returning parsed JSON."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "going-yard/1.0"})
            req.add_header("x-api-key", api_key)
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if last_err:
        raise last_err
    return []


def fetch_props(api_key: str, retries: int = 2) -> list:
    q = urllib.parse.urlencode({
        "markets": ",".join(ALL_MARKETS),   # HR + hits + hrr + pitcher Ks, one call = 3 credits
        "apiKey": api_key,
    })
    return _fetch(f"{API_BASE}/sports/{SPORT}/props?{q}", api_key, retries)


def fetch_game_lines(api_key: str, retries: int = 2) -> list:
    """Game lines (moneyline + totals) from the /odds endpoint. Costs markets x regions
    credits = 2 markets x 1 region = 2 credits. Returns TOA-format events with bookmakers."""
    q = urllib.parse.urlencode({
        "markets": "h2h,totals",   # moneyline + game total
        "regions": "us",
        "apiKey": api_key,
    })
    return _fetch(f"{API_BASE}/sports/{SPORT}/odds?{q}", api_key, retries)


def build_game_lines(events: list) -> dict:
    """Parse /odds TOA-format events into {normalized_matchup: {home, away, ml:{home,away},
    total:{line, over, under}, books}}. Line-shopped best price across DK/Fanatics, FanDuel
    fallback. The matchup key is a normalized 'away@home' so the frontend can join to its
    game_projections."""
    out = {}
    if not isinstance(events, list):
        return out
    for ev in events:
        try:
            if not isinstance(ev, dict):
                continue
            home = ev.get("home_team") or ""
            away = ev.get("away_team") or ""
            if not home or not away:
                continue
            key = f"{_norm_name(away)}@{_norm_name(home)}"
            slot = out.setdefault(key, {
                "home": home, "away": away,
                "ml": {"home": None, "away": None, "home_book": None, "away_book": None},
                "total": {"line": None, "over": None, "under": None, "over_book": None, "under_book": None},
                "commence_time": ev.get("commence_time"),
                "fallback_only": True,   # flipped False if any primary book prices it
            })
            for bm in ev.get("bookmakers", []):
                raw_book = bm.get("key") or bm.get("title") or ""
                book = _canon_book(raw_book)
                is_fallback = book is None
                if is_fallback:
                    book = _canon_fallback(raw_book)
                if book is None:
                    continue
                if not is_fallback:
                    slot["fallback_only"] = False
                for mk in bm.get("markets", []):
                    mkey = mk.get("key")
                    outcomes = mk.get("outcomes", [])
                    if mkey == "h2h":
                        for o in outcomes:
                            price = _safe_int(o.get("price"))
                            if price is None:
                                continue
                            side = "home" if o.get("name") == home else ("away" if o.get("name") == away else None)
                            if side is None:
                                continue
                            # best (longest) price for a bettor
                            cur = slot["ml"][side]
                            if cur is None or _better_over(cur, price) == price:
                                slot["ml"][side] = price
                                slot["ml"][side + "_book"] = book
                    elif mkey == "totals":
                        for o in outcomes:
                            price = _safe_int(o.get("price"))
                            line = o.get("point")
                            if price is None or line is None:
                                continue
                            name = (o.get("name") or "").lower()
                            side = "over" if "over" in name else ("under" if "under" in name else None)
                            if side is None:
                                continue
                            # anchor to the first line seen; keep best price at that line
                            if slot["total"]["line"] is None:
                                slot["total"]["line"] = line
                            if slot["total"]["line"] != line:
                                continue
                            cur = slot["total"][side]
                            if cur is None or _better_over(cur, price) == price:
                                slot["total"][side] = price
                                slot["total"][side + "_book"] = book
        except Exception:
            continue
    return out


def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


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


def build_prop_odds(rows: list) -> dict:
    """Build two-sided prop odds for hits / hrr / pitcher-K markets.

    Unlike HR (a yes/no milestone), these are over/under markets with a real line. For each
    (prop, player) we keep the book's line and BOTH over and under prices, best-priced across
    primary books (DK/Fanatics), with FanDuel as a labeled fallback. Structure:
      {prop_key: {normalized_name: {name, line, over, under, over_book, under_book,
                                     fallback, books:{book:{line,over,under}}}}}
    The frontend compares its model estimate to `line` and prices whichever side is +EV.
    """
    # market_key -> our prop key
    key_map = {v: k for k, v in PROP_MARKETS.items()}
    out = {k: {} for k in PROP_MARKETS}
    fb = {k: {} for k in PROP_MARKETS}   # FanDuel fallback per prop

    for row in rows:
        try:
            if not isinstance(row, dict):
                continue
            mk = row.get("market_key")
            prop = key_map.get(mk)
            if prop is None:
                continue
            line = row.get("line")
            if line is None:
                continue
            # Per-prop line gate. HITS must be the 0.5 line (1+ hits) — NOT 1.5 (2+ hits) or
            # 2.5. HRR standard market is the 1.5 line. Pitcher Ks keep their real varying line
            # (each pitcher has a different K total). This is the fix for "hits showing 2-hit
            # lines": we were accepting any line for player_hits.
            if prop == "hits" and line != 0.5:
                continue
            if prop == "hrr" and line != 1.5:
                continue
            if prop == "pk" and (line is None or line < 2.5):
                continue   # skip the 0.5 "1+ K" novelty; keep real O/U totals
            over = row.get("over_price")
            under = row.get("under_price")
            # need at least one priced side
            if over is None and under is None:
                continue
            try:
                over = int(over) if over is not None else None
                under = int(under) if under is not None else None
            except (TypeError, ValueError):
                continue
            # absurd-price guard
            for px in (over, under):
                if px is not None and (px == 0 or px < -100000 or px > 100000):
                    over = over if over != px else None
                    under = under if under != px else None
            name = row.get("player") or ""
            if "(" in name:
                name = name.split("(")[0].strip()
            key = _norm_name(name)
            if not key:
                continue

            raw_book = row.get("bookmaker") or ""
            book = _canon_book(raw_book)
            target = out if book is not None else None
            if target is None:
                fbk = _canon_fallback(raw_book)
                if fbk is None:
                    continue
                book = fbk
                target = fb

            slot = target[prop].setdefault(key, {
                "name": name, "line": line,
                "over": None, "under": None, "over_book": None, "under_book": None,
                "books": {},
                "home_team": row.get("home_team"), "away_team": row.get("away_team"),
                "last_update": row.get("last_update"),
            })
            # only compare prices at the SAME line (mixing lines would be apples/oranges)
            if slot["line"] != line and slot["over"] is not None:
                # a different line already recorded; prefer the more common (keep first seen)
                continue
            slot["line"] = line
            slot["books"][book] = {"line": line, "over": over, "under": under}
            # best over = longest (bettor-friendliest); best under = longest too
            if over is not None and (slot["over"] is None or _better_over(slot["over"], over) == over):
                slot["over"], slot["over_book"] = over, book
            if under is not None and (slot["under"] is None or _better_over(slot["under"], under) == under):
                slot["under"], slot["under_book"] = under, book
        except Exception:
            continue

    # apply FanDuel fallback for players no primary book priced
    for prop in PROP_MARKETS:
        for key, slot in fb[prop].items():
            if key in out[prop]:
                continue
            slot["fallback"] = True
            out[prop][key] = slot
    return out


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

    # build the two-sided prop markets (hits/hrr/pitcher-K) from the same fetched rows
    try:
        prop_odds = build_prop_odds(rows)
    except Exception as e:
        print(f"[odds] prop build failed (non-fatal): {e}", file=sys.stderr)
        prop_odds = {k: {} for k in PROP_MARKETS}

    # game lines (moneyline + totals) from the /odds endpoint — separate 2-credit call.
    # Non-fatal: if it fails, we keep the prop odds and just skip game lines.
    game_lines = {}
    try:
        gl_events = fetch_game_lines(api_key)
        game_lines = build_game_lines(gl_events)
    except Exception as e:
        print(f"[odds] game-line fetch failed (non-fatal): {e}", file=sys.stderr)

    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slate_date": slate,
        "market": MARKET,
        "books": BOOKS,
        "count": len(prices),
        "prices": prices,
        "props": prop_odds,
        "game_lines": game_lines,
    }
    _write(payload)
    for pk, pv in prop_odds.items():
        print(f"[odds] prop '{pk}': {len(pv)} players priced", file=sys.stderr)
    print(f"[odds] game lines: {len(game_lines)} games priced", file=sys.stderr)
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
