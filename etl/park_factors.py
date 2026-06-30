"""
Baseball Savant park factors — pulled automatically, zero manual entry.

Savant publishes empirical HR park factors built from actual batted-ball outcomes,
on a rolling multi-year window and split by batter handedness. Because it's empirical
and continuously refreshed, it (a) reflects each park's CURRENT configuration as walls
move year to year, and (b) needs no hand-entered dimensions. We use it as the source of
truth for each park's HR-friendliness, and anchor the physics/weather model to it so the
per-hitter spray and live-weather nuance ride on top of a number that's always current.

Output: {venue_name: {"all": mult, "R": mult, "L": mult}} where mult is a multiplier
around 1.00 (1.18 = 18% more HR than league average). Falls back to the last cached
pull, then to neutral (1.0), so the board never breaks.

NOTE: this runs in GitHub Actions (open network). It can't be exercised from the build
sandbox, so the Savant endpoint/columns below are parsed defensively and the first live
run logs exactly what it pulled.
"""
import csv
import io
import json
import os
import time
import urllib.request

BASE = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
CACHE_MAX_AGE_DAYS = 7


def _normalize(name):
    return "".join(c for c in (name or "").lower() if c.isalnum())


def _fetch_csv(year, bat_side="", rolling=3, timeout=20):
    """Pull the park-factors CSV for one handedness ('' = all, 'R', 'L')."""
    params = (f"?type=year&year={year}&batSide={bat_side}&stat=index_hr"
              f"&condition=All&rolling={rolling}&sortColumn=hr&sortDir=desc&csv=true")
    req = urllib.request.Request(BASE + params, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _parse(text):
    """Flexible parse: find the venue-name column and the HR-factor column by header."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return {}
    headers = list(rows[0].keys())
    low = {h: h.lower() for h in headers}
    venue_col = next((h for h in headers if low[h] in ("venue_name", "name", "venue")), None) \
        or next((h for h in headers if "venue" in low[h] or "name" in low[h]), None)
    hr_col = next((h for h in headers if low[h] in ("index_hr", "hr_factor", "hr")), None) \
        or next((h for h in headers if "hr" in low[h]), None)
    if not venue_col or not hr_col:
        return {}
    out = {}
    for row in rows:
        v = (row.get(venue_col) or "").strip()
        raw = (row.get(hr_col) or "").strip()
        if not v or not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        # Savant publishes an index where 100 = average; normalize to a multiplier.
        mult = val / 100.0 if val > 5 else val
        out[v] = round(mult, 3)
    return out


def fetch_park_factors(year, rolling=3):
    """Pull all/R/L HR park factors. Returns {venue: {'all','R','L'}} or {} on failure."""
    try:
        all_pf = _parse(_fetch_csv(year, "", rolling))
        if not all_pf:
            return {}
        r_pf = _parse(_fetch_csv(year, "R", rolling))
        l_pf = _parse(_fetch_csv(year, "L", rolling))
        merged = {}
        for v, m in all_pf.items():
            merged[v] = {"all": m, "R": r_pf.get(v, m), "L": l_pf.get(v, m)}
        print(f"[park_factors] pulled {len(merged)} venues from Savant (year={year}, rolling={rolling})")
        return merged
    except Exception as e:
        print(f"[park_factors] Savant fetch failed: {e}")
        return {}


def load_park_factors(cache_path, year, rolling=3):
    """
    Fresh cache -> use it. Otherwise fetch from Savant and rewrite the cache. If the fetch
    fails, fall back to whatever cache exists (even if stale). Returns the venue dict.
    """
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    fresh = (cache.get("year") == year
             and (time.time() - cache.get("fetched", 0)) < CACHE_MAX_AGE_DAYS * 86400
             and cache.get("venues"))
    if fresh:
        return cache["venues"]
    venues = fetch_park_factors(year, rolling)
    if venues:
        try:
            json.dump({"year": year, "rolling": rolling, "fetched": time.time(),
                       "venues": venues}, open(cache_path, "w"))
        except Exception:
            pass
        return venues
    return cache.get("venues", {})   # stale cache beats nothing


def factor_for(venues, venue_name, hand="all"):
    """HR multiplier for a venue + batter hand. 1.0 if unknown (neutral / physics-only)."""
    if not venues:
        return 1.0
    rec = venues.get(venue_name)
    if rec is None:
        target = _normalize(venue_name)
        for k, v in venues.items():
            if _normalize(k) == target:
                rec = v
                break
    if rec is None:
        return 1.0
    h = hand if hand in ("R", "L") else "all"
    return rec.get(h, rec.get("all", 1.0))
