"""
Game-time weather for the park HR model, via Open-Meteo (free, no API key).

get_weather(lat, lon, iso_time) -> {temp_f, wind_mph, wind_from_deg, pressure_pa,
rh_pct, precip_prob, precip_mm} averaged over the game window (first pitch + 3 hours,
wind vector-averaged) so shifting conditions during the game are priced in
or None if anything fails (no key, network hiccup, bad coords). The caller treats
None as "neutral conditions", so a weather outage degrades gracefully to a
no-weather park-only model rather than breaking the build.

Results are cached per (lat, lon, hour) so every hitter in the same game reuses one
fetch. Domed/climate-controlled parks short-circuit to neutral indoor conditions.
"""
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

_CACHE = {}

# fixed-roof / climate-controlled: always ~72F, calm, regardless of outside weather.
# (Retractable-roof parks are treated as outdoor — most play open in summer — which is
#  an approximation when the roof is shut; flagged in the model output.)
INDOOR = {"Tropicana Field"}

# retractable-roof parks: roof state is PREDICTED from the forecast (closed on extreme
# heat/cold or real rain risk). T-Mobile's canopy is special: it blocks rain and wind
# but the air stays outdoor, so a "closed" call there kills wind and keeps temperature.
RETRACTABLE = {"Chase Field", "Daikin Park", "Minute Maid Park", "loanDepot park",
               "American Family Field", "Globe Life Field", "Rogers Centre"}
CANOPY = {"T-Mobile Park"}


def _canon(venue):
    try:
        from etl import park_geometry as _PG
        return _PG.canonical(venue) or venue
    except Exception:
        return venue


def roof_call(venue, wx):
    """Predicted roof state: 'dome' | 'closed' | 'canopy' | 'open' | None (open-air park).
    Heuristic: retractables close on temp >= 96F, <= 54F, or precip risk >= 45%."""
    venue = _canon(venue)
    if venue in INDOOR:
        return "dome"
    shut = False
    if wx:
        t = wx.get("temp_f"); pp = wx.get("precip_prob")
        shut = (t is not None and (t >= 96 or t <= 54)) or (pp is not None and pp >= 45)
    if venue in RETRACTABLE:
        return "closed" if shut else "open"
    if venue in CANOPY:
        return "canopy" if shut else "open"
    return None

NEUTRAL_INDOOR = {"temp_f": 72.0, "wind_mph": 0.0, "wind_from_deg": 0.0, "pressure_pa": 101325.0,
                  "rh_pct": 45.0, "precip_prob": 0.0, "precip_mm": 0.0}


def _hour_key(iso_time):
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H")
    except Exception:
        return None


def get_weather(lat, lon, iso_time, venue=None, timeout=8):
    if venue in INDOOR:
        return dict(NEUTRAL_INDOOR)
    if not lat and not lon:
        return None
    hk = _hour_key(iso_time)
    if hk is None:
        return None
    ck = (round(lat, 3), round(lon, 3), hk)
    if ck in _CACHE:
        return _CACHE[ck]
    try:
        day = hk.split("T")[0]
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation_probability,precipitation",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "start_date": day, "end_date": day, "timezone": "UTC",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        data = None
        for _try in (1, 2):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception:
                if _try == 2:
                    raise
                time.sleep(1.5)
        out = _parse(data, hk)
        _CACHE[ck] = out
        return out
    except Exception:
        return None


def _parse(data, hour_key, window=3):
    """Average the game window (first-pitch hour + the next `window`-1 hours) out of an
    Open-Meteo hourly payload. Wind is vector-averaged so direction shifts blend
    correctly; precip probability takes the window MAX (rain any hour disrupts)."""
    import math
    h = data.get("hourly", {})
    times = h.get("time", [])
    idx = None
    for i, t in enumerate(times):
        if t.startswith(hour_key):
            idx = i
            break
    if idx is None:
        return None
    idxs = [i for i in range(idx, min(idx + window, len(times)))]
    def vals(k):
        arr = h.get(k) or []
        return [arr[i] for i in idxs if i < len(arr) and arr[i] is not None]
    def avg(k):
        v = vals(k)
        return sum(v) / len(v) if v else None
    temp = avg("temperature_2m")
    rh = avg("relative_humidity_2m")
    psurf = avg("surface_pressure")   # hPa
    pp = vals("precipitation_probability")
    pmm = vals("precipitation")
    ws = vals("wind_speed_10m"); wd = vals("wind_direction_10m")
    wind_mph = wind_from = None
    if ws and wd and len(ws) == len(wd):
        u = sum(s_ * math.sin(math.radians(d_)) for s_, d_ in zip(ws, wd)) / len(ws)
        v = sum(s_ * math.cos(math.radians(d_)) for s_, d_ in zip(ws, wd)) / len(ws)
        wind_mph = math.hypot(u, v)
        wind_from = math.degrees(math.atan2(u, v)) % 360.0
    return {
        "temp_f": float(temp) if temp is not None else None,
        "rh_pct": float(rh) if rh is not None else None,
        "wind_mph": float(wind_mph) if wind_mph is not None else None,
        "wind_from_deg": float(wind_from) if wind_from is not None else None,
        "pressure_pa": float(psurf) * 100.0 if psurf is not None else None,
        "precip_prob": float(max(pp)) if pp else None,
        "precip_mm": float(sum(pmm)) if pmm else None,
    }
