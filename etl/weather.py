"""
Game-time weather for the park HR model, via Open-Meteo (free, no API key).

get_weather(lat, lon, iso_time) -> {temp_f, wind_mph, wind_from_deg, pressure_pa}
or None if anything fails (no key, network hiccup, bad coords). The caller treats
None as "neutral conditions", so a weather outage degrades gracefully to a
no-weather park-only model rather than breaking the build.

Results are cached per (lat, lon, hour) so every hitter in the same game reuses one
fetch. Domed/climate-controlled parks short-circuit to neutral indoor conditions.
"""
import json
import urllib.request
import urllib.parse
from datetime import datetime

_CACHE = {}

# fixed-roof / climate-controlled: always ~72F, calm, regardless of outside weather.
# (Retractable-roof parks are treated as outdoor — most play open in summer — which is
#  an approximation when the roof is shut; flagged in the model output.)
INDOOR = {"Tropicana Field"}

NEUTRAL_INDOOR = {"temp_f": 72.0, "wind_mph": 0.0, "wind_from_deg": 0.0, "pressure_pa": 101325.0}


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
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,surface_pressure",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "start_date": day, "end_date": day, "timezone": "UTC",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        out = _parse(data, hk)
        _CACHE[ck] = out
        return out
    except Exception:
        return None


def _parse(data, hour_key):
    """Pull the row matching the game hour out of an Open-Meteo hourly payload."""
    h = data.get("hourly", {})
    times = h.get("time", [])
    target = hour_key + ":00"
    idx = None
    for i, t in enumerate(times):
        if t.startswith(hour_key):
            idx = i
            break
    if idx is None:
        return None
    def g(k):
        arr = h.get(k) or []
        return arr[idx] if idx < len(arr) else None
    temp = g("temperature_2m")
    wspd = g("wind_speed_10m")
    wdir = g("wind_direction_10m")
    psurf = g("surface_pressure")   # hPa
    return {
        "temp_f": float(temp) if temp is not None else None,
        "wind_mph": float(wspd) if wspd is not None else None,
        "wind_from_deg": float(wdir) if wdir is not None else None,
        "pressure_pa": float(psurf) * 100.0 if psurf is not None else None,
    }
