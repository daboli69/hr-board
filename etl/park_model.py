"""
Per-hitter park + weather HR model.

Runs a hitter's ACTUAL batted balls (exit velocity, launch angle, spray) through the
trajectory engine under (a) today's park geometry + weather and (b) a neutral baseline
park at 70F, calm, sea level. The fraction that clear the fence in each gives a
hitter-specific park factor — capturing the thing a flat park factor can't: a dead-pull
hitter benefits more from a short porch on his side, and wind/temperature scale with how
often he already lives near the wall.

Output per hitter:
  exp100     : expected HR per 100 batted balls in today's park+weather
  neutral100 : same in a neutral park (his baseline)
  boost      : % more/fewer HR than neutral (the "+62%" number) for these conditions
  score      : 0-100 for ranking/coloring (maps exp100)
  weather    : whether live weather was applied (vs park-only)
  temp_f, wind_mph : conditions used (for display)

This is SECONDARY data — its own lens. It never touches the Heat score or the grader.
"""
import numpy as np
from etl import trajectory as T, park_geometry as PG, weather as W

MIN_BALLS = 20          # need a real sample to model a hitter
NEUTRAL_VENUE = "__neutral__"   # not in PARK_GEO -> falls back to neutral geometry


def _conditions(venue, iso_time):
    """Return (air_density, field_wind_vec, used_weather, temp_f, wind_mph) for a park."""
    lat, lon, elev = PG.park_coords(venue)
    wx = W.get_weather(lat, lon, iso_time, venue=venue)
    if wx and wx.get("temp_f") is not None:
        temp = wx["temp_f"]
        rho = T.air_density(temp, elev, wx.get("pressure_pa"))
        windv = PG.field_wind_vector(wx.get("wind_mph"), wx.get("wind_from_deg"), venue)
        return rho, windv, True, temp, wx.get("wind_mph")
    # no weather -> park-only at a mild default temperature
    return T.air_density(70.0, elev, None), np.zeros(3), False, None, None


def _clears(ev, la, spray, venue, rho, windv):
    """Boolean per-ball: does it clear the fence at this park under these conditions?"""
    wd, wh = PG.wall_at(venue, spray)
    dist, zw = T.carry_batch(ev, la, spray, rho, windv, wd)
    return (dist >= wd) & (np.nan_to_num(zw, nan=-1.0) >= wh)


def evaluate_game(ev, la, spray, venue, iso_time):
    """
    Vectorized over ALL batted balls in a game (concatenated across its hitters).
    Returns (hr_park, hr_neutral, meta) where the first two are boolean arrays aligned
    to the inputs, and meta carries the conditions used for display.
    """
    ev = np.asarray(ev, float); la = np.asarray(la, float); spray = np.asarray(spray, float)
    rho_p, wind_p, used, temp, wind_mph = _conditions(venue, iso_time)
    hr_park = _clears(ev, la, spray, venue, rho_p, wind_p)
    rho_n = T.air_density(70.0, 0.0, None)
    hr_neut = _clears(ev, la, spray, NEUTRAL_VENUE, rho_n, np.zeros(3))
    meta = {"weather": used, "temp_f": round(temp) if temp is not None else None,
            "wind_mph": round(wind_mph) if wind_mph is not None else None,
            "venue": venue, "indoor": venue in W.INDOOR}
    return hr_park, hr_neut, meta


def aggregate_hitter(hr_park_slice, hr_neutral_slice, meta):
    """Turn a hitter's slice of the game arrays into the park_hr output dict."""
    n = len(hr_park_slice)
    if n < MIN_BALLS:
        return None
    park_exp = float(hr_park_slice.sum())
    neut_exp = float(hr_neutral_slice.sum())
    exp100 = 100.0 * park_exp / n
    neutral100 = 100.0 * neut_exp / n
    boost = 100.0 * (park_exp - neut_exp) / max(neut_exp, 2.0)
    return {
        "exp100": round(exp100, 1),
        "neutral100": round(neutral100, 1),
        "boost": int(round(float(np.clip(boost, -60, 150)))),
        "score": int(round(float(np.clip(exp100 / 12.0 * 100, 0, 100)))),
        "n": int(n),
        "weather": meta["weather"],
        "temp_f": meta["temp_f"],
        "wind_mph": meta["wind_mph"],
        "indoor": meta["indoor"],
    }
