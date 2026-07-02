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

# A fixed league-average batted-ball reference per handedness, used only to measure the
# physics-implied GEOMETRY factor of a park (so we can rescale it to match Savant's
# empirical factor). R hitters pull to LF (negative spray), L hitters to RF (positive).
def _ref_sample(hand, n=900, seed=17):
    rng = np.random.default_rng(seed + (1 if hand == "L" else 0))
    ev = np.clip(rng.normal(88, 13, n), 40, 118)
    la = rng.normal(12, 24, n)
    center = 8.0 if hand == "L" else -8.0
    spray = rng.normal(center, 20, n)
    return ev, la, spray

_REF = {"R": _ref_sample("R"), "L": _ref_sample("L")}
_GEOM_CACHE = {}


def geometry_factor(venue, hand):
    """
    Physics-implied PERSISTENT park factor vs neutral, for a league-average hitter of this
    hand, at the park's real elevation but average weather (70F, calm). It captures what
    physics CAN see — wall geometry + altitude — so that anchoring to Savant corrects only
    the residual physics can't see (foul territory, marine layer, batter's eye, microclimate)
    and never double-counts altitude or today's wind. Cached per (venue, hand).
    """
    key = (venue, hand if hand in ("R", "L") else "R")
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    ev, la, spray = _REF[key[1]]
    _, _, elev = PG.park_coords(venue)
    rho_park = T.air_density(70.0, elev, None)         # real altitude, average weather
    rho_neut = T.air_density(70.0, 0.0, None)          # fixed sea-level neutral
    calm = np.zeros(3)
    park = _clears(ev, la, spray, venue, rho_park, calm).sum()
    neut = _clears(ev, la, spray, NEUTRAL_VENUE, rho_neut, calm).sum()
    f = float(park) / max(float(neut), 1.0)
    _GEOM_CACHE[key] = f
    return f


def savant_anchor(venue, hand, savant_factor):
    """
    Correction c so that physics geometry rescaled by c matches Savant's empirical factor.
    c = savant_factor / physics_geometry_factor, clamped so a single park can't swing wildly.
    Returns 1.0 (no anchor) when Savant has no number for the park.
    """
    if not savant_factor or savant_factor <= 0:
        return 1.0
    g = geometry_factor(venue, hand)
    if g <= 0:
        return 1.0
    return float(np.clip(savant_factor / g, 0.5, 2.0))


_WIND_SENS = {}


def set_wind_sens(d):
    """Install the learned per-park wind sensitivities (build calls this once)."""
    global _WIND_SENS
    _WIND_SENS = d or {}


def _conditions(venue, iso_time):
    """Return (air_density, field_wind_vec, used_weather, wxmeta) for a park.
    Applies the predicted roof state (closed retractable -> indoor air; canopy ->
    outdoor air, no wind) and the park's learned wind sensitivity."""
    lat, lon, elev = PG.park_coords(venue)
    wx = W.get_weather(lat, lon, iso_time, venue=venue)
    roof = W.roof_call(venue, wx)
    if wx and wx.get("temp_f") is not None:
        temp, rh = wx["temp_f"], wx.get("rh_pct")
        if roof in ("dome", "closed"):
            temp, rh = 72.0, 45.0
            windv = np.zeros(3)
        elif roof == "canopy":
            windv = np.zeros(3)                     # rain/wind shielded, outdoor air
        else:
            sens = _WIND_SENS.get(venue, 1.0) if isinstance(_WIND_SENS, dict) else 1.0
            windv = PG.field_wind_vector(wx.get("wind_mph"), wx.get("wind_from_deg"), venue) * sens
        rho = T.air_density(temp, elev, wx.get("pressure_pa") if roof not in ("dome", "closed") else None,
                            rh_pct=rh)
        meta = {"temp_f": temp, "wind_mph": (0.0 if roof in ("dome", "closed", "canopy") else wx.get("wind_mph")),
                "rh_pct": rh, "precip_prob": wx.get("precip_prob"), "roof": roof}
        return rho, windv, True, meta
    # no weather -> park-only at a mild default temperature
    return T.air_density(70.0, elev, None), np.zeros(3), False, {"temp_f": None, "wind_mph": None,
                                                                 "rh_pct": None, "precip_prob": None,
                                                                 "roof": roof}


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
    rho_p, wind_p, used, wxm = _conditions(venue, iso_time)
    hr_park = _clears(ev, la, spray, venue, rho_p, wind_p)
    rho_n = T.air_density(70.0, 0.0, None)
    hr_neut = _clears(ev, la, spray, NEUTRAL_VENUE, rho_n, np.zeros(3))
    meta = {"weather": used,
            "temp_f": round(wxm["temp_f"]) if wxm.get("temp_f") is not None else None,
            "wind_mph": round(wxm["wind_mph"]) if wxm.get("wind_mph") is not None else None,
            "rh_pct": round(wxm["rh_pct"]) if wxm.get("rh_pct") is not None else None,
            "precip_prob": round(wxm["precip_prob"]) if wxm.get("precip_prob") is not None else None,
            "roof": wxm.get("roof"),
            "venue": venue, "indoor": (wxm.get("roof") in ("dome", "closed"))}
    return hr_park, hr_neut, meta


def aggregate_hitter(hr_park_slice, hr_neutral_slice, meta, anchor=1.0, savant_factor=None):
    """Turn a hitter's slice of the game arrays into the park_hr output dict.

    `anchor` rescales the physics park expectation onto Savant's empirical park factor
    (1.0 = no Savant number available, physics only). `savant_factor` is stored for display.
    """
    n = len(hr_park_slice)
    if n < MIN_BALLS:
        return None
    park_exp = float(hr_park_slice.sum()) * anchor
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
        "savant_pf": round(savant_factor, 2) if savant_factor else None,
        "anchored": anchor != 1.0,
    }
