"""
Park geometry for the trajectory HR model.

For each venue we store the outfield wall as control points across the spray arc
(LF line -> LF gap -> CF -> RF gap -> RF line), each with a distance (ft) and a
wall height (ft), plus the stadium's latitude/longitude/elevation and field
orientation (compass bearing from home plate to dead center). Distance and height
are linearly interpolated by spray angle. Anything unlisted falls back to a neutral
park so the build never breaks on a venue rename or a new stadium.

Spray convention matches the rest of the ETL: 0 = CF, negative = LF (3B side),
positive = RF (1B side). Foul poles at +/-45 deg.

Dimensions are real published park dimensions; orientations are approximate compass
bearings (good enough to get wind direction right within a sector). Relocation
venues in use for 2025-26 (Sacramento for the A's, Steinbrenner Field for the Rays)
are included.
"""
import numpy as np

# control spray angles (deg): LF line, LF gap, CF, RF gap, RF line
_ANG = np.array([-45.0, -27.0, 0.0, 27.0, 45.0])

# name -> (d_lf,d_lfg,d_cf,d_rfg,d_rf, h_lf,h_lfg,h_cf,h_rfg,h_rf, lat, lon, cf_bearing, elev_m)
# cf_bearing = compass azimuth (deg from true N, clockwise) of the home-plate -> dead-center
# line. This is what decides whether a given wind blows OUT, IN, or across. It only modulates
# the weather term; the park's overall HR level is anchored to Baseball Savant's park factors
# (see park_factors.py), so an orientation that's a little off can't make the park factor wrong.
PARK_GEO = {
    "Coors Field":             (347,390,415,375,350,  8, 8, 8,14, 8, 39.7559,-104.9942,  12, 1580),
    "Fenway Park":             (310,379,420,380,302, 37,17, 8, 5, 3, 42.3467, -71.0972,  45,    6),
    "Yankee Stadium":          (318,399,408,385,314,  8, 8, 8, 8, 8, 40.8296, -73.9262,  30,   16),
    "Wrigley Field":           (355,368,400,368,353, 11,11,11,11,11, 41.9484, -87.6553,  28,  200),
    "Dodger Stadium":          (330,385,395,385,330,  8, 8, 8, 8, 8, 34.0739,-118.2400,  25,   80),
    "Oracle Park":             (339,364,391,415,309,  8, 8, 8, 8,24, 37.7786,-122.3893,  65,    0),
    "Petco Park":              (334,367,396,391,322,  8, 8, 8, 8, 8, 32.7073,-117.1566,  38,   20),
    "Citizens Bank Park":      (329,374,401,369,330, 12, 8, 8, 6,13, 39.9061, -75.1665,  15,   20),
    "Great American Ball Park":(328,379,404,370,325, 12, 8, 8, 8, 8, 39.0975, -84.5067, 120,  150),
    "Globe Life Field":        (329,372,407,374,326, 14, 8, 8, 8, 8, 32.7473, -97.0837,  75,  170),
    "Truist Park":             (335,385,400,375,325,  8, 8, 8, 8,16, 33.8907, -84.4677,  25,  320),
    "Chase Field":             (330,374,407,374,334,  8, 8,25, 8, 8, 33.4455,-112.0667,  65,  340),
    "Oriole Park at Camden Yards":(333,364,410,373,318, 7, 7, 7, 7,25, 39.2839,-76.6217, 32,   10),
    "Camden Yards":            (333,364,410,373,318,  7, 7, 7, 7,25, 39.2839, -76.6217,  32,   10),
    "Rogers Centre":           (328,375,400,375,328, 10,10,10,10,10, 43.6414, -79.3894,   0,   90),
    "American Family Field":   (342,371,400,374,345,  8, 8, 8, 8, 8, 43.0280, -87.9712, 130,  200),
    "Nationals Park":          (336,377,402,370,335,  8,14, 8,14,14, 38.8730, -77.0074,  30,   10),
    "Citi Field":              (335,379,408,383,330,  8, 8, 8, 8, 8, 40.7571, -73.8458,  30,   10),
    "loanDepot park":          (344,386,400,387,335,  8, 8, 8, 8, 8, 25.7781, -80.2197,  40,    2),
    "T-Mobile Park":           (331,378,401,381,326,  8, 8, 8, 8, 8, 47.5914,-122.3325,  60,    5),
    "Kauffman Stadium":        (330,387,410,387,330,  8, 8, 8, 8, 8, 39.0517, -94.4803,  45,  230),
    "Comerica Park":           (342,370,412,365,330,  8, 8, 8, 8, 8, 42.3390, -83.0485, 150,  180),
    "Progressive Field":       (325,370,405,375,325, 19, 8, 8, 8, 8, 41.4962, -81.6852,   0,  200),
    "Rate Field":              (330,377,400,372,335,  8, 8, 8, 8, 8, 41.8299, -87.6338, 130,  180),
    "Guaranteed Rate Field":   (330,377,400,372,335,  8, 8, 8, 8, 8, 41.8299, -87.6338, 130,  180),
    "Target Field":            (339,377,404,367,328,  8, 8, 8,23, 8, 44.9817, -93.2776,  90,  250),
    "Busch Stadium":           (336,375,400,375,335,  8, 8, 8, 8, 8, 38.6226, -90.1928,  60,  140),
    "PNC Park":                (325,389,399,375,320,  6, 8, 8, 8,21, 40.4469, -80.0057, 120,  220),
    "Angel Stadium":           (330,387,396,370,330,  8, 8, 8,18,18, 33.8003,-117.8827,  45,   50),
    "Daikin Park":             (315,366,409,373,326, 19, 8, 8, 8, 7, 29.7572, -95.3555, 345,   10),
    "Minute Maid Park":        (315,366,409,373,326, 19, 8, 8, 8, 7, 29.7572, -95.3555, 345,   10),
    "Sutter Health Park":      (330,375,403,375,325,  8, 8, 8, 8, 8, 38.5802,-121.5141,  60,   10),
    "George M. Steinbrenner Field":(318,399,408,385,314, 8,8,8,8,8, 27.9802,-82.5067,    30,    5),
    "Tropicana Field":         (315,370,404,370,322,  8, 8, 8, 8, 8, 27.7683, -82.6534,   0,    3),
}

# neutral league-average park for unknown venues
_NEUTRAL = (332,376,402,376,330, 8,8,8,8,8, 0.0, 0.0, 0, 100)


def _row(venue):
    return PARK_GEO.get(venue, _NEUTRAL)


def wall_at(venue, spray_deg):
    """(distance_ft, height_ft) of the outfield wall at a given spray angle (array-safe)."""
    r = _row(venue)
    dist_pts = np.array(r[0:5], dtype=float)
    h_pts = np.array(r[5:10], dtype=float)
    s = np.clip(np.asarray(spray_deg, dtype=float), -45, 45)
    d = np.interp(s, _ANG, dist_pts)
    h = np.interp(s, _ANG, h_pts)
    return d, h


def park_coords(venue):
    """(lat, lon, elev_m). lat/lon are 0,0 for unknown venues (weather will no-op)."""
    r = _row(venue)
    return r[10], r[11], r[13]


def cf_bearing(venue):
    """Compass bearing (deg from North) from home plate to dead center field."""
    return _row(venue)[12]


def known(venue):
    return venue in PARK_GEO


def field_wind_vector(wind_mph, wind_from_deg, venue):
    """
    Convert meteorological wind (speed + direction FROM which it blows, deg from N)
    into a FIELD-frame velocity vector (m/s): +x out to CF, +y to RF (1B side).

    A wind blowing FROM the SW (toward the NE) at a park whose CF also faces NE is
    blowing out to center -> large +x. Returns (3,) array.
    """
    from etl.trajectory import MPH
    if wind_mph is None or wind_from_deg is None:
        return np.zeros(3)
    blow_to = (wind_from_deg + 180.0) % 360.0          # direction wind blows toward
    cf = cf_bearing(venue)
    rel = np.radians(blow_to - cf)                     # angle of wind vs the CF axis
    spd = wind_mph * MPH
    # +x along CF axis (out), +y to the right of that axis = RF side
    return np.array([spd * np.cos(rel), spd * np.sin(rel), 0.0])
