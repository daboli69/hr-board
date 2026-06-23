"""
Park HR factors by venue, split by batter handedness.

These move slowly (essentially once a year), so a static table is the right
call — no need to hit an API for them. 1.00 = neutral; 1.10 means ~10% more
HR than league average for that handedness in that park.

Values are reasonable starting estimates. Tune them once with a season of
Statcast park splits if you want; the loader falls back to 1.00 for any park
not listed so nothing breaks if a venue name changes.
"""

# venue name (as StatsAPI returns it) -> {"L": factor, "R": factor}
PARK_HR_FACTOR = {
    "Coors Field":            {"L": 1.18, "R": 1.20},
    "Great American Ball Park":{"L": 1.18, "R": 1.16},
    "Yankee Stadium":         {"L": 1.22, "R": 1.02},
    "Fenway Park":            {"L": 0.96, "R": 1.04},
    "Citizens Bank Park":     {"L": 1.10, "R": 1.12},
    "Globe Life Field":       {"L": 1.06, "R": 1.05},
    "Dodger Stadium":         {"L": 1.08, "R": 1.10},
    "Wrigley Field":          {"L": 1.02, "R": 1.03},
    "Truist Park":            {"L": 1.04, "R": 1.05},
    "Chase Field":            {"L": 1.03, "R": 1.04},
    "Camden Yards":           {"L": 1.04, "R": 0.92},
    "Rogers Centre":          {"L": 1.05, "R": 1.06},
    "American Family Field":  {"L": 1.10, "R": 1.08},
    "Nationals Park":         {"L": 1.02, "R": 1.01},
    "Citi Field":             {"L": 0.96, "R": 0.95},
    "loanDepot park":         {"L": 0.88, "R": 0.86},
    "Oracle Park":            {"L": 0.78, "R": 0.95},
    "Petco Park":             {"L": 0.94, "R": 0.93},
    "T-Mobile Park":          {"L": 0.92, "R": 0.94},
    "Kauffman Stadium":       {"L": 0.90, "R": 0.92},
    "Comerica Park":          {"L": 0.94, "R": 0.93},
    "Progressive Field":      {"L": 0.98, "R": 0.99},
    "Guaranteed Rate Field":  {"L": 1.08, "R": 1.10},
    "Rate Field":             {"L": 1.08, "R": 1.10},
    "Target Field":           {"L": 1.00, "R": 1.01},
    "Busch Stadium":          {"L": 0.94, "R": 0.93},
    "PNC Park":               {"L": 0.96, "R": 0.92},
    "Minute Maid Park":       {"L": 1.02, "R": 1.08},
    "Daikin Park":            {"L": 1.02, "R": 1.08},
    "Angel Stadium":          {"L": 1.00, "R": 1.01},
    "Oakland Coliseum":       {"L": 0.90, "R": 0.90},
    "Sutter Health Park":     {"L": 1.05, "R": 1.05},
    "Tropicana Field":        {"L": 0.96, "R": 0.97},
    "George M. Steinbrenner Field": {"L": 1.05, "R": 1.05},
}


def park_factor(venue: str, bats: str) -> float:
    side = "L" if bats == "L" else "R"   # switch hitters scored as R by default
    return PARK_HR_FACTOR.get(venue, {}).get(side, 1.00)
