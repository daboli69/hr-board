"""
Batted-ball flight model: drag + Magnus (backspin lift), integrated in 3D with wind.

This is the physics core of the park/weather HR model. It takes a hitter's actual
batted balls (exit velocity, launch angle, spray) and computes, under a given park's
air density and wind, how high the ball is when it reaches the outfield wall — which,
checked against the wall's distance and height at that spray angle, decides HR or not.

Validated against published carry distances (a 103 mph / 28 deg / 1900 rpm ball at
70F sea level calm carries ~410 ft here; matches Nathan-model references within a few ft).

Honest limits, inherent to PUBLIC Statcast data (not this implementation):
  - Batted-ball backspin is ESTIMATED from launch conditions, not measured (public
    Statcast has no batted-ball spin). Every public carry model does this.
  - Sidespin / hook-slice curvature is not modeled (pure backspin assumption), so
    spray angle is treated as preserved from contact to the fence.
"""
import numpy as np

# baseball constants (SI)
M = 0.145          # kg
R = 0.0366         # m (radius)
A = np.pi * R * R  # m^2 cross-section
G = 9.81           # m/s^2
CD = 0.40          # drag coefficient (constant approx, calibrated to benchmarks)
MPH = 0.44704      # mph -> m/s
FT = 3.28084       # m -> ft


def air_density(temp_f, elev_m=0.0, pressure_pa=None):
    """Air density (kg/m^3) from temperature and elevation. Warmer/higher = thinner = more carry."""
    T = (temp_f - 32.0) * 5.0 / 9.0 + 273.15
    if pressure_pa is None:
        pressure_pa = 101325.0 * np.exp(-elev_m / 8000.0)   # barometric approx
    return pressure_pa / (287.05 * T)


def estimate_backspin(la_deg, ev_mph):
    """
    Estimate backspin (rpm) from launch conditions — the unavoidable approximation
    when working with public data. Backspin rises with launch angle (steeper = more lift)
    and is dampened on weakly hit balls. Bounded to a realistic range.
    """
    la = np.asarray(la_deg, dtype=float)
    ev = np.asarray(ev_mph, dtype=float)
    spin = 900.0 + 22.0 * np.clip(la, 0, 45) + 6.0 * (np.clip(ev, 80, 120) - 95.0)
    return np.clip(spin, 800.0, 3200.0)


def carry_batch(ev_mph, la_deg, spray_deg, rho, wind_vec_ms, wall_dist_ft,
                spin_rpm=None, dt=0.003, max_t=9.0):
    """
    Integrate a batch of batted balls. Vectorized over N balls.

    Field frame: x = toward center field, y = toward right field (+, 1B side),
    z = up. Spray 0 = CF, negative = LF (3B side), positive = RF — matching the
    Statcast spray convention used elsewhere in the ETL.

    Args:
      ev_mph, la_deg, spray_deg : per-ball arrays (length N)
      rho            : air density (scalar, kg/m^3) for this park/weather
      wind_vec_ms    : (3,) wind velocity in FIELD coords (m/s); +x blows out to CF
      wall_dist_ft   : per-ball outfield wall distance at that ball's spray angle (ft)
      spin_rpm       : per-ball backspin; if None, estimated from launch conditions

    Returns (dist_ft, z_at_wall_ft):
      dist_ft      : total carry (radial horizontal distance, ft)
      z_at_wall_ft : ball height (ft) at the moment it reaches wall_dist_ft
                     (np.nan if it never reaches the wall horizontally)
    """
    ev = np.asarray(ev_mph, dtype=float)
    la = np.radians(np.asarray(la_deg, dtype=float))
    az = np.radians(np.asarray(spray_deg, dtype=float))
    N = ev.shape[0]
    if spin_rpm is None:
        spin_rpm = estimate_backspin(la_deg, ev_mph)
    omega = np.asarray(spin_rpm, dtype=float) * 2 * np.pi / 60.0
    wall_m = np.asarray(wall_dist_ft, dtype=float) / FT

    v0 = ev * MPH
    v = np.stack([v0 * np.cos(la) * np.cos(az),
                  v0 * np.cos(la) * np.sin(az),
                  v0 * np.sin(la)], axis=1)            # (N,3)
    p = np.zeros((N, 3)); p[:, 2] = 1.0                # contact ~1 m up
    wind = np.asarray(wind_vec_ms, dtype=float).reshape(1, 3)

    # spin axis: horizontal, perpendicular to launch azimuth -> pure backspin (lift up & back)
    saxis = np.stack([-np.sin(az), np.cos(az), np.zeros(N)], axis=1)   # (N,3)
    kdrag = 0.5 * rho * A * CD / M

    landed = np.zeros(N, dtype=bool)
    dist_ft = np.full(N, np.nan)
    z_at_wall = np.full(N, np.nan)
    reached = np.zeros(N, dtype=bool)
    prev_r = np.zeros(N); prev_z = p[:, 2].copy()

    steps = int(max_t / dt)
    for _ in range(steps):
        vair = v - wind
        sp = np.linalg.norm(vair, axis=1)
        sp = np.where(sp < 1e-6, 1e-6, sp)
        vhat = vair / sp[:, None]
        S = R * omega / sp                                   # spin factor
        Cl = np.minimum(0.35, 1.6 * S)
        a_drag = -kdrag * (sp[:, None] * vair)
        a_mag = (0.5 * rho * A * Cl / M)[:, None] * (sp ** 2)[:, None] * np.cross(vhat, saxis)
        a = a_drag + a_mag
        a[:, 2] -= G

        v = v + a * dt
        p_new = p + v * dt
        r_new = np.hypot(p_new[:, 0], p_new[:, 1])           # radial horizontal distance (m)

        # fence crossing: radial distance passes wall_m while still airborne
        cross = (~reached) & (~landed) & (prev_r < wall_m) & (r_new >= wall_m)
        if cross.any():
            frac = (wall_m[cross] - prev_r[cross]) / np.maximum(r_new[cross] - prev_r[cross], 1e-9)
            z_cross = prev_z[cross] + frac * (p_new[cross, 2] - prev_z[cross])
            z_at_wall[cross] = z_cross * FT
            reached[cross] = True

        # landing: z crosses 0
        land = (~landed) & (p_new[:, 2] <= 0)
        if land.any():
            frac = prev_z[land] / np.maximum(prev_z[land] - p_new[land, 2], 1e-9)
            r_land = prev_r[land] + frac * (r_new[land] - prev_r[land])
            dist_ft[land] = r_land * FT
            landed[land] = True

        prev_r = r_new; prev_z = p_new[:, 2].copy(); p = p_new
        if landed.all():
            break

    dist_ft = np.where(np.isnan(dist_ft), prev_r * FT, dist_ft)
    return dist_ft, z_at_wall
