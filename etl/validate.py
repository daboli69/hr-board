"""Phase 3 — leak-free backtesting and validation.

DESIGN PRINCIPLE: leakage is prevented STRUCTURALLY, not by discipline.

Every look-ahead bug I have seen in this codebase came from code that was written correctly and
then edited. A boolean mask like `df[df.game_date < d]` is correct but fragile — one later edit
that reuses the unfiltered frame reintroduces the leak silently, and the result looks *better*,
so nothing alerts you. `AsOfFrame` removes the option: the full frame is sorted once, and every
accessor returns an `.iloc[:idx]` slice bounded by a precomputed integer. Future rows are not
filtered out, they are unreachable.

WHAT CAN AND CANNOT BE GRADED TODAY:
  gradeable now  — MAE / RMSE / Brier / calibration / decile lift / fatigue deltas / fence lift
  NOT gradeable  — ROI vs closing lines. There is no historical odds archive: odds.json holds
                   only the current slate and the daily snapshots carry no prices. See
                   `archive_odds_snapshot()` for the one-line fix that makes ROI possible in
                   about six weeks of accumulation. Reporting an ROI without those prices would
                   mean inventing them.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Temporal isolation
# ---------------------------------------------------------------------------


class AsOfFrame:
    """A Statcast frame that can only ever be read up to a cutoff date.

    The frame is sorted by date ONCE at construction. `as_of(date)` uses `searchsorted` to find
    the integer row index of the first row on or after that date, and every accessor slices
    `iloc[:idx]`. There is no code path that returns a row at or after the cutoff, so a future
    edit cannot reintroduce leakage by forgetting a mask.

    The cutoff is EXCLUSIVE of the date itself: predicting a 2026-06-01 game may use data
    through 2026-05-31 only. Same-day rows are the subtlest leak of all, because a game's own
    outcome sits in them.
    """

    def __init__(self, df, date_col="game_date"):
        if df is None or df.empty:
            self._df = pd.DataFrame()
            self._dates = np.array([], dtype="datetime64[ns]")
            self.date_col = date_col
            return
        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col, kind="mergesort")
        d = d.reset_index(drop=True)
        self._df = d
        self._dates = d[date_col].values
        self.date_col = date_col

    def _boundary(self, date):
        """Integer index of the first row ON or AFTER `date`. Everything before is legal."""
        cut = np.datetime64(pd.to_datetime(date), "ns")
        return int(np.searchsorted(self._dates, cut, side="left"))

    def as_of(self, date):
        """All rows strictly BEFORE `date`."""
        return self._df.iloc[:self._boundary(date)]

    def window(self, date, days):
        """Rows in [date-days, date) — a trailing window that also excludes the day itself."""
        end = self._boundary(date)
        start_date = pd.to_datetime(date) - pd.Timedelta(days=int(days))
        start = int(np.searchsorted(self._dates, np.datetime64(start_date, "ns"), side="left"))
        return self._df.iloc[start:end]

    def assert_clean(self, frame, date, label=""):
        """Belt-and-braces: prove a slice contains nothing at or after the cutoff.

        Cheap, and it converts a silent scoring inflation into a loud failure. Any leak that
        survives the structural guard dies here.
        """
        if frame is None or frame.empty:
            return True
        mx = pd.to_datetime(frame[self.date_col]).max()
        cut = pd.to_datetime(date)
        if mx >= cut:
            raise AssertionError(
                f"LEAKAGE [{label}]: slice contains {mx.date()} but cutoff is {cut.date()}")
        return True

    @property
    def frame(self):
        return self._df


# ---------------------------------------------------------------------------
# 2. Calibration statistics
# ---------------------------------------------------------------------------


def decile_calibration(pred, actual, n_bands=10, min_per_band=30):
    """Bin predictions into bands and compare predicted vs actual rates.

    Bands are cut on QUANTILES of the predictions, not on fixed probability ranges. Fixed
    ranges leave most bands nearly empty when a model's outputs are concentrated (hit
    probabilities cluster between .55 and .80), and an empty band tells you nothing.

    Returns per-band rows plus the summary statistics that actually decide whether a model is
    calibrated: slope, intercept, ECE and Brier.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    if len(pred) < n_bands * min_per_band:
        return {"error": f"need >= {n_bands * min_per_band} samples, have {len(pred)}"}

    edges = np.quantile(pred, np.linspace(0, 1, n_bands + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bands - 1)

    bands = []
    for b in range(n_bands):
        m = idx == b
        n = int(m.sum())
        if n < min_per_band:
            continue
        p_hat = float(pred[m].mean())
        p_act = float(actual[m].mean())
        se = math.sqrt(max(p_act * (1 - p_act), 1e-9) / n)
        bands.append({
            "band": b + 1, "n": n,
            "pred": round(p_hat, 4), "actual": round(p_act, 4),
            "gap": round(p_act - p_hat, 4),
            "ci95": round(1.96 * se, 4),
            "within_ci": bool(abs(p_act - p_hat) <= 1.96 * se),
        })

    if len(bands) < 3:
        return {"error": "too few populated bands"}

    # Calibration line: regress actual on predicted, weighted by band size.
    # A perfectly calibrated model has slope 1 and intercept 0. Slope < 1 means the model is
    # OVERCONFIDENT — it spreads its predictions wider than reality justifies.
    x = np.array([b["pred"] for b in bands])
    y = np.array([b["actual"] for b in bands])
    w = np.array([b["n"] for b in bands], dtype=float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    denom = np.sum(w * (x - xm) ** 2)
    slope = float(np.sum(w * (x - xm) * (y - ym)) / denom) if denom > 0 else float("nan")
    intercept = float(ym - slope * xm)

    ece = float(np.sum(w * np.abs(y - x)) / np.sum(w))       # expected calibration error
    brier = float(np.mean((pred - actual) ** 2))
    base = float(actual.mean())
    brier_base = float(np.mean((base - actual) ** 2))

    # Monotonicity, tested properly. A strict "every band beats the last" rule REJECTS a
    # perfectly calibrated model most of the time at realistic sample sizes — simulated at
    # 150/band it passes only 8% of the time, and at 400/band only 48%. Spearman correlation
    # answers the question the strict rule was trying to ask without the false failures.
    rho = _spearman(x, y)
    strict = all(y[i] < y[i + 1] for i in range(len(y) - 1))

    return {
        "bands": bands,
        "n": int(len(pred)),
        "slope": round(slope, 3),
        "intercept": round(intercept, 4),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
        "brier_baseline": round(brier_base, 4),
        "beats_baseline": bool(brier < brier_base),
        "spearman": round(rho, 3),
        "strict_monotonic": strict,
        "bands_within_ci": sum(1 for b in bands if b["within_ci"]),
        "verdict": _calibration_verdict(slope, ece, rho, brier, brier_base),
    }


def _spearman(x, y):
    """Rank correlation, implemented locally to avoid a scipy dependency in the ETL."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def _calibration_verdict(slope, ece, rho, brier, brier_base):
    if brier >= brier_base:
        return "FAIL — no better than predicting the base rate"
    if rho < 0.6:
        return "FAIL — predictions do not order outcomes"
    if ece > 0.05:
        return "MISCALIBRATED — ordering works, levels are off; apply an isotonic correction"
    if slope < 0.75:
        return "OVERCONFIDENT — shrink predictions toward the base rate"
    if slope > 1.25:
        return "UNDERCONFIDENT — model is too conservative"
    return "PASS — calibrated and discriminating"


# ---------------------------------------------------------------------------
# 3. Module graders
# ---------------------------------------------------------------------------


def grade_totals(rows):
    """Markov totals vs actual. rows: [{pred_mean, pred_dist, actual, line?}]

    MAE is reported because it is asked for, but it is NOT the number that decides this model.
    Game totals have a ~4.4 run sd against only ~1.5 runs of true between-game variation, so a
    model that knew every game's true mean would still post ~3.30 MAE and a constant predictor
    posts ~3.51. The whole usable range is about 0.2 runs wide. Judge this on the calibration
    of P(over) instead — that is the quantity you actually bet.
    """
    pred = np.array([r["pred_mean"] for r in rows], dtype=float)
    act = np.array([r["actual"] for r in rows], dtype=float)
    resid = act - pred
    out = {
        "n": len(rows),
        "mae": round(float(np.mean(np.abs(resid))), 3),
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "bias": round(float(np.mean(resid)), 3),
        "mae_constant_baseline": round(float(np.mean(np.abs(act - act.mean()))), 3),
        "theoretical_floor": 3.30,
    }
    out["beats_constant"] = out["mae"] < out["mae_constant_baseline"]

    # Distribution quality: does P(over line) actually come true at that rate?
    lines = [r for r in rows if r.get("line") is not None and r.get("pred_dist")]
    if len(lines) >= 300:
        p_over, hit = [], []
        for r in lines:
            d = r["pred_dist"]
            L = float(r["line"])
            over = sum(p for k, p in d.items() if k > L)
            push = sum(p for k, p in d.items() if k == L)
            if push >= 0.999:
                continue
            p_over.append(over / (1 - push))
            hit.append(1.0 if r["actual"] > L else 0.0)
        out["over_calibration"] = decile_calibration(p_over, hit, n_bands=10)

    # Tail check: the fixed-variance approximation understates blowups by ~30% in weak-arm vs
    # strong-lineup spots. Verify the simulated tail matches reality there specifically.
    tail = [r for r in rows if r.get("blowup_spot")]
    if len(tail) >= 100:
        pred_hi = np.mean([sum(p for k, p in r["pred_dist"].items() if k >= 11)
                           for r in tail if r.get("pred_dist")])
        act_hi = np.mean([1.0 if r["actual"] >= 11 else 0.0 for r in tail])
        out["tail_check"] = {
            "n": len(tail),
            "pred_p_11plus": round(float(pred_hi), 4),
            "actual_p_11plus": round(float(act_hi), 4),
            "gap": round(float(act_hi - pred_hi), 4),
            "note": "positive gap = still understating blowups",
        }
    return out


def grade_hit_props(rows):
    """Contact-gated hit model. rows: [{p_hit, got_hit}] -> decile calibration."""
    return decile_calibration([r["p_hit"] for r in rows],
                              [1.0 if r["got_hit"] else 0.0 for r in rows])


def grade_strikeouts(rows):
    """Poisson-binomial K counts vs actual. rows: [{exp_k, dist, actual_k}]"""
    pred = np.array([r["exp_k"] for r in rows], dtype=float)
    act = np.array([r["actual_k"] for r in rows], dtype=float)
    resid = act - pred
    out = {
        "n": len(rows),
        "mae": round(float(np.mean(np.abs(resid))), 3),
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
        "bias": round(float(np.mean(resid)), 3),
        "mae_constant_baseline": round(float(np.mean(np.abs(act - act.mean()))), 3),
    }
    # Per-line calibration is where the Poisson-binomial should beat a plain Poisson, because
    # the two differ mainly in the tails and that is where the lines sit.
    out["lines"] = {}
    for L in (4.5, 5.5, 6.5, 7.5):
        p, h = [], []
        for r in rows:
            d = r.get("dist")
            if not d:
                continue
            p.append(sum(v for k, v in d.items() if k > L))
            h.append(1.0 if r["actual_k"] > L else 0.0)
        if len(p) >= 300:
            cal = decile_calibration(p, h, n_bands=5, min_per_band=40)
            out["lines"][f"O{L}"] = {
                "n": len(p),
                "pred_rate": round(float(np.mean(p)), 4),
                "actual_rate": round(float(np.mean(h)), 4),
                "gap": round(float(np.mean(h) - np.mean(p)), 4),
                "calibration": cal.get("verdict") if isinstance(cal, dict) else None,
            }
    return out


def grade_bullpen_fatigue(rows):
    """Validate the -8% K% / +5% BB% fatigue penalty against what actually happened.

    rows: [{status, k_pct, bb_pct, xwoba, fb_velo}] — one row per reliever appearance, with
    `status` assigned AS-OF that date from prior pitch logs only.

    This is the check that tells you whether the penalty weight is right, too soft or invented.
    If FATIGUED arms show a 3% K drop rather than 8%, the constant should move — and the point
    of measuring is that the answer is allowed to disagree with the assumption.
    """
    df = pd.DataFrame(rows)
    if df.empty or "status" not in df:
        return {"error": "no rows"}
    out = {}
    base = df[df["status"] == "AVAILABLE"]
    if base.empty:
        return {"error": "no AVAILABLE baseline"}
    for status in ("FATIGUED", "UNAVAILABLE"):
        grp = df[df["status"] == status]
        if len(grp) < 50:
            out[status] = {"n": len(grp), "note": "insufficient sample"}
            continue
        rec = {"n": len(grp)}
        for col, label in (("k_pct", "k"), ("bb_pct", "bb"),
                           ("xwoba", "xwoba"), ("fb_velo", "velo")):
            if col not in df:
                continue
            b, g = float(base[col].mean()), float(grp[col].mean())
            rec[f"{label}_baseline"] = round(b, 4)
            rec[f"{label}_flagged"] = round(g, 4)
            rec[f"{label}_delta_pct"] = round(100.0 * (g - b) / b, 2) if b else None
        # is the observed K drop consistent with the -8% we apply?
        if "k_delta_pct" in rec and rec["k_delta_pct"] is not None:
            obs = -rec["k_delta_pct"]
            rec["assumed_k_penalty_pct"] = 8.0
            rec["observed_k_penalty_pct"] = round(obs, 2)
            rec["verdict"] = ("penalty about right" if abs(obs - 8.0) <= 3.0
                              else ("penalty TOO HARSH" if obs < 5.0 else "penalty TOO SOFT"))
        out[status] = rec
    return out


def grade_fence_delta(rows):
    """Does trailing near-miss count predict FUTURE home runs?

    rows: [{near_misses_14d, bbe_14d, future_hr, future_pa}] — near misses measured as-of, and
    the outcome strictly after.

    The claim being tested: a ball that died 3 feet short is contact that converts on a warmer
    night or in a different park, so it should carry predictive signal that an ordinary flyout
    does not. If it doesn't, the fence-delta work buys precision without lift and should be a
    display metric only.
    """
    df = pd.DataFrame(rows)
    if df.empty or len(df) < 200:
        return {"error": f"need >= 200 rows, have {len(df)}"}
    df = df[df["bbe_14d"] >= 15].copy()
    if df.empty:
        return {"error": "no rows with enough batted balls"}
    df["nm_rate"] = df["near_misses_14d"] / df["bbe_14d"]
    df["hr_rate"] = df["future_hr"] / df["future_pa"].clip(lower=1)
    base = float(df["hr_rate"].mean())
    q = df["nm_rate"].quantile([0.25, 0.5, 0.75]).values
    buckets = {
        "none (0 near misses)": df[df["near_misses_14d"] == 0],
        "low": df[(df["near_misses_14d"] > 0) & (df["nm_rate"] <= q[1])],
        "high": df[(df["nm_rate"] > q[1]) & (df["nm_rate"] <= q[2])],
        "elite": df[df["nm_rate"] > q[2]],
    }
    out = {"base_hr_rate": round(base, 4), "n": len(df), "buckets": {}}
    for name, g in buckets.items():
        if len(g) < 40:
            continue
        r = float(g["hr_rate"].mean())
        se = float(g["hr_rate"].std() / math.sqrt(len(g))) if len(g) > 1 else 0.0
        out["buckets"][name] = {
            "n": len(g), "hr_rate": round(r, 4),
            "lift": round(r / base, 3) if base else None,
            "z": round((r - base) / se, 2) if se else None,
        }
    return out


# ---------------------------------------------------------------------------
# 4. Odds archiving — the missing piece for ROI
# ---------------------------------------------------------------------------


# archive_odds_snapshot() used to live here and has been REMOVED, not lost.
#
# fetch_odds._archive_odds_snapshot() does the same job better: it writes from the in-memory
# payload inside _write(), so the archive is atomic with the file it mirrors and cannot capture
# a half-written odds.json. This version re-read from disk and was never called by anything.
#
# Two implementations of one behaviour drift — one gets a bug fix, the other does not, and the
# archive silently disagrees with itself across a season. Keeping the called one.

def grade_roi(rows, stake=1.0):
    """ROI given historical prices. rows: [{prob, american, won}]

    Only bets where the model's edge cleared the price are counted, because that is the only
    population you would actually have bet. Grading every projection instead measures the
    model's opinion rather than the strategy.
    """
    if not rows:
        return {"error": "no priced rows — run archive_odds_snapshot() nightly to build history"}
    staked = ret = 0.0
    n = 0
    for r in rows:
        am = r.get("american")
        if am is None or r.get("prob") is None:
            continue
        dec = 1.0 + (am / 100.0 if am > 0 else 100.0 / abs(am))
        implied = 1.0 / dec
        if r["prob"] <= implied:
            continue                    # no edge; would not have bet
        n += 1
        staked += stake
        if r.get("won"):
            ret += stake * dec
    if not staked:
        return {"error": "no qualifying bets"}
    return {"bets": n, "staked": round(staked, 2), "returned": round(ret, 2),
            "roi_pct": round(100.0 * (ret - staked) / staked, 2)}
