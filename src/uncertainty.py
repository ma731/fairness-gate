"""Confidence intervals for every rate and every gap.

A confidence interval is just the range a number could plausibly have been, given how
many people it was worked out from. A recall of 0.694 computed on 926 people is not the
same claim as one computed on 434,022, and printing both to three decimals pretends
otherwise. Worse, a
policy gate comparing a point estimate to a threshold can fire on noise, which is the
one thing a gate must never do: a red build nobody believes is a gate that has already
stopped working.

Method: a parametric bootstrap over the binomial counts rather than resampling 600,000
rows. For a rate metric the two are equivalent, because the only randomness that matters
is how many successes fall out of a fixed number of trials, and the counts version runs
in a second instead of minutes.

The gap between groups is resampled jointly: each draw perturbs every group's rate, then
the gap is recomputed from those perturbed rates. Bootstrapping the maximum and the
minimum separately would understate the spread, because which group is the extreme is
itself uncertain.
"""

from __future__ import annotations

import numpy as np

DRAWS = 4000
SEED = 20260922
LEVEL = 0.95


def wilson(successes: int, trials: int, level: float = LEVEL) -> tuple[float, float]:
    """Wilson score interval for a single rate.

    Preferred over the textbook normal interval because it stays inside [0, 1] and keeps
    its coverage on small groups, which is exactly where this project has groups.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + level / 2))
    p = successes / trials
    d = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / d
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _draw_rates(rng, successes: np.ndarray, trials: np.ndarray, draws: int) -> np.ndarray:
    """draws x groups matrix of resampled rates."""
    out = np.empty((draws, len(trials)), dtype=float)
    for j, (s, n) in enumerate(zip(successes, trials, strict=True)):
        if n <= 0:
            out[:, j] = np.nan
            continue
        out[:, j] = rng.binomial(int(n), min(max(s / n, 0.0), 1.0), size=draws) / n
    return out


def gap_interval(successes, trials, draws: int = DRAWS, level: float = LEVEL,
                 seed: int = SEED) -> dict:
    """Interval for max-minus-min across groups, resampled jointly."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if len(trials) < 2 or not np.isfinite(trials).all():
        return {}
    rng = np.random.default_rng(seed)
    rates = _draw_rates(rng, successes, trials, draws)
    gaps = np.nanmax(rates, axis=1) - np.nanmin(rates, axis=1)
    lo, hi = np.percentile(gaps, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    point = float(np.nanmax(successes / trials) - np.nanmin(successes / trials))
    return {
        "point": round(point, 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "level": level,
        "draws": draws,
    }


def build(table, attribute: str) -> dict:
    """Intervals for the gaps this project gates on, for one protected attribute."""
    sub = table[(table["attribute"] == attribute) & table["reportable"]]
    if len(sub) < 2:
        return {}

    # trials and successes for each rate, from counts rather than the rounded rate
    n = sub["n"].to_numpy(dtype=float)
    base = sub["base_rate"].to_numpy(dtype=float)
    pos = n * base
    neg = n - pos

    out = {"attribute": attribute, "level": LEVEL, "draws": DRAWS}
    out["tpr_gap"] = gap_interval(sub["tpr"].to_numpy() * pos, pos)
    out["fpr_gap"] = gap_interval(sub["fpr"].to_numpy() * neg, neg)
    out["selection_gap"] = gap_interval(sub["selection_rate"].to_numpy() * n, n)

    # per-group Wilson intervals on the rate the whole project turns on
    groups = []
    for _, r in sub.iterrows():
        p = float(r["n"]) * float(r["base_rate"])
        qualifying = round(p)
        lo, hi = wilson(round(float(r["tpr"]) * p), qualifying)
        groups.append({
            "code": int(r["code"]),
            "group": str(r["group"]),
            "qualifying": qualifying,
            "tpr": round(float(r["tpr"]), 4),
            "lo": round(lo, 4),
            "hi": round(hi, 4),
            "width": round(hi - lo, 4),
        })
    out["tpr_by_group"] = sorted(groups, key=lambda g: -g["width"])
    return out
