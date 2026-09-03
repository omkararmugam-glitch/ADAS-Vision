"""Time-to-collision from a tracked object.

Two independent estimators, because they fail in different ways and a
disagreement between them is itself information.

**Scale expansion** (no calibration needed). Under a pinhole camera the apparent
height of an object is ``h = f*H/Z``, so ``1/h`` is proportional to distance
``Z``. If the closing speed is constant, ``1/h`` therefore falls *linearly* with
time and reaches zero at the moment of contact. Fitting a straight line to
``1/h(t)`` and extrapolating to zero is thus not an approximation -- it is the
exact model for constant relative velocity, which is why this is fitted in
inverse-height space rather than differentiating ``h`` directly.

**Ground-plane range** (needs calibration). The bottom-centre of the box
back-projects through the homography to a real distance in metres; fitting that
against time gives a range rate and hence a TTC in seconds.

Neither is trustworthy alone:

* Scale expansion needs no calibration and no flat road, but says nothing about
  *distance* -- only time -- and is very noisy for small boxes, where a
  one-pixel change in height is a large fractional change.
* Ground-plane range gives metres, but assumes the road is flat and that the
  box bottom really is the ground contact point. Both fail exactly when it
  matters: on a crest or dip, and under braking when the car pitches forward.

See ``docs/ARCHITECTURE.md`` for the full error analysis. The headline caveat is
that monocular distance error grows with the square of distance, so a TTC at
50 m is a far weaker claim than the same number at 10 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Returned when the object is not approaching, or there is too little evidence.
#: Positive infinity is the honest value -- "no collision is implied" -- and it
#: compares correctly against any threshold without special-casing.
NO_COLLISION = float("inf")


@dataclass(frozen=True)
class TTCEstimate:
    """Time-to-collision for one tracked object.

    Attributes:
        ttc_s: Seconds until contact, or ``inf`` when not approaching or when
            the evidence is too weak. Never negative.
        source: Which estimator produced ``ttc_s`` -- ``scale``, ``range`` or
            ``none``.
        distance_m: Ground-plane distance now, if calibration was available.
        range_rate_mps: Closing speed in m/s, negative when approaching.
        scale_ttc_s: The scale-expansion estimate on its own.
        range_ttc_s: The ground-plane estimate on its own.
        n_samples: Track history points the fit used.
        fit_quality: Coefficient of determination of the inverse-height fit, in
            ``[0, 1]``. Low means the object is not moving the way the constant-
            velocity model assumes, so the TTC is not meaningful.
        disagreement_s: Absolute gap between the two estimates when both exist.
            A large gap usually means the flat-road assumption is being violated.
    """

    ttc_s: float = NO_COLLISION
    source: str = "none"
    distance_m: float | None = None
    range_rate_mps: float | None = None
    scale_ttc_s: float = NO_COLLISION
    range_ttc_s: float = NO_COLLISION
    n_samples: int = 0
    fit_quality: float = 0.0
    disagreement_s: float | None = None

    @property
    def is_approaching(self) -> bool:
        return np.isfinite(self.ttc_s)


def _linear_fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares ``y = a + b*t``. Returns ``(a, b, r_squared)``."""
    n = len(t)
    if n < 2:
        return float(y[0]) if n else 0.0, 0.0, 0.0
    t_mean, y_mean = t.mean(), y.mean()
    dt = t - t_mean
    denom = float((dt * dt).sum())
    if denom <= 1e-12:          # every sample at the same instant
        return float(y_mean), 0.0, 0.0
    b = float((dt * (y - y_mean)).sum() / denom)
    a = float(y_mean - b * t_mean)
    ss_tot = float(((y - y_mean) ** 2).sum())
    ss_res = float(((y - (a + b * t)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    return a, b, max(0.0, min(1.0, r2))


def scale_expansion_ttc(
    history: list[tuple[int, float, float]], window: int, min_samples: int
) -> tuple[float, int, float]:
    """TTC from apparent size growth alone. No calibration required.

    Fits ``1/h`` against time and extrapolates to zero, which is the exact
    constant-relative-velocity model rather than a finite-difference estimate --
    a difference that matters because differentiating a noisy height directly
    produces a wildly unstable TTC.

    Args:
        history: ``(frame_index, timestamp_s, height_px)`` samples, oldest first.
        window: How many of the most recent samples to fit over.
        min_samples: Fewer than this and no estimate is made.

    Returns:
        ``(ttc_s, n_samples, fit_quality)``.
    """
    if len(history) < min_samples:
        return NO_COLLISION, len(history), 0.0
    recent = history[-window:]
    t = np.array([s[1] for s in recent], dtype=np.float64)
    h = np.array([s[2] for s in recent], dtype=np.float64)
    if np.any(h <= 1e-6):
        return NO_COLLISION, len(recent), 0.0

    inv = 1.0 / h
    a, b, r2 = _linear_fit(t, inv)
    # b >= 0 means 1/h is flat or growing: the object is receding or static.
    if b >= -1e-12:
        return NO_COLLISION, len(recent), r2
    t_now = float(t[-1])
    inv_now = a + b * t_now
    if inv_now <= 0:            # already past the extrapolated contact point
        return 0.0, len(recent), r2
    return float(inv_now / -b), len(recent), r2


def range_rate_ttc(
    distances_m: list[tuple[float, float]], window: int, min_samples: int
) -> tuple[float, float | None, float | None]:
    """TTC from ground-plane distance over time.

    Args:
        distances_m: ``(timestamp_s, distance_m)`` samples, oldest first.
        window: Most recent samples to fit over.
        min_samples: Minimum needed for an estimate.

    Returns:
        ``(ttc_s, distance_now_m, range_rate_mps)``.
    """
    if len(distances_m) < min_samples:
        now = distances_m[-1][1] if distances_m else None
        return NO_COLLISION, now, None
    recent = distances_m[-window:]
    t = np.array([s[0] for s in recent], dtype=np.float64)
    d = np.array([s[1] for s in recent], dtype=np.float64)
    a, b, _ = _linear_fit(t, d)
    d_now = float(a + b * float(t[-1]))
    if b >= -1e-9:              # not closing
        return NO_COLLISION, d_now, float(b)
    if d_now <= 0:
        return 0.0, d_now, float(b)
    return float(d_now / -b), d_now, float(b)


def combine(
    scale_ttc: float,
    range_ttc: float,
    distance_m: float | None,
    range_rate: float | None,
    n_samples: int,
    fit_quality: float,
    min_fit_quality: float,
    prefer_range_within_m: float,
) -> TTCEstimate:
    """Choose between the two estimates, and record how far apart they were.

    Ground-plane range is preferred close in, where the flat-road assumption
    holds best and metres are meaningful. Further out, pixel quantisation makes
    the homography distance unreliable and scale expansion is the steadier
    signal. The two are never averaged: when they disagree it is because an
    assumption has broken, and averaging would hide exactly the case worth
    flagging.
    """
    both = np.isfinite(scale_ttc) and np.isfinite(range_ttc)
    disagreement = abs(scale_ttc - range_ttc) if both else None

    if fit_quality < min_fit_quality and np.isfinite(scale_ttc):
        # The object is not moving the way the model assumes; the number would
        # be arithmetic without meaning.
        scale_ttc = NO_COLLISION

    use_range = (
        np.isfinite(range_ttc)
        and distance_m is not None
        and distance_m <= prefer_range_within_m
    )
    if use_range:
        chosen, source = range_ttc, "range"
    elif np.isfinite(scale_ttc):
        chosen, source = scale_ttc, "scale"
    elif np.isfinite(range_ttc):
        chosen, source = range_ttc, "range"
    else:
        chosen, source = NO_COLLISION, "none"

    return TTCEstimate(
        ttc_s=chosen,
        source=source,
        distance_m=distance_m,
        range_rate_mps=range_rate,
        scale_ttc_s=scale_ttc,
        range_ttc_s=range_ttc,
        n_samples=n_samples,
        fit_quality=fit_quality,
        disagreement_s=disagreement,
    )
