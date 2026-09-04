"""Turn relative monocular depth into metres, using the calibrated ground plane.

MiDaS and DPT predict *relative inverse depth*: the map is correct up to an
unknown scale and shift, so a raw MiDaS value is not a distance and cannot be
compared between frames. Two facts make it usable anyway.

First, the standard MiDaS relation is affine in inverse depth::

    1 / Z  ~=  a * d + b

where ``d`` is the network's output. Fitting ``(a, b)`` needs at least two points
whose true ``Z`` is known.

Second, this pipeline already has metric truth for a subset of pixels: any object
whose bottom edge touches the road back-projects through the homography to a real
distance. So the homography supplies the anchors, and the fit transfers that scale
to the whole image -- including the pixels the homography cannot handle.

That combination is worth more than either alone:

* the **homography** is accurate for objects standing on flat road and says
  nothing at all about anything else -- an occluded contact point, a vehicle on a
  crest, the back of a truck whose wheels are hidden;
* **MiDaS** covers every pixel but has no scale;
* fitted together, MiDaS inherits the homography's scale and extends it to the
  cases the homography cannot see.

The fit is re-estimated per frame and rejected when the anchors do not support it,
because a bad scale silently produces confident, wrong distances -- the worst
failure mode available to a collision warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this many usable anchors, no fit is attempted. Two points define a line
#: through any noise; the extra points are what make the fit mean something.
MIN_ANCHORS = 4


@dataclass(frozen=True)
class DepthScale:
    """An affine map from relative inverse depth to metres.

    Attributes:
        a, b: Coefficients of ``1/Z = a*d + b``.
        n_anchors: Ground-plane points the fit used.
        r_squared: Quality of the fit in inverse-depth space.
        residual_m: Median absolute error against the anchors, in metres. The
            number to judge the fit by -- ``r_squared`` can look healthy while
            the metric error is large, because the fit lives in inverse space
            where distant points contribute almost nothing.
        valid: Whether this scale should be used at all.
    """

    a: float = 0.0
    b: float = 0.0
    n_anchors: int = 0
    r_squared: float = 0.0
    residual_m: float = float("inf")
    valid: bool = False

    def to_metres(self, d: np.ndarray, max_m: float = 200.0) -> np.ndarray:
        """Convert relative depth values to metres.

        Returns ``inf`` where the affine map gives a non-positive inverse depth,
        which means "further than this fit can speak to" rather than a distance.
        """
        if not self.valid:
            return np.full(np.shape(d), np.inf, dtype=np.float64)
        inv = self.a * np.asarray(d, dtype=np.float64) + self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(inv > 1e-9, 1.0 / np.maximum(inv, 1e-9), np.inf)
        return np.clip(z, 0.0, max_m)


def fit_scale(
    depth_values: np.ndarray,
    distances_m: np.ndarray,
    min_anchors: int = MIN_ANCHORS,
    max_residual_m: float = 6.0,
    min_r_squared: float = 0.5,
) -> DepthScale:
    """Fit ``1/Z = a*d + b`` from anchors whose metric distance is known.

    Args:
        depth_values: Relative depth the network predicted at each anchor.
        distances_m: True distance at each anchor, from the homography.
        min_anchors: Fewer than this and no fit is attempted.
        max_residual_m: Median absolute metric error above which the fit is
            rejected. Guards the case that matters: a fit that is numerically
            fine but metrically wrong.
        min_r_squared: Minimum correlation in inverse-depth space.

    Returns:
        A :class:`DepthScale`, with ``valid`` False when the anchors do not
        support a trustworthy fit.
    """
    d = np.asarray(depth_values, dtype=np.float64).reshape(-1)
    z = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    if d.size != z.size:
        raise ValueError(f"got {d.size} depth values but {z.size} distances")

    good = np.isfinite(d) & np.isfinite(z) & (z > 0.5)
    d, z = d[good], z[good]
    if d.size < min_anchors:
        return DepthScale(n_anchors=int(d.size))

    inv_z = 1.0 / z
    d_mean, inv_mean = d.mean(), inv_z.mean()
    dd = d - d_mean
    denom = float((dd * dd).sum())
    if denom <= 1e-12:
        # Every anchor at the same predicted depth: the slope is unconstrained.
        return DepthScale(n_anchors=int(d.size))

    a = float((dd * (inv_z - inv_mean)).sum() / denom)
    b = float(inv_mean - a * d_mean)

    pred_inv = a * d + b
    ss_tot = float(((inv_z - inv_mean) ** 2).sum())
    ss_res = float(((inv_z - pred_inv) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
    r2 = max(0.0, min(1.0, r2))

    with np.errstate(divide="ignore", invalid="ignore"):
        pred_z = np.where(pred_inv > 1e-9, 1.0 / np.maximum(pred_inv, 1e-9), np.inf)
    finite = np.isfinite(pred_z)
    residual = float(np.median(np.abs(pred_z[finite] - z[finite]))) if finite.any() else float("inf")

    # A negative slope means the network's depth ordering disagrees with the
    # homography's -- the two are describing different scenes, and no scaling
    # reconciles that.
    valid = (
        a > 0.0
        and r2 >= min_r_squared
        and residual <= max_residual_m
        and np.isfinite(residual)
    )
    return DepthScale(a=a, b=b, n_anchors=int(d.size), r_squared=r2,
                      residual_m=residual, valid=bool(valid))


def sample_depth(depth_map: np.ndarray, points_px: np.ndarray, patch: int = 5) -> np.ndarray:
    """Median depth in a small patch around each image point.

    A median over a patch rather than the single pixel: depth maps are noisiest
    exactly at object boundaries, which is where a bounding box's bottom edge
    sits, and one unlucky pixel there would anchor the whole frame's scale wrongly.
    """
    dm = np.asarray(depth_map)
    h, w = dm.shape[:2]
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    half = max(0, patch // 2)
    out = np.empty(len(pts), dtype=np.float64)
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            out[i] = np.nan
            continue
        x0, x1 = max(0, xi - half), min(w, xi + half + 1)
        y0, y1 = max(0, yi - half), min(h, yi + half + 1)
        window = dm[y0:y1, x0:x1]
        out[i] = float(np.median(window)) if window.size else np.nan
    return out
