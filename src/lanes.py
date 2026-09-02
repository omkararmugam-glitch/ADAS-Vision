"""Lane fitting: binary lane mask -> polynomial boundaries in bird's-eye metres.

Pipeline for one frame:

1. Warp the lane mask from image pixels to the bird's-eye window using the
   Stage 0 :class:`~src.calibrate.CameraModel`. In that view a real lane
   boundary is close to a straight vertical line, which is what makes a
   low-order polynomial the right model.
2. Convert the surviving mask pixels to ego metres (X right, Y forward).
3. Locate boundary bases from a lateral histogram of the near field.
4. Track each boundary away from the camera with sliding windows.
5. Fit ``x = a*y^2 + b*y + c`` per boundary by least squares.

Coordinate frames: input mask is **image px**; everything returned is in
**ego metres**, X right and Y forward from the camera's ground projection.

This module is stateless by design. Temporal smoothing belongs to a later stage,
and mixing it in here would make a single frame's fit unreproducible.

Degraded input is reported, never faked. If the mask is empty, too sparse, or
supports only one boundary, the returned :class:`~src.types.LaneSet` says so and
leaves the coefficients absent -- downstream code must not be handed a garbage
polynomial that looks like a real measurement.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.calibrate import CameraModel
from src.config import BEVConfig, LaneFitConfig
from src.logging_utils import get_logger, log_kv
from src.types import LaneFit, LaneSet, LaneStatus

_log = get_logger(__name__)

#: A quadratic needs the samples to span some forward distance; fitting one to a
#: sliver of rows produces wild coefficients that mean nothing.
_MIN_Y_SPAN_M = 3.0


class LaneFitter:
    """Fits lane boundaries from a segmentation mask. Stateless across frames.

    Args:
        camera: Calibration, already matched to the frame size being processed.
        bev: Bird's-eye window definition.
        cfg: Fitting parameters.
    """

    def __init__(self, camera: CameraModel, bev: BEVConfig, cfg: LaneFitConfig) -> None:
        self.camera = camera
        self.bev = bev
        self.cfg = cfg

    # -- geometry helpers --------------------------------------------------- #

    def warp_mask(self, lane_mask_px: np.ndarray) -> np.ndarray:
        """Warp a lane mask from image pixels into the bird's-eye window.

        Nearest-neighbour: these are labels, and interpolating between 0 and 1
        would invent partial lane pixels at every boundary.
        """
        M = self.camera.bev_warp_matrix(self.bev)
        return cv2.warpPerspective(
            lane_mask_px.astype(np.uint8), M, self.bev.size_px, flags=cv2.INTER_NEAREST
        )

    def _bev_pixels_to_metres(self, bev_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nonzero bird's-eye pixels as ``(x_m, y_m)`` arrays in ego metres."""
        vs, us = np.nonzero(bev_mask)
        x_m = us / self.bev.px_per_m + self.bev.x_min_m
        y_m = self.bev.y_max_m - vs / self.bev.px_per_m
        return x_m, y_m

    # -- boundary search ---------------------------------------------------- #

    def _find_bases(self, x_m: np.ndarray, y_m: np.ndarray) -> list[float]:
        """Lateral positions of candidate boundaries, from a near-field histogram.

        Returns centres in metres, sorted left to right.
        """
        near_limit = self.bev.y_min_m + 0.45 * (self.bev.y_max_m - self.bev.y_min_m)
        near = y_m <= near_limit
        # Fall back to the whole window if the near field is empty; a boundary
        # visible only at distance is still worth reporting.
        sample = x_m[near] if np.count_nonzero(near) >= self.cfg.min_pixels_per_window else x_m
        if sample.size == 0:
            return []

        bin_w = 0.10
        edges = np.arange(-self.cfg.search_half_width_m, self.cfg.search_half_width_m + bin_w, bin_w)
        hist, _ = np.histogram(sample, bins=edges)
        if hist.sum() == 0:
            return []
        # Smooth so one marking does not fragment into several peaks.
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")
        centres = (edges[:-1] + edges[1:]) / 2.0

        threshold = max(self.cfg.min_pixels_per_window / 5.0, 0.18 * smooth.max())
        peaks: list[float] = []
        min_separation_m = 1.0
        order = np.argsort(smooth)[::-1]
        for idx in order:
            if smooth[idx] < threshold:
                break
            c = float(centres[idx])
            if all(abs(c - p) >= min_separation_m for p in peaks):
                peaks.append(c)
        return sorted(peaks)

    def _slide(self, x_m: np.ndarray, y_m: np.ndarray, base_x: float) -> np.ndarray:
        """Select pixels belonging to one boundary by sliding windows outward.

        Returns a boolean mask over the input arrays.
        """
        y_lo, y_hi = self.bev.y_min_m, self.bev.y_max_m
        step = (y_hi - y_lo) / self.cfg.n_windows
        centre = base_x
        selected = np.zeros(len(x_m), dtype=bool)
        for i in range(self.cfg.n_windows):
            lo = y_lo + i * step
            hi = lo + step
            in_window = (
                (y_m >= lo) & (y_m < hi) & (np.abs(x_m - centre) <= self.cfg.window_half_width_m)
            )
            n = int(np.count_nonzero(in_window))
            if n:
                selected |= in_window
            if n >= self.cfg.min_pixels_per_window:
                # Re-centre on this window's pixels so the search follows a curve.
                centre = float(np.mean(x_m[in_window]))
        return selected

    def _fit_one(self, x_m: np.ndarray, y_m: np.ndarray, role: str) -> LaneFit | None:
        """Least-squares quadratic for one boundary, or None if unsupportable."""
        n = len(x_m)
        if n < self.cfg.min_pixels_for_fit:
            return None
        y_span = float(y_m.max() - y_m.min())
        if y_span < _MIN_Y_SPAN_M:
            # Samples crowded into a few rows cannot constrain a quadratic.
            return None
        coeffs = np.polyfit(y_m, x_m, 2)
        if not np.all(np.isfinite(coeffs)):
            return None
        return LaneFit(
            coeffs=np.asarray(coeffs, dtype=np.float64),
            role=role,
            n_pixels=n,
            y_range_m=(float(y_m.min()), float(y_m.max())),
        )

    # -- public API --------------------------------------------------------- #

    def fit(self, lane_mask_px: np.ndarray) -> LaneSet:
        """Fit lane boundaries for one frame.

        Args:
            lane_mask_px: ``(H, W)`` binary lane mask in image pixels, at the
                frame size this fitter's camera is calibrated for.

        Returns:
            A :class:`~src.types.LaneSet`. Check ``status`` before trusting any
            field; degraded frames carry an explicit state, not a fake fit.
        """
        bev_mask = self.warp_mask(lane_mask_px)
        total = int(np.count_nonzero(bev_mask))
        if total == 0:
            return LaneSet(status=LaneStatus.NOT_DETECTED, eval_distance_m=self.cfg.eval_distance_m)
        if total < self.cfg.min_mask_pixels:
            return LaneSet(
                status=LaneStatus.MASK_TOO_SPARSE, eval_distance_m=self.cfg.eval_distance_m
            )

        x_m, y_m = self._bev_pixels_to_metres(bev_mask)
        bases = self._find_bases(x_m, y_m)
        if not bases:
            return LaneSet(status=LaneStatus.NOT_DETECTED, eval_distance_m=self.cfg.eval_distance_m)

        # The ego boundaries are the candidates straddling the camera axis.
        left_bases = [b for b in bases if b < 0]
        right_bases = [b for b in bases if b >= 0]
        ego_left_base = max(left_bases) if left_bases else None
        ego_right_base = min(right_bases) if right_bases else None

        fits: dict[str, LaneFit] = {}
        for base, role in ((ego_left_base, "ego_left"), (ego_right_base, "ego_right")):
            if base is None:
                continue
            sel = self._slide(x_m, y_m, base)
            fit = self._fit_one(x_m[sel], y_m[sel], role)
            if fit is not None:
                fits[role] = fit

        adjacent: list[LaneFit] = []
        if self.cfg.adjacent_lanes:
            for base in bases:
                if base in (ego_left_base, ego_right_base):
                    continue
                role = "adjacent_left" if base < 0 else "adjacent_right"
                sel = self._slide(x_m, y_m, base)
                fit = self._fit_one(x_m[sel], y_m[sel], role)
                if fit is not None:
                    adjacent.append(fit)

        left, right = fits.get("ego_left"), fits.get("ego_right")

        # Reject an implausible pairing rather than reporting a nonsense width.
        if left is not None and right is not None:
            width = float(right.x_at(self.cfg.eval_distance_m)[0] - left.x_at(self.cfg.eval_distance_m)[0])
            if not self.cfg.min_lane_width_m <= width <= self.cfg.max_lane_width_m:
                log_kv(
                    _log, logging.DEBUG, "lanes.implausible_width",
                    width_m=f"{width:.2f}",
                    bounds=f"{self.cfg.min_lane_width_m}-{self.cfg.max_lane_width_m}",
                    action="demoting the weaker boundary to adjacent",
                )
                weaker, stronger = (
                    (left, right) if left.n_pixels < right.n_pixels else (right, left)
                )
                adjacent.append(
                    LaneFit(
                        coeffs=weaker.coeffs,
                        role="adjacent_left" if weaker.role == "ego_left" else "adjacent_right",
                        n_pixels=weaker.n_pixels,
                        y_range_m=weaker.y_range_m,
                    )
                )
                left, right = (stronger, None) if stronger.role == "ego_left" else (None, stronger)

        if left is None and right is None:
            return LaneSet(
                status=LaneStatus.MASK_TOO_SPARSE,
                adjacent=tuple(adjacent),
                eval_distance_m=self.cfg.eval_distance_m,
            )
        if left is None or right is None:
            return LaneSet(
                status=LaneStatus.SINGLE_BOUNDARY,
                ego_left=left,
                ego_right=right,
                adjacent=tuple(adjacent),
                eval_distance_m=self.cfg.eval_distance_m,
            )

        y_eval = self.cfg.eval_distance_m
        x_left = float(left.x_at(y_eval)[0])
        x_right = float(right.x_at(y_eval)[0])
        centre_coeffs = (left.coeffs + right.coeffs) / 2.0
        centre = LaneFit(
            coeffs=centre_coeffs, role="ego_centre", n_pixels=left.n_pixels + right.n_pixels,
            y_range_m=(
                max(left.y_range_m[0], right.y_range_m[0]),
                min(left.y_range_m[1], right.y_range_m[1]),
            ),
        )
        return LaneSet(
            status=LaneStatus.OK,
            ego_left=left,
            ego_right=right,
            adjacent=tuple(adjacent),
            curvature_radius_m=centre.curvature_radius_m(y_eval),
            # The camera sits at x = 0. A lane centre to our right means we are
            # left of centre, hence the sign flip: positive = ego right of centre.
            ego_offset_m=-0.5 * (x_left + x_right),
            lane_width_m=x_right - x_left,
            eval_distance_m=y_eval,
        )


def fit_to_image_points(
    fit: LaneFit,
    camera: CameraModel,
    y_range_m: tuple[float, float] | None = None,
    n_samples: int = 40,
) -> np.ndarray:
    """Reproject a fitted polynomial from bird's-eye metres back into the image.

    This is the main visual check that calibration and fitting agree: the
    reprojected curve should lie along the lane pixels it was fitted to. Points
    that fall on or above the horizon are dropped rather than raising, because a
    boundary fitted to 40 m will routinely reach the vanishing line.

    Returns:
        ``(M, 2)`` image pixel coordinates, ordered near to far. May be empty.
    """
    lo, hi = y_range_m if y_range_m is not None else fit.y_range_m
    if hi <= lo:
        return np.zeros((0, 2), dtype=np.float64)
    ys = np.linspace(lo, hi, n_samples)
    xs = fit.x_at(ys)
    ground = np.column_stack([xs, ys])

    # Filter to points that actually have a finite image, then project.
    H = camera.H_ground_to_image
    homogeneous = np.hstack([ground, np.ones((len(ground), 1))])
    w = homogeneous @ H[2]
    ok = np.abs(w) > 1e-9
    if not np.any(ok):
        return np.zeros((0, 2), dtype=np.float64)
    out = (homogeneous[ok] @ H.T)
    return out[:, :2] / out[:, 2:3]
