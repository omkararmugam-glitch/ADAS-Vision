"""Tests for the parallel lane model and the far field it buys.

The problem this solves is invisible near the car and severe far from it. Fitted
independently, each boundary is extrapolated on its own pixels; a boundary
supported by a short stub of dashed paint gives a quadratic that is fine over the
stub and flies off the road beyond it. That is precisely the part of the image a
driver is looking at.

The risk it introduces is the opposite one: a shared curvature *cannot* express
a fork, so applying it blindly would erase the divergence the split detector
exists to find. Hence the model selection, and hence the tests here come in
pairs -- one that the parallel model is adopted when the road is parallel, one
that it is rejected when the road is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.lane_seg.fitting import (
    _PARALLEL_SLACK_M,
    _PARALLEL_TOLERANCE,
    fit_parallel_pair,
)


def samples(a: float, b: float, c: float, lo: float, hi: float, n: int = 40):
    """Points on x = a*y^2 + b*y + c over [lo, hi]."""
    y = np.linspace(lo, hi, n)
    return y, a * y**2 + b * y + c


class TestTheJointSolve:
    def test_it_recovers_a_known_parallel_pair(self) -> None:
        a, b = 0.0016, -0.03
        y_l, x_l = samples(a, b, -1.85, 5.0, 35.0)
        y_r, x_r = samples(a, b, 1.85, 5.0, 35.0)
        got = fit_parallel_pair(y_l, x_l, y_r, x_r)
        assert got is not None
        cl, cr, rms = got
        assert rms == pytest.approx(0.0, abs=1e-6)
        assert cl == pytest.approx([a, b, -1.85], abs=1e-6)
        assert cr == pytest.approx([a, b, 1.85], abs=1e-6)

    def test_the_two_boundaries_share_curvature_exactly(self) -> None:
        y_l, x_l = samples(0.002, 0.01, -1.75, 5.0, 30.0)
        y_r, x_r = samples(0.002, 0.01, 1.75, 5.0, 30.0)
        cl, cr, _ = fit_parallel_pair(y_l, x_l, y_r, x_r)
        assert cl[0] == cr[0]
        assert cl[1] == cr[1]

    def test_lane_width_is_constant_by_construction(self) -> None:
        """The corridor cannot fan open into the nine-metre lane."""
        y_l, x_l = samples(0.003, -0.05, -1.9, 4.0, 32.0)
        y_r, x_r = samples(0.003, -0.05, 1.9, 4.0, 32.0)
        cl, cr, _ = fit_parallel_pair(y_l, x_l, y_r, x_r)
        ys = np.linspace(4.0, 80.0, 25)          # far past the fitted range
        widths = np.polyval(cr, ys) - np.polyval(cl, ys)
        assert np.allclose(widths, widths[0], atol=1e-9)

    def test_a_short_boundary_inherits_the_long_one_s_curvature(self) -> None:
        """The whole point. The right boundary sees 5-10 m and must still bend
        correctly at 35 m, because the left boundary saw that far."""
        a, b, n_wide = 0.0025, -0.04, 60
        y_l, x_l = samples(a, b, -1.85, 5.0, 35.0, n_wide)
        y_r, x_r = samples(a, b, 1.85, 5.0, 10.0, 8)      # a stub
        cl, cr, _ = fit_parallel_pair(y_l, x_l, y_r, x_r)

        truth_at_35 = a * 35**2 + b * 35 + 1.85
        joint_at_35 = float(np.polyval(cr, 35.0))
        alone = np.polyfit(y_r, x_r, 2)
        alone_at_35 = float(np.polyval(alone, 35.0))

        assert abs(joint_at_35 - truth_at_35) < 0.05
        # The independent fit on a stub is wrong by metres out there.
        assert abs(alone_at_35 - truth_at_35) > abs(joint_at_35 - truth_at_35)

    def test_noise_is_averaged_across_both_boundaries(self) -> None:
        rng = np.random.default_rng(0)
        a, b = 0.002, -0.02
        y_l, x_l = samples(a, b, -1.85, 5.0, 35.0)
        y_r, x_r = samples(a, b, 1.85, 5.0, 35.0)
        x_l = x_l + rng.normal(0, 0.08, len(x_l))
        x_r = x_r + rng.normal(0, 0.08, len(x_r))
        cl, cr, _ = fit_parallel_pair(y_l, x_l, y_r, x_r)
        assert abs(cl[0] - a) < 5e-4

    def test_too_few_points_is_refused(self) -> None:
        y_l, x_l = samples(0.0, 0.0, -1.75, 5.0, 10.0, 2)
        y_r, x_r = samples(0.0, 0.0, 1.75, 5.0, 35.0, 40)
        assert fit_parallel_pair(y_l, x_l, y_r, x_r) is None
        assert fit_parallel_pair(y_r, x_r, y_l, x_l) is None

    def test_a_fork_fits_the_parallel_model_badly(self) -> None:
        """The signal the selection rule reads: a diverging pair leaves a large
        residual under a model that cannot represent divergence."""
        y_l, x_l = samples(0.0, 0.0, -1.85, 5.0, 35.0)
        y_r, x_r = samples(0.0, 0.12, 1.85, 5.0, 35.0)     # peeling away
        _, _, rms_fork = fit_parallel_pair(y_l, x_l, y_r, x_r)

        y_r2, x_r2 = samples(0.0, 0.0, 1.85, 5.0, 35.0)
        _, _, rms_ok = fit_parallel_pair(y_l, x_l, y_r2, x_r2)
        assert rms_fork > 0.5
        assert rms_ok < 1e-6
        # And the fork's residual is well outside the acceptance budget.
        assert rms_fork > rms_ok * _PARALLEL_TOLERANCE + _PARALLEL_SLACK_M


class TestModelSelectionInTheFitter:
    """End to end through LaneFitter, on synthetic bird's-eye masks."""

    @staticmethod
    def _fitter():
        from core.autocalib import focal_from_fov, homography_from_pose
        from core.calibrate import CameraModel
        from core.config import BEVConfig, LaneFitConfig
        from models.lane_seg.fitting import LaneFitter

        W, H = 1280, 720
        Hm = homography_from_pose(focal_from_fov(W, 90.0), (W / 2, H / 2),
                                  np.deg2rad(6.0), 0.0, 1.35)
        cam = CameraModel(H_image_to_ground=Hm, image_size=(W, H))
        return LaneFitter(cam, BEVConfig(), LaneFitConfig()), cam, (W, H)

    @staticmethod
    def _mask(cam, size, curves, thickness=7, dash=None):
        """Paint boundaries into an image-space mask.

        ``dash`` is ``(role_index, y_max)``: that boundary stops early, which is
        what dashed paint running out looks like to the fitter.
        """
        import cv2

        W, H = size
        mask = np.zeros((H, W), np.uint8)
        for i, (a, b, c, lo, hi) in enumerate(curves):
            if dash is not None and i == dash[0]:
                hi = dash[1]
            ys = np.linspace(lo, hi, 200)
            xs = a * ys**2 + b * ys + c
            pts = cam.ground_to_image(np.column_stack([xs, ys]))
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if len(pts) >= 2:
                cv2.polylines(mask, [pts.reshape(-1, 1, 2).astype(np.int32)],
                              False, 255, thickness)
        return mask

    def test_a_parallel_road_adopts_the_shared_model(self) -> None:
        fitter, cam, size = self._fitter()
        mask = self._mask(cam, size, [(0.0015, -0.02, -1.85, 4.0, 38.0),
                                      (0.0015, -0.02, 1.85, 4.0, 38.0)])
        lanes = fitter.fit(mask)
        assert lanes.parallel_model is True

    def test_adopting_it_extends_the_weaker_boundary(self) -> None:
        """The right boundary's paint stops at 14 m; its curve must not."""
        fitter, cam, size = self._fitter()
        mask = self._mask(cam, size, [(0.0015, -0.02, -1.85, 4.0, 38.0),
                                      (0.0015, -0.02, 1.85, 4.0, 38.0)],
                          dash=(1, 14.0))
        lanes = fitter.fit(mask)
        assert lanes.parallel_model is True
        assert lanes.ego_right is not None
        assert lanes.ego_right.y_range_m[1] > 30.0

    def test_a_fork_keeps_the_independent_fits(self) -> None:
        """Model selection must not erase the divergence detect_split reads."""
        fitter, cam, size = self._fitter()
        mask = self._mask(cam, size, [(0.0, 0.0, -1.85, 4.0, 38.0),
                                      (0.0, 0.10, 1.85, 4.0, 38.0)])
        lanes = fitter.fit(mask)
        assert lanes.parallel_model is False

    def test_a_fork_is_still_detected_after_the_change(self) -> None:
        from models.lane_seg.split import detect_split

        fitter, cam, size = self._fitter()
        mask = self._mask(cam, size, [(0.0, 0.0, -1.85, 4.0, 38.0),
                                      (0.0, 0.10, 1.85, 4.0, 38.0)])
        lanes = fitter.fit(mask)
        assert detect_split(lanes) is not None

    def test_the_flag_survives_a_json_round_trip(self) -> None:
        from core.types import LaneSet

        fitter, cam, size = self._fitter()
        mask = self._mask(cam, size, [(0.0015, -0.02, -1.85, 4.0, 38.0),
                                      (0.0015, -0.02, 1.85, 4.0, 38.0)])
        lanes = fitter.fit(mask)
        assert LaneSet.from_dict(lanes.to_dict()).parallel_model is lanes.parallel_model

    def test_an_older_cached_laneset_still_loads(self) -> None:
        """The field was added after v3 runs were cached; absence means False."""
        from core.types import LaneSet

        d = LaneSet(status=__import__(
            "core.types", fromlist=["LaneStatus"]).LaneStatus.OK).to_dict()
        d.pop("parallel_model")
        assert LaneSet.from_dict(d).parallel_model is False
