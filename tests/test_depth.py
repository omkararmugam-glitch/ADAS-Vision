"""Tests for monocular depth scale alignment.

The alignment is where monocular depth becomes a *distance*, and a wrong scale
produces confident, wrong metres -- the worst failure available to a collision
warning. So the tests concentrate on rejection: too few anchors, anchors that do
not constrain the fit, a depth ordering that disagrees with the ground plane, and
a fit that is numerically healthy but metrically wrong.

Anchors are generated from the exact pinhole relation, so the recoverable answer
is known analytically rather than approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from depth.align import DepthScale, fit_scale, sample_depth
from depth.midas import PlanarStubDepth, build_depth_estimator
from core.config import DepthConfig

# MiDaS-like output: relative inverse depth, d = A/Z + B for unknown A, B.
TRUE_A, TRUE_B = 240.0, 3.0


def synthetic(distances_m: np.ndarray) -> np.ndarray:
    return TRUE_A / np.asarray(distances_m, dtype=np.float64) + TRUE_B


class TestFitScale:
    def test_recovers_the_exact_affine_map(self) -> None:
        z = np.array([5.0, 8.0, 12.0, 18.0, 25.0, 40.0])
        scale = fit_scale(synthetic(z), z)
        assert scale.valid
        # 1/Z = (d - B)/A, so a = 1/A and b = -B/A
        assert scale.a == pytest.approx(1.0 / TRUE_A, rel=1e-6)
        assert scale.b == pytest.approx(-TRUE_B / TRUE_A, rel=1e-6)
        assert scale.r_squared == pytest.approx(1.0, abs=1e-9)
        assert scale.residual_m == pytest.approx(0.0, abs=1e-6)

    def test_round_trips_distances(self) -> None:
        z = np.array([6.0, 10.0, 15.0, 22.0, 33.0])
        scale = fit_scale(synthetic(z), z)
        assert scale.to_metres(synthetic(z)) == pytest.approx(z, rel=1e-6)

    def test_extrapolates_to_an_unseen_distance(self) -> None:
        """The whole point: transfer the anchors' scale to other pixels."""
        z = np.array([5.0, 9.0, 14.0, 20.0, 30.0])
        scale = fit_scale(synthetic(z), z)
        assert float(scale.to_metres(synthetic(np.array([17.5])))[0]) == pytest.approx(17.5, rel=1e-5)

    def test_survives_mild_noise(self) -> None:
        rng = np.random.default_rng(0)
        z = np.linspace(5.0, 40.0, 20)
        d = synthetic(z) + rng.normal(0.0, 0.4, size=z.size)
        scale = fit_scale(d, z)
        assert scale.valid
        assert scale.residual_m < 2.0

    def test_reports_the_anchor_count(self) -> None:
        z = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        assert fit_scale(synthetic(z), z).n_anchors == 5


class TestRejection:
    def test_too_few_anchors_is_invalid(self) -> None:
        z = np.array([10.0, 20.0])
        s = fit_scale(synthetic(z), z, min_anchors=4)
        assert not s.valid and s.n_anchors == 2

    def test_identical_depths_do_not_constrain_the_slope(self) -> None:
        s = fit_scale(np.full(6, 12.0), np.array([5.0, 9, 14, 20, 28, 35]))
        assert not s.valid

    def test_inverted_depth_ordering_is_rejected(self) -> None:
        """If the network says far is near, no scaling reconciles that."""
        z = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        s = fit_scale(-synthetic(z), z)
        assert not s.valid
        assert s.a <= 0.0

    def test_a_metrically_wrong_fit_is_rejected(self) -> None:
        """r_squared can look healthy while the metres are badly wrong."""
        rng = np.random.default_rng(3)
        z = np.linspace(5.0, 60.0, 25)
        d = synthetic(z) + rng.normal(0.0, 25.0, size=z.size)
        s = fit_scale(d, z, max_residual_m=2.0)
        assert not s.valid

    def test_non_finite_anchors_are_dropped(self) -> None:
        z = np.array([5.0, 10.0, np.nan, 20.0, 25.0, np.inf, 30.0])
        d = synthetic(np.array([5.0, 10.0, 1.0, 20.0, 25.0, 1.0, 30.0]))
        d[2] = np.nan
        s = fit_scale(d, z)
        assert s.n_anchors == 5 and s.valid

    def test_zero_and_negative_distances_are_dropped(self) -> None:
        z = np.array([0.0, -3.0, 8.0, 12.0, 18.0, 26.0])
        d = synthetic(np.array([1.0, 1.0, 8.0, 12.0, 18.0, 26.0]))
        s = fit_scale(d, z)
        assert s.n_anchors == 4

    def test_mismatched_input_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="depth values"):
            fit_scale(np.zeros(4), np.zeros(5))


class TestInvalidScaleIsSafe:
    def test_an_invalid_scale_returns_infinity_not_a_guess(self) -> None:
        """Silence is safe; a fabricated distance is not."""
        s = DepthScale()
        out = s.to_metres(np.array([1.0, 2.0, 3.0]))
        assert np.all(np.isinf(out))

    def test_distances_are_clipped_to_a_maximum(self) -> None:
        z = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        s = fit_scale(synthetic(z), z)
        # A depth value at the horizon maps beyond any useful range.
        assert float(s.to_metres(np.array([TRUE_B + 1e-6]), max_m=200.0)[0]) <= 200.0


class TestSampling:
    def test_samples_the_median_of_a_patch(self) -> None:
        dm = np.zeros((20, 20), np.float32)
        dm[9:12, 9:12] = 5.0
        assert float(sample_depth(dm, np.array([[10.0, 10.0]]), patch=3)[0]) == 5.0

    def test_an_outlier_pixel_does_not_dominate(self) -> None:
        """Depth maps are noisiest exactly at object boundaries."""
        dm = np.full((20, 20), 2.0, np.float32)
        dm[10, 10] = 900.0
        assert float(sample_depth(dm, np.array([[10.0, 10.0]]), patch=5)[0]) == 2.0

    def test_points_outside_the_image_give_nan(self) -> None:
        dm = np.ones((10, 10), np.float32)
        out = sample_depth(dm, np.array([[-5.0, 5.0], [50.0, 5.0], [5.0, 5.0]]))
        assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 1.0

    def test_patches_are_clipped_at_the_border(self) -> None:
        dm = np.ones((10, 10), np.float32)
        assert float(sample_depth(dm, np.array([[0.0, 0.0]]), patch=5)[0]) == 1.0


class TestEstimators:
    def test_the_stub_is_monotonic_down_the_image(self) -> None:
        """Nearer road is lower in the frame, so inverse depth must grow."""
        d = PlanarStubDepth().infer(np.zeros((120, 160, 3), np.uint8))
        col = d[:, 80]
        assert col[-1] > col[70] >= col[60]
        assert np.all(np.diff(col[60:]) >= -1e-6)

    def test_the_stub_matches_the_frame_size(self) -> None:
        d = PlanarStubDepth().infer(np.zeros((90, 160, 3), np.uint8))
        assert d.shape == (90, 160)

    def test_disabled_depth_builds_nothing(self) -> None:
        assert build_depth_estimator(DepthConfig(enabled=False)) is None

    def test_the_stub_kind_builds_without_weights(self) -> None:
        est = build_depth_estimator(DepthConfig(enabled=True, kind="stub"))
        assert est is not None and est.name == "depth-stub"

    def test_stub_depth_aligns_back_to_true_distances(self) -> None:
        """End to end: a planar depth map plus ground anchors recovers metres."""
        h, w = 240, 320
        est = PlanarStubDepth(horizon_frac=0.5, scale=1.0)
        depth = est.infer(np.zeros((h, w, 3), np.uint8))
        # Under the pinhole model with the stub's construction, row y maps to
        # Z proportional to 1/(y - horizon); pick rows and derive their Z.
        rows = np.array([140, 160, 180, 200, 220], dtype=np.float64)
        horizon = 0.5 * h
        z_true = 100.0 / (rows - horizon)
        pts = np.column_stack([np.full_like(rows, w / 2), rows])
        d = sample_depth(depth, pts, patch=3)
        scale = fit_scale(d, z_true)
        assert scale.valid
        assert scale.to_metres(d) == pytest.approx(z_true, rel=1e-3)


class TestCollisionDepthFallback:
    """Depth must fill the gap the homography leaves, and never override it."""

    @staticmethod
    def _camera():
        from core.calibrate import CameraModel, rectangle_world_points, solve_ground_homography
        W, H, F, CAM_H, PITCH = 1280, 720, 900.0, 1.5, np.deg2rad(-4.0)

        def project(pts):
            pts = np.asarray(pts, float).reshape(-1, 2)
            x, y = pts[:, 0], pts[:, 1]
            z = -CAM_H
            c, s = np.cos(PITCH), np.sin(PITCH)
            yc, zc = c * y - s * z, s * y + c * z
            return np.column_stack([F * (x / yc) + W / 2, F * (-zc / yc) + H / 2])

        world = rectangle_world_points(3.5, 6.0, 30.0)
        Hm, _ = solve_ground_homography(project(world), world)
        return CameraModel(H_image_to_ground=Hm, image_size=(W, H)), project

    def _tracks(self, project, distances, above_horizon_extra=False):
        from core.config import TrackerConfig
        from core.types import FrameResult
        from tracking.bytetrack import ByteTracker
        boxes = []
        for z in distances:
            c = project(np.array([[-0.9, z], [0.9, z]]))
            boxes.append([c[0, 0], c[0, 1] - 60, c[1, 0], c[0, 1]])
        if above_horizon_extra:
            # A box whose bottom edge sits above the horizon: the homography
            # cannot place it at all.
            boxes.append([600.0, 40.0, 680.0, 120.0])
        blank = np.zeros((720, 1280), np.uint8)
        tr = ByteTracker(TrackerConfig(n_init=1))
        r = FrameResult(
            boxes_px=np.array(boxes, np.float32),
            class_ids=np.zeros(len(boxes), np.int32),
            scores=np.full(len(boxes), 0.9, np.float32),
            drivable_mask=blank, lane_mask=blank.copy(),
        )
        return tr.update(r, 0, 0.0)

    def test_homography_is_used_when_available(self) -> None:
        from core.config import DepthConfig, TTCConfig
        from tracking.collision import CollisionEstimator
        cam, project = self._camera()
        tracks = self._tracks(project, [10.0, 15.0, 22.0, 30.0])
        est = CollisionEstimator(TTCConfig(), cam, DepthConfig(enabled=True, kind="stub"))
        depth = PlanarStubDepth().infer(np.zeros((720, 1280, 3), np.uint8))
        states = est.update(tracks, 0.0, depth_map=depth)
        assert all(s.distance_source == "homography" for s in states)

    def test_depth_fills_in_where_the_homography_cannot(self) -> None:
        from core.config import DepthConfig, TTCConfig
        from tracking.collision import CollisionEstimator
        cam, project = self._camera()
        tracks = self._tracks(project, [8.0, 12.0, 18.0, 25.0, 34.0], above_horizon_extra=True)
        est = CollisionEstimator(TTCConfig(), cam, DepthConfig(enabled=True, kind="stub"))
        depth = PlanarStubDepth().infer(np.zeros((720, 1280, 3), np.uint8))
        states = est.update(tracks, 0.0, depth_map=depth)
        sources = {s.distance_source for s in states}
        assert "homography" in sources
        # The off-plane object gets a distance from depth, or honestly none --
        # never a fabricated homography value.
        assert sources <= {"homography", "depth", "none"}

    def test_without_a_depth_map_nothing_changes(self) -> None:
        from core.config import DepthConfig, TTCConfig
        from tracking.collision import CollisionEstimator
        cam, project = self._camera()
        tracks = self._tracks(project, [10.0, 16.0, 24.0])
        est = CollisionEstimator(TTCConfig(), cam, DepthConfig(enabled=True, kind="stub"))
        states = est.update(tracks, 0.0, depth_map=None)
        assert est.depth_scale is None
        assert all(s.distance_source in ("homography", "none") for s in states)

    def test_too_few_anchors_leaves_the_scale_invalid(self) -> None:
        from core.config import DepthConfig, TTCConfig
        from tracking.collision import CollisionEstimator
        cam, project = self._camera()
        tracks = self._tracks(project, [12.0])
        est = CollisionEstimator(TTCConfig(), cam,
                                 DepthConfig(enabled=True, kind="stub", min_anchors=4))
        depth = PlanarStubDepth().infer(np.zeros((720, 1280, 3), np.uint8))
        est.update(tracks, 0.0, depth_map=depth)
        assert est.depth_scale is not None and not est.depth_scale.valid
