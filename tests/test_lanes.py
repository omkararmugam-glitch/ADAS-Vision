"""Tests for lane fitting, against synthetic masks with known ground truth.

Masks are painted by taking a *known* polynomial in ego metres, projecting it
through the calibration into image pixels, and filling a marking-width band --
the same route a real marking takes. The fitter then has to recover the
coefficients it was never told, so curvature, width and offset are all checked
against numbers the fitter never saw.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.calibrate import CameraModel, rectangle_world_points, solve_ground_homography
from src.config import BEVConfig, LaneFitConfig
from src.lanes import LaneFitter, fit_to_image_points
from src.types import LaneStatus
from tests.test_calibrate import HEIGHT, WIDTH, project_ground_to_image

MARKING_WIDTH_M = 0.16
LANE_RANGE_M = (5.0, 38.0)


@pytest.fixture
def camera() -> CameraModel:
    world = rectangle_world_points(3.5, 6.0, 30.0)
    H, _ = solve_ground_homography(project_ground_to_image(world), world)
    return CameraModel(H_image_to_ground=H, image_size=(WIDTH, HEIGHT))


@pytest.fixture
def bev() -> BEVConfig:
    return BEVConfig(x_min_m=-6, x_max_m=6, y_min_m=4, y_max_m=40, px_per_m=20, grid_step_m=5)


@pytest.fixture
def cfg() -> LaneFitConfig:
    return LaneFitConfig(eval_distance_m=10.0)


def paint_lane_mask(
    camera: CameraModel,
    polys: list[np.ndarray],
    y_range_m: tuple[float, float] = LANE_RANGE_M,
    width_m: float = MARKING_WIDTH_M,
    size: tuple[int, int] = (WIDTH, HEIGHT),
) -> np.ndarray:
    """Render lane markings into an image-space mask from ground polynomials.

    Args:
        polys: Each is ``[a, b, c]`` for ``x = a*y^2 + b*y + c`` in ego metres.
    """
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    ys = np.linspace(*y_range_m, 200)
    for coeffs in polys:
        xs = np.polyval(coeffs, ys)
        inner = camera.ground_to_image(np.column_stack([xs - width_m / 2, ys]))
        outer = camera.ground_to_image(np.column_stack([xs + width_m / 2, ys]))
        band = np.vstack([inner, outer[::-1]])
        good = np.isfinite(band).all(axis=1)
        band = band[good]
        if len(band) >= 3:
            cv2.fillPoly(mask, [np.int32(np.round(band))], 1)
    return mask


class TestStraightLanes:
    def test_recovers_both_boundaries(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.OK
        assert lanes.ego_left is not None and lanes.ego_right is not None

    def test_recovers_lane_width(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.lane_width_m == pytest.approx(3.5, abs=0.12)

    def test_recovers_lateral_positions(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert float(lanes.ego_left.x_at(10.0)[0]) == pytest.approx(-1.75, abs=0.10)
        assert float(lanes.ego_right.x_at(10.0)[0]) == pytest.approx(1.75, abs=0.10)

    def test_straight_road_reports_a_huge_radius(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.curvature_radius_m > 2000.0

    def test_centred_ego_has_near_zero_offset(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.ego_offset_m == pytest.approx(0.0, abs=0.10)


class TestLateralOffset:
    @pytest.mark.parametrize("shift", [-0.5, -0.25, 0.3, 0.6])
    def test_offset_matches_the_painted_shift(self, camera, bev, cfg, shift: float) -> None:
        """Ego sits at x=0; shifting both markings right means we are left of centre."""
        mask = paint_lane_mask(
            camera,
            [np.array([0, 0, -1.75 + shift]), np.array([0, 0, 1.75 + shift])],
        )
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.OK
        assert lanes.ego_offset_m == pytest.approx(-shift, abs=0.12)

    def test_sign_convention_positive_is_right_of_centre(self, camera, bev, cfg) -> None:
        # Markings shifted left => the lane centre is left of us => we are right of it.
        mask = paint_lane_mask(
            camera, [np.array([0, 0, -1.75 - 0.4]), np.array([0, 0, 1.75 - 0.4])]
        )
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.ego_offset_m > 0


class TestCurvature:
    @pytest.mark.parametrize("radius_m", [300.0, 600.0, 1000.0])
    def test_recovers_a_known_curvature_radius(self, camera, bev, cfg, radius_m: float) -> None:
        """For x = a*y^2 with small slope, the radius at y=0 is 1/(2a)."""
        a = 1.0 / (2.0 * radius_m)
        mask = paint_lane_mask(
            camera, [np.array([a, 0, -1.75]), np.array([a, 0, 1.75])]
        )
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.OK
        # 25% tolerance: curvature is a second-derivative estimate off a
        # rasterised, perspective-warped mask, so it is inherently noisy.
        assert lanes.curvature_radius_m == pytest.approx(radius_m, rel=0.25)

    def test_curve_direction_is_preserved(self, camera, bev, cfg) -> None:
        left_curve = paint_lane_mask(
            camera, [np.array([-0.002, 0, -1.75]), np.array([-0.002, 0, 1.75])]
        )
        lanes = LaneFitter(camera, bev, cfg).fit(left_curve)
        assert lanes.ego_left.coeffs[0] < 0, "a curve bending left must have a < 0"

    def test_curved_lane_still_measures_the_right_width(self, camera, bev, cfg) -> None:
        a = 1.0 / (2.0 * 400.0)
        mask = paint_lane_mask(camera, [np.array([a, 0, -1.75]), np.array([a, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.lane_width_m == pytest.approx(3.5, abs=0.15)


class TestDegradedInput:
    """Degraded evidence must produce an explicit state, never a garbage fit."""

    def test_empty_mask_is_not_detected(self, camera, bev, cfg) -> None:
        lanes = LaneFitter(camera, bev, cfg).fit(np.zeros((HEIGHT, WIDTH), dtype=np.uint8))
        assert lanes.status is LaneStatus.NOT_DETECTED
        assert lanes.ego_left is None and lanes.ego_right is None
        assert lanes.curvature_radius_m is None
        assert lanes.ego_offset_m is None
        assert not lanes.detected

    def test_sparse_mask_is_reported_as_sparse(self, camera, bev, cfg) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        mask[700:703, 600:610] = 1  # a handful of pixels near the bottom
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.MASK_TOO_SPARSE
        assert lanes.ego_left is None and lanes.ego_right is None

    def test_single_boundary_is_reported(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.SINGLE_BOUNDARY
        assert lanes.ego_left is not None
        assert lanes.ego_right is None
        assert lanes.lane_width_m is None, "width is meaningless with one boundary"
        assert lanes.ego_offset_m is None, "offset is meaningless with one boundary"

    def test_implausible_width_is_rejected_not_reported(self, camera, bev, cfg) -> None:
        """Two markings 8 m apart are not one lane; the pair must be refused."""
        mask = paint_lane_mask(camera, [np.array([0, 0, -4.2]), np.array([0, 0, 4.2])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is not LaneStatus.OK
        assert lanes.lane_width_m is None

    def test_no_crash_on_a_full_white_mask(self, camera, bev, cfg) -> None:
        lanes = LaneFitter(camera, bev, cfg).fit(np.ones((HEIGHT, WIDTH), dtype=np.uint8))
        assert isinstance(lanes.status, LaneStatus)

    def test_pure_noise_does_not_produce_a_confident_lane(self, camera, bev, cfg) -> None:
        rng = np.random.default_rng(0)
        mask = (rng.random((HEIGHT, WIDTH)) < 0.002).astype(np.uint8)
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        if lanes.status is LaneStatus.OK:
            # If it does fit something, the width bound must still have held.
            assert cfg.min_lane_width_m <= lanes.lane_width_m <= cfg.max_lane_width_m


class TestAdjacentLanes:
    def test_finds_boundaries_outside_the_ego_lane(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(
            camera,
            [
                np.array([0, 0, -5.25]),
                np.array([0, 0, -1.75]),
                np.array([0, 0, 1.75]),
                np.array([0, 0, 5.25]),
            ],
        )
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.status is LaneStatus.OK
        assert lanes.lane_width_m == pytest.approx(3.5, abs=0.15)
        assert len(lanes.adjacent) >= 1, "adjacent boundaries should be reported"

    def test_adjacent_fitting_can_be_disabled(self, camera, bev) -> None:
        cfg = LaneFitConfig(adjacent_lanes=False)
        mask = paint_lane_mask(
            camera,
            [np.array([0, 0, -5.25]), np.array([0, 0, -1.75]), np.array([0, 0, 1.75])],
        )
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert lanes.adjacent == ()


class TestReprojection:
    def test_fitted_curve_reprojects_onto_the_pixels_it_was_fitted_to(
        self, camera, bev, cfg
    ) -> None:
        """The visual check, asserted numerically: curve must lie on the mask."""
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        pts = fit_to_image_points(lanes.ego_left, camera, y_range_m=(8.0, 30.0), n_samples=25)
        assert len(pts) > 10

        # Every sampled point should land on or beside a painted lane pixel.
        hits = 0
        for u, v in pts:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < HEIGHT and 0 <= ui < WIDTH:
                window = mask[max(0, vi - 4) : vi + 5, max(0, ui - 4) : ui + 5]
                hits += int(window.any())
        assert hits >= 0.85 * len(pts), f"only {hits}/{len(pts)} curve points hit the mask"

    def test_reprojection_is_empty_for_a_degenerate_range(self, camera, bev, cfg) -> None:
        mask = paint_lane_mask(camera, [np.array([0, 0, -1.75]), np.array([0, 0, 1.75])])
        lanes = LaneFitter(camera, bev, cfg).fit(mask)
        assert len(fit_to_image_points(lanes.ego_left, camera, y_range_m=(20.0, 20.0))) == 0


class TestLaneFitMath:
    def test_x_at_evaluates_the_polynomial(self) -> None:
        from src.types import LaneFit

        fit = LaneFit(coeffs=np.array([0.001, 0.02, -1.75]), role="ego_left",
                      n_pixels=500, y_range_m=(5, 40))
        assert float(fit.x_at(10.0)[0]) == pytest.approx(0.001 * 100 + 0.02 * 10 - 1.75)

    def test_straight_line_has_infinite_radius(self) -> None:
        from src.types import LaneFit

        fit = LaneFit(coeffs=np.array([0.0, 0.0, 1.75]), role="ego_right",
                      n_pixels=500, y_range_m=(5, 40))
        assert np.isinf(fit.curvature_radius_m(10.0))

    def test_radius_formula_matches_the_analytic_value(self) -> None:
        from src.types import LaneFit

        a = 0.001
        fit = LaneFit(coeffs=np.array([a, 0.0, 0.0]), role="x", n_pixels=1, y_range_m=(0, 1))
        assert fit.curvature_radius_m(0.0) == pytest.approx(1.0 / (2 * a))

    def test_infinite_radius_survives_json_round_trip(self) -> None:
        from src.types import LaneSet

        s = LaneSet(status=LaneStatus.OK, curvature_radius_m=float("inf"),
                    ego_offset_m=0.1, lane_width_m=3.5, eval_distance_m=10.0)
        assert np.isinf(LaneSet.from_dict(s.to_dict()).curvature_radius_m)

    def test_wrong_coefficient_count_raises(self) -> None:
        from src.types import LaneFit

        with pytest.raises(ValueError, match="3 polynomial coefficients"):
            LaneFit(coeffs=np.array([1.0, 2.0]), role="x", n_pixels=1, y_range_m=(0, 1))
