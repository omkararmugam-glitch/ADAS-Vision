"""Tests for intrinsics, the ground-plane homography, and CameraModel.

The synthetic camera below is a real pinhole projection of a flat road, built
from an explicit pose (height, pitch, focal length). That gives every geometry
test an independent ground truth: the homography is *solved* from four points,
then checked against distances the solver never saw.

Coordinate frames follow src/calibrate.py: image px, and ego metres with X right
and Y forward from the camera's ground projection.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.calibrate import (
    CameraModel,
    Intrinsics,
    calibrate_intrinsics,
    draw_bev_grid,
    measure_width_between_verticals,
    rectangle_world_points,
    solve_ground_homography,
)
from core.config import BEVConfig, ChessboardConfig

WIDTH, HEIGHT = 1280, 720
#: Tolerance for quantities recovered through a *solved* homography. cv2's
#: normalised DLT is accurate to a few microns over a 45 m range, which is far
#: below any physical meaning here; 0.1 mm is a bound with room to spare that
#: would still catch a genuine geometry error.
SOLVE_TOL_M = 1e-4
FOCAL_PX = 900.0
CAM_HEIGHT_M = 1.5
PITCH_RAD = np.deg2rad(-4.0)  # nose-down: the road fills the lower image


def project_ground_to_image(points_m: np.ndarray) -> np.ndarray:
    """Ground-truth pinhole projection of road points, independent of the solver.

    Camera at ``(0, 0, CAM_HEIGHT_M)`` looking along +Y with a small downward
    pitch. Road points are ``(x, y)`` in ego metres at Z = 0.
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    z = -CAM_HEIGHT_M  # road is below the camera
    # Rotate about the camera's X axis by the pitch angle.
    c, s = np.cos(PITCH_RAD), np.sin(PITCH_RAD)
    y_cam = c * y - s * z
    z_cam = s * y + c * z
    # Camera frame: +x right, +y down, +z forward.
    u = FOCAL_PX * (x / y_cam) + WIDTH / 2.0
    v = FOCAL_PX * (-z_cam / y_cam) + HEIGHT / 2.0
    return np.column_stack([u, v])


@pytest.fixture
def truth_camera() -> CameraModel:
    """A CameraModel solved from 4 points of the synthetic pinhole projection."""
    world = rectangle_world_points(width_m=3.5, near_m=6.0, far_m=30.0)
    image = project_ground_to_image(world)
    H, _residual = solve_ground_homography(image, world)
    return CameraModel(H_image_to_ground=H, image_size=(WIDTH, HEIGHT))


class TestSolveGroundHomography:
    def test_recovers_the_four_calibration_points(self) -> None:
        world = rectangle_world_points(3.5, 6.0, 30.0)
        image = project_ground_to_image(world)
        H, residual = solve_ground_homography(image, world)
        assert residual < SOLVE_TOL_M
        assert H.shape == (3, 3)

    def test_generalises_to_points_it_never_saw(self, truth_camera: CameraModel) -> None:
        """The real test: distances at ranges outside the calibration rectangle."""
        unseen = np.array(
            [[0.0, 10.0], [1.0, 15.0], [-1.5, 22.0], [0.5, 45.0], [-3.0, 8.0]]
        )
        px = project_ground_to_image(unseen)
        recovered = truth_camera.image_to_ground(px)
        assert recovered == pytest.approx(unseen, abs=SOLVE_TOL_M)

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError, match="at least 4 points"):
            solve_ground_homography(np.zeros((3, 2)), np.zeros((3, 2)))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            solve_ground_homography(np.zeros((4, 2)), np.zeros((5, 2)))

    def test_rejects_collinear_points(self) -> None:
        collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        world = rectangle_world_points(3.5, 6.0, 30.0)
        with pytest.raises(ValueError):
            solve_ground_homography(collinear, world)

    def test_overdetermined_fit_reports_a_real_residual(self) -> None:
        """With more than 4 points, a perturbed point produces a nonzero residual."""
        world = np.array(
            [[-1.75, 6.0], [1.75, 6.0], [-1.75, 30.0], [1.75, 30.0], [0.0, 15.0]]
        )
        image = project_ground_to_image(world)
        image[4] += np.array([6.0, 6.0])  # nudge the extra point off the true ray
        _H, residual = solve_ground_homography(image, world)
        assert residual > 0.01


class TestRoundTrip:
    """project -> unproject -> compare, in both directions."""

    def test_ground_to_image_to_ground(self, truth_camera: CameraModel) -> None:
        xs = np.linspace(-4.0, 4.0, 9)
        ys = np.linspace(5.0, 60.0, 12)
        grid = np.array([[x, y] for x in xs for y in ys])
        px = truth_camera.ground_to_image(grid)
        back = truth_camera.image_to_ground(px)
        assert back == pytest.approx(grid, abs=1e-6)

    def test_image_to_ground_to_image(self, truth_camera: CameraModel) -> None:
        us = np.linspace(50, WIDTH - 50, 9)
        vs = np.linspace(HEIGHT * 0.62, HEIGHT - 2, 8)  # strictly below the horizon
        px = np.array([[u, v] for u in us for v in vs])
        assert np.all(truth_camera.is_on_ground(px))
        ground = truth_camera.image_to_ground(px)
        back = truth_camera.ground_to_image(ground)
        assert back == pytest.approx(px, abs=1e-6)

    def test_round_trip_holds_at_realistic_precision(self, truth_camera: CameraModel) -> None:
        """A tolerance stated in metres, which is what actually matters downstream."""
        targets = np.array([[0.0, 8.0], [0.0, 20.0], [0.0, 40.0], [2.0, 35.0]])
        px = truth_camera.ground_to_image(targets)
        back = truth_camera.image_to_ground(px)
        error_m = np.linalg.norm(back - targets, axis=1)
        assert np.max(error_m) < 1e-3  # sub-millimetre; the limit is float precision


class TestHorizon:
    def test_points_above_the_horizon_raise(self, truth_camera: CameraModel) -> None:
        horizon = float(truth_camera.horizon_v(WIDTH / 2)[0])
        above = np.array([[WIDTH / 2, horizon - 50.0]])
        assert not truth_camera.is_on_ground(above)[0]
        with pytest.raises(ValueError, match="horizon"):
            truth_camera.image_to_ground(np.array([[WIDTH / 2, horizon]]))

    def test_is_on_ground_separates_road_from_sky(self, truth_camera: CameraModel) -> None:
        horizon = float(truth_camera.horizon_v(WIDTH / 2)[0])
        pts = np.array(
            [[WIDTH / 2, horizon + 60], [WIDTH / 2, HEIGHT - 1], [WIDTH / 2, horizon - 60]]
        )
        assert truth_camera.is_on_ground(pts).tolist() == [True, True, False]

    def test_horizon_matches_the_synthetic_camera_geometry(
        self, truth_camera: CameraModel
    ) -> None:
        """A point infinitely far ahead should image at the computed horizon row."""
        far = project_ground_to_image(np.array([[0.0, 1e7]]))[0]
        horizon = float(truth_camera.horizon_v(WIDTH / 2)[0])
        assert horizon == pytest.approx(far[1], abs=0.01)


class TestForImageSize:
    def test_half_resolution_yields_the_same_metres(self, truth_camera: CameraModel) -> None:
        """Rescaling the model must undo rescaling the pixels, exactly."""
        ground = np.array([[0.0, 12.0], [1.5, 25.0], [-2.0, 40.0]])
        px_full = truth_camera.ground_to_image(ground)

        half = truth_camera.for_image_size((WIDTH // 2, HEIGHT // 2))
        px_half = px_full * 0.5
        assert half.image_to_ground(px_half) == pytest.approx(ground, abs=1e-6)

    def test_same_size_returns_the_same_object(self, truth_camera: CameraModel) -> None:
        assert truth_camera.for_image_size((WIDTH, HEIGHT)) is truth_camera

    def test_non_uniform_rescale_is_handled(self, truth_camera: CameraModel) -> None:
        ground = np.array([[1.0, 18.0]])
        px = truth_camera.ground_to_image(ground)
        squished = truth_camera.for_image_size((WIDTH // 2, HEIGHT))
        assert squished.image_to_ground(px * np.array([0.5, 1.0])) == pytest.approx(
            ground, abs=1e-6
        )


class TestPersistence:
    def test_camera_model_round_trips_through_yaml(self, truth_camera, tmp_path) -> None:
        model = CameraModel(
            H_image_to_ground=truth_camera.H_image_to_ground,
            image_size=(WIDTH, HEIGHT),
            undistort_applied=False,
            residual_m=0.0123,
            source_points_px=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            source_points_m=rectangle_world_points(3.5, 6.0, 30.0),
        )
        path = model.save(tmp_path / "homography.yaml")
        loaded = CameraModel.load(path)
        assert loaded.H_image_to_ground == pytest.approx(model.H_image_to_ground)
        assert loaded.image_size == (WIDTH, HEIGHT)
        assert loaded.residual_m == pytest.approx(0.0123)
        assert loaded.source_points_m == pytest.approx(model.source_points_m)

        ground = np.array([[0.0, 20.0]])
        assert loaded.ground_to_image(ground) == pytest.approx(
            model.ground_to_image(ground)
        )

    def test_missing_homography_raises_and_says_what_to_run(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="run.py calibrate homography"):
            CameraModel.load(tmp_path / "absent.yaml")

    def test_missing_intrinsics_raises_and_says_what_to_run(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="run.py calibrate intrinsics"):
            Intrinsics.load(tmp_path / "absent.yaml")

    def test_singular_homography_rejected(self) -> None:
        with pytest.raises(ValueError, match="singular"):
            CameraModel(H_image_to_ground=np.zeros((3, 3)), image_size=(WIDTH, HEIGHT))

    def test_wrong_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            CameraModel(H_image_to_ground=np.eye(4), image_size=(WIDTH, HEIGHT))


class TestBEV:
    def test_warp_matrix_places_metres_where_expected(self, truth_camera: CameraModel) -> None:
        bev = BEVConfig(x_min_m=-6, x_max_m=6, y_min_m=3, y_max_m=40, px_per_m=20)
        M = truth_camera.bev_warp_matrix(bev)

        # The ego origin projected forward to y_max should land on the top edge,
        # and x = 0 should land at the horizontal centre of the window.
        px = truth_camera.ground_to_image(np.array([[0.0, 40.0]]))
        homogeneous = np.array([px[0, 0], px[0, 1], 1.0])
        out = M @ homogeneous
        out = out[:2] / out[2]
        assert out[0] == pytest.approx(bev.size_px[0] / 2, abs=1e-6)
        assert out[1] == pytest.approx(0.0, abs=1e-6)

    def test_forward_distance_increases_upward(self, truth_camera: CameraModel) -> None:
        bev = BEVConfig()
        M = truth_camera.bev_warp_matrix(bev)
        near = truth_camera.ground_to_image(np.array([[0.0, 5.0]]))[0]
        far = truth_camera.ground_to_image(np.array([[0.0, 35.0]]))[0]
        v_near = (M @ [near[0], near[1], 1.0])
        v_far = (M @ [far[0], far[1], 1.0])
        assert (v_near[1] / v_near[2]) > (v_far[1] / v_far[2])

    def test_warp_rejects_a_mismatched_frame_size(self, truth_camera: CameraModel) -> None:
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="calibrated for"):
            truth_camera.warp_to_bev(small, BEVConfig())

    def test_grid_overlay_draws_without_changing_size(self) -> None:
        bev = BEVConfig()
        w, h = bev.size_px
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        gridded = draw_bev_grid(blank, bev)
        assert gridded.shape == blank.shape
        assert gridded.any(), "grid overlay drew nothing"

    def test_bev_size_matches_the_configured_window(self) -> None:
        bev = BEVConfig(x_min_m=-5, x_max_m=5, y_min_m=4, y_max_m=44, px_per_m=10)
        assert bev.size_px == (100, 400)


class TestWidthVerification:
    def test_constant_width_for_a_correct_homography(self, truth_camera: CameraModel) -> None:
        """Two image columns bounding a real 3.5 m lane must stay 3.5 m apart."""
        lane_left = project_ground_to_image(np.array([[-1.75, 20.0]]))[0]
        lane_right = project_ground_to_image(np.array([[1.75, 20.0]]))[0]
        # Sample the vertical band the lane occupies at that one row only: the
        # lane edges converge in the image, so a fixed column pair is only
        # 3.5 m apart at the row where it was measured.
        table = measure_width_between_verticals(
            truth_camera, lane_left[0], lane_right[0], [lane_left[1]]
        )
        assert len(table) == 1
        dist_m, sep_m = table[0]
        assert dist_m == pytest.approx(20.0, abs=SOLVE_TOL_M)
        assert sep_m == pytest.approx(3.5, abs=SOLVE_TOL_M)

    def test_skips_rows_above_the_horizon(self, truth_camera: CameraModel) -> None:
        horizon = float(truth_camera.horizon_v(WIDTH / 2)[0])
        table = measure_width_between_verticals(
            truth_camera, 400.0, 800.0, [horizon - 20.0, HEIGHT - 1.0]
        )
        assert len(table) == 1  # the above-horizon row was dropped, not projected


class TestRectangleWorldPoints:
    def test_order_and_values(self) -> None:
        pts = rectangle_world_points(3.5, 6.0, 30.0)
        assert pts.tolist() == [[-1.75, 6.0], [1.75, 6.0], [-1.75, 30.0], [1.75, 30.0]]

    @pytest.mark.parametrize(
        "args", [(0.0, 6.0, 30.0), (-3.5, 6.0, 30.0), (3.5, 30.0, 6.0), (3.5, -1.0, 30.0)]
    )
    def test_invalid_geometry_raises(self, args) -> None:
        with pytest.raises(ValueError):
            rectangle_world_points(*args)


class TestIntrinsics:
    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="directory not found"):
            calibrate_intrinsics(tmp_path / "nope", ChessboardConfig())

    def test_empty_directory_raises(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no images"):
            calibrate_intrinsics(tmp_path / "empty", ChessboardConfig())

    def test_too_few_detections_raises_with_a_useful_hint(self, tmp_path) -> None:
        import cv2

        d = tmp_path / "noise"
        d.mkdir()
        for i in range(4):
            cv2.imwrite(str(d / f"{i}.png"), np.zeros((200, 200, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="INNER corners"):
            calibrate_intrinsics(d, ChessboardConfig())

    def test_intrinsics_round_trip_through_yaml(self, tmp_path) -> None:
        intr = Intrinsics(
            camera_matrix=np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]]),
            dist_coeffs=np.array([[-0.24, 0.09, 0.0, 0.0, -0.02]]),
            image_size=(WIDTH, HEIGHT),
            rms_reprojection_error_px=0.4212,
            n_images_used=17,
            n_images_total=20,
        )
        loaded = Intrinsics.load(intr.save(tmp_path / "intrinsics.yaml"))
        assert loaded.camera_matrix == pytest.approx(intr.camera_matrix)
        assert loaded.dist_coeffs.ravel() == pytest.approx(intr.dist_coeffs.ravel())
        assert loaded.image_size == (WIDTH, HEIGHT)
        assert loaded.rms_reprojection_error_px == pytest.approx(0.4212)
        assert loaded.n_images_used == 17

    def test_undistort_rejects_a_mismatched_size(self) -> None:
        intr = Intrinsics(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros((1, 5)),
            image_size=(WIDTH, HEIGHT),
            rms_reprojection_error_px=0.1,
            n_images_used=5,
            n_images_total=5,
        )
        with pytest.raises(ValueError, match="calibrated at"):
            intr.undistort(np.zeros((10, 10, 3), dtype=np.uint8))
