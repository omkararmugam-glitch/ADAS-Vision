"""Tests for target-free calibration from road geometry.

The property that matters most is REFUSAL. An invented homography produces
confident, wrong metres, which is worse than no calibration at all -- the
pipeline already degrades gracefully when calibration is absent. So most of
these check that implausible evidence is rejected rather than fitted.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.autocalib import (
    HORIZON_BAND,
    MAX_PLAUSIBLE_PITCH_DEG,
    MAX_PLAUSIBLE_YAW_DEG,
    estimate,
    focal_from_fov,
    homography_from_pose,
    vanishing_point,
)

W, H = 1280, 720


def road_frame(pitch_deg=6.0, height=1.35, hfov=90.0, lanes=(-5.25, -1.75, 1.75, 5.25)):
    """A synthetic straight road, drawn through a camera of known pose."""
    focal = focal_from_fov(W, hfov)
    pitch = np.deg2rad(pitch_deg)

    def project(pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]
        z = -height
        c, s = np.cos(pitch), np.sin(pitch)
        yc, zc = c * y - s * z, s * y + c * z
        return np.column_stack([focal * (x / yc) + W / 2, focal * (-zc / yc) + H / 2])

    img = np.full((H, W, 3), 60, np.uint8)
    img[: H // 2 - 40] = (150, 140, 130)
    ys = np.arange(6.0, 70.0, 0.25)
    for lane_x in lanes:
        for i in range(0, len(ys) - 4, 12):
            seg = project(np.column_stack([np.full(4, lane_x), ys[i:i + 4]]))
            cv2.polylines(img, [seg.reshape(-1, 1, 2).astype(np.int32)],
                          False, (245, 245, 245), 6)
    return img, project


class TestFocal:
    def test_ninety_degrees_gives_half_width(self) -> None:
        assert focal_from_fov(1280, 90.0) == pytest.approx(640.0)

    def test_a_narrower_field_of_view_means_a_longer_focal(self) -> None:
        assert focal_from_fov(1280, 60.0) > focal_from_fov(1280, 120.0)


class TestVanishingPoint:
    def test_converging_lines_intersect_where_expected(self) -> None:
        lines = [(600.0, 700.0, 640.0, 300.0), (700.0, 700.0, 645.0, 300.0),
                 (500.0, 700.0, 635.0, 300.0), (800.0, 700.0, 650.0, 300.0)]
        found = vanishing_point(lines, (W, H))
        assert found is not None
        (u, v), n, ratio = found
        assert 550 < u < 750 and 200 < v < 400
        assert n >= 4 and 0.0 < ratio <= 1.0

    def test_too_few_lines_is_refused(self) -> None:
        assert vanishing_point([(0.0, 0, 1, 1), (2.0, 2, 3, 3)], (W, H)) is None

    def test_parallel_lines_do_not_yield_a_point(self) -> None:
        lines = [(float(x), 700.0, float(x), 300.0) for x in (400, 500, 600, 700)]
        found = vanishing_point(lines, (W, H))
        assert found is None or abs(found[0][1]) > H


class TestHomographyFromPose:
    def test_it_inverts_a_known_projection(self) -> None:
        pitch, height, hfov = 6.0, 1.35, 90.0
        _, project = road_frame(pitch, height, hfov)
        Hm = homography_from_pose(focal_from_fov(W, hfov), (W / 2, H / 2),
                                  np.deg2rad(pitch), 0.0, height)
        from core.calibrate import CameraModel
        cam = CameraModel(H_image_to_ground=Hm, image_size=(W, H))
        truth = np.array([[0.0, 10.0], [0.0, 20.0], [1.75, 15.0], [-1.75, 30.0]])
        got = np.asarray(cam.image_to_ground(project(truth))).reshape(-1, 2)
        assert got == pytest.approx(truth, abs=0.35)


class TestEstimateRefuses:
    def test_no_frames_is_refused(self) -> None:
        assert estimate([]) is None

    def test_a_blank_frame_is_refused(self) -> None:
        assert estimate([np.full((H, W, 3), 90, np.uint8)] * 4) is None

    def test_noise_is_refused(self) -> None:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(4)]
        assert estimate(frames) is None

    def test_an_implausible_yaw_is_refused(self) -> None:
        """Building edges converge too, and their pose is nonsense."""
        assert MAX_PLAUSIBLE_YAW_DEG < 90.0

    def test_the_plausibility_gates_are_tighter_than_the_geometry_allows(self) -> None:
        assert 0 < MAX_PLAUSIBLE_YAW_DEG <= 20
        assert 0 < MAX_PLAUSIBLE_PITCH_DEG <= 30
        assert 0.0 < HORIZON_BAND[0] < HORIZON_BAND[1] < 1.0


class TestEstimateSucceeds:
    def test_a_clean_road_yields_an_estimate(self) -> None:
        frames = [road_frame(6.0)[0] for _ in range(4)]
        est = estimate(frames)
        assert est is not None
        assert est.n_lines >= 4
        assert HORIZON_BAND[0] * H <= est.horizon_v <= HORIZON_BAND[1] * H

    def test_the_assumptions_are_recorded(self) -> None:
        """They scale every distance, so they must travel with the result."""
        est = estimate([road_frame(6.0)[0] for _ in range(4)])
        assert est is not None
        blob = " ".join(est.assumptions).lower()
        assert "camera height" in blob and "lane width" in blob
        assert "not a measured calibration" in blob

    def test_camera_height_scales_distance_linearly(self) -> None:
        """The single largest error source, so it is pinned."""
        a = estimate([road_frame(6.0)[0] for _ in range(4)], camera_height_m=1.2)
        b = estimate([road_frame(6.0)[0] for _ in range(4)], camera_height_m=2.4)
        assert a is not None and b is not None
        from core.calibrate import CameraModel
        px = np.array([[float(W / 2), float(H * 0.8)]])
        ga = np.asarray(CameraModel(H_image_to_ground=a.H_image_to_ground,
                                    image_size=(W, H)).image_to_ground(px)).reshape(-1, 2)
        gb = np.asarray(CameraModel(H_image_to_ground=b.H_image_to_ground,
                                    image_size=(W, H)).image_to_ground(px)).reshape(-1, 2)
        assert gb[0, 1] == pytest.approx(2.0 * ga[0, 1], rel=1e-6)


class TestPitchSignRegression:
    """The horizon of the produced homography must equal the detected VP row.

    A sign error between `estimate` and `homography_from_pose` put the horizon on
    the wrong side of the image centre. Distances still increased monotonically
    with range, so nothing looked broken -- but 5 m was reported where the
    geometry says 39 m. There is no symptom to notice; only this equality catches
    it.
    """

    def test_horizon_matches_the_detected_vanishing_point(self) -> None:
        est = estimate([road_frame(6.0)[0] for _ in range(4)])
        assert est is not None
        implied = H / 2 - est.focal_px * np.tan(est.pitch_rad)
        assert implied == pytest.approx(est.horizon_v, abs=1.0)

    def test_recovered_pitch_matches_the_truth(self) -> None:
        est = estimate([road_frame(6.0)[0] for _ in range(4)])
        assert est is not None
        assert np.rad2deg(est.pitch_rad) == pytest.approx(6.0, abs=1.0)

    def test_distances_are_accurate_near_and_degrade_with_range(self) -> None:
        """Monocular error grows with the square of distance. Pin both facts."""
        from core.calibrate import CameraModel
        img, project = road_frame(6.0)
        est = estimate([img] * 4)
        assert est is not None
        cam = CameraModel(H_image_to_ground=est.H_image_to_ground, image_size=(W, H))
        truth = np.array([[0.0, 8.0], [0.0, 40.0]])
        got = np.asarray(cam.image_to_ground(project(truth))).reshape(-1, 2)
        near = abs(got[0, 1] - 8.0)
        far = abs(got[1, 1] - 40.0)
        assert near < 0.5, "near-field distance should be good"
        assert far > near, "error must grow with range, as the geometry dictates"

    def test_forward_distance_is_positive(self) -> None:
        """A negative forward distance makes is_on_ground reject every point."""
        from core.calibrate import CameraModel
        est = estimate([road_frame(6.0)[0] for _ in range(4)])
        assert est is not None
        cam = CameraModel(H_image_to_ground=est.H_image_to_ground, image_size=(W, H))
        pt = np.array([[float(W / 2), float(H * 0.85)]])
        assert bool(cam.is_on_ground(pt)[0])
        assert np.asarray(cam.image_to_ground(pt)).reshape(-1, 2)[0, 1] > 0
