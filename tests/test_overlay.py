"""Tests for annotation rendering and MP4 output.

Drawing is checked by its effect on pixels -- that something changed, in the
right region, without mutating the input -- rather than by exact appearance,
which would make the tests brittle against harmless cosmetic edits.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.calibrate import CameraModel, rectangle_world_points, solve_ground_homography
from src.config import OverlayConfig, VideoConfig
from src.overlay import (
    VideoWriter,
    annotate,
    draw_boxes,
    draw_hud,
    draw_lane_fits,
    draw_mask,
    label_color,
    lane_color,
    lane_hud_lines,
)
from src.types import CLASS_IDS, FrameResult, LaneFit, LaneSet, LaneStatus
from tests.test_calibrate import HEIGHT, WIDTH, project_ground_to_image


@pytest.fixture
def camera() -> CameraModel:
    world = rectangle_world_points(3.5, 6.0, 30.0)
    H, _ = solve_ground_homography(project_ground_to_image(world), world)
    return CameraModel(H_image_to_ground=H, image_size=(WIDTH, HEIGHT))


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)


@pytest.fixture
def cfg() -> OverlayConfig:
    return OverlayConfig()


def _result(boxes=None, mask_rows: slice | None = None) -> FrameResult:
    boxes = np.zeros((0, 4), np.float32) if boxes is None else np.asarray(boxes, np.float32)
    drivable = np.zeros((HEIGHT, WIDTH), np.uint8)
    if mask_rows is not None:
        drivable[mask_rows] = 1
    return FrameResult(
        boxes_px=boxes,
        class_ids=np.full(len(boxes), CLASS_IDS["vehicle"], np.int32),
        scores=np.full(len(boxes), 0.9, np.float32),
        drivable_mask=drivable,
        lane_mask=np.zeros((HEIGHT, WIDTH), np.uint8),
    )


@pytest.fixture
def lane_set() -> LaneSet:
    return LaneSet(
        status=LaneStatus.OK,
        ego_left=LaneFit(np.array([0.0, 0.0, -1.75]), "ego_left", 900, (6.0, 32.0)),
        ego_right=LaneFit(np.array([0.0, 0.0, 1.75]), "ego_right", 850, (6.0, 32.0)),
        curvature_radius_m=float("inf"),
        ego_offset_m=-0.14,
        lane_width_m=3.5,
        eval_distance_m=10.0,
    )


class TestDrawBoxes:
    def test_draws_without_mutating_the_input(self, frame, cfg) -> None:
        original = frame.copy()
        vis = draw_boxes(frame, _result([[500, 300, 700, 450]]), cfg)
        assert not np.array_equal(vis, original)
        assert np.array_equal(frame, original), "draw_boxes mutated its input"

    def test_marks_the_box_edges(self, frame, cfg) -> None:
        vis = draw_boxes(frame, _result([[500, 300, 700, 450]]), cfg)
        changed = np.any(vis != frame, axis=2)
        assert changed[300, 600] and changed[450, 600]
        assert not changed[650, 100]

    def test_label_stays_inside_the_frame_at_the_top_edge(self, frame, cfg) -> None:
        vis = draw_boxes(frame, _result([[400, 0, 600, 90]]), cfg)
        assert np.any(vis[0:12, 400:600] != frame[0:12, 400:600])

    def test_distance_appears_in_the_label(self, frame, cfg) -> None:
        r = _result([[500, 300, 700, 450]])
        assert not np.array_equal(
            draw_boxes(frame, r, cfg), draw_boxes(frame, r, cfg, distances_m={0: 23.4})
        )

    def test_no_detections_is_a_no_op(self, frame, cfg) -> None:
        assert np.array_equal(draw_boxes(frame, _result(), cfg), frame)

    def test_classes_get_distinct_colours(self) -> None:
        assert label_color("vehicle") != label_color("person")
        assert label_color("rickshaw") == (200, 200, 200)


class TestDrawMask:
    def test_blends_only_inside_the_mask(self, frame, cfg) -> None:
        mask = np.zeros((HEIGHT, WIDTH), bool)
        mask[400:500, 100:200] = True
        vis = draw_mask(frame, mask, cfg)
        assert not np.array_equal(vis[450, 150], frame[450, 150])
        assert np.array_equal(vis[10, 10], frame[10, 10])

    def test_alpha_zero_changes_nothing(self, frame) -> None:
        mask = np.ones((HEIGHT, WIDTH), bool)
        assert np.array_equal(draw_mask(frame, mask, OverlayConfig(mask_alpha=0.0)), frame)

    def test_alpha_override_is_honoured(self, frame, cfg) -> None:
        mask = np.ones((HEIGHT, WIDTH), bool)
        assert not np.array_equal(
            draw_mask(frame, mask, cfg, alpha=0.1), draw_mask(frame, mask, cfg, alpha=0.9)
        )

    def test_empty_mask_is_a_no_op(self, frame, cfg) -> None:
        assert np.array_equal(draw_mask(frame, np.zeros((HEIGHT, WIDTH), bool), cfg), frame)

    def test_mismatched_shape_raises(self, frame, cfg) -> None:
        with pytest.raises(ValueError, match="does not match frame"):
            draw_mask(frame, np.zeros((10, 10), bool), cfg)


class TestDrawLaneFits:
    def test_reprojected_curves_are_drawn(self, frame, cfg, lane_set, camera) -> None:
        vis = draw_lane_fits(frame, lane_set, camera, cfg)
        assert not np.array_equal(vis, frame)

    def test_does_not_mutate_input(self, frame, cfg, lane_set, camera) -> None:
        original = frame.copy()
        draw_lane_fits(frame, lane_set, camera, cfg)
        assert np.array_equal(frame, original)

    def test_curves_land_where_the_geometry_says(self, frame, cfg, lane_set, camera) -> None:
        """A boundary at x=-1.75 m must be drawn left of one at x=+1.75 m."""
        left_only = LaneSet(status=LaneStatus.SINGLE_BOUNDARY, ego_left=lane_set.ego_left)
        right_only = LaneSet(status=LaneStatus.SINGLE_BOUNDARY, ego_right=lane_set.ego_right)
        lv = draw_lane_fits(frame, left_only, camera, cfg)
        rv = draw_lane_fits(frame, right_only, camera, cfg)
        left_cols = np.nonzero(np.any(lv != frame, axis=2).any(axis=0))[0]
        right_cols = np.nonzero(np.any(rv != frame, axis=2).any(axis=0))[0]
        assert left_cols.mean() < right_cols.mean()

    def test_undetected_lanes_draw_nothing(self, frame, cfg, camera) -> None:
        empty = LaneSet(status=LaneStatus.NOT_DETECTED)
        assert np.array_equal(draw_lane_fits(frame, empty, camera, cfg), frame)

    def test_roles_get_distinct_colours(self) -> None:
        assert lane_color("ego_left") != lane_color("adjacent_left")
        assert lane_color("nonsense") == (255, 255, 255)


class TestHud:
    def test_draws_a_panel(self, frame, cfg) -> None:
        assert not np.array_equal(draw_hud(frame, ["frame 12"], cfg)[20, 20], frame[20, 20])

    def test_disabled_hud_is_a_no_op(self, frame) -> None:
        assert np.array_equal(draw_hud(frame, ["x"], OverlayConfig(hud=False)), frame)

    def test_warnings_render_in_addition(self, frame, cfg) -> None:
        assert not np.array_equal(
            draw_hud(frame, ["a"], cfg), draw_hud(frame, ["a"], cfg, warnings=["! NO LANE"])
        )

    def test_lane_lines_report_geometry_when_ok(self, lane_set) -> None:
        text = " ".join(lane_hud_lines(lane_set))
        assert "3.50" in text and "-0.14" in text and "straight" in text

    def test_lane_lines_are_honest_when_degraded(self) -> None:
        for status in (LaneStatus.NOT_DETECTED, LaneStatus.MASK_TOO_SPARSE):
            text = " ".join(lane_hud_lines(LaneSet(status=status)))
            assert "n/a" in text and status.value in text

    def test_single_boundary_reports_which_side(self) -> None:
        s = LaneSet(
            status=LaneStatus.SINGLE_BOUNDARY,
            ego_left=LaneFit(np.array([0.0, 0.0, -1.75]), "ego_left", 500, (6, 30)),
        )
        assert "left boundary only" in " ".join(lane_hud_lines(s))

    def test_finite_radius_is_shown_in_metres(self) -> None:
        s = LaneSet(status=LaneStatus.OK, curvature_radius_m=420.0,
                    ego_offset_m=0.0, lane_width_m=3.5, eval_distance_m=10.0)
        assert "420 m" in " ".join(lane_hud_lines(s))


class TestAnnotate:
    def test_composes_every_layer(self, frame, cfg, lane_set, camera) -> None:
        result = _result([[500, 300, 700, 450]], mask_rows=slice(500, 700))
        vis = annotate(frame, result, lane_set, camera, cfg, ["frame 0"], [])
        assert vis.shape == frame.shape
        assert not np.array_equal(vis, frame)
        assert not np.array_equal(vis[600, 640], frame[600, 640]), "drivable fill missing"
        assert not np.array_equal(vis[20, 20], frame[20, 20]), "HUD missing"


class TestVideoWriter:
    def test_writes_a_playable_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.mp4"
        with VideoWriter(out, 25.0, VideoConfig()) as w:
            for i in range(10):
                w.write(np.full((240, 320, 3), i * 20, np.uint8))
        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
            assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, abs=0.1)
        finally:
            cap.release()

    def test_config_fps_overrides(self, tmp_path: Path) -> None:
        out = tmp_path / "slow.mp4"
        with VideoWriter(out, 25.0, VideoConfig(fps=10.0)) as w:
            w.write(np.zeros((240, 320, 3), np.uint8))
        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
        finally:
            cap.release()

    def test_size_change_mid_video_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="frame size changed mid-video"):
            with VideoWriter(tmp_path / "x.mp4", 25.0, VideoConfig()) as w:
                w.write(np.zeros((240, 320, 3), np.uint8))
                w.write(np.zeros((100, 100, 3), np.uint8))

    def test_zero_fps_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fps must be > 0"):
            VideoWriter(tmp_path / "x.mp4", 0.0, VideoConfig())

    def test_creates_missing_parents(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "out.mp4"
        with VideoWriter(out, 25.0, VideoConfig()) as w:
            w.write(np.zeros((240, 320, 3), np.uint8))
        assert out.is_file()
