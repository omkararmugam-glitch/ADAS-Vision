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

from src.config import OverlayConfig, VideoConfig
from src.overlay import (
    VideoWriter,
    draw_boxes,
    draw_hud,
    draw_mask,
    draw_polylines,
    label_color,
    lane_color,
)
from src.types import Detection, LaneLine

WIDTH, HEIGHT = 320, 240


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)


@pytest.fixture
def cfg() -> OverlayConfig:
    return OverlayConfig()


class TestDrawBoxes:
    def test_draws_and_leaves_the_input_untouched(self, frame, cfg) -> None:
        original = frame.copy()
        dets = [Detection(bbox_px=(50.0, 80.0, 150.0, 180.0), label="car", score=0.9)]
        vis = draw_boxes(frame, dets, cfg)
        assert not np.array_equal(vis, original)
        assert np.array_equal(frame, original), "draw_boxes mutated its input"

    def test_marks_the_box_edges(self, frame, cfg) -> None:
        dets = [Detection(bbox_px=(50.0, 80.0, 150.0, 180.0), label="car", score=0.9)]
        vis = draw_boxes(frame, dets, cfg)
        changed = np.any(vis != frame, axis=2)
        assert changed[80, 100], "top edge not drawn"
        assert changed[180, 100], "bottom edge not drawn"
        assert not changed[220, 300], "drew outside the box region"

    def test_label_stays_inside_the_frame_for_a_box_at_the_top(self, frame, cfg) -> None:
        """A box touching y=0 must not push its label off-image and vanish."""
        dets = [Detection(bbox_px=(40.0, 0.0, 140.0, 60.0), label="person", score=0.7)]
        vis = draw_boxes(frame, dets, cfg)
        assert np.any(vis[0:12, 40:140] != frame[0:12, 40:140]), "label was clipped away"

    def test_distance_appears_in_the_label(self, frame, cfg) -> None:
        dets = [Detection(bbox_px=(50.0, 80.0, 150.0, 180.0), label="car", score=0.9)]
        plain = draw_boxes(frame, dets, cfg)
        with_dist = draw_boxes(frame, dets, cfg, distances_m={0: 23.4})
        assert not np.array_equal(plain, with_dist)

    def test_empty_detections_is_a_no_op(self, frame, cfg) -> None:
        assert np.array_equal(draw_boxes(frame, [], cfg), frame)

    def test_unknown_class_gets_the_default_colour(self) -> None:
        assert label_color("car") != label_color("rickshaw")
        assert label_color("rickshaw") == (200, 200, 200)


class TestDrawMask:
    def test_blends_only_inside_the_mask(self, frame, cfg) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[100:150, 40:80] = True
        vis = draw_mask(frame, mask, cfg)
        assert not np.array_equal(vis[120, 60], frame[120, 60]), "mask region unchanged"
        assert np.array_equal(vis[10, 10], frame[10, 10]), "changed pixels outside the mask"

    def test_alpha_zero_changes_nothing(self, frame) -> None:
        mask = np.ones((HEIGHT, WIDTH), dtype=bool)
        assert np.array_equal(draw_mask(frame, mask, OverlayConfig(mask_alpha=0.0)), frame)

    def test_empty_mask_is_a_no_op(self, frame, cfg) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        assert np.array_equal(draw_mask(frame, mask, cfg), frame)

    def test_mismatched_mask_shape_raises(self, frame, cfg) -> None:
        with pytest.raises(ValueError, match="does not match frame"):
            draw_mask(frame, np.zeros((10, 10), dtype=bool), cfg)

    def test_accepts_a_uint8_mask(self, frame, cfg) -> None:
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        mask[50:60, 50:60] = 1
        assert not np.array_equal(draw_mask(frame, mask, cfg), frame)


class TestDrawPolylines:
    def test_solid_and_dashed_render_differently(self, frame, cfg) -> None:
        pts = np.array([[20.0, 230.0], [60.0, 190.0], [100.0, 150.0], [140.0, 110.0]])
        solid = draw_polylines(frame, [LaneLine(pts, role="ego_left", kind="solid")], cfg)
        dashed = draw_polylines(frame, [LaneLine(pts, role="ego_left", kind="dashed")], cfg)
        assert not np.array_equal(solid, dashed)
        assert np.count_nonzero(np.any(dashed != frame, axis=2)) < np.count_nonzero(
            np.any(solid != frame, axis=2)
        ), "dashed should mark fewer pixels than solid"

    def test_roles_get_distinct_colours(self) -> None:
        assert lane_color("ego_left") != lane_color("adjacent_left")
        assert lane_color("nonsense") == (255, 255, 255)

    def test_does_not_mutate_input(self, frame, cfg) -> None:
        original = frame.copy()
        draw_polylines(frame, [LaneLine(np.array([[10.0, 10.0], [50.0, 50.0]]))], cfg)
        assert np.array_equal(frame, original)


class TestDrawHud:
    def test_draws_a_panel(self, frame, cfg) -> None:
        vis = draw_hud(frame, ["frame 12", "objects 3"], cfg)
        assert not np.array_equal(vis[20, 20], frame[20, 20])

    def test_disabled_hud_is_a_no_op(self, frame) -> None:
        vis = draw_hud(frame, ["ignored"], OverlayConfig(hud=False))
        assert np.array_equal(vis, frame)

    def test_warnings_render_in_addition_to_status(self, frame, cfg) -> None:
        plain = draw_hud(frame, ["a", "b"], cfg)
        warned = draw_hud(frame, ["a", "b"], cfg, warnings=["! CLOSE: 8.0 m"])
        assert not np.array_equal(plain, warned)

    def test_panel_fits_a_narrow_frame(self, cfg) -> None:
        narrow = np.zeros((80, 100, 3), dtype=np.uint8)
        vis = draw_hud(narrow, ["x"], cfg)
        assert vis.shape == narrow.shape


class TestVideoWriter:
    def test_writes_a_playable_file(self, tmp_path: Path, frame) -> None:
        out = tmp_path / "out.mp4"
        with VideoWriter(out, 25.0, VideoConfig()) as w:
            for i in range(10):
                w.write(np.full_like(frame, i * 20))
        assert out.is_file() and out.stat().st_size > 0

        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
            assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == WIDTH
            assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(25.0, abs=0.1)
        finally:
            cap.release()

    def test_config_fps_overrides_the_source_rate(self, tmp_path: Path, frame) -> None:
        out = tmp_path / "slow.mp4"
        with VideoWriter(out, 25.0, VideoConfig(fps=10.0)) as w:
            w.write(frame)
        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
        finally:
            cap.release()

    def test_size_change_mid_video_raises(self, tmp_path: Path, frame) -> None:
        with pytest.raises(ValueError, match="frame size changed mid-video"):
            with VideoWriter(tmp_path / "x.mp4", 25.0, VideoConfig()) as w:
                w.write(frame)
                w.write(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_zero_fps_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fps must be > 0"):
            VideoWriter(tmp_path / "x.mp4", 0.0, VideoConfig())

    def test_creates_missing_parent_directories(self, tmp_path: Path, frame) -> None:
        out = tmp_path / "deep" / "nested" / "out.mp4"
        with VideoWriter(out, 25.0, VideoConfig()) as w:
            w.write(frame)
        assert out.is_file()
