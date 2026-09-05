"""Tests for the composed scene view.

Rendering tests are usually worth little -- "it drew some pixels" catches
nothing. These target the specific ways this view has been observed to lie:

* extrapolating a quadratic **toward** the camera, where the projection is most
  sensitive, which threw a confident red line across the bonnet;
* extrapolating one far enough sideways that it crossed unrelated scenery;
* extending the two corridor edges independently until they crossed, filling a
  bowtie that is not a lane;
* drawing a corridor over a range where one edge is measured and the other
  invented, which looks exactly as solid as a real one.

Each of those produced a picture that looked plausible and was wrong, which is
the failure mode a perception display exists to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.autocalib import focal_from_fov, homography_from_pose
from core.calibrate import CameraModel
from core.config import OverlayConfig
from core.scene import (
    MAX_BOTTOM_EXTENSION_FRAC,
    MAX_LATERAL_M,
    SceneStyle,
    _extend_pair_to_bottom,
    _extend_to_bottom,
    _project,
    draw_boundaries,
    draw_drivable_area,
    draw_legend,
    draw_predicted_path,
    draw_scene,
    draw_split_badge,
    scene_status_rows,
    trusted_range,
)
from core.types import LaneFit, LaneSet, LaneStatus

W, H = 1280, 720


def camera() -> CameraModel:
    Hm = homography_from_pose(focal_from_fov(W, 90.0), (W / 2, H / 2),
                              np.deg2rad(6.0), 0.0, 1.35)
    return CameraModel(H_image_to_ground=Hm, image_size=(W, H))


def boundary(role: str, c: float, b: float = 0.0, a: float = 0.0,
             y_range=(5.0, 35.0)) -> LaneFit:
    return LaneFit(coeffs=np.array([a, b, c]), role=role, n_pixels=400,
                   y_range_m=y_range)


def lanes_ok(parallel: bool = True, **kw) -> LaneSet:
    return LaneSet(
        status=LaneStatus.OK,
        ego_left=kw.pop("left", boundary("ego_left", -1.85)),
        ego_right=kw.pop("right", boundary("ego_right", 1.85)),
        lane_width_m=3.7, ego_offset_m=0.0, curvature_radius_m=float("inf"),
        eval_distance_m=15.0, parallel_model=parallel, **kw)


def frame() -> np.ndarray:
    return np.full((H, W, 3), 60, np.uint8)


class TestTrustedRange:
    def test_the_parallel_model_earns_the_union_of_supports(self) -> None:
        lanes = lanes_ok(True,
                         left=boundary("ego_left", -1.85, y_range=(4.0, 36.0)),
                         right=boundary("ego_right", 1.85, y_range=(6.0, 14.0)))
        assert trusted_range(lanes) == (4.0, 36.0)

    def test_without_it_only_the_overlap_may_be_drawn(self) -> None:
        """Beyond the overlap one edge is measured and the other invented."""
        lanes = lanes_ok(False,
                         left=boundary("ego_left", -1.85, y_range=(4.0, 36.0)),
                         right=boundary("ego_right", 1.85, y_range=(6.0, 14.0)))
        assert trusted_range(lanes) == (6.0, 14.0)

    def test_a_missing_boundary_has_no_corridor(self) -> None:
        assert trusted_range(LaneSet(status=LaneStatus.SINGLE_BOUNDARY,
                                     ego_left=boundary("ego_left", -1.85))) is None

    def test_a_degenerate_overlap_is_refused(self) -> None:
        lanes = lanes_ok(False,
                         left=boundary("ego_left", -1.85, y_range=(4.0, 10.0)),
                         right=boundary("ego_right", 1.85, y_range=(9.9, 30.0)))
        assert trusted_range(lanes) is None


class TestProjectionGuards:
    def test_a_diverging_quadratic_is_truncated_not_drawn(self) -> None:
        """A curve that leaves the road stops; it does not carry on confidently."""
        wild = boundary("ego_right", 1.85, b=0.9, y_range=(5.0, 35.0))
        assert abs(float(wild.x_at(35.0)[0])) > MAX_LATERAL_M
        pts = _project(wild, camera(), (5.0, 35.0), 100)
        full = _project(boundary("ego_right", 1.85), camera(), (5.0, 35.0), 100)
        assert 0 < len(pts) < len(full)

    def test_a_curve_that_starts_off_road_yields_nothing(self) -> None:
        absurd = boundary("ego_right", 40.0)
        assert len(_project(absurd, camera(), (5.0, 35.0), 50)) == 0

    def test_an_empty_range_yields_nothing(self) -> None:
        assert len(_project(boundary("ego_left", -1.85), camera(), (20.0, 5.0), 50)) == 0


class TestBonnetContinuation:
    def test_it_reaches_the_bottom_of_the_frame(self) -> None:
        pts = np.array([[600.0, 620.0], [610.0, 600.0], [620.0, 580.0]])
        assert (H - 1 - 620) < MAX_BOTTOM_EXTENSION_FRAC * H   # within the cap
        out = _extend_to_bottom(pts, H)
        assert out[0][1] == pytest.approx(H - 1)
        assert len(out) == len(pts) + 1

    def test_it_follows_the_curve_s_own_direction(self) -> None:
        """Leaning left going down must continue leaning left, not jump."""
        pts = np.array([[600.0, 600.0], [640.0, 560.0]])
        out = _extend_to_bottom(pts, H)
        assert out[0][0] < pts[0][0]

    def test_a_fit_starting_high_in_the_frame_is_not_extended(self) -> None:
        """Continuing it would invent more road than the fit ever measured."""
        high = np.array([[600.0, 200.0], [610.0, 190.0]])
        assert (H - 1 - 200) > MAX_BOTTOM_EXTENSION_FRAC * H
        assert np.array_equal(_extend_to_bottom(high, H), high)

    def test_a_curve_already_at_the_bottom_is_untouched(self) -> None:
        pts = np.array([[600.0, float(H - 1)], [610.0, 600.0]])
        assert np.array_equal(_extend_to_bottom(pts, H), pts)

    def test_a_single_point_is_untouched(self) -> None:
        pts = np.array([[600.0, 500.0]])
        assert np.array_equal(_extend_to_bottom(pts, H), pts)


class TestNoBowtie:
    def test_edges_that_would_cross_are_not_extended(self) -> None:
        """A filled polygon with crossing edges renders as an hourglass."""
        lp = np.array([[600.0, 600.0], [560.0, 560.0]])   # heading right, going down
        rp = np.array([[620.0, 600.0], [660.0, 560.0]])   # heading left, going down
        le, re = _extend_pair_to_bottom(lp, rp, H)
        assert np.array_equal(le, lp) and np.array_equal(re, rp)

    def test_edges_that_stay_ordered_are_extended(self) -> None:
        lp = np.array([[600.0, 600.0], [620.0, 560.0]])
        rp = np.array([[700.0, 600.0], [680.0, 560.0]])
        le, re = _extend_pair_to_bottom(lp, rp, H)
        assert len(le) == len(lp) + 1 and len(re) == len(rp) + 1
        assert le[0][0] < re[0][0]


class TestDrawing:
    def test_the_corridor_is_filled(self) -> None:
        out = draw_drivable_area(frame(), lanes_ok(), camera())
        assert not np.array_equal(out, frame())

    def test_nothing_is_drawn_without_both_boundaries(self) -> None:
        lanes = LaneSet(status=LaneStatus.SINGLE_BOUNDARY,
                        ego_left=boundary("ego_left", -1.85))
        f = frame()
        assert np.array_equal(draw_drivable_area(f, lanes, camera()), f)
        assert np.array_equal(draw_predicted_path(f, lanes, camera()), f)

    def test_left_is_green_and_right_is_red(self) -> None:
        """The colour language is the contract; a swap inverts its meaning."""
        from core.scene import LEFT_BOUNDARY, RIGHT_BOUNDARY

        out = draw_boundaries(frame(), lanes_ok(), camera())
        painted = out.reshape(-1, 3)
        assert (painted == np.array(LEFT_BOUNDARY)).all(axis=1).any()
        assert (painted == np.array(RIGHT_BOUNDARY)).all(axis=1).any()

    def test_the_path_is_dashed_not_solid(self) -> None:
        """Ground-spaced gaps: the centreline must not be one continuous line."""
        from core.scene import PATH_DASH

        out = draw_predicted_path(frame(), lanes_ok(), camera())
        hit = (out == np.array(PATH_DASH)).all(axis=2)
        rows = np.flatnonzero(hit.any(axis=1))
        assert len(rows) > 0
        # A solid line would occupy every row between its ends.
        assert len(rows) < (rows.max() - rows.min() + 1)

    def test_the_legend_is_drawn_bottom_left(self) -> None:
        out = draw_legend(frame())
        assert not np.array_equal(out[H - 120:, :300], frame()[H - 120:, :300])
        assert np.array_equal(out[:100, W - 300:], frame()[:100, W - 300:])

    def test_the_badge_is_absent_without_a_split(self) -> None:
        f = frame()
        assert np.array_equal(draw_split_badge(f, None), f)

    def test_the_badge_names_the_side_and_distance(self) -> None:
        from models.lane_seg.split import RoadSplit, SplitSide

        split = RoadSplit(side=SplitSide.RIGHT, near_separation_m=3.5,
                          far_separation_m=7.0, divergence_ratio=2.0,
                          gore_distance_m=24.0, left_role="ego_left",
                          right_role="ego_right")
        out = draw_split_badge(frame(), split)
        assert not np.array_equal(out[:200, W - 260:], frame()[:200, W - 260:])

    def test_the_whole_scene_composes(self) -> None:
        out = draw_scene(frame(), lanes_ok(), camera(), OverlayConfig(),
                         status_rows=[("Lane", "centre")])
        assert out.shape == (H, W, 3)
        assert not np.array_equal(out, frame())

    def test_a_scene_without_a_camera_still_returns_a_frame(self) -> None:
        out = draw_scene(frame(), lanes_ok(), None, OverlayConfig(),
                         status_rows=[("Lane", "centre")])
        assert out.shape == (H, W, 3)

    def test_an_undetected_lane_draws_no_geometry(self) -> None:
        out = draw_scene(frame(), LaneSet(status=LaneStatus.NOT_DETECTED),
                         camera(), OverlayConfig(), status_rows=None,
                         show_legend=False, show_badge=False, show_panel=False)
        assert np.array_equal(out, frame())

    def test_a_fork_and_a_divided_road_are_not_tinted_at_once(self) -> None:
        """They are different claims about the same region."""
        from models.lane_seg.carriageway import DividerKind, DualCarriageway
        from models.lane_seg.split import RoadSplit, SplitSide

        split = RoadSplit(side=SplitSide.RIGHT, near_separation_m=3.5,
                          far_separation_m=7.0, divergence_ratio=2.0,
                          gore_distance_m=24.0, left_role="ego_left",
                          right_role="ego_right")
        cw = DualCarriageway(divider_role="ego_left", divider_x_m=-1.85,
                             divider_side="left", kind=DividerKind.MEDIAN_GAP,
                             yellow_fraction=None, median_width_m=6.0,
                             ego_lanes=1, oncoming_lanes=1, confidence=0.6,
                             eval_distance_m=15.0)
        lanes = lanes_ok()
        both = draw_scene(frame(), lanes, camera(), OverlayConfig(), split=split,
                          carriageway=cw, show_legend=False, show_panel=False,
                          show_badge=False, show_labels=False)
        split_only = draw_scene(frame(), lanes, camera(), OverlayConfig(),
                                split=split, show_legend=False, show_panel=False,
                                show_badge=False, show_labels=False)
        assert np.array_equal(both, split_only)


class TestStatusRows:
    def test_unknowns_are_omitted_not_printed(self) -> None:
        rows = scene_status_rows(lanes_ok(), weather="unknown", visibility=None)
        assert all(k != "Weather" for k, _ in rows)
        assert all(k != "Visibility" for k, _ in rows)

    def test_a_known_weather_appears(self) -> None:
        rows = dict(scene_status_rows(lanes_ok(), weather="rainy"))
        assert rows["Weather"] == "rainy"

    def test_a_straight_road_says_straight_not_a_huge_radius(self) -> None:
        rows = dict(scene_status_rows(lanes_ok()))
        assert rows["Curvature"] == "straight"

    def test_a_degraded_lane_reports_its_status(self) -> None:
        rows = dict(scene_status_rows(LaneSet(status=LaneStatus.MASK_TOO_SPARSE)))
        assert rows["Lane"] == "mask too sparse"

    def test_a_split_and_a_divided_road_do_not_both_claim_the_road_row(self) -> None:
        from models.lane_seg.carriageway import DividerKind, DualCarriageway
        from models.lane_seg.split import RoadSplit, SplitSide

        split = RoadSplit(side=SplitSide.LEFT, near_separation_m=3.5,
                          far_separation_m=7.0, divergence_ratio=2.0,
                          gore_distance_m=24.0, left_role="ego_left",
                          right_role="ego_right")
        cw = DualCarriageway(divider_role="ego_left", divider_x_m=-1.85,
                             divider_side="left", kind=DividerKind.MEDIAN_GAP,
                             yellow_fraction=None, median_width_m=6.0,
                             ego_lanes=1, oncoming_lanes=1, confidence=0.6,
                             eval_distance_m=15.0)
        rows = [k for k, _ in scene_status_rows(lanes_ok(), split, cw)]
        assert rows.count("Road") == 1
