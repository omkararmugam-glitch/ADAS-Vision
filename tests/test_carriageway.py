"""Tests for divided-highway detection.

Two things are being guarded, and they pull in opposite directions.

The first is that a divided highway must not be mistaken for a fork, or a fork
for a divided highway. Both present as two boundaries with a large gap; the
difference is whether the gap *opens*. Getting this backwards would tell the
driver a road is dividing ahead when in fact they have been on a dual
carriageway for the last ten miles.

The second is that the side must be right. Everything beyond the divider is
oncoming traffic, and a divider reported on the wrong side inverts the meaning
of every object on half the road -- turning a car in the next lane into a
head-on closing threat, or worse, the reverse.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.autocalib import focal_from_fov, homography_from_pose
from core.calibrate import CameraModel
from core.types import LaneFit, LaneSet, LaneStatus
from models.lane_seg.carriageway import (
    CarriagewayTracker,
    DividerKind,
    DriveSide,
    DualCarriageway,
    detect_carriageway,
)

NEAR, FAR = 6.0, 36.0
W, H = 1280, 720


def boundary(role: str, c: float, b: float = 0.0, a: float = 0.0,
             y_range: tuple[float, float] = (NEAR, FAR)) -> LaneFit:
    """x = a*y^2 + b*y + c, in bird's-eye metres."""
    return LaneFit(coeffs=np.array([a, b, c]), role=role,
                   n_pixels=400, y_range_m=y_range)


def lane_set(*fits: LaneFit, status: LaneStatus = LaneStatus.OK) -> LaneSet:
    ego = [f for f in fits if f.role.startswith("ego")]
    adj = tuple(f for f in fits if not f.role.startswith("ego"))
    return LaneSet(
        status=status,
        ego_left=next((f for f in ego if "left" in f.role), None),
        ego_right=next((f for f in ego if "right" in f.role), None),
        adjacent=adj, eval_distance_m=15.0,
    )


def camera() -> CameraModel:
    """A plausible dashcam pose, so projections land inside the frame."""
    Hm = homography_from_pose(focal_from_fov(W, 90.0), (W / 2, H / 2),
                              np.deg2rad(6.0), 0.0, 1.35)
    return CameraModel(H_image_to_ground=Hm, image_size=(W, H))


def paint(fit: LaneFit, cam: CameraModel, thickness: int = 9) -> np.ndarray:
    """A mask with ``fit`` drawn on it, as the segmenter would produce."""
    import cv2

    from models.lane_seg.fitting import fit_to_image_points

    mask = np.zeros((H, W), np.uint8)
    pts = fit_to_image_points(fit, cam, n_samples=120)
    if len(pts) >= 2:
        cv2.polylines(mask, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                      255, thickness)
    return mask


class TestNoFalsePositives:
    def test_an_ordinary_two_boundary_lane_is_not_divided(self) -> None:
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75))
        assert detect_carriageway(lanes) is None

    def test_no_lane_means_no_answer(self) -> None:
        assert detect_carriageway(LaneSet(status=LaneStatus.NOT_DETECTED)) is None

    def test_a_wide_urban_road_is_not_a_median(self) -> None:
        """Four metres is a generous lane, not a central reservation."""
        lanes = lane_set(boundary("ego_left", -2.0), boundary("ego_right", 2.0))
        assert detect_carriageway(lanes) is None

    def test_a_white_marking_is_not_a_divider(self) -> None:
        """The whole point of the yellow cue: white paint separates lanes."""
        cam = camera()
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75))
        empty_yellow = np.zeros((H, W), np.uint8)
        assert detect_carriageway(lanes, cam, empty_yellow) is None

    def test_an_opening_gap_is_a_fork_not_a_median(self) -> None:
        """A gap that widens belongs to detect_split; this must decline it."""
        lanes = lane_set(boundary("ego_left", -3.0),
                         boundary("adjacent_left", -6.0, b=-0.35))
        assert detect_carriageway(lanes) is None

    def test_a_short_overlap_is_not_enough_to_judge(self) -> None:
        """Two fits sharing three metres cannot show a gap holding its width."""
        lanes = lane_set(
            boundary("ego_left", -3.0, y_range=(6.0, 12.0)),
            boundary("adjacent_left", -9.0, y_range=(10.0, 30.0)),
        )
        assert detect_carriageway(lanes) is None


class TestMedianGap:
    def _divided(self) -> LaneSet:
        """Ego lane, then a six-metre median, then the opposing carriageway."""
        return lane_set(
            boundary("ego_left", -1.75), boundary("ego_right", 1.75),
            boundary("adjacent_left", -7.75),
        )

    def test_a_constant_wide_gap_is_a_divided_highway(self) -> None:
        got = detect_carriageway(self._divided())
        assert got is not None
        assert got.kind is DividerKind.MEDIAN_GAP

    def test_the_divider_is_the_near_edge_of_the_median(self) -> None:
        """Not the far edge: the divider is the boundary the ego drives beside."""
        got = detect_carriageway(self._divided())
        assert got is not None
        assert got.divider_role == "ego_left"

    def test_the_median_width_is_measured(self) -> None:
        got = detect_carriageway(self._divided())
        assert got is not None
        assert got.median_width_m == pytest.approx(6.0, abs=0.01)

    def test_oncoming_traffic_is_on_the_left_for_right_hand_traffic(self) -> None:
        got = detect_carriageway(self._divided())
        assert got is not None
        assert got.divider_side == "left"
        assert got.oncoming_is_left()
        assert got.agrees_with_convention(DriveSide.RIGHT)

    def test_a_divider_on_the_right_is_reported_but_doubted(self) -> None:
        """Measured honestly, scored lower: usually a bad fit, not a contraflow."""
        mirrored = lane_set(
            boundary("ego_left", -1.75), boundary("ego_right", 1.75),
            boundary("adjacent_right", 7.75),
        )
        got = detect_carriageway(mirrored)
        assert got is not None
        assert got.divider_side == "right"
        assert not got.oncoming_is_left()
        assert not got.agrees_with_convention(DriveSide.RIGHT)
        assert got.confidence < detect_carriageway(self._divided()).confidence

    def test_driving_inside_the_median_is_rejected(self) -> None:
        """Both gap edges straddling the ego means the fit is wrong."""
        lanes = lane_set(boundary("ego_left", -4.0), boundary("ego_right", 4.0))
        assert detect_carriageway(lanes, min_median_gap_m=5.0) is None


class TestYellowCue:
    def test_a_yellow_boundary_is_the_divider(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        got = detect_carriageway(lanes, cam, paint(left, cam))
        assert got is not None
        assert got.kind is DividerKind.YELLOW_LINE
        assert got.divider_role == "ego_left"
        assert got.yellow_fraction is not None and got.yellow_fraction > 0.8

    def test_a_yellow_line_on_the_right_is_an_edge_line_not_a_divider(self) -> None:
        """The commonest false positive on real US frames, and the reason
        ``drive_side`` gates the cue instead of merely discounting it.

        A yellow line to the driver's right marks the roadway's edge -- a kerb,
        a no-parking boundary. Reading it as a direction divider puts oncoming
        traffic on the pavement.
        """
        cam = camera()
        right = boundary("ego_right", 1.75)
        lanes = lane_set(boundary("ego_left", -1.75), right)
        assert detect_carriageway(lanes, cam, paint(right, cam)) is None

    def test_the_gate_follows_the_convention_it_is_given(self) -> None:
        """In left-hand traffic the divider is on the right, and this flips."""
        cam = camera()
        right = boundary("ego_right", 1.75)
        lanes = lane_set(boundary("ego_left", -1.75), right)
        got = detect_carriageway(lanes, cam, paint(right, cam),
                                 drive_side=DriveSide.LEFT)
        assert got is not None
        assert got.divider_role == "ego_right"
        assert got.agrees_with_convention(DriveSide.LEFT)

    def test_a_yellow_object_is_not_a_yellow_line(self) -> None:
        """A taxi ahead fills most of the projected curve and scores a fraction
        near one. Only its short longitudinal span gives it away."""
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        full = paint(left, cam, thickness=40)
        blob = np.zeros_like(full)
        # Keep only the part of the marking between 8 m and 13 m: a thing at a
        # place, rather than a line running the length of the road.
        from models.lane_seg.fitting import fit_to_image_points
        band = fit_to_image_points(left, cam, (8.0, 13.0), n_samples=40)
        v_lo, v_hi = int(band[:, 1].min()), int(band[:, 1].max())
        blob[v_lo:v_hi + 1] = full[v_lo:v_hi + 1]
        assert blob.any()
        assert detect_carriageway(lanes, cam, blob) is None
        # ... while the same paint over the full length is accepted.
        assert detect_carriageway(lanes, cam, full) is not None

    def test_a_few_stray_yellow_pixels_do_not_convince(self) -> None:
        """Autumn leaves, a skip, a taxi. Sparse yellow is not a painted line."""
        cam = camera()
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75))
        speckle = np.zeros((H, W), np.uint8)
        rng = np.random.default_rng(0)
        ys = rng.integers(H // 2, H, 40)
        xs = rng.integers(0, W, 40)
        speckle[ys, xs] = 255
        assert detect_carriageway(lanes, cam, speckle) is None

    def test_both_cues_together_score_highest(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        both = detect_carriageway(lanes, cam, paint(left, cam))
        gap_only = detect_carriageway(lanes)
        assert both is not None and gap_only is not None
        assert both.kind is DividerKind.YELLOW_AND_MEDIAN
        assert both.confidence > gap_only.confidence

    def test_yellow_needs_a_camera(self) -> None:
        """Without calibration the mask cannot be related to the fit at all."""
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        assert detect_carriageway(lanes, None, paint(left, camera())) is None


class TestLaneCounting:
    def test_lanes_are_the_spaces_between_boundaries(self) -> None:
        """Divider, ego lane, one more lane to the right: two ego lanes."""
        lanes = lane_set(
            boundary("ego_left", -1.75), boundary("ego_right", 1.75),
            boundary("adjacent_right", 5.25),
            boundary("adjacent_left", -7.75),
        )
        got = detect_carriageway(lanes)
        assert got is not None
        assert got.ego_lanes == 2

    def test_one_far_boundary_yields_a_lower_bound_of_one(self) -> None:
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        got = detect_carriageway(lanes)
        assert got is not None
        assert got.oncoming_lanes == 1
        assert got.total_lanes == got.ego_lanes + got.oncoming_lanes


class TestTracker:
    @staticmethod
    def _obs(side: str = "left", conf: float = 0.7) -> DualCarriageway:
        return DualCarriageway(
            divider_role="ego_left", divider_x_m=-1.75 if side == "left" else 1.75,
            divider_side=side, kind=DividerKind.MEDIAN_GAP, yellow_fraction=None,
            median_width_m=6.0, ego_lanes=1, oncoming_lanes=1,
            confidence=conf, eval_distance_m=15.0)

    def test_one_frame_does_not_confirm(self) -> None:
        t = CarriagewayTracker(min_frames=4, window=10)
        assert t.update(self._obs()) is None

    def test_sustained_evidence_confirms(self) -> None:
        t = CarriagewayTracker(min_frames=4, window=10)
        out = [t.update(self._obs()) for _ in range(5)]
        assert out[3] is not None

    def test_it_keeps_reporting_because_it_is_a_state_not_an_event(self) -> None:
        """Unlike a fork, a dual carriageway stays true. Report it every frame."""
        t = CarriagewayTracker(min_frames=3, window=10)
        for _ in range(3):
            t.update(self._obs())
        assert all(t.update(self._obs()) is not None for _ in range(20))

    def test_a_brief_dropout_does_not_lose_the_belief(self) -> None:
        """A few frames where the median is hidden behind a lorry."""
        t = CarriagewayTracker(min_frames=3, window=10, forget_after=20)
        for _ in range(3):
            t.update(self._obs())
        assert all(t.update(None) is not None for _ in range(10))

    def test_a_long_absence_forgets(self) -> None:
        t = CarriagewayTracker(min_frames=3, window=10, forget_after=8)
        for _ in range(3):
            t.update(self._obs())
        out = [t.update(None) for _ in range(12)]
        assert out[-1] is None

    def test_the_majority_side_wins(self) -> None:
        t = CarriagewayTracker(min_frames=4, window=10)
        t.update(self._obs("left", 0.6))
        t.update(self._obs("left", 0.6))
        t.update(self._obs("right", 0.99))      # confident, but outnumbered
        got = t.update(self._obs("left", 0.7))
        assert got is not None and got.divider_side == "left"

    def test_an_impossible_config_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="never be met"):
            CarriagewayTracker(min_frames=10, window=4)

    def test_reset_clears_state(self) -> None:
        t = CarriagewayTracker(min_frames=2, window=6)
        for _ in range(3):
            t.update(self._obs())
        t.reset()
        assert t.confirmed is None
        assert t.update(self._obs()) is None


class TestDistinctFromFork:
    """The two detectors must partition the cases, not overlap on them."""

    def test_a_fork_is_a_split_and_not_a_carriageway(self) -> None:
        from models.lane_seg.split import detect_split

        lanes = lane_set(boundary("ego_left", -1.75),
                         boundary("ego_right", 1.75, b=0.12))
        assert detect_split(lanes) is not None
        assert detect_carriageway(lanes) is None

    def test_a_median_is_a_carriageway_and_not_a_split(self) -> None:
        from models.lane_seg.split import detect_split

        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        assert detect_split(lanes) is None
        assert detect_carriageway(lanes) is not None


class TestRendering:
    def test_it_draws_something(self) -> None:
        import cv2

        from core.config import OverlayConfig
        from core.overlay import draw_carriageway

        cam = camera()
        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        got = detect_carriageway(lanes)
        assert got is not None
        frame = np.full((H, W, 3), 70, np.uint8)
        out = draw_carriageway(frame, got, lanes, cam, OverlayConfig())
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)
        del cv2

    def test_no_detection_leaves_the_frame_untouched(self) -> None:
        from core.config import OverlayConfig
        from core.overlay import draw_carriageway

        frame = np.full((H, W, 3), 70, np.uint8)
        lanes = lane_set(boundary("ego_left", -1.75))
        out = draw_carriageway(frame, None, lanes, camera(), OverlayConfig())
        assert np.array_equal(out, frame)

    def test_the_hud_line_names_the_side(self) -> None:
        from core.overlay import carriageway_hud_lines

        lanes = lane_set(boundary("ego_left", -1.75), boundary("ego_right", 1.75),
                         boundary("adjacent_left", -7.75))
        got = detect_carriageway(lanes)
        assert got is not None
        line = carriageway_hud_lines(got)[0]
        assert "divided" in line and "left" in line
        assert carriageway_hud_lines(None) == []


class TestObjectExclusion:
    """A yellow car is not a yellow line.

    On real BDD100K frames this was the last remaining false divider: New York
    taxis. No colour or shape test separates a taxi's flank from paint, but the
    detector already knows where the vehicles are, so the lane logic can simply
    be told to ignore what is inside them.
    """

    def test_a_box_over_the_paint_removes_the_detection(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        yellow = paint(left, cam, thickness=15)
        assert detect_carriageway(lanes, cam, yellow) is not None

        ys, xs = np.nonzero(yellow)
        box = np.array([[xs.min() - 5, ys.min() - 5, xs.max() + 5, ys.max() + 5]])
        assert detect_carriageway(lanes, cam, yellow,
                                  exclude_boxes_px=box) is None

    def test_a_box_elsewhere_leaves_the_detection_alone(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        yellow = paint(left, cam, thickness=15)
        far_away = np.array([[0.0, 0.0, 40.0, 40.0]])
        assert detect_carriageway(lanes, cam, yellow,
                                  exclude_boxes_px=far_away) is not None

    def test_the_callers_mask_is_not_modified(self) -> None:
        """The exclusion is local to the call; the segmenter's output stands."""
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        yellow = paint(left, cam, thickness=15)
        before = yellow.copy()
        detect_carriageway(lanes, cam, yellow,
                           exclude_boxes_px=np.array([[0.0, 0.0, W, H]]))
        assert np.array_equal(yellow, before)

    def test_boxes_reaching_outside_the_frame_are_clipped(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        yellow = paint(left, cam, thickness=15)
        huge = np.array([[-500.0, -500.0, W + 500.0, H + 500.0]])
        assert detect_carriageway(lanes, cam, yellow, exclude_boxes_px=huge) is None

    def test_no_boxes_is_the_same_as_none(self) -> None:
        cam = camera()
        left = boundary("ego_left", -1.75)
        lanes = lane_set(left, boundary("ego_right", 1.75))
        yellow = paint(left, cam, thickness=15)
        a = detect_carriageway(lanes, cam, yellow)
        b = detect_carriageway(lanes, cam, yellow,
                               exclude_boxes_px=np.zeros((0, 4)))
        assert (a is None) == (b is None)
        assert a is not None and b is not None
        assert a.divider_role == b.divider_role
