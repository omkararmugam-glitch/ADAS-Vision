"""Tests for ByteTrack association and the track lifecycle.

The interesting cases are the ones that make tracking worth having at all:
an object surviving occlusion with its id intact, two objects crossing without
swapping ids, and a low-confidence detection continuing a track but never
starting one. Those are asserted directly rather than inferred from a count.

Everything here is synthetic geometry -- no model, no GPU, no video.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import TrackerConfig
from core.types import FrameResult
from tracking import ByteTracker, KalmanBoxFilter, TrackState, iou_matrix
from tracking.kalman import cxah_to_xyxy, xyxy_to_cxah

W, H = 1280, 720


def frame(boxes, scores=None, classes=None) -> FrameResult:
    """A FrameResult carrying only boxes; masks are empty."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    blank = np.zeros((H, W), np.uint8)
    return FrameResult(
        boxes_px=boxes,
        class_ids=np.zeros(n, np.int32) if classes is None else np.asarray(classes, np.int32),
        scores=np.full(n, 0.9, np.float32) if scores is None else np.asarray(scores, np.float32),
        drivable_mask=blank,
        lane_mask=blank.copy(),
    )


def box(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class TestIoU:
    def test_identical_boxes_are_one(self) -> None:
        b = np.array([[10.0, 10, 50, 50]])
        assert iou_matrix(b, b)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_are_zero(self) -> None:
        a = np.array([[0.0, 0, 10, 10]])
        b = np.array([[20.0, 20, 30, 30]])
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap(self) -> None:
        a = np.array([[0.0, 0, 10, 10]])       # area 100
        b = np.array([[5.0, 0, 15, 10]])       # area 100, intersection 50
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)

    def test_empty_inputs_give_empty_matrix(self) -> None:
        assert iou_matrix(np.zeros((0, 4)), np.array([[0.0, 0, 1, 1]])).shape == (0, 1)


class TestBoxConversion:
    @pytest.mark.parametrize("b", [[0, 0, 10, 20], [100, 50, 140, 130], [-5, -5, 5, 5]])
    def test_round_trip(self, b: list[float]) -> None:
        assert cxah_to_xyxy(xyxy_to_cxah(np.array(b, float))) == pytest.approx(b)

    def test_aspect_is_width_over_height(self) -> None:
        cx, cy, a, h = xyxy_to_cxah(np.array([0.0, 0, 30, 10]))
        assert (a, h) == pytest.approx((3.0, 10.0))

    def test_zero_height_does_not_divide_by_zero(self) -> None:
        out = xyxy_to_cxah(np.array([0.0, 5, 10, 5]))
        assert np.isfinite(out).all()


class TestKalman:
    def test_a_static_box_stays_put(self) -> None:
        kf = KalmanBoxFilter()
        m, c = kf.initiate(np.array([100.0, 100, 1.0, 50]))
        for _ in range(10):
            m, c = kf.predict(m, c)
            m, c = kf.update(m, c, np.array([100.0, 100, 1.0, 50]))
        assert m[:4] == pytest.approx([100, 100, 1.0, 50], abs=0.5)

    def test_constant_velocity_is_learned(self) -> None:
        """After a few frames the filter should predict ahead, not lag behind."""
        kf = KalmanBoxFilter()
        m, c = kf.initiate(np.array([100.0, 100, 1.0, 50]))
        for i in range(1, 12):
            m, c = kf.predict(m, c)
            m, c = kf.update(m, c, np.array([100.0 + 10 * i, 100, 1.0, 50]))
        assert m[4] == pytest.approx(10.0, abs=1.5)   # vx ~ 10 px/frame
        m_pred, _ = kf.predict(m, c)
        assert m_pred[0] == pytest.approx(100 + 10 * 12, abs=3.0)

    def test_gating_rejects_a_distant_measurement(self) -> None:
        kf = KalmanBoxFilter()
        m, c = kf.initiate(np.array([100.0, 100, 1.0, 50]))
        for _ in range(5):
            m, c = kf.predict(m, c)
            m, c = kf.update(m, c, np.array([100.0, 100, 1.0, 50]))
        near = kf.gating_distance(m, c, np.array([[102.0, 100, 1.0, 50]]))[0]
        far = kf.gating_distance(m, c, np.array([[900.0, 600, 1.0, 50]]))[0]
        assert near < far
        assert far > 9.4877     # beyond the chi-square gate


class TestLifecycle:
    def test_a_track_is_not_reported_before_n_init(self) -> None:
        """One stray detection must never reach the collision logic."""
        tr = ByteTracker(TrackerConfig(n_init=3))
        assert tr.update(frame([box(100, 100, 40, 40)]), 0, 0.0) == []
        assert tr.update(frame([box(100, 100, 40, 40)]), 1, 0.04) == []
        assert len(tr.update(frame([box(100, 100, 40, 40)]), 2, 0.08)) == 1

    def test_a_single_false_detection_is_discarded(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=3))
        tr.update(frame([box(100, 100, 40, 40)]), 0, 0.0)
        tr.update(frame([]), 1, 0.04)
        assert tr.tracks == []

    def test_ids_are_stable_across_frames(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=2))
        ids = set()
        for i in range(12):
            active = tr.update(frame([box(100 + 5 * i, 200, 40, 40)]), i, i / 25)
            ids.update(t.track_id for t in active)
        assert ids == {1}

    def test_ids_are_never_reused_after_removal(self) -> None:
        cfg = TrackerConfig(n_init=2, max_age=2)
        tr = ByteTracker(cfg)
        for i in range(4):
            tr.update(frame([box(100, 200, 40, 40)]), i, i / 25)
        for i in range(4, 12):                      # object leaves
            tr.update(frame([]), i, i / 25)
        for i in range(12, 16):                     # a different object arrives
            active = tr.update(frame([box(100, 200, 40, 40)]), i, i / 25)
        assert active[0].track_id != 1, "a new object must not inherit a dead id"

    def test_reset_restarts_ids(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=1))
        tr.update(frame([box(100, 200, 40, 40)]), 0, 0.0)
        tr.reset()
        active = tr.update(frame([box(100, 200, 40, 40)]), 0, 0.0)
        assert active[0].track_id == 1


class TestOcclusion:
    def test_a_track_survives_a_gap_and_keeps_its_id(self) -> None:
        """A vehicle passing behind a sign must come back as the same object."""
        tr = ByteTracker(TrackerConfig(n_init=2, max_age=30))
        for i in range(6):
            tr.update(frame([box(100 + 10 * i, 200, 40, 40)]), i, i / 25)
        for i in range(6, 12):                       # occluded: no detections
            tr.update(frame([]), i, i / 25)
        active = tr.update(frame([box(100 + 10 * 12, 200, 40, 40)]), 12, 12 / 25)
        assert len(active) == 1
        assert active[0].track_id == 1

    def test_a_track_is_dropped_after_max_age(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=2, max_age=5))
        for i in range(4):
            tr.update(frame([box(100, 200, 40, 40)]), i, i / 25)
        for i in range(4, 20):
            tr.update(frame([]), i, i / 25)
        assert tr.tracks == []

    def test_a_lost_track_is_not_reported_while_lost(self) -> None:
        """Predicting through an occlusion must not fabricate a detection."""
        tr = ByteTracker(TrackerConfig(n_init=2, max_age=30))
        for i in range(5):
            tr.update(frame([box(100, 200, 40, 40)]), i, i / 25)
        assert tr.update(frame([]), 5, 0.2) == []
        assert any(t.state is TrackState.LOST for t in tr.tracks)


class TestTwoPassAssociation:
    def test_a_weak_detection_continues_a_track(self) -> None:
        """The point of ByteTrack: a half-occluded object scores low but is real."""
        cfg = TrackerConfig(n_init=2, high_threshold=0.5, low_threshold=0.1)
        tr = ByteTracker(cfg)
        for i in range(4):
            tr.update(frame([box(100 + 5 * i, 200, 40, 40)], scores=[0.9]), i, i / 25)
        active = tr.update(frame([box(120, 200, 40, 40)], scores=[0.2]), 4, 0.16)
        assert len(active) == 1 and active[0].track_id == 1

    def test_a_weak_detection_never_starts_a_track(self) -> None:
        """Weak evidence continues belief; it must not create it."""
        cfg = TrackerConfig(n_init=1, high_threshold=0.5, low_threshold=0.1,
                            new_track_threshold=0.6)
        tr = ByteTracker(cfg)
        for i in range(6):
            assert tr.update(frame([box(100, 200, 40, 40)], scores=[0.2]), i, i / 25) == []
        assert tr.tracks == []

    def test_a_detection_below_low_threshold_is_ignored_entirely(self) -> None:
        cfg = TrackerConfig(n_init=2, high_threshold=0.5, low_threshold=0.1)
        tr = ByteTracker(cfg)
        for i in range(4):
            tr.update(frame([box(100, 200, 40, 40)], scores=[0.9]), i, i / 25)
        tr.update(frame([box(100, 200, 40, 40)], scores=[0.05]), 4, 0.16)
        assert tr.active_tracks() == []


class TestIdentityUnderCrossing:
    def test_two_objects_crossing_keep_distinct_ids(self) -> None:
        """Hungarian assignment exists for exactly this case."""
        tr = ByteTracker(TrackerConfig(n_init=2, max_age=30))
        seen: dict[int, list[float]] = {}
        for i in range(14):
            left_x = 200 + 30 * i
            right_x = 620 - 30 * i
            active = tr.update(
                frame([box(left_x, 300, 50, 50), box(right_x, 340, 50, 50)]), i, i / 25
            )
            for t in active:
                seen.setdefault(t.track_id, []).append(t.box_px[0])
        assert len(seen) == 2, f"expected 2 ids through the crossing, got {sorted(seen)}"

    def test_two_separated_objects_get_two_ids(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=2))
        for i in range(5):
            active = tr.update(
                frame([box(200, 300, 50, 50), box(900, 300, 50, 50)]), i, i / 25
            )
        assert {t.track_id for t in active} == {1, 2}


class TestTrackGeometry:
    def test_bottom_centre_is_the_ground_contact_point(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=1))
        active = tr.update(frame([[100.0, 200, 180, 300]]), 0, 0.0)
        bc = active[0].bottom_centre_px
        assert bc[0] == pytest.approx(140, abs=2)
        assert bc[1] == pytest.approx(300, abs=2)

    def test_height_history_accumulates_with_timestamps(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=1))
        for i in range(5):
            tr.update(frame([box(100, 200, 40, 40 + 4 * i)]), i, i / 25)
        hist = tr.tracks[0].height_history
        assert len(hist) == 5
        assert [h[0] for h in hist] == [0, 1, 2, 3, 4]
        assert hist[-1][1] == pytest.approx(4 / 25)

    def test_height_history_is_bounded(self) -> None:
        """An hour-long clip must not grow this without limit."""
        tr = ByteTracker(TrackerConfig(n_init=1))
        for i in range(200):
            tr.update(frame([box(100, 200, 40, 40)]), i, i / 25)
        assert len(tr.tracks[0].height_history) <= tr.tracks[0].MAX_HISTORY

    def test_class_id_is_carried_through(self) -> None:
        tr = ByteTracker(TrackerConfig(n_init=1))
        active = tr.update(frame([box(100, 200, 40, 40)], classes=[1]), 0, 0.0)
        assert active[0].class_id == 1


class TestEmptyInput:
    def test_no_detections_ever_is_safe(self) -> None:
        tr = ByteTracker(TrackerConfig())
        for i in range(5):
            assert tr.update(frame([]), i, i / 25) == []
