"""ByteTrack association.

The idea ByteTrack is named for: most trackers throw away low-confidence
detections before association, but a *partially occluded* object produces exactly
that -- a real object with a weak score. Discarding it breaks the track; keeping
it as a candidate for *already-established* tracks recovers the object without
letting noise create new ones.

So association runs in two passes:

1. High-scoring detections against all confirmed and lost tracks.
2. Low-scoring detections against only the tracks that went unmatched in pass 1.

Low-scoring detections never start a track. That asymmetry is the whole design:
weak evidence is enough to *continue* believing in an object, never enough to
*start* believing in one.

Chosen over DeepSORT because it needs no appearance-embedding network -- no
second model on the critical path, which matters on an 8 GB laptop GPU already
hosting the backbone and MiDaS.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.config import TrackerConfig
from core.types import FrameResult
from tracking.kalman import CHI2_INV_95_4DOF, KalmanBoxFilter, xyxy_to_cxah
from tracking.track import Track, TrackState


#: Cost assigned to a pair the motion model vetoes. Finite so the assignment
#: stays solvable, far above any real cost (which lives in [0, 1]) so it is
#: always rejected by the threshold check afterwards.
GATED = 1e6


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``(N, 4)`` xyxy boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _associate(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal one-to-one assignment, rejecting pairs above ``max_cost``.

    Hungarian rather than greedy: greedy matching takes the single best pair
    first and can strand two tracks in a crossing, which is precisely the case
    where identities get swapped.
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))
    rows, cols = linear_sum_assignment(cost)
    matches, matched_r, matched_c = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_r.add(int(r))
            matched_c.add(int(c))
    return (
        matches,
        [r for r in range(n_rows) if r not in matched_r],
        [c for c in range(n_cols) if c not in matched_c],
    )


class ByteTracker:
    """Multi-object tracker giving detections persistent ids across frames.

    Two downstream capabilities depend on this and cannot be built without it:
    time-to-collision needs the *same* object's box on consecutive frames to
    measure how fast it is growing, and lane-change detection needs a history to
    tell a real crossing from a one-frame lane-fit wobble.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_index = -1

    # ------------------------------------------------------------------ #

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _gate(self, cost: np.ndarray, tracks: list[Track], boxes: np.ndarray) -> np.ndarray:
        """Veto matches the motion model says are implausible.

        IoU alone will happily match a track to a detection that overlaps it but
        sits far outside where the object could have moved. The Mahalanobis gate
        uses the filter's own uncertainty, so a confident track is gated tightly
        and a freshly-created one leniently.
        """
        if not self.cfg.use_mahalanobis_gate or len(tracks) == 0 or len(boxes) == 0:
            return cost
        measurements = xyxy_to_cxah(boxes)
        out = cost.copy()
        for i, t in enumerate(tracks):
            d2 = self.kf.gating_distance(t.mean, t.covariance, measurements)
            # A large finite sentinel, never np.inf: a row or column that is
            # entirely infinite makes the assignment problem infeasible and
            # scipy raises. GATED is above any reachable threshold, so these
            # pairs are still rejected -- they just no longer break the solver.
            out[i, d2 > CHI2_INV_95_4DOF] = GATED
        return out

    def update(
        self, result: FrameResult, frame_index: int, timestamp_s: float
    ) -> list[Track]:
        """Advance the tracker one frame.

        Args:
            result: Detections for this frame, in original image pixels.
            frame_index: Index of this frame.
            timestamp_s: Its timestamp, used for the TTC time base. Real
                timestamps matter here -- assuming a constant frame interval on
                variable-frame-rate footage biases every velocity estimate.

        Returns:
            The confirmed tracks matched on this frame.
        """
        self.frame_index = frame_index
        boxes = np.asarray(result.boxes_px, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(result.scores, dtype=np.float64).reshape(-1)
        classes = np.asarray(result.class_ids, dtype=np.int32).reshape(-1)

        high = scores >= self.cfg.high_threshold
        low = (scores >= self.cfg.low_threshold) & ~high

        for t in self.tracks:
            t.predict(self.kf)

        # --- pass 1: strong detections against everything still alive -----
        pool = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state is TrackState.TENTATIVE]

        matches, un_tracks, un_dets = self._match(
            pool, boxes[high], self.cfg.match_threshold
        )
        hi_idx = np.flatnonzero(high)
        for ti, di in matches:
            d = hi_idx[di]
            pool[ti].update(self.kf, boxes[d], classes[d], scores[d],
                            frame_index, timestamp_s, self.cfg.n_init)

        # --- pass 2: weak detections rescue tracks that just went unmatched -
        remaining = [pool[i] for i in un_tracks if pool[i].state is not TrackState.LOST]
        lo_idx = np.flatnonzero(low)
        m2, un2, _ = self._match(remaining, boxes[low], self.cfg.match_threshold_low)
        for ti, di in m2:
            d = lo_idx[di]
            remaining[ti].update(self.kf, boxes[d], classes[d], scores[d],
                                 frame_index, timestamp_s, self.cfg.n_init)
        rescued = {id(remaining[ti]) for ti, _ in m2}

        # --- pass 3: leftover strong detections may extend tentative tracks -
        leftover = hi_idx[un_dets] if len(un_dets) else np.empty(0, dtype=int)
        m3, un3_t, un3_d = self._match(
            tentative, boxes[leftover], self.cfg.match_threshold
        )
        for ti, di in m3:
            d = leftover[di]
            tentative[ti].update(self.kf, boxes[d], classes[d], scores[d],
                                 frame_index, timestamp_s, self.cfg.n_init)

        # --- ageing --------------------------------------------------------
        for t in pool:
            if t.time_since_update > 0 and id(t) not in rescued:
                t.mark_missed(self.cfg.max_age)
        for i in un3_t:
            tentative[i].mark_missed(self.cfg.max_age)

        # --- births: only strong, still-unmatched detections ---------------
        for di in un3_d:
            d = leftover[di]
            if scores[d] < self.cfg.new_track_threshold:
                continue
            self.tracks.append(Track.create(
                self._new_id(), boxes[d], classes[d], scores[d],
                self.kf, frame_index, timestamp_s, self.cfg.n_init,
            ))

        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        return self.active_tracks()

    def _match(
        self, tracks: list[Track], boxes: np.ndarray, threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """IoU association between tracks and boxes, gated by the motion model."""
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        predicted = np.stack([t.box_px for t in tracks])
        cost = 1.0 - iou_matrix(predicted, boxes)
        cost = self._gate(cost, tracks, boxes)
        return _associate(cost, 1.0 - threshold)

    # ------------------------------------------------------------------ #

    def active_tracks(self) -> list[Track]:
        """Confirmed tracks matched on the current frame, in id order."""
        return sorted(
            (t for t in self.tracks if t.is_active), key=lambda t: t.track_id
        )

    def reset(self) -> None:
        """Forget everything. Ids restart, so a new clip never inherits them."""
        self.tracks.clear()
        self._next_id = 1
        self.frame_index = -1
