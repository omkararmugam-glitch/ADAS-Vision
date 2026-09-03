"""One tracked object and its lifecycle.

A track is deliberately not "a detection with an id". It has a state machine,
because the two failure modes either side of it are both expensive:

* Confirming instantly means one false detection becomes a tracked object, and a
  tracked object is what raises a collision warning.
* Never forgetting means an object that left the scene keeps a stale id, and the
  next object to appear nearby inherits it.

So a track is TENTATIVE until it has been seen ``n_init`` times, and a confirmed
track that stops matching goes LOST rather than being deleted -- it keeps
predicting through the occlusion and can be re-matched -- until ``max_age``
frames pass with no support, at which point it is REMOVED for good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from tracking.kalman import KalmanBoxFilter, cxah_to_xyxy, xyxy_to_cxah


class TrackState(str, Enum):
    """Lifecycle stage of a track."""

    TENTATIVE = "tentative"
    """Seen, but not yet enough times to be trusted or reported."""

    CONFIRMED = "confirmed"
    """Matched on ``n_init`` frames; reported downstream."""

    LOST = "lost"
    """Was confirmed, currently unmatched. Still predicted, still re-matchable."""

    REMOVED = "removed"
    """Unmatched past ``max_age``, or a tentative track that missed once."""


@dataclass
class Track:
    """A single object followed across frames.

    Attributes:
        track_id: Stable identifier for the lifetime of this track.
        class_id: Class from the detection that started it.
        score: Confidence of the most recent matching detection.
        state: Lifecycle stage.
        mean, covariance: Kalman state.
        hits: Number of frames this track has been matched on.
        age: Frames since the track was created.
        time_since_update: Frames since the last successful match.
        first_frame, last_frame: Frame indices bounding the track's support.
        height_history: Recent ``(frame_index, timestamp_s, box height px)``.
            Time-to-collision reads this; it lives here because only the track
            knows which boxes across frames belong to the same object.
    """

    track_id: int
    class_id: int
    score: float
    mean: np.ndarray
    covariance: np.ndarray
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    first_frame: int = 0
    last_frame: int = 0
    height_history: list[tuple[int, float, float]] = field(default_factory=list)

    #: Bound on retained history. Time-to-collision fits over a short recent
    #: window, so keeping more would grow without limit for no benefit.
    MAX_HISTORY: int = 60

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    @property
    def box_px(self) -> np.ndarray:
        """Current estimate as ``(x0, y0, x1, y1)``."""
        return cxah_to_xyxy(self.mean[:4])

    @property
    def height_px(self) -> float:
        """Current box height. The quantity TTC differentiates."""
        return float(self.mean[3])

    @property
    def bottom_centre_px(self) -> np.ndarray:
        """Ground-contact point: where the object meets the road, if it does.

        This is the point the homography turns into a distance in metres, so it
        is the box edge that matters most for collision logic.
        """
        b = self.box_px
        return np.array([(b[0] + b[2]) / 2.0, b[3]])

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        """Confirmed and matched on the current frame."""
        return self.state is TrackState.CONFIRMED and self.time_since_update == 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    @classmethod
    def create(
        cls, track_id: int, box_px: np.ndarray, class_id: int, score: float,
        kf: KalmanBoxFilter, frame_index: int, timestamp_s: float,
        n_init: int = 3,
    ) -> Track:
        mean, cov = kf.initiate(xyxy_to_cxah(box_px))
        t = cls(
            track_id=track_id, class_id=int(class_id), score=float(score),
            mean=mean, covariance=cov,
            first_frame=frame_index, last_frame=frame_index,
        )
        t.height_history.append((frame_index, timestamp_s, float(mean[3])))
        # Creation counts as the first hit. Without this, n_init=1 -- "trust a
        # detection immediately" -- would never confirm, because confirmation
        # otherwise only happens inside update(), which a new track has not
        # reached yet.
        if t.hits >= n_init:
            t.state = TrackState.CONFIRMED
        return t

    def predict(self, kf: KalmanBoxFilter) -> None:
        """Advance one frame. Called for every track, matched or not."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self, kf: KalmanBoxFilter, box_px: np.ndarray, class_id: int, score: float,
        frame_index: int, timestamp_s: float, n_init: int,
    ) -> None:
        """Correct with a matched detection and advance the state machine."""
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, xyxy_to_cxah(box_px)
        )
        self.class_id = int(class_id)
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = frame_index
        self.height_history.append((frame_index, timestamp_s, float(self.mean[3])))
        if len(self.height_history) > self.MAX_HISTORY:
            del self.height_history[: -self.MAX_HISTORY]
        # A lost track that matches again is immediately trustworthy: it already
        # earned confirmation once, and re-proving it would drop the object for
        # n_init frames every time it passed behind something.
        if self.state in (TrackState.TENTATIVE, TrackState.LOST):
            if self.state is TrackState.LOST or self.hits >= n_init:
                self.state = TrackState.CONFIRMED

    def mark_missed(self, max_age: int) -> None:
        """Handle a frame with no matching detection."""
        if self.state is TrackState.TENTATIVE:
            # Never confirmed, already missing: it was almost certainly noise.
            self.state = TrackState.REMOVED
        elif self.time_since_update > max_age:
            self.state = TrackState.REMOVED
        elif self.state is TrackState.CONFIRMED:
            self.state = TrackState.LOST
