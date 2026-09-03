"""Multi-object tracking: persistent ids across frames, via ByteTrack.

Two downstream capabilities require this and cannot be built without it:

  * **time-to-collision** needs the *same* object's box across consecutive frames
    to measure scale growth; per-frame detections alone give no rate.
  * **lane-change detection** needs the ego vehicle's lateral offset tracked over
    time to tell a real crossing from a one-frame lane-fit wobble.

ByteTrack over DeepSORT because it associates using detection confidence alone --
no appearance-embedding network, so no second model on the critical path, which
matters on an 8 GB laptop GPU already hosting the backbone and MiDaS.

Tracking is post-processing over cached detections, not part of inference: the
`tracker` config section is excluded from the cache key, so retuning it costs
seconds rather than a re-run of the network.
"""

from tracking.bytetrack import ByteTracker, iou_matrix
from tracking.collision import CollisionEstimator, ObjectState
from tracking.kalman import KalmanBoxFilter, cxah_to_xyxy, xyxy_to_cxah
from tracking.track import Track, TrackState
from tracking.ttc import NO_COLLISION, TTCEstimate

__all__ = [
    "ByteTracker",
    "CollisionEstimator",
    "ObjectState",
    "TTCEstimate",
    "Track",
    "TrackState",
    "KalmanBoxFilter",
    "iou_matrix",
    "xyxy_to_cxah",
    "cxah_to_xyxy",
    "NO_COLLISION",
]
