"""Per-track collision state: distance, TTC, and whether the object is in path.

Holds the per-track distance history that :mod:`tracking.ttc` fits against. That
history cannot live on the ``Track`` itself, because a distance only exists once
a calibrated camera is available and tracking must work without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.calibrate import CameraModel
from core.config import TTCConfig
from core.types import CLASS_NAMES
from tracking.track import Track
from tracking.ttc import TTCEstimate, combine, range_rate_ttc, scale_expansion_ttc


@dataclass(frozen=True)
class ObjectState:
    """Everything the decision engine needs to know about one tracked object.

    Attributes:
        track_id: Persistent id.
        class_name: Resolved class name.
        box_px: Current box.
        score: Detection confidence.
        distance_m: Ground-plane distance, if calibrated and on the road plane.
        lateral_offset_m: Signed lateral position, positive to the right of the
            camera axis. This is what decides "in my path" versus "in the next
            lane" -- a vehicle 3 m to the left at 15 m is not a collision risk.
        ttc: Time-to-collision estimate.
        in_path: Whether the object lies within the ego corridor.
    """

    track_id: int
    class_name: str
    box_px: np.ndarray
    score: float
    distance_m: float | None
    lateral_offset_m: float | None
    ttc: TTCEstimate
    in_path: bool


class CollisionEstimator:
    """Turns tracks into distances and time-to-collision.

    One instance per run; it accumulates per-track distance history keyed by
    track id, and forgets a track once it disappears so a long clip does not
    grow this without bound.
    """

    def __init__(self, cfg: TTCConfig, camera: CameraModel | None = None) -> None:
        self.cfg = cfg
        self.camera = camera
        self._distances: dict[int, list[tuple[float, float]]] = {}

    def _ground_position(self, track: Track) -> tuple[float | None, float | None]:
        """Back-project the box's ground-contact point to ego metres.

        Returns ``(forward_m, lateral_m)``, or ``(None, None)`` when there is no
        calibration or the point does not lie on the road plane -- which happens
        for anything at or above the horizon, where the projection is meaningless
        rather than merely inaccurate.
        """
        if self.camera is None:
            return None, None
        pt = track.bottom_centre_px.reshape(1, 2)
        if not bool(self.camera.is_on_ground(pt)[0]):
            return None, None
        ground = np.asarray(self.camera.image_to_ground(pt)).reshape(-1, 2)[0]
        lateral, forward = float(ground[0]), float(ground[1])
        if not np.isfinite(forward) or forward <= 0:
            return None, None
        return forward, lateral

    def update(self, tracks: list[Track], timestamp_s: float) -> list[ObjectState]:
        """Compute collision state for every active track this frame."""
        alive = {t.track_id for t in tracks}
        for tid in list(self._distances):
            if tid not in alive:
                del self._distances[tid]

        out: list[ObjectState] = []
        for t in tracks:
            forward, lateral = self._ground_position(t)
            if forward is not None:
                hist = self._distances.setdefault(t.track_id, [])
                hist.append((timestamp_s, forward))
                if len(hist) > self.cfg.history_window:
                    del hist[: -self.cfg.history_window]

            scale_ttc, n, quality = scale_expansion_ttc(
                t.height_history, self.cfg.fit_window, self.cfg.min_samples
            )
            r_ttc, d_now, rate = range_rate_ttc(
                self._distances.get(t.track_id, []),
                self.cfg.fit_window,
                self.cfg.min_samples,
            )
            estimate = combine(
                scale_ttc, r_ttc, d_now if d_now is not None else forward, rate,
                n, quality, self.cfg.min_fit_quality, self.cfg.prefer_range_within_m,
            )

            out.append(ObjectState(
                track_id=t.track_id,
                class_name=(CLASS_NAMES[t.class_id]
                            if 0 <= t.class_id < len(CLASS_NAMES) else f"class:{t.class_id}"),
                box_px=t.box_px,
                score=t.score,
                distance_m=estimate.distance_m,
                lateral_offset_m=lateral,
                ttc=estimate,
                in_path=self.in_path(lateral, forward),
            ))
        return out

    def in_path(self, lateral_m: float | None, forward_m: float | None) -> bool:
        """Whether an object sits inside the ego corridor.

        The corridor widens with distance rather than being a fixed lane width.
        That is not about the road: it is about *error*. Lateral position is
        derived from the same monocular geometry whose accuracy falls off with
        distance, so a fixed-width corridor would drop genuinely dangerous
        distant objects on the strength of a lateral estimate too noisy to
        justify the exclusion.
        """
        if lateral_m is None or forward_m is None:
            # Without calibration nothing can be excluded on geometry, and
            # silently dropping objects would be the dangerous failure.
            return True
        half = self.cfg.corridor_half_width_m + self.cfg.corridor_widening * forward_m
        return abs(lateral_m) <= half

    def reset(self) -> None:
        self._distances.clear()
