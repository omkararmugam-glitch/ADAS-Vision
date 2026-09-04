"""Per-track collision state: distance, TTC, and whether the object is in path.

Holds the per-track distance history that :mod:`tracking.ttc` fits against. That
history cannot live on the ``Track`` itself, because a distance only exists once
a calibrated camera is available and tracking must work without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.calibrate import CameraModel
from core.config import DepthConfig, TTCConfig
from core.types import CLASS_NAMES
from depth.align import DepthScale, fit_scale, sample_depth
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
        distance_source: ``homography``, ``depth`` or ``none`` -- which estimator
            produced ``distance_m``. Recorded because the two carry very
            different error characteristics and an alert should be auditable
            back to which one was used.
    """

    track_id: int
    class_name: str
    box_px: np.ndarray
    score: float
    distance_m: float | None
    lateral_offset_m: float | None
    ttc: TTCEstimate
    in_path: bool
    distance_source: str = "none"


class CollisionEstimator:
    """Turns tracks into distances and time-to-collision.

    One instance per run; it accumulates per-track distance history keyed by
    track id, and forgets a track once it disappears so a long clip does not
    grow this without bound.
    """

    def __init__(
        self,
        cfg: TTCConfig,
        camera: CameraModel | None = None,
        depth_cfg: DepthConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.depth_cfg = depth_cfg
        self._distances: dict[int, list[tuple[float, float]]] = {}
        #: Scale fitted on the most recent frame, exposed for logging and tests.
        self.depth_scale: DepthScale | None = None

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

    def _fit_depth_scale(
        self, tracks: list[Track], ground: dict[int, tuple[float, float]],
        depth_map: np.ndarray,
    ) -> DepthScale:
        """Fit metres onto the relative depth map, using on-road tracks as anchors.

        The anchors come from the scene itself: every tracked object whose
        ground-contact point the homography accepted is a point where the true
        distance is known. No extra sensor, no assumption beyond the one the
        homography already makes.
        """
        cfg = self.depth_cfg
        pts, dists = [], []
        for t in tracks:
            g = ground.get(t.track_id)
            if g is None:
                continue
            pts.append(t.bottom_centre_px)
            dists.append(g[0])
        if len(pts) < cfg.min_anchors:
            return DepthScale(n_anchors=len(pts))
        values = sample_depth(depth_map, np.asarray(pts), patch=cfg.patch_px)
        return fit_scale(
            values, np.asarray(dists),
            min_anchors=cfg.min_anchors,
            max_residual_m=cfg.max_residual_m,
            min_r_squared=cfg.min_r_squared,
        )

    def update(
        self,
        tracks: list[Track],
        timestamp_s: float,
        depth_map: np.ndarray | None = None,
    ) -> list[ObjectState]:
        """Compute collision state for every active track this frame.

        Args:
            tracks: Confirmed tracks for this frame.
            timestamp_s: Frame time, for the TTC time base.
            depth_map: Optional relative depth. Used only to supply a distance
                for objects the homography could not place -- an occluded
                ground-contact point, or anything off the road plane.
        """
        alive = {t.track_id for t in tracks}
        for tid in list(self._distances):
            if tid not in alive:
                del self._distances[tid]

        ground = {}
        for t in tracks:
            fwd, lat = self._ground_position(t)
            if fwd is not None:
                ground[t.track_id] = (fwd, lat)

        self.depth_scale = None
        if depth_map is not None and self.depth_cfg is not None:
            self.depth_scale = self._fit_depth_scale(tracks, ground, depth_map)

        out: list[ObjectState] = []
        for t in tracks:
            forward, lateral = ground.get(t.track_id, (None, None))
            source = "homography" if forward is not None else "none"
            if (
                forward is None
                and depth_map is not None
                and self.depth_scale is not None
                and self.depth_scale.valid
            ):
                value = sample_depth(
                    depth_map, t.bottom_centre_px.reshape(1, 2),
                    patch=self.depth_cfg.patch_px,
                )
                z = float(self.depth_scale.to_metres(value)[0])
                if np.isfinite(z) and z > 0:
                    forward, source = z, "depth"
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
                distance_source=source,
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
        self.depth_scale = None
