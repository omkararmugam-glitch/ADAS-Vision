"""A fake detector, purely to prove the pipeline end to end before any model.

This file is scaffolding for Stage 1 and is expected to be deleted once the real
YOLOPv2 backbone lands in ``src/detect.py`` and ``src/lanes.py``. It lives here,
rather than squatting on those filenames, so that the real modules arrive clean.

It is not random noise. Every output is *generated through the calibration*: a
synthetic lead vehicle is placed at a known ego-frame distance in metres and
projected into the image, and the lane boundaries are ground-plane lines at a
fixed lateral offset projected the same way. That makes the annotated video an
end-to-end check on the geometry -- if the homography is right, the stub lane
lines lie along the real road markings and the vehicle box sits on the road at
the distance the HUD claims. If calibration is wrong, it is obvious on sight.

Coordinate frames: ego metres (X right, Y forward) for the scene definition,
image pixels for everything returned.
"""

from __future__ import annotations

import numpy as np

from src.calibrate import CameraModel
from src.config import DetectorConfig
from src.io import Frame
from src.types import Detection, FrameResult, LaneLine

#: Nominal geometry of the synthetic lead vehicle, in metres.
_VEHICLE_WIDTH_M = 1.8
_VEHICLE_ASPECT = 0.85  # height / width, as it appears in the image

#: Synthetic approach: the fake vehicle closes from this distance at this speed.
_START_DISTANCE_M = 45.0
_CLOSING_SPEED_MPS = 4.0
_MIN_DISTANCE_M = 8.0


class StubDetector:
    """Produces deterministic fake detections and lanes from the calibration.

    Args:
        camera: Calibration used to project the synthetic scene into the image.
            Must already match the frame size being processed.
        cfg: Detector settings; ``seed`` makes the jitter reproducible so that
            cache reuse can be tested.
        lane_half_width_m: Lateral offset of the ego lane boundaries.
        sample_range_m: ``(near, far)`` extent over which lane polylines are drawn.
    """

    #: Bumped whenever the fake output changes, so caches invalidate correctly.
    VERSION = "stub-0.1"

    def __init__(
        self,
        camera: CameraModel,
        cfg: DetectorConfig,
        lane_half_width_m: float = 1.75,
        sample_range_m: tuple[float, float] = (5.0, 40.0),
    ) -> None:
        self.camera = camera
        self.cfg = cfg
        self.lane_half_width_m = lane_half_width_m
        self.sample_range_m = sample_range_m
        self._rng = np.random.default_rng(cfg.seed)

    # -- scene definition (ego metres) -------------------------------------- #

    def _vehicle_ground_position(self, t: float) -> tuple[float, float]:
        """Ego-frame ``(x, y)`` metres of the synthetic lead vehicle at time ``t``.

        Closes steadily from :data:`_START_DISTANCE_M`, drifting laterally on a
        slow sinusoid so that lane assignment and cut-in logic have something to
        chew on later.
        """
        y = max(_MIN_DISTANCE_M, _START_DISTANCE_M - _CLOSING_SPEED_MPS * t)
        x = 1.2 * np.sin(0.35 * t)
        return float(x), float(y)

    def _project_ground_box(self, x_m: float, y_m: float) -> tuple[float, float, float, float] | None:
        """Project a ground-plane vehicle footprint to an image box.

        The homography maps the road plane only, so the box width comes from
        projecting the vehicle's ground footprint and the height is taken as a
        fixed multiple of that width. That is an approximation -- a genuine 3-D
        box needs the full projection matrix -- and it is good enough for a stub
        whose purpose is to exercise the plumbing.

        Returns None if the footprint does not project to a usable image box.
        """
        half = _VEHICLE_WIDTH_M / 2.0
        footprint = np.array([[x_m - half, y_m], [x_m + half, y_m]], dtype=np.float64)
        # The footprint is already in ground metres, so it is on the road by
        # construction; what has to be checked is that it is ahead of the camera
        # and projects to a finite pixel. (is_on_ground() takes image pixels and
        # must not be handed metres.)
        if y_m <= 0:
            return None
        try:
            px = self.camera.ground_to_image(footprint)
        except ValueError:
            return None
        u1, u2 = float(px[0, 0]), float(px[1, 0])
        if u2 <= u1:
            return None
        v_bottom = float(px[:, 1].mean())
        width_px = u2 - u1
        height_px = width_px * _VEHICLE_ASPECT
        v_top = v_bottom - height_px
        if not np.isfinite([u1, u2, v_top, v_bottom]).all() or height_px < 2:
            return None
        return (u1, v_top, u2, v_bottom)

    def _lane_polyline(self, x_offset_m: float) -> np.ndarray | None:
        """Image polyline for a ground line at constant lateral offset.

        Returns points ordered near-to-far, or None if fewer than two samples
        project into finite image coordinates.
        """
        near, far = self.sample_range_m
        ys = np.linspace(near, far, 24)
        ground = np.column_stack([np.full_like(ys, x_offset_m), ys])
        px = self.camera.ground_to_image(ground)
        finite = np.isfinite(px).all(axis=1)
        px = px[finite]
        return px if len(px) >= 2 else None

    def drivable_mask(self, height: int, width: int) -> np.ndarray:
        """Boolean mask of the ego lane's road surface, as a filled polygon.

        Stands in for the drivable-area head of the real multi-task network.
        """
        left = self._lane_polyline(-self.lane_half_width_m)
        right = self._lane_polyline(self.lane_half_width_m)
        mask = np.zeros((height, width), dtype=bool)
        if left is None or right is None:
            return mask
        import cv2  # local: only needed for polygon fill

        polygon = np.int32(np.round(np.vstack([left, right[::-1]])))
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(filled, [polygon], 1)
        return filled.astype(bool)

    # -- public API --------------------------------------------------------- #

    def detect(self, frame: Frame) -> FrameResult:
        """Return the synthetic detections and lanes for one frame.

        The forward distance used to build the vehicle box is recorded in
        ``extras['stub_vehicle_distance_m']``, so the annotated video can be
        checked against the ground truth that generated it.
        """
        t = frame.timestamp_s
        detections: list[Detection] = []
        lanes: list[LaneLine] = []
        extras: dict[str, float | bool] = {}

        x_m, y_m = self._vehicle_ground_position(t)
        box = self._project_ground_box(x_m, y_m)
        if box is not None:
            # Deterministic wobble, so the boxes do not look suspiciously perfect.
            score = float(np.clip(0.85 + 0.1 * np.sin(2.1 * t), 0.0, 1.0))
            if score >= self.cfg.score_threshold:
                detections.append(
                    Detection(bbox_px=box, label="car", score=score, track_id=1)
                )
                extras["stub_vehicle_distance_m"] = y_m
                extras["stub_vehicle_lateral_m"] = x_m

        for role, offset, kind in (
            ("ego_left", -self.lane_half_width_m, "solid"),
            ("ego_right", self.lane_half_width_m, "dashed"),
        ):
            pts = self._lane_polyline(offset)
            if pts is not None:
                lanes.append(LaneLine(points_px=pts, role=role, kind=kind, score=0.9))

        extras["stub"] = True
        return FrameResult(
            frame_index=frame.index,
            source_index=frame.source_index,
            timestamp_s=frame.timestamp_s,
            detections=detections,
            lanes=lanes,
            extras=extras,
        )
