"""A fake detector, so the pipeline can be exercised without loading a model.

This implements the same :class:`~models.perception.Detector` interface as the real
backbone, which keeps the end-to-end tests fast (no 156 MB of weights, no GPU)
and keeps ``predict`` runnable on a machine with no model at all.

It is not random noise. Every output is *generated through the calibration*: a
synthetic lead vehicle is placed at a known ego-frame distance in metres and
projected into the image, and the lane markings are ground-plane lines at fixed
lateral offsets rendered the same way. So the annotated video remains an
end-to-end check on the geometry -- if the homography is right, the stub's lane
mask lies where real markings would, and the fitted polynomial should recover
the exact offsets the stub drew.

Coordinate frames: ego metres (X right, Y forward) for the scene definition,
image pixels for everything returned.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.calibrate import CameraModel
from core.config import DetectorConfig
from models.perception import Detector
from core.types import CLASS_IDS, FrameResult, NativeMasks

#: Nominal geometry of the synthetic lead vehicle, in metres.
_VEHICLE_WIDTH_M = 1.8
_VEHICLE_ASPECT = 0.85  # height / width, as it appears in the image

#: Synthetic approach: the fake vehicle closes from this distance at this speed.
_START_DISTANCE_M = 45.0
_CLOSING_SPEED_MPS = 4.0
_MIN_DISTANCE_M = 8.0

#: Painted width of a synthetic lane marking, metres.
_MARKING_WIDTH_M = 0.15


class StubDetector(Detector):
    """Deterministic fake detections and segmentation, generated from calibration.

    Args:
        camera: Calibration used to project the synthetic scene. Must already
            match the frame size being processed.
        cfg: Detector settings; ``seed`` and ``score_threshold`` apply.
        lane_half_width_m: Lateral offset of the synthetic ego lane boundaries.
        sample_range_m: ``(near, far)`` extent the synthetic scene spans.
    """

    VERSION = "stub-0.2"

    def __init__(
        self,
        camera: CameraModel,
        cfg: DetectorConfig,
        lane_half_width_m: float = 1.85,
        sample_range_m: tuple[float, float] = (5.0, 40.0),
    ) -> None:
        self.camera = camera
        self.cfg = cfg
        self.lane_half_width_m = lane_half_width_m
        self.sample_range_m = sample_range_m
        self._t = 0.0

    @property
    def name(self) -> str:
        return "stub"

    @property
    def weights_hash(self) -> str:
        """No weights; the version stands in so a behaviour change invalidates caches."""
        return self.VERSION

    @property
    def class_names(self) -> tuple[str, ...]:
        return ("vehicle",)

    def set_time(self, t_s: float) -> None:
        """Set the scene time, since :meth:`infer` receives only an array.

        The pipeline calls this from the frame timestamp before inferring, which
        keeps the synthetic vehicle's motion tied to real time rather than to
        how many frames happen to have been processed.
        """
        self._t = float(t_s)

    # -- scene definition (ego metres) -------------------------------------- #

    def _vehicle_ground_position(self, t: float) -> tuple[float, float]:
        """Ego-frame ``(x, y)`` metres of the synthetic lead vehicle at time ``t``."""
        y = max(_MIN_DISTANCE_M, _START_DISTANCE_M - _CLOSING_SPEED_MPS * t)
        x = 1.2 * np.sin(0.35 * t)
        return float(x), float(y)

    def _project_ground_box(
        self, x_m: float, y_m: float
    ) -> tuple[float, float, float, float] | None:
        """Project a ground footprint to an image box, or None if unusable.

        The homography maps the road plane only, so the box width comes from the
        projected footprint and the height is a fixed multiple of it. A real 3-D
        box needs the full projection matrix; this is a stub.
        """
        if y_m <= 0:
            return None
        half = _VEHICLE_WIDTH_M / 2.0
        footprint = np.array([[x_m - half, y_m], [x_m + half, y_m]], dtype=np.float64)
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
        if not np.isfinite([u1, u2, v_bottom]).all() or height_px < 2:
            return None
        return (u1, v_bottom - height_px, u2, v_bottom)

    def _ground_polyline(self, x_offset_m: float, n: int = 40) -> np.ndarray | None:
        """Image polyline for a ground line at constant lateral offset."""
        near, far = self.sample_range_m
        ys = np.linspace(near, far, n)
        ground = np.column_stack([np.full_like(ys, x_offset_m), ys])
        try:
            px = self.camera.ground_to_image(ground)
        except ValueError:
            return None
        finite = np.isfinite(px).all(axis=1)
        px = px[finite]
        return px if len(px) >= 2 else None

    def _render_masks(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Paint synthetic drivable-area and lane-marking masks."""
        drivable = np.zeros((height, width), dtype=np.uint8)
        lane = np.zeros((height, width), dtype=np.uint8)
        left = self._ground_polyline(-self.lane_half_width_m)
        right = self._ground_polyline(self.lane_half_width_m)
        if left is None or right is None:
            return drivable, lane

        cv2.fillPoly(drivable, [np.int32(np.round(np.vstack([left, right[::-1]])))], 1)

        # Paint markings with real width so the fitter sees a band, not a hairline.
        for offset in (-self.lane_half_width_m, self.lane_half_width_m):
            inner = self._ground_polyline(offset - _MARKING_WIDTH_M / 2)
            outer = self._ground_polyline(offset + _MARKING_WIDTH_M / 2)
            if inner is None or outer is None:
                continue
            cv2.fillPoly(lane, [np.int32(np.round(np.vstack([inner, outer[::-1]])))], 1)
        return drivable, lane

    # -- Detector API ------------------------------------------------------- #

    def infer(self, frame: np.ndarray) -> FrameResult:
        """Synthesise one frame's results. ``frame`` supplies only the size."""
        h, w = frame.shape[:2]
        drivable, lane = self._render_masks(h, w)

        boxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        x_m, y_m = self._vehicle_ground_position(self._t)
        box = self._project_ground_box(x_m, y_m)
        if box is not None:
            score = float(np.clip(0.85 + 0.1 * np.sin(2.1 * self._t), 0.0, 1.0))
            if score >= self.cfg.score_threshold:
                boxes.append(box)
                scores.append(score)

        return FrameResult(
            boxes_px=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            class_ids=np.full(len(boxes), CLASS_IDS["vehicle"], dtype=np.int32),
            scores=np.array(scores, dtype=np.float32),
            drivable_mask=drivable,
            lane_mask=lane,
            # Halve the stored resolution: the stub exercises the same
            # native-mask caching path the real network uses.
            native_masks=NativeMasks(
                drivable=cv2.resize(drivable, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
                lane=cv2.resize(lane, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
                original_size=(w, h),
            ),
        )

    def ground_truth(self) -> dict[str, float]:
        """The scene parameters used for the current frame, for assertions."""
        x_m, y_m = self._vehicle_ground_position(self._t)
        return {
            "stub_vehicle_distance_m": y_m,
            "stub_vehicle_lateral_m": x_m,
            "stub_lane_half_width_m": self.lane_half_width_m,
        }
