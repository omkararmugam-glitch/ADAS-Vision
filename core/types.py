"""Shared result types passed between pipeline stages and through the cache.

Coordinate frames:
  * ``*_px`` fields are **image pixel** coordinates at the canonical frame size
    (origin top-left, +x right, +y down).
  * ``*_m`` fields are **ego-vehicle metres** (X right, Y forward, origin at the
    camera's ground projection).
  * Lane polynomials are ``x = a*y^2 + b*y + c`` in **bird's-eye metres**, i.e.
    lateral position as a function of forward distance. Fitting lateral-on-
    forward rather than the reverse is what makes a near-vertical lane boundary
    representable at all.

Everything here round-trips through ``to_dict``/``from_dict`` so a frame can be
cached and reloaded without re-running any model. Masks are the exception: they
are bulk arrays and are persisted alongside the JSON by :mod:`core.cache`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

#: Bumped whenever the on-disk serialisation changes. A cache written under a
#: different value is stale and is never reused.
SCHEMA_VERSION = 2

#: Unified class vocabulary across every detector in the pipeline. YOLOPv2
#: supplies vehicles only; vulnerable road users come from a second COCO model.
#: The vocabulary is recorded in the cache manifest so that a cached run is never
#: ambiguous about which classes it was *able* to see.
CLASS_NAMES: tuple[str, ...] = ("vehicle", "person", "bicycle", "motorcycle")
CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def class_name(class_id: int) -> str:
    """Name for a class id, or ``"class:<id>"`` if it is outside the vocabulary."""
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class:{class_id}"


@dataclass(frozen=True)
class NativeMasks:
    """Segmentation masks at the resolution the network actually produced them.

    The network predicts masks at its letterboxed input size; upsampling them to
    the frame size adds interpolated pixels but no information. Keeping the
    native arrays lets the cache persist what was predicted rather than what was
    interpolated -- roughly a 4x saving at 720p -- while
    :class:`FrameResult` still exposes full-resolution masks to callers.

    Attributes:
        drivable: ``(h, w)`` uint8 in {0, 1}, letterbox padding already removed.
        lane: ``(h, w)`` uint8 in {0, 1}, same geometry as ``drivable``.
        original_size: ``(width, height)`` these masks upsample back to.
    """

    drivable: np.ndarray
    lane: np.ndarray
    original_size: tuple[int, int]


@dataclass(frozen=True)
class FrameResult:
    """What a :class:`~models.perception.Detector` produced for one frame.

    Attributes:
        boxes_px: ``(N, 4)`` float32 ``(x1, y1, x2, y2)`` in **original image
            pixels** -- the letterbox transform has already been inverted.
        class_ids: ``(N,)`` int32 indices into :data:`CLASS_NAMES`.
        scores: ``(N,)`` float32 confidences in ``[0, 1]``.
        drivable_mask: ``(H, W)`` uint8 in {0, 1} at the original frame size.
        lane_mask: ``(H, W)`` uint8 in {0, 1} at the original frame size.
        native_masks: The same masks before upsampling, for cheap caching.
            None for detectors that produce no segmentation.
    """

    boxes_px: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray
    drivable_mask: np.ndarray
    lane_mask: np.ndarray
    native_masks: NativeMasks | None = None

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes_px, dtype=np.float32).reshape(-1, 4)
        ids = np.asarray(self.class_ids, dtype=np.int32).reshape(-1)
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if not (len(boxes) == len(ids) == len(scores)):
            raise ValueError(
                f"boxes/class_ids/scores must be the same length, got "
                f"{len(boxes)}/{len(ids)}/{len(scores)}"
            )
        if len(boxes) and not np.all((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])):
            bad = int(np.sum(~((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))))
            raise ValueError(f"{bad} degenerate boxes: need x1 < x2 and y1 < y2")
        if self.drivable_mask.shape != self.lane_mask.shape:
            raise ValueError(
                f"drivable and lane masks must share a shape, got "
                f"{self.drivable_mask.shape} and {self.lane_mask.shape}"
            )
        object.__setattr__(self, "boxes_px", boxes)
        object.__setattr__(self, "class_ids", ids)
        object.__setattr__(self, "scores", scores)

    def __len__(self) -> int:
        return len(self.boxes_px)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the masks, i.e. of the original frame."""
        h, w = self.drivable_mask.shape[:2]
        return w, h

    def bottom_centres_px(self) -> np.ndarray:
        """``(N, 2)`` ground-contact points: the midpoint of each box's bottom edge.

        The only part of a 2-D box that lies on the road plane, and therefore the
        only point that may be projected through the homography for a distance.
        """
        if not len(self):
            return np.zeros((0, 2), dtype=np.float64)
        return np.column_stack(
            [0.5 * (self.boxes_px[:, 0] + self.boxes_px[:, 2]), self.boxes_px[:, 3]]
        ).astype(np.float64)

    def filter(self, keep: np.ndarray) -> FrameResult:
        """Return a copy keeping only the detections selected by a boolean mask."""
        return FrameResult(
            boxes_px=self.boxes_px[keep],
            class_ids=self.class_ids[keep],
            scores=self.scores[keep],
            drivable_mask=self.drivable_mask,
            lane_mask=self.lane_mask,
            native_masks=self.native_masks,
        )

    def to_dict(self) -> dict[str, Any]:
        """Detections only; masks are persisted separately by the cache."""
        return {
            "boxes_px": self.boxes_px.tolist(),
            "class_ids": self.class_ids.tolist(),
            "scores": [round(float(s), 5) for s in self.scores],
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], drivable_mask: np.ndarray, lane_mask: np.ndarray
    ) -> FrameResult:
        return cls(
            boxes_px=np.asarray(d["boxes_px"], dtype=np.float32).reshape(-1, 4),
            class_ids=np.asarray(d["class_ids"], dtype=np.int32),
            scores=np.asarray(d["scores"], dtype=np.float32),
            drivable_mask=drivable_mask,
            lane_mask=lane_mask,
        )

    @classmethod
    def empty(cls, size: tuple[int, int]) -> FrameResult:
        """A result with no detections and empty masks, for a frame that failed."""
        w, h = size
        blank = np.zeros((h, w), dtype=np.uint8)
        return cls(
            boxes_px=np.zeros((0, 4), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int32),
            scores=np.zeros((0,), dtype=np.float32),
            drivable_mask=blank,
            lane_mask=blank.copy(),
        )


# --------------------------------------------------------------------------- #
# Lanes
# --------------------------------------------------------------------------- #


class LaneStatus(str, Enum):
    """Outcome of lane fitting for one frame.

    A fit is never faked. When the evidence does not support a lane model, the
    status says so and the coefficients are absent, rather than a garbage
    polynomial being returned that downstream code would treat as real.
    """

    OK = "ok"
    """Both ego boundaries fitted."""

    SINGLE_BOUNDARY = "single_boundary"
    """Exactly one ego boundary fitted; width and offset are unavailable."""

    MASK_TOO_SPARSE = "mask_too_sparse"
    """Lane pixels present but too few to support a fit."""

    NOT_DETECTED = "not_detected"
    """No lane pixels survived warping into the bird's-eye window."""


@dataclass(frozen=True)
class LaneFit:
    """One fitted lane boundary, in bird's-eye metres.

    The polynomial is ``x = a*y^2 + b*y + c`` where ``y`` is forward distance and
    ``x`` is lateral offset, both in ego metres.

    Attributes:
        coeffs: ``(3,)`` array ``[a, b, c]``.
        role: ``"ego_left"``, ``"ego_right"``, ``"adjacent_left"``, ...
        n_pixels: Lane pixels that supported the fit.
        y_range_m: ``(near, far)`` extent actually covered by those pixels.
            Evaluating outside this range is extrapolation.
    """

    coeffs: np.ndarray
    role: str
    n_pixels: int
    y_range_m: tuple[float, float]

    def __post_init__(self) -> None:
        c = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        if c.shape != (3,):
            raise ValueError(f"expected 3 polynomial coefficients, got {c.shape}")
        object.__setattr__(self, "coeffs", c)

    def x_at(self, y_m: float | np.ndarray) -> np.ndarray:
        """Lateral position in metres at forward distance(s) ``y_m``."""
        y = np.atleast_1d(np.asarray(y_m, dtype=np.float64))
        a, b, c = self.coeffs
        return a * y**2 + b * y + c

    def heading_at(self, y_m: float) -> float:
        """``dx/dy`` at ``y_m``: the boundary's slope, dimensionless."""
        a, b, _ = self.coeffs
        return float(2.0 * a * y_m + b)

    def curvature_radius_m(self, y_m: float) -> float:
        """Radius of curvature in metres at ``y_m``.

        Returns ``inf`` for a straight boundary, where the second derivative
        vanishes and the radius is genuinely unbounded.
        """
        a, b, _ = self.coeffs
        if abs(a) < 1e-9:
            return float("inf")
        slope = 2.0 * a * y_m + b
        return float((1.0 + slope**2) ** 1.5 / abs(2.0 * a))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coeffs": self.coeffs.tolist(),
            "role": self.role,
            "n_pixels": int(self.n_pixels),
            "y_range_m": [float(self.y_range_m[0]), float(self.y_range_m[1])],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LaneFit:
        return cls(
            coeffs=np.asarray(d["coeffs"], dtype=np.float64),
            role=d["role"],
            n_pixels=int(d["n_pixels"]),
            y_range_m=(float(d["y_range_m"][0]), float(d["y_range_m"][1])),
        )


@dataclass(frozen=True)
class LaneSet:
    """All lane geometry fitted for one frame, in bird's-eye metres.

    Attributes:
        status: Why the fields below are or are not populated.
        ego_left: Left boundary of the ego lane, or None.
        ego_right: Right boundary of the ego lane, or None.
        adjacent: Further boundaries found outside the ego lane.
        curvature_radius_m: Radius of the ego lane centreline, evaluated at
            ``eval_distance_m``. None unless both boundaries were fitted;
            ``inf`` for a straight road.
        ego_offset_m: Signed lateral offset of the camera from the lane centre,
            positive when the vehicle sits right of centre. None unless both
            boundaries were fitted.
        lane_width_m: Measured separation of the two ego boundaries at
            ``eval_distance_m``. A sanity signal: a plausible road lane is
            roughly 2.5-4.5 m, and a value outside that suggests the fit latched
            onto the wrong markings.
        eval_distance_m: Forward distance the three scalars above were
            evaluated at.
    """

    status: LaneStatus
    ego_left: LaneFit | None = None
    ego_right: LaneFit | None = None
    adjacent: tuple[LaneFit, ...] = ()
    curvature_radius_m: float | None = None
    ego_offset_m: float | None = None
    lane_width_m: float | None = None
    eval_distance_m: float = 0.0

    @property
    def detected(self) -> bool:
        """True when at least one ego boundary was fitted."""
        return self.ego_left is not None or self.ego_right is not None

    def all_fits(self) -> list[LaneFit]:
        """Every fitted boundary, ego and adjacent."""
        out = [f for f in (self.ego_left, self.ego_right) if f is not None]
        out.extend(self.adjacent)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ego_left": None if self.ego_left is None else self.ego_left.to_dict(),
            "ego_right": None if self.ego_right is None else self.ego_right.to_dict(),
            "adjacent": [f.to_dict() for f in self.adjacent],
            "curvature_radius_m": _json_float(self.curvature_radius_m),
            "ego_offset_m": _json_float(self.ego_offset_m),
            "lane_width_m": _json_float(self.lane_width_m),
            "eval_distance_m": float(self.eval_distance_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LaneSet:
        return cls(
            status=LaneStatus(d["status"]),
            ego_left=None if d["ego_left"] is None else LaneFit.from_dict(d["ego_left"]),
            ego_right=None if d["ego_right"] is None else LaneFit.from_dict(d["ego_right"]),
            adjacent=tuple(LaneFit.from_dict(x) for x in d["adjacent"]),
            curvature_radius_m=_from_json_float(d["curvature_radius_m"]),
            ego_offset_m=_from_json_float(d["ego_offset_m"]),
            lane_width_m=_from_json_float(d["lane_width_m"]),
            eval_distance_m=float(d["eval_distance_m"]),
        )


def _json_float(v: float | None) -> float | str | None:
    """JSON has no infinity; a straight road's radius must survive the round trip."""
    if v is None:
        return None
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return float(v)


def _from_json_float(v: float | str | None) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        return float("inf") if v == "inf" else float("-inf")
    return float(v)


# --------------------------------------------------------------------------- #
# Cache record
# --------------------------------------------------------------------------- #


@dataclass
class FrameRecord:
    """One frame's complete cached state: metadata, detections and lane geometry.

    Masks are not held here; the cache stores them beside the JSON and rebuilds
    :class:`FrameResult` on read.

    Attributes:
        frame_index: Index in the emitted sequence, contiguous from 0.
        source_index: Index in the underlying source, before stride/start.
        timestamp_s: Seconds from the start of the source.
        result: Detections for this frame.
        lanes: Fitted lane geometry for this frame.
        extras: Stage-specific JSON-serialisable scalars.
    """

    frame_index: int
    source_index: int
    timestamp_s: float
    result: FrameResult
    lanes: LaneSet
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_index": self.source_index,
            "timestamp_s": self.timestamp_s,
            "result": self.result.to_dict(),
            "lanes": self.lanes.to_dict(),
            "extras": self.extras,
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], drivable_mask: np.ndarray, lane_mask: np.ndarray
    ) -> FrameRecord:
        return cls(
            frame_index=int(d["frame_index"]),
            source_index=int(d["source_index"]),
            timestamp_s=float(d["timestamp_s"]),
            result=FrameResult.from_dict(d["result"], drivable_mask, lane_mask),
            lanes=LaneSet.from_dict(d["lanes"]),
            extras=d.get("extras", {}),
        )
