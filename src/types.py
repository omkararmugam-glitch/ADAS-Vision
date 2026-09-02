"""Shared result types passed between pipeline stages and through the cache.

Coordinate frames:
  * ``*_px`` fields are **image pixel** coordinates in the canonical frame size
    (origin top-left, +x right, +y down).
  * ``*_m`` fields are **ego-vehicle metres** (X right, Y forward, origin at the
    camera's ground projection).

Every type round-trips through ``to_dict``/``from_dict`` so that a frame's result
can be cached as JSON and reloaded without re-running any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Detection:
    """One detected object in a single frame.

    Attributes:
        bbox_px: ``(x1, y1, x2, y2)`` in image pixels, x1 < x2 and y1 < y2.
        label: Class name, e.g. ``"car"`` or ``"person"``.
        score: Detector confidence in ``[0, 1]``.
        track_id: Stable identity across frames, or None if untracked.
    """

    bbox_px: tuple[float, float, float, float]
    label: str
    score: float
    track_id: int | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_px
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"degenerate bbox_px {self.bbox_px}: need x1 < x2 and y1 < y2")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    @property
    def bottom_centre_px(self) -> tuple[float, float]:
        """Midpoint of the box's bottom edge: the object's ground contact point.

        This is the only point of a 2-D box that lies on the road plane, so it is
        what gets projected through the homography to obtain a distance.
        """
        x1, _, x2, y2 = self.bbox_px
        return (0.5 * (x1 + x2), y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_px": [float(v) for v in self.bbox_px],
            "label": self.label,
            "score": float(self.score),
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Detection:
        x1, y1, x2, y2 = d["bbox_px"]
        return cls(
            bbox_px=(float(x1), float(y1), float(x2), float(y2)),
            label=d["label"],
            score=float(d["score"]),
            track_id=d["track_id"],
        )


@dataclass(frozen=True)
class LaneLine:
    """A lane boundary as an ordered polyline in image pixels.

    Attributes:
        points_px: ``(N, 2)`` float array of ``(u, v)``, ordered near-to-far.
        role: Which boundary this is, e.g. ``"ego_left"``, ``"ego_right"``.
        kind: Marking type -- ``"solid"``, ``"dashed"`` or ``"unknown"``.
        score: Confidence in ``[0, 1]``.
    """

    points_px: np.ndarray
    role: str = "unknown"
    kind: str = "unknown"
    score: float = 1.0

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_px, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points_px must be (N, 2), got shape {pts.shape}")
        if len(pts) < 2:
            raise ValueError(f"a polyline needs at least 2 points, got {len(pts)}")
        object.__setattr__(self, "points_px", pts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points_px": self.points_px.tolist(),
            "role": self.role,
            "kind": self.kind,
            "score": float(self.score),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LaneLine:
        return cls(
            points_px=np.asarray(d["points_px"], dtype=np.float64),
            role=d["role"],
            kind=d["kind"],
            score=float(d["score"]),
        )


@dataclass
class FrameResult:
    """Everything a per-frame stage produced, as cached to disk.

    Attributes:
        frame_index: Index in the emitted sequence (contiguous from 0).
        source_index: Index in the underlying source, before stride/start.
        timestamp_s: Seconds from the start of the source.
        detections: Objects found in this frame.
        lanes: Lane boundaries found in this frame.
        extras: Stage-specific scalars; must be JSON-serialisable.
    """

    frame_index: int
    source_index: int
    timestamp_s: float
    detections: list[Detection] = field(default_factory=list)
    lanes: list[LaneLine] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "source_index": self.source_index,
            "timestamp_s": self.timestamp_s,
            "detections": [d.to_dict() for d in self.detections],
            "lanes": [line.to_dict() for line in self.lanes],
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameResult:
        return cls(
            frame_index=int(d["frame_index"]),
            source_index=int(d["source_index"]),
            timestamp_s=float(d["timestamp_s"]),
            detections=[Detection.from_dict(x) for x in d["detections"]],
            lanes=[LaneLine.from_dict(x) for x in d["lanes"]],
            extras=d.get("extras", {}),
        )
