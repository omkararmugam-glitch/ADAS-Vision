"""Alert vocabulary and the structured record the engine emits.

Alerts are data, not log lines. Every one carries the frame and timestamp it
fired on, the evidence behind it, and the threshold it crossed -- so a warning
can be audited after the fact, which is the whole reason the decision layer is
rules-based rather than learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertType(str, Enum):
    """Every alert the fusion engine can raise."""

    LANE_DEPARTURE = "lane_departure"
    LANE_CHANGE = "lane_change"
    FORWARD_COLLISION = "forward_collision"
    PEDESTRIAN_IN_PATH = "pedestrian_in_path"
    RED_LIGHT_RUNNING = "red_light_running"
    ROAD_HAZARD = "road_hazard"
    ROAD_SPLIT = "road_split"
    LANE_LOST = "lane_lost"


class Severity(str, Enum):
    """How urgent an alert is.

    Three levels, not five: the engine has to map each one to a distinct driver
    response, and a scale finer than "be aware / act soon / act now" cannot be
    acted on differently.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass(frozen=True)
class Alert:
    """One alert on one frame.

    Attributes:
        type: Which alert.
        severity: Urgency.
        frame_index: Frame it fired on.
        timestamp_s: Time within the clip.
        message: Short human-readable text, also used for the on-frame overlay.
        track_id: Object responsible, when there is one.
        evidence: The numbers behind the decision -- TTC, distance, offset,
            whichever applied. Written to the JSON log so a firing can be
            explained without re-running anything.
    """

    type: AlertType
    severity: Severity
    frame_index: int
    timestamp_s: float
    message: str
    track_id: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 4),
            "message": self.message,
            "track_id": self.track_id,
            "evidence": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.evidence.items()
            },
        }


@dataclass(frozen=True)
class FrameDecision:
    """Everything the engine concluded about one frame."""

    frame_index: int
    timestamp_s: float
    alerts: tuple[Alert, ...] = ()
    weather: str = "unknown"
    ttc_threshold_s: float = 0.0
    n_objects: int = 0
    n_in_path: int = 0

    @property
    def worst(self) -> Alert | None:
        """The most urgent alert on this frame, if any."""
        return max(self.alerts, key=lambda a: a.severity.rank, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 4),
            "weather": self.weather,
            "ttc_threshold_s": round(self.ttc_threshold_s, 3),
            "n_objects": self.n_objects,
            "n_in_path": self.n_in_path,
            "alerts": [a.to_dict() for a in self.alerts],
        }
