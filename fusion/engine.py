"""The rules-based decision engine.

Consumes per-frame perception output -- tracked objects with distances and TTC,
lane geometry, weather -- and emits structured alerts.

Rules rather than a learned policy, for three reasons. Every alert has to be
explainable after the fact, which a network's output is not. Thresholds have to
be tunable without retraining, because the right TTC threshold is a product
decision, not a statistical one. And there is no labelled alert dataset to learn
from in the first place.

Weather feeds back into the thresholds rather than sitting beside them: stopping
distance on wet tarmac is roughly double the dry figure, so a TTC threshold that
is right in the dry fires far too late in the rain.

Every rule here is a pure function of typed records, so the whole engine is
testable with no video, no GPU and no weights.
"""

from __future__ import annotations

from core.config import FusionConfig
from core.types import LaneSet, LaneStatus
from fusion.alerts import Alert, AlertType, FrameDecision, Severity
from fusion.lane_change import LaneChangeDetector, LaneEvent, LaneState
from tracking.collision import ObjectState

#: Classes treated as vulnerable road users. A pedestrian anywhere near the path
#: is escalated harder than a vehicle at the same distance, because the
#: consequences are not symmetric and pedestrians change direction abruptly.
VULNERABLE = frozenset({"person", "bicycle", "motorcycle"})


class DecisionEngine:
    """Turns perception output into alerts, one frame at a time."""

    def __init__(self, cfg: FusionConfig) -> None:
        self.cfg = cfg
        self.lane_detector = LaneChangeDetector(cfg.lane_change)
        self._weather = "unknown"

    def set_weather(self, weather: str) -> None:
        """Set the current weather label, which scales the TTC thresholds."""
        self._weather = weather or "unknown"

    # ------------------------------------------------------------------ #

    def ttc_thresholds(self) -> tuple[float, float]:
        """``(warning, critical)`` TTC thresholds for the current weather.

        Scaled up in conditions that lengthen stopping distance, so the warning
        arrives earlier rather than at the same too-late moment.
        """
        factor = self.cfg.weather_ttc_factor.get(self._weather, 1.0)
        return (
            self.cfg.ttc_warning_s * factor,
            self.cfg.ttc_critical_s * factor,
        )

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        objects: list[ObjectState],
        lanes: LaneSet,
    ) -> FrameDecision:
        """Evaluate every rule against one frame."""
        alerts: list[Alert] = []
        warn_s, crit_s = self.ttc_thresholds()

        lane_state = self.lane_detector.update(lanes)
        alerts.extend(self._lane_alerts(frame_index, timestamp_s, lane_state, lanes))
        alerts.extend(
            self._collision_alerts(frame_index, timestamp_s, objects, warn_s, crit_s)
        )

        # Most urgent first, so an overlay that shows only one shows the worst.
        alerts.sort(key=lambda a: -a.severity.rank)
        return FrameDecision(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            alerts=tuple(alerts),
            weather=self._weather,
            ttc_threshold_s=warn_s,
            n_objects=len(objects),
            n_in_path=sum(1 for o in objects if o.in_path),
        )

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #

    def _lane_alerts(
        self, frame_index: int, timestamp_s: float, state: LaneState, lanes: LaneSet
    ) -> list[Alert]:
        out: list[Alert] = []
        ev = state.event
        if ev in (LaneEvent.DEPARTURE_LEFT, LaneEvent.DEPARTURE_RIGHT):
            side = "left" if ev is LaneEvent.DEPARTURE_LEFT else "right"
            out.append(Alert(
                type=AlertType.LANE_DEPARTURE,
                severity=Severity.WARNING,
                frame_index=frame_index, timestamp_s=timestamp_s,
                message=f"Lane departure {side}",
                evidence={
                    "normalised_offset": state.normalised_offset,
                    "offset_m": state.offset_m,
                    "lane_width_m": state.lane_width_m,
                    "frames_departed": state.frames_departed,
                },
            ))
        elif ev in (LaneEvent.CHANGE_LEFT, LaneEvent.CHANGE_RIGHT):
            side = "left" if ev is LaneEvent.CHANGE_LEFT else "right"
            out.append(Alert(
                type=AlertType.LANE_CHANGE,
                severity=Severity.INFO,
                frame_index=frame_index, timestamp_s=timestamp_s,
                message=f"Lane change {side}",
                evidence={"normalised_offset": state.normalised_offset,
                          "lane_width_m": state.lane_width_m},
            ))

        if self.cfg.warn_on_lane_loss and lanes.status is LaneStatus.NOT_DETECTED:
            out.append(Alert(
                type=AlertType.LANE_LOST,
                severity=Severity.INFO,
                frame_index=frame_index, timestamp_s=timestamp_s,
                message="No lane detected",
                evidence={"status": lanes.status.value},
            ))
        return out

    def _collision_alerts(
        self, frame_index: int, timestamp_s: float,
        objects: list[ObjectState], warn_s: float, crit_s: float,
    ) -> list[Alert]:
        out: list[Alert] = []
        for obj in objects:
            if not obj.in_path:
                continue
            ttc = obj.ttc.ttc_s
            vulnerable = obj.class_name in VULNERABLE

            if vulnerable and self._pedestrian_is_a_concern(obj, warn_s):
                out.append(Alert(
                    type=AlertType.PEDESTRIAN_IN_PATH,
                    severity=Severity.CRITICAL,
                    frame_index=frame_index, timestamp_s=timestamp_s,
                    message=f"{obj.class_name.capitalize()} in path",
                    track_id=obj.track_id,
                    evidence=self._evidence(obj),
                ))
                continue

            if ttc <= crit_s:
                severity = Severity.CRITICAL
            elif ttc <= warn_s:
                severity = Severity.WARNING
            else:
                continue
            out.append(Alert(
                type=AlertType.FORWARD_COLLISION,
                severity=severity,
                frame_index=frame_index, timestamp_s=timestamp_s,
                message=f"Collision risk: {ttc:.1f}s to {obj.class_name}",
                track_id=obj.track_id,
                evidence=self._evidence(obj),
            ))
        return out

    def _pedestrian_is_a_concern(self, obj: ObjectState, warn_s: float) -> bool:
        """Whether a vulnerable road user in the corridor warrants an alert.

        Deliberately not TTC-only. A pedestrian standing still in the path has an
        infinite TTC -- nothing is closing -- yet is exactly the situation that
        must raise a warning. Proximity alone is therefore sufficient.
        """
        if obj.ttc.ttc_s <= warn_s:
            return True
        d = obj.distance_m
        if d is not None and d <= self.cfg.pedestrian_distance_m:
            return True
        # No calibration: fall back to TTC alone rather than inventing a
        # distance, and say so by not firing on proximity.
        return False

    @staticmethod
    def _evidence(obj: ObjectState) -> dict:
        return {
            "class": obj.class_name,
            "ttc_s": obj.ttc.ttc_s if obj.ttc.ttc_s != float("inf") else None,
            "ttc_source": obj.ttc.source,
            "distance_m": obj.distance_m,
            "lateral_offset_m": obj.lateral_offset_m,
            "range_rate_mps": obj.ttc.range_rate_mps,
            "fit_quality": obj.ttc.fit_quality,
            "score": obj.score,
        }

    def reset(self) -> None:
        self.lane_detector.reset()
        self._weather = "unknown"
