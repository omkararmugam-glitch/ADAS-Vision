"""Rules-based decision engine over all module outputs.

Consumes per-frame perception -- tracked objects with distance and TTC, lane
geometry, weather -- and emits structured, timestamped alerts: lane departure,
lane change, forward collision, pedestrian in path, and road hazards.

Rules rather than a learned policy, because every alert has to be explainable
after the fact and tunable without retraining, and because there is no labelled
alert dataset to learn from. The rules are pure functions over typed records,
which is what makes them unit-testable with no video and no GPU.
"""

from fusion.alerts import Alert, AlertType, FrameDecision, Severity
from fusion.engine import DecisionEngine
from fusion.lane_change import LaneChangeDetector, LaneEvent, LaneState
from fusion.log import AlertLog, Episode, to_episodes

__all__ = [
    "DecisionEngine",
    "Alert",
    "AlertType",
    "Severity",
    "FrameDecision",
    "LaneChangeDetector",
    "LaneEvent",
    "LaneState",
    "AlertLog",
    "Episode",
    "to_episodes",
]
