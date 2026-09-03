"""Lane departure and lane-change detection from tracked lane geometry.

Both events come from one signal: the ego vehicle's lateral offset from the lane
centre, normalised by half the lane width, so that ``u = 0`` is centred and
``|u| = 1`` sits on a boundary. Normalising matters -- 0.9 m off centre is
harmless in a 3.7 m motorway lane and most of the way across a 2.6 m urban one.

The two events are genuinely different and are detected differently:

* **Departure** is a *sustained* state: the vehicle is riding a boundary. It is
  detected by ``|u|`` staying high for several consecutive frames.
* **A lane change** is an *event*, and it announces itself by a sign flip. As the
  vehicle crosses, the lane fitter re-locks onto the new lane, and the offset
  jumps from "far right of the old lane" to "left of the new one". A single
  frame's fit wobble cannot produce that, because the flip must be preceded by a
  sustained departure in the matching direction.

Everything here is a pure function of a geometry history: no video, no model, no
GPU. That is what makes the thresholds testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.config import LaneChangeConfig
from core.types import LaneSet, LaneStatus


class LaneEvent(str, Enum):
    """What the lane state machine reported on a frame."""

    NONE = "none"
    DEPARTURE_LEFT = "departure_left"
    DEPARTURE_RIGHT = "departure_right"
    CHANGE_LEFT = "change_left"
    CHANGE_RIGHT = "change_right"


@dataclass(frozen=True)
class LaneState:
    """Lane-keeping state for one frame.

    Attributes:
        event: What happened on this frame, if anything.
        normalised_offset: Offset from lane centre, where 1.0 is the boundary.
            None when the lane was not fitted well enough to say.
        offset_m: Raw signed offset in metres, positive to the right.
        lane_width_m: Fitted lane width, for sanity.
        departing: True while the vehicle is riding a boundary.
        frames_departed: Consecutive frames spent departing.
        confidence: 0-1, how much of the fit window carried a usable lane fit.
    """

    event: LaneEvent = LaneEvent.NONE
    normalised_offset: float | None = None
    offset_m: float | None = None
    lane_width_m: float | None = None
    departing: bool = False
    frames_departed: int = 0
    confidence: float = 0.0


@dataclass
class LaneChangeDetector:
    """State machine over successive :class:`~core.types.LaneSet` fits."""

    cfg: LaneChangeConfig
    _history: list[float | None] = field(default_factory=list)
    _frames_departed: int = 0
    _departure_direction: int = 0        # -1 left, +1 right, 0 none
    _frames_since_departure: int = 0
    _cooldown: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._frames_departed = 0
        self._departure_direction = 0
        self._frames_since_departure = 0
        self._cooldown = 0

    # ------------------------------------------------------------------ #

    def _normalise(self, lanes: LaneSet) -> float | None:
        """Offset as a fraction of half the lane width, or None if unusable.

        A fit whose width is implausible for a road lane is rejected outright:
        it means the fitter latched onto the wrong markings, and an offset
        computed from it would be confidently wrong.
        """
        if lanes.status is not LaneStatus.OK:
            return None
        if lanes.ego_offset_m is None or lanes.lane_width_m is None:
            return None
        width = lanes.lane_width_m
        if not (self.cfg.min_plausible_width_m <= width <= self.cfg.max_plausible_width_m):
            return None
        half = width / 2.0
        if half <= 1e-6:
            return None
        return lanes.ego_offset_m / half

    def update(self, lanes: LaneSet) -> LaneState:
        """Advance one frame and report any event."""
        u = self._normalise(lanes)
        self._history.append(u)
        if len(self._history) > self.cfg.history_window:
            del self._history[: -self.cfg.history_window]

        usable = [x for x in self._history if x is not None]
        confidence = len(usable) / max(1, len(self._history))

        if self._cooldown > 0:
            self._cooldown -= 1

        if u is None:
            # No usable fit. Hold the departure state briefly rather than
            # clearing it -- lane markings vanish for a few frames all the time
            # (worn paint, a shadow), and forgetting instantly would drop a
            # departure that is still happening.
            self._frames_since_departure += 1
            if self._frames_since_departure > self.cfg.max_gap_frames:
                self._frames_departed = 0
                self._departure_direction = 0
            return LaneState(
                normalised_offset=None,
                offset_m=lanes.ego_offset_m,
                lane_width_m=lanes.lane_width_m,
                departing=self._frames_departed >= self.cfg.min_departure_frames,
                frames_departed=self._frames_departed,
                confidence=confidence,
            )

        self._frames_since_departure = 0
        previous = self._previous_usable()
        event = LaneEvent.NONE

        # --- crossing: a sign flip after a sustained departure -------------
        if (
            previous is not None
            and self._cooldown == 0
            and self._departure_direction != 0
            and self._frames_departed >= self.cfg.min_departure_frames
        ):
            d = self._departure_direction
            flipped = (previous * d >= self.cfg.cross_threshold) and (u * d <= -self.cfg.cross_threshold)
            if flipped:
                event = LaneEvent.CHANGE_RIGHT if d > 0 else LaneEvent.CHANGE_LEFT
                self._cooldown = self.cfg.cooldown_frames
                self._frames_departed = 0
                self._departure_direction = 0
                return LaneState(
                    event=event, normalised_offset=u, offset_m=lanes.ego_offset_m,
                    lane_width_m=lanes.lane_width_m, departing=False,
                    frames_departed=0, confidence=confidence,
                )

        # --- departure: riding a boundary ----------------------------------
        if abs(u) >= self.cfg.departure_threshold:
            direction = 1 if u > 0 else -1
            if direction == self._departure_direction:
                self._frames_departed += 1
            else:
                self._departure_direction = direction
                self._frames_departed = 1
            if (
                self._frames_departed == self.cfg.min_departure_frames
                and self._cooldown == 0
            ):
                event = (LaneEvent.DEPARTURE_RIGHT if direction > 0
                         else LaneEvent.DEPARTURE_LEFT)
        elif abs(u) <= self.cfg.recover_threshold:
            # Back near the centre: the departure is over.
            self._frames_departed = 0
            self._departure_direction = 0

        return LaneState(
            event=event,
            normalised_offset=u,
            offset_m=lanes.ego_offset_m,
            lane_width_m=lanes.lane_width_m,
            departing=self._frames_departed >= self.cfg.min_departure_frames,
            frames_departed=self._frames_departed,
            confidence=confidence,
        )

    def _previous_usable(self) -> float | None:
        """The most recent usable offset before the current one."""
        for x in reversed(self._history[:-1]):
            if x is not None:
                return x
        return None
