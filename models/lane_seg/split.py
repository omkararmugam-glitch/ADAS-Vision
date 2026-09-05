"""Detect a road split: a fork, an exit ramp, or a carriageway dividing.

The lane fitter assumes one continuous road with roughly parallel boundaries.
Where the road actually divides, that assumption fails quietly and in the worst
possible way: the gore area between the two carriageways reads as one very wide
lane, ``lane_width_m`` grows past anything plausible, and the ego-offset the
lane-departure rule depends on becomes meaningless. Nothing raises. The system
simply believes it is centred in a nine-metre lane.

Detection works on the **fitted boundaries**, not the mask. Two boundaries that
belong to one road stay roughly parallel: their separation at 30 m is close to
their separation at 10 m. Two boundaries either side of a fork do not -- the gap
opens with distance, which is exactly what a driver sees as the road dividing.

So the test is a ratio, not an absolute width. A ratio is invariant to how wide
the lanes happen to be, which matters because a motorway lane and an urban lane
differ by more than a metre and a fixed threshold would fire on one and miss the
other.

Two things this deliberately does not do:

* It does not claim a split from a single frame's evidence alone. The caller
  debounces over consecutive frames (see :class:`SplitTracker`), because one
  bad fit produces exactly the same signature as a real fork.
* It does not try to decide which branch the vehicle will take. That needs
  indicator state or a route, neither of which this system has, and guessing
  would produce a confident wrong answer where "the road divides ahead" is
  both true and useful.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.types import LaneFit, LaneSet, LaneStatus


class SplitSide(str, Enum):
    """Which way the diverging branch departs."""

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RoadSplit:
    """A detected divergence between two lane boundaries.

    Attributes:
        side: Which side the branch peels off toward.
        near_separation_m: Gap between the two boundaries at the near sample.
        far_separation_m: Gap at the far sample.
        divergence_ratio: ``far / near``. The quantity actually tested.
        gore_distance_m: Forward distance at which the gap starts opening -- the
            nose of the gore area, approximately.
        left_role, right_role: Which fitted boundaries diverged, for auditing.
    """

    side: SplitSide
    near_separation_m: float
    far_separation_m: float
    divergence_ratio: float
    gore_distance_m: float
    left_role: str
    right_role: str

    @property
    def widening_m(self) -> float:
        return self.far_separation_m - self.near_separation_m


def _separation(a: LaneFit, b: LaneFit, y_m: float) -> float:
    return float(abs(float(b.x_at(y_m)[0]) - float(a.x_at(y_m)[0])))


def _shared_range(a: LaneFit, b: LaneFit) -> tuple[float, float] | None:
    """Forward range both fits actually cover.

    Evaluating a quadratic outside the range its pixels supported is
    extrapolation, and two extrapolated curves will diverge for numerical
    reasons that have nothing to do with the road.
    """
    lo = max(a.y_range_m[0], b.y_range_m[0])
    hi = min(a.y_range_m[1], b.y_range_m[1])
    return (lo, hi) if hi - lo >= 5.0 else None


def detect_split(
    lanes: LaneSet,
    min_divergence_ratio: float = 1.6,
    min_widening_m: float = 1.5,
    max_plausible_near_m: float = 6.0,
) -> RoadSplit | None:
    """Find the most strongly diverging pair of boundaries, if any.

    Args:
        lanes: Fitted lane geometry for one frame.
        min_divergence_ratio: Far separation over near separation, above which a
            pair counts as diverging. 1.6 is well past what lane-width variation
            or fit noise produces on a straight road.
        min_widening_m: Absolute gap growth required as well. The ratio alone
            fires on two boundaries that start very close together, where a
            small absolute change is a large proportional one.
        max_plausible_near_m: Pairs already further apart than this at the near
            sample are different carriageways, not a fork opening.

    Returns:
        The strongest divergence, or None.
    """
    if lanes.status is LaneStatus.NOT_DETECTED:
        return None
    fits = lanes.all_fits()
    if len(fits) < 2:
        return None

    # Order left to right at a common near distance so "side" is meaningful.
    ordered = sorted(fits, key=lambda f: float(f.x_at(max(f.y_range_m[0], 8.0))[0]))

    best: RoadSplit | None = None
    for a, b in zip(ordered, ordered[1:]):
        rng = _shared_range(a, b)
        if rng is None:
            continue
        near_y, far_y = rng[0], rng[1]
        near = _separation(a, b, near_y)
        far = _separation(a, b, far_y)
        if near < 0.5 or near > max_plausible_near_m:
            continue
        ratio = far / near
        if ratio < min_divergence_ratio or (far - near) < min_widening_m:
            continue

        # Walk outward to find where the gap starts opening: the first sample
        # whose separation exceeds the near value by a quarter of the total
        # widening. That is the gore nose, near enough for a warning.
        gore = far_y
        for y in np.linspace(near_y, far_y, 24):
            if _separation(a, b, float(y)) - near > 0.25 * (far - near):
                gore = float(y)
                break

        # Whichever boundary moved further from the centre is the branch.
        centre_near = 0.5 * (float(a.x_at(near_y)[0]) + float(b.x_at(near_y)[0]))
        drift_a = abs(float(a.x_at(far_y)[0]) - centre_near)
        drift_b = abs(float(b.x_at(far_y)[0]) - centre_near)
        if abs(drift_a - drift_b) < 0.3:
            side = SplitSide.UNKNOWN
        elif drift_a > drift_b:
            side = SplitSide.LEFT if float(a.x_at(far_y)[0]) < centre_near else SplitSide.RIGHT
        else:
            side = SplitSide.LEFT if float(b.x_at(far_y)[0]) < centre_near else SplitSide.RIGHT

        candidate = RoadSplit(
            side=side, near_separation_m=near, far_separation_m=far,
            divergence_ratio=ratio, gore_distance_m=gore,
            left_role=a.role, right_role=b.role,
        )
        if best is None or candidate.divergence_ratio > best.divergence_ratio:
            best = candidate
    return best


class SplitTracker:
    """Debounces split detections over consecutive frames.

    A single frame's lane fit going wrong -- a kerb picked up as a boundary, a
    shadow edge -- produces exactly the signature of a fork. Requiring the
    divergence to persist is what separates the two, and costs only the few
    frames it takes to confirm, against a road split that stays visible for
    seconds.
    """

    def __init__(self, min_frames: int = 4, window: int = 8, cooldown: int = 30) -> None:
        if min_frames > window:
            raise ValueError(
                f"min_frames ({min_frames}) exceeds window ({window}); the "
                f"condition could never be met"
            )
        self.min_frames = min_frames
        self.window = window
        self.cooldown = cooldown
        self._recent: deque[RoadSplit | None] = deque(maxlen=window)
        self._cooldown_left = 0
        self._active = False

    def update(self, split: RoadSplit | None) -> RoadSplit | None:
        """Feed one frame; returns the split only when newly confirmed."""
        self._recent.append(split)
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return None

        hits = [s for s in self._recent if s is not None]
        if len(hits) < self.min_frames:
            self._active = False
            return None
        if self._active:
            return None

        self._active = True
        self._cooldown_left = self.cooldown

        # Side by MAJORITY across the supporting frames, not from whichever
        # single frame diverged most. Taking the strongest frame let the alert
        # say "right branch" while the HUD, reading the current frame, said
        # "left" -- two answers to the same question on one screen. A majority
        # is also more robust: the clearest frame is not always the correct one,
        # and a fork's side is the one thing that stays constant while its
        # measured ratio does not.
        sides = Counter(h.side for h in hits)
        majority = sides.most_common(1)[0][0]
        agreeing = [h for h in hits if h.side is majority] or hits
        return max(agreeing, key=lambda s: s.divergence_ratio)

    def reset(self) -> None:
        self._recent.clear()
        self._cooldown_left = 0
        self._active = False
