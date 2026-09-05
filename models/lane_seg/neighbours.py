"""Enumerate the lanes across the road and say what each one is.

The lane fitter answers "where are the boundaries". This answers the question a
driver actually has: **which of those lanes could I move into right now.**

A boundary on its own does not carry that. Two adjacent boundaries bound a lane,
and that lane is one of five things:

``EGO``       the lane being driven
``FREE``      same direction, nothing in it -- a lane change is available
``OCCUPIED``  same direction, but a vehicle is in it
``ONCOMING``  beyond the divider of a divided road; never an option
``BRANCH``    the exit lane of a fork; an option, but a different journey

The distinction that matters most is FREE versus ONCOMING, because geometrically
they look identical -- an empty strip of road alongside the ego lane. Only
:mod:`models.lane_seg.carriageway` can tell them apart, so this consumes its
answer rather than guessing. Getting it wrong would invite a lane change into
head-on traffic, which is the worst output this system could produce.

Occupancy is decided at the **ground contact point** -- the bottom centre of the
detection box -- not the box centre. A box centre floats at bumper height and
back-projects to a point well beyond the vehicle, which on a three-lane road
puts a lorry in the wrong lane. The bottom edge is the only part of a 2-D box
that is genuinely on the road plane.

Occupancy is checked **further out than the boundaries were fitted**, over the
whole stretch that gets painted. Checking only the fitted range meant declaring
a lane free over road that had never been examined, which showed up immediately
on real frames as an orange "lane change available" laid across three visible
cars. The boundaries are extrapolated to run the test, which is acceptable in
this direction only: a wrongly-included vehicle downgrades a lane to occupied,
and being conservative about an invitation to change lanes costs nothing.

What this deliberately does not do is decide whether a lane change is *safe*.
Free means unoccupied ahead. There is nothing behind: a vehicle closing fast
from the rear is invisible to a forward camera, and no amount of geometry here
can supply it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.calibrate import CameraModel
from core.types import LaneFit, LaneSet, LaneStatus


class LaneKind(str, Enum):
    """What a lane is, from the ego vehicle's point of view."""

    EGO = "ego"
    FREE = "free"
    OCCUPIED = "occupied"
    ONCOMING = "oncoming"
    BRANCH = "branch"

    @property
    def is_change_option(self) -> bool:
        """Whether moving into it is a lane change this system would offer."""
        return self is LaneKind.FREE


@dataclass(frozen=True)
class Lane:
    """One lane of the road, bounded by two fitted boundaries.

    Attributes:
        left, right: The bounding fits, ordered left to right in ego metres.
        kind: What the lane is.
        side: ``"ego"``, or which side of the ego vehicle it lies on.
        width_m: Separation of the bounds at the evaluation distance.
        y_range_m: Forward range over which both bounds are supported.
        occupants: Indices into the object list of whatever is in the lane.
    """

    left: LaneFit
    right: LaneFit
    kind: LaneKind
    side: str
    width_m: float
    y_range_m: tuple[float, float]
    occupants: tuple[int, ...] = ()

    def centre_x_at(self, y_m: float) -> float:
        return 0.5 * (float(self.left.x_at(y_m)[0]) + float(self.right.x_at(y_m)[0]))

    def contains(self, x_m: float, y_m: float, far_m: float | None = None,
                 near_m: float | None = None) -> bool:
        """Whether a ground point lies in this lane.

        ``near_m`` and ``far_m`` override the fitted ends. Occupancy is checked
        over a wider range than the boundaries were *fitted*, and deliberately
        so: the lane is painted for the whole length of road it covers, and
        claiming a lane is free over road that was never examined is the one
        error here with a cost attached.

        The near end matters as much as the far one and is easier to overlook. A
        fit that begins at 5.6 m ignored a car sitting at 4.3 m -- directly
        alongside, the single most important vehicle in the frame for a lane
        change -- while the lane was painted right down to the bonnet.

        The boundaries are extrapolated to run the test, acceptable in this
        direction only: a wrongly-included vehicle downgrades the lane to
        occupied, which is merely conservative.
        """
        lo, hi = self.y_range_m
        if far_m is not None:
            hi = max(hi, far_m)
        if near_m is not None:
            lo = min(lo, near_m)
        if not lo <= y_m <= hi:
            return False
        return float(self.left.x_at(y_m)[0]) < x_m < float(self.right.x_at(y_m)[0])


def _ground_contact(box: np.ndarray, camera: CameraModel) -> tuple[float, float] | None:
    """Back-project the bottom-centre of a detection box onto the road."""
    x1, y1, x2, y2 = (float(v) for v in np.asarray(box).reshape(4))
    pt = np.array([[0.5 * (x1 + x2), y2]])
    if not bool(np.asarray(camera.is_on_ground(pt)).reshape(-1)[0]):
        return None
    g = np.asarray(camera.image_to_ground(pt)).reshape(-1)
    if not np.all(np.isfinite(g)):
        return None
    return float(g[0]), float(g[1])


def enumerate_lanes(
    lanes: LaneSet,
    camera: CameraModel | None = None,
    objects=None,
    *,
    split=None,
    carriageway=None,
    min_width_m: float = 2.2,
    max_width_m: float = 5.5,
    max_width_variation: float = 1.6,
    occupancy_far_m: float = 80.0,
    occupancy_near_m: float = 0.0,
    road_user_classes: frozenset[str] | None = None,
) -> list[Lane]:
    """Build the lanes of the road, left to right, and classify each.

    Args:
        lanes: Fitted boundaries for one frame.
        camera: Needed only to place detected objects on the road.
        objects: Sequence with a ``box_px`` attribute (``ObjectState`` or any
            stand-in). Without it every same-direction lane reads as FREE, which
            is why the caller should pass detections whenever it has them.
        split: A detected fork, so its branch is not offered as a lane change.
        carriageway: A detected divided road, so the far side is not either.
        min_width_m, max_width_m: Plausibility bounds. A gap outside these is
            not a lane -- it is a gore, a median, or a mis-pairing.
        max_width_variation: Far width over near width, above which the gap is
            opening rather than running alongside. A real lane holds its width.
        occupancy_far_m, occupancy_near_m: The range searched for occupants,
            independent of how far the boundaries were fitted. See
            :meth:`Lane.contains`.
        road_user_classes: Which classes can occupy a lane. Defaults to
            :data:`core.types.ROAD_USER_CLASSES`. A traffic light's box has a
            ground contact point too, and it is meaningless -- signals hang over
            the road rather than sitting on it, so back-projecting one places a
            phantom vehicle in whichever lane the gantry happens to cross.

    Returns:
        Lanes ordered left to right. Empty when nothing could be established.
    """
    if lanes.status is LaneStatus.NOT_DETECTED:
        return []
    fits = lanes.all_fits()
    if len(fits) < 2:
        return []

    y_eval = lanes.eval_distance_m or 15.0
    ordered = sorted(fits, key=lambda f: float(f.x_at(y_eval)[0]))

    ego_roles = {f.role for f in (lanes.ego_left, lanes.ego_right) if f is not None}
    divider_role = getattr(carriageway, "divider_role", None)
    divider_side = getattr(carriageway, "divider_side", None)
    divider_x = getattr(carriageway, "divider_x_m", None)

    split_side = getattr(getattr(split, "side", None), "value", None)
    branch_roles = ({split.left_role, split.right_role} if split is not None
                    else set())

    # Where each object sits on the road, computed once.
    if road_user_classes is None:
        from core.types import ROAD_USER_CLASSES

        road_user_classes = frozenset(ROAD_USER_CLASSES)
    contacts: list[tuple[int, float, float]] = []
    if objects and camera is not None:
        for i, obj in enumerate(objects):
            name = getattr(obj, "class_name", None)
            if name is not None and name not in road_user_classes:
                continue
            box = getattr(obj, "box_px", None)
            g = _ground_contact(box, camera) if box is not None else None
            if g is not None:
                contacts.append((i, g[0], g[1]))

    out: list[Lane] = []
    for a, b in zip(ordered, ordered[1:]):
        lo = max(a.y_range_m[0], b.y_range_m[0])
        hi = min(a.y_range_m[1], b.y_range_m[1])
        if hi - lo < 4.0:
            continue
        near = abs(float(b.x_at(lo)[0]) - float(a.x_at(lo)[0]))
        far = abs(float(b.x_at(hi)[0]) - float(a.x_at(hi)[0]))
        width = abs(float(b.x_at(y_eval)[0]) - float(a.x_at(y_eval)[0]))
        if not min_width_m <= width <= max_width_m:
            continue
        if max(near, far) / max(min(near, far), 1e-6) > max_width_variation:
            continue

        centre = 0.5 * (float(a.x_at(y_eval)[0]) + float(b.x_at(y_eval)[0]))
        is_ego = {a.role, b.role} == ego_roles and len(ego_roles) == 2
        side = "ego" if is_ego else ("left" if centre < 0 else "right")

        # Classification, most-restrictive first: a lane that is oncoming is
        # never also a change option, whatever else is true of it.
        if is_ego:
            kind = LaneKind.EGO
        elif (divider_x is not None and divider_side is not None
              and ((divider_side == "left" and centre < divider_x)
                   or (divider_side == "right" and centre > divider_x))):
            kind = LaneKind.ONCOMING
        elif divider_role is not None and divider_role in (a.role, b.role) and (
                (divider_side == "left" and centre < 0)
                or (divider_side == "right" and centre > 0)):
            kind = LaneKind.ONCOMING
        elif branch_roles and (a.role in branch_roles or b.role in branch_roles) \
                and split_side == side:
            kind = LaneKind.BRANCH
        else:
            kind = LaneKind.FREE

        lane = Lane(left=a, right=b, kind=kind, side=side, width_m=width,
                    y_range_m=(lo, hi))
        occupants = tuple(i for i, gx, gy in contacts
                          if lane.contains(gx, gy, occupancy_far_m,
                                           occupancy_near_m))
        if occupants and kind is LaneKind.FREE:
            kind = LaneKind.OCCUPIED
        out.append(Lane(left=a, right=b, kind=kind, side=side, width_m=width,
                        y_range_m=(lo, hi), occupants=occupants))
    return out


def change_options(lanes_list: list[Lane]) -> list[Lane]:
    """The lanes a change could be made into, nearest side first."""
    return [ln for ln in lanes_list if ln.kind.is_change_option]
