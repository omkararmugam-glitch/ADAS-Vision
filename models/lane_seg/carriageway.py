"""Detect a divided highway: two carriageways carrying opposing traffic.

This answers a different question from :mod:`models.lane_seg.split`, and the two
are easy to confuse. ``split`` finds a road *dividing ahead* -- a fork, an exit
ramp -- where the gap between two boundaries opens with distance. This finds a
road *already divided*, where the left-bound and right-bound carriageways run
alongside each other, separated by a painted line or a physical median, with the
gap staying constant. One is an event the driver is approaching; the other is a
standing property of the road.

The distinction matters for behaviour, not taxonomy. Everything beyond the
divider is **oncoming traffic**. A vehicle there is closing at the sum of both
speeds, so a time-to-collision computed from its range rate is right, while the
lane-change logic must never propose crossing to it and the tracker should not
be surprised when it leaves the frame quickly. Treating a divided highway as one
wide road gets all three wrong.

Two independent cues, either sufficient, stronger together:

**Yellow paint.** Under the US marking standard -- which is what BDD100K shows
-- yellow separates opposing directions and white separates same-direction
lanes. So a yellow boundary *is* the direction divider, by definition rather
than by inference. :mod:`models.lane_seg.classical` already thresholds yellow
separately; this consumes that mask.

**A median gap.** Two boundaries five or more metres apart whose separation
stays constant are not one very wide lane; they are the inner edges of two
carriageways with a median between them. The constancy is what separates this
from a fork, and it is also why ``detect_split`` explicitly skips pairs already
that far apart -- it hands them here.

Both cues are regional. The yellow convention is a right-hand-traffic marking
standard and does not hold in the UK, Japan, Australia or much of Europe, where
white separates everything and the divider carries no colour cue at all. A
median gap is universal, but only exists where a physical median does. Hence
``drive_side``: geometry is reported as measured, and only the interpretation of
which side is oncoming depends on the convention.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.calibrate import CameraModel
from core.types import LaneFit, LaneSet, LaneStatus


class DriveSide(str, Enum):
    """Which side of the road traffic drives on, for interpretation only."""

    RIGHT = "right"     # US, BDD100K, most of the world
    LEFT = "left"       # UK, Japan, India, Australia


class DividerKind(str, Enum):
    """What evidence identified the divider."""

    YELLOW_LINE = "yellow_line"
    MEDIAN_GAP = "median_gap"
    YELLOW_AND_MEDIAN = "yellow_and_median"


@dataclass(frozen=True)
class DualCarriageway:
    """A road found to be divided into opposing directions.

    Attributes:
        divider_role: Which fitted boundary is the divider.
        divider_x_m: Its lateral offset from the camera at ``eval_distance_m``,
            signed, positive to the right.
        divider_side: ``"left"`` or ``"right"`` -- which side of the ego vehicle
            the divider sits on. Measured, not assumed.
        kind: The evidence that identified it.
        yellow_fraction: Share of samples along the divider that fell on yellow
            paint, or None when no yellow mask was supplied.
        median_width_m: Width of the physical gap, or None for a painted
            divider with no median.
        ego_lanes: Lanes counted on the ego's carriageway.
        oncoming_lanes: Lanes counted on the far carriageway. A **lower bound**:
            the far carriageway's outer edge is often unmarked, out of frame, or
            hidden behind the median, so one boundary there yields a count of
            one whatever the road really has.
        confidence: 0-1. Both cues agreeing scores highest.
        eval_distance_m: Forward distance the geometry was evaluated at.
    """

    divider_role: str
    divider_x_m: float
    divider_side: str
    kind: DividerKind
    yellow_fraction: float | None
    median_width_m: float | None
    ego_lanes: int
    oncoming_lanes: int
    confidence: float
    eval_distance_m: float

    def oncoming_is_left(self, drive_side: DriveSide = DriveSide.RIGHT) -> bool:
        """Whether opposing traffic is on the ego's left.

        This reads the *measured* side rather than deriving it from
        ``drive_side``. For right-hand traffic the two agree; when they do not,
        the honest answer is where the divider actually is, because a fit that
        put it on the wrong side is exactly the case a caller needs to see
        rather than have smoothed over.
        """
        return self.divider_side == "left"

    def agrees_with_convention(self, drive_side: DriveSide = DriveSide.RIGHT) -> bool:
        """Whether the measured divider side matches the marking convention."""
        expected_left = drive_side is DriveSide.RIGHT
        return (self.divider_side == "left") == expected_left

    @property
    def total_lanes(self) -> int:
        return self.ego_lanes + self.oncoming_lanes


def _yellow_evidence(
    fit: LaneFit,
    camera: CameraModel,
    yellow_mask: np.ndarray,
    n_samples: int = 40,
) -> tuple[float, float]:
    """How much of ``fit`` runs along yellow paint, and over what distance.

    Returns ``(fraction, span_m)``: the share of sampled points that landed on
    yellow, and the forward distance between the nearest and furthest hit.

    The span is what separates paint from a yellow *object*. A taxi, a skip or a
    road sign fills a narrow band of range -- it is a thing at a place -- while a
    painted line runs the length of the fit. Fraction alone cannot tell them
    apart: a taxi directly ahead occupies most of the projected curve and scores
    a fraction of one. Measured on real frames this was the single largest
    source of false dividers.

    The search window at each sample is sized from the projection itself -- the
    pixel distance spanned by 0.35 m of ground at that range -- rather than
    being a constant. A marking 30 m away is a couple of pixels wide and a fixed
    window would swamp it with road surface; the same window near the bonnet
    would miss a fit that is off by a hand's breadth. Scaling with perspective
    keeps the test asking the same physical question at every distance.
    """
    h, w = yellow_mask.shape[:2]
    lo, hi = fit.y_range_m
    ys = np.linspace(lo, hi, n_samples)
    xs = fit.x_at(ys)

    on_line = camera.ground_to_image(np.column_stack([xs, ys]))
    offset = camera.ground_to_image(np.column_stack([xs + 0.35, ys]))

    hit_ys: list[float] = []
    valid = 0
    for y, (u, v), (u2, v2) in zip(ys, on_line, offset):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        r = int(np.clip(round(np.hypot(u2 - u, v2 - v)), 2, 25))
        patch = yellow_mask[max(0, vi - r): vi + r + 1, max(0, ui - r): ui + r + 1]
        valid += 1
        # Any yellow pixel in the window counts. Markings are thin and the fit
        # is a smoothed approximation of them, so requiring the centre pixel to
        # be yellow would reject a correct fit that is two pixels off.
        if patch.size and patch.any():
            hit_ys.append(float(y))
    if not valid or not hit_ys:
        return 0.0, 0.0
    return len(hit_ys) / valid, max(hit_ys) - min(hit_ys)


def _median_gap(
    ordered: list[LaneFit],
    eval_distance_m: float,
    min_gap_m: float,
    max_divergence_ratio: float,
) -> tuple[LaneFit, LaneFit, float] | None:
    """The widest constant-width gap between adjacent boundaries, if any.

    Constant is the operative word. A gap that opens with distance is a fork,
    which :mod:`models.lane_seg.split` reports; a gap that holds its width for
    the length of the fit is a median.
    """
    best: tuple[LaneFit, LaneFit, float] | None = None
    for a, b in zip(ordered, ordered[1:]):
        lo = max(a.y_range_m[0], b.y_range_m[0])
        hi = min(a.y_range_m[1], b.y_range_m[1])
        if hi - lo < 5.0:
            continue
        near = abs(float(b.x_at(lo)[0]) - float(a.x_at(lo)[0]))
        far = abs(float(b.x_at(hi)[0]) - float(a.x_at(hi)[0]))
        if min(near, far) < min_gap_m:
            continue
        if max(near, far) / max(min(near, far), 1e-6) > max_divergence_ratio:
            continue                                  # diverging: a fork
        width = abs(float(b.x_at(eval_distance_m)[0])
                    - float(a.x_at(eval_distance_m)[0]))
        if best is None or width > best[2]:
            best = (a, b, width)
    return best


def detect_carriageway(
    lanes: LaneSet,
    camera: CameraModel | None = None,
    yellow_mask: np.ndarray | None = None,
    *,
    exclude_boxes_px: np.ndarray | None = None,
    min_yellow_fraction: float = 0.35,
    min_yellow_span_frac: float = 0.6,
    min_fit_length_m: float = 6.0,
    min_median_gap_m: float = 5.0,
    max_median_divergence_ratio: float = 1.35,
    drive_side: DriveSide = DriveSide.RIGHT,
    eval_distance_m: float | None = None,
) -> DualCarriageway | None:
    """Decide whether the road is divided into opposing carriageways.

    Args:
        lanes: Fitted lane geometry for one frame.
        camera: Needed only for the yellow cue, which has to project the fitted
            boundary back into the image to read paint colour.
        yellow_mask: ``LaneMaskResult.yellow_mask`` from the classical
            segmenter. Without it only the median-gap cue runs.
        exclude_boxes_px: ``(N, 4)`` detected object boxes, ``xyxy``. Yellow
            inside them is discounted. This is the one place the detector feeds
            back into lane understanding, and it earns its keep: a yellow *car*
            is the dominant false divider on these frames -- BDD100K is full of
            New York, and New York is full of taxis -- and no colour or shape
            test can separate a taxi's flank from paint, while a box around it
            says so directly.
        min_yellow_fraction: Share of samples along a boundary that must land on
            yellow. Well below half, because dividers are frequently dashed and
            a double yellow is often partly worn.
        min_yellow_span_frac: Share of the boundary's own fitted range the
            yellow must extend across. Rejects yellow *objects* -- a taxi, a
            skip, a sign -- which fill a band of range rather than running the
            length of the road. Relative rather than an absolute distance,
            because fits on dashed markings are routinely only seven or eight
            metres long and an absolute floor rejected correct ones.
        min_fit_length_m: Shorter fits than this are not asked the question at
            all. A boundary supported by four metres of road cannot establish
            how the road is divided, whatever colour it is.
        min_median_gap_m: Gap width that stops being a wide lane and starts
            being a median. Five metres is wider than any legal lane.
        max_median_divergence_ratio: Above this the gap is opening, so it is a
            fork and belongs to ``detect_split``, not here.
        drive_side: Which side traffic drives on. This **gates** the yellow cue
            rather than merely discounting it: under the US standard a yellow
            line on the driver's right is an edge line marking the roadway's
            boundary, not a direction divider, so accepting one there is simply
            reading the standard backwards. The median-gap cue is left
            unrestricted, because a median is a median wherever it sits.
        eval_distance_m: Where to evaluate the geometry. Defaults to the
            LaneSet's own evaluation distance, or 15 m.

    Returns:
        The detection, or None when the road shows no sign of being divided.
    """
    if lanes.status is LaneStatus.NOT_DETECTED:
        return None
    fits = lanes.all_fits()
    if not fits:
        return None

    y_eval = eval_distance_m if eval_distance_m is not None else (
        lanes.eval_distance_m or 15.0)
    ordered = sorted(fits, key=lambda f: float(f.x_at(y_eval)[0]))

    # --- cue 1: yellow paint -------------------------------------------------
    yellow_best: tuple[LaneFit, float] | None = None
    if yellow_mask is not None and camera is not None and np.any(yellow_mask):
        if exclude_boxes_px is not None and len(exclude_boxes_px):
            yellow_mask = yellow_mask.copy()
            h_m, w_m = yellow_mask.shape[:2]
            for x1, y1, x2, y2 in np.asarray(exclude_boxes_px,
                                             dtype=np.float64).reshape(-1, 4):
                a = max(0, int(np.floor(y1)))
                b = min(h_m, int(np.ceil(y2)) + 1)
                c = max(0, int(np.floor(x1)))
                d = min(w_m, int(np.ceil(x2)) + 1)
                if b > a and d > c:
                    yellow_mask[a:b, c:d] = 0
            if not np.any(yellow_mask):
                yellow_mask = None

    if yellow_mask is not None and camera is not None and np.any(yellow_mask):
        divider_is_left = drive_side is DriveSide.RIGHT
        for f in fits:
            # Only a boundary on the correct side can be a direction divider.
            # Checked before the mask is sampled, which also saves the work.
            if (float(f.x_at(y_eval)[0]) < 0.0) != divider_is_left:
                continue
            length = f.y_range_m[1] - f.y_range_m[0]
            if length < min_fit_length_m:
                continue
            frac, span = _yellow_evidence(f, camera, yellow_mask)
            if frac < min_yellow_fraction or span < min_yellow_span_frac * length:
                continue
            if yellow_best is None or frac > yellow_best[1]:
                yellow_best = (f, frac)

    # --- cue 2: a median gap -------------------------------------------------
    gap = _median_gap(ordered, y_eval, min_median_gap_m,
                      max_median_divergence_ratio)
    gap_divider: LaneFit | None = None
    if gap is not None:
        a, b, _width = gap
        xa, xb = float(a.x_at(y_eval)[0]), float(b.x_at(y_eval)[0])
        # The divider is the near edge of the median -- the boundary between the
        # ego and the gap. If the ego sits *inside* the gap it is driving on the
        # median, which means the fit is wrong, not that the road is strange.
        if xa >= 0.0:
            gap_divider = a
        elif xb <= 0.0:
            gap_divider = b

    if yellow_best is None and gap_divider is None:
        return None

    if yellow_best is not None and gap_divider is not None:
        kind = (DividerKind.YELLOW_AND_MEDIAN
                if yellow_best[0].role == gap_divider.role
                else DividerKind.YELLOW_LINE)
        divider = yellow_best[0]
    elif yellow_best is not None:
        kind, divider = DividerKind.YELLOW_LINE, yellow_best[0]
    else:
        assert gap_divider is not None
        kind, divider = DividerKind.MEDIAN_GAP, gap_divider

    x_div = float(divider.x_at(y_eval)[0])
    side = "left" if x_div < 0 else "right"

    # Lanes are the spaces between boundaries, so n boundaries bound n-1 lanes;
    # a carriageway with one visible boundary still has at least one lane.
    xs = [float(f.x_at(y_eval)[0]) for f in ordered]
    if side == "left":
        ego_bounds = sum(1 for x in xs if x >= x_div)
        far_bounds = sum(1 for x in xs if x < x_div) + 1   # + the divider itself
    else:
        ego_bounds = sum(1 for x in xs if x <= x_div)
        far_bounds = sum(1 for x in xs if x > x_div) + 1

    median_w = (gap[2] if gap is not None and kind is not DividerKind.YELLOW_LINE
                else None)

    confidence = 0.0
    if yellow_best is not None:
        confidence = max(confidence, 0.5 + 0.4 * min(1.0, yellow_best[1]))
    if gap_divider is not None:
        confidence = max(confidence, 0.55)
    if kind is DividerKind.YELLOW_AND_MEDIAN:
        confidence = min(0.98, confidence + 0.15)
    # A divider on the wrong side for the marking convention is possible but
    # unusual, and far more often a bad fit than a genuine contraflow.
    if side == "right":
        confidence *= 0.7

    return DualCarriageway(
        divider_role=divider.role,
        divider_x_m=x_div,
        divider_side=side,
        kind=kind,
        yellow_fraction=None if yellow_best is None else yellow_best[1],
        median_width_m=median_w,
        ego_lanes=max(1, ego_bounds - 1),
        oncoming_lanes=max(1, far_bounds - 1),
        confidence=round(float(confidence), 3),
        eval_distance_m=float(y_eval),
    )


class CarriagewayTracker:
    """Holds a stable answer across frames.

    Deliberately *not* the same shape as
    :class:`~models.lane_seg.split.SplitTracker`. A fork is an event: it happens
    once, is announced once, and then the vehicle is past it. A divided highway
    is a **state** -- true for the whole length of the road -- so once confirmed
    it is reported on every frame, and it decays only after the evidence has
    been absent for a while. Announcing it once and forgetting would leave the
    fusion layer with no idea, ten seconds later, whether the traffic to its
    left is oncoming.
    """

    def __init__(self, min_frames: int = 4, window: int = 10,
                 forget_after: int = 20) -> None:
        if min_frames > window:
            raise ValueError(
                f"min_frames ({min_frames}) exceeds window ({window}); the "
                f"condition could never be met")
        self.min_frames = min_frames
        self.window = window
        self.forget_after = forget_after
        self._recent: deque[DualCarriageway | None] = deque(maxlen=window)
        self._misses = 0
        self._held: DualCarriageway | None = None

    def update(self, obs: DualCarriageway | None) -> DualCarriageway | None:
        """Feed one frame; returns the current belief, held or None."""
        self._recent.append(obs)

        if obs is None:
            self._misses += 1
            if self._misses >= self.forget_after:
                self._held = None
            return self._held
        self._misses = 0

        hits = [o for o in self._recent if o is not None]
        if len(hits) >= self.min_frames:
            # Side by majority: the divider's side is the one thing that should
            # not flicker, and a single frame's fit reversing it would flip the
            # meaning of every object on that half of the road.
            side = Counter(h.divider_side for h in hits).most_common(1)[0][0]
            agreeing = [h for h in hits if h.divider_side == side] or hits
            self._held = max(agreeing, key=lambda o: o.confidence)
        return self._held

    @property
    def confirmed(self) -> DualCarriageway | None:
        return self._held

    def reset(self) -> None:
        self._recent.clear()
        self._misses = 0
        self._held = None
