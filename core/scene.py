"""The ADAS scene view: drivable corridor, boundaries, branches, HUD.

``core.overlay`` holds the primitive drawing operations -- one function per
thing, each drawing only what it is asked to. This module composes them into
the single annotated view a driver-facing system actually shows, with a fixed
colour language and a fixed layout, so that the same fact always looks the same
whichever script produced the frame.

The colour language, which is the whole point and is worth stating once:

======================  =========================================
Green                   your side of the road: the ego corridor
                        and its left boundary
Red                     do not enter: the right boundary, an exit
                        branch peeling off, the oncoming carriageway
Blue, dashed            the predicted path down the lane centre --
                        a *path*, deliberately not an edge
======================  =========================================

Red therefore means two different-looking things -- "this is the edge of your
lane" and "this region belongs to someone else" -- but one consistent idea:
crossing it is what you must not do. That is worth more than a larger palette
in which every region has its own hue and none of them mean anything.

Dashes are spaced in **ground metres**, not image pixels, so they foreshorten
with distance the way real paint does. A pixel-spaced dash reads as a dotted
line drawn on the screen; a metre-spaced one reads as markings lying on the
road, which is the difference between an overlay and a perception display.

How far the curves are drawn is not a style choice. It comes from
``LaneSet.parallel_model``: with a shared curvature the pair's far field rests
on the pair's pixels and the union of their supports is honest, while without
it each boundary is only as good as its own paint. See
:func:`models.lane_seg.fitting.fit_parallel_pair`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.calibrate import CameraModel
from core.config import OverlayConfig
from core.types import LaneFit, LaneSet, LaneStatus

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: BGR. See the module docstring for what each one means.
EGO_FILL = (120, 235, 120)
LEFT_BOUNDARY = (80, 225, 80)
RIGHT_BOUNDARY = (60, 60, 240)
PATH_DASH = (245, 180, 70)
FORBIDDEN_FILL = (70, 70, 225)
#: Orange: a same-direction lane with nothing in it -- a lane change is
#: available. Deliberately between the green of "yours" and the red of "not
#: yours", because that is exactly what it is: not yours, but you may take it.
CHANGE_FILL = (40, 165, 255)
PANEL_BG = (22, 22, 22)
PANEL_FG = (245, 245, 245)
ACCENT = (60, 220, 250)

#: Ground-space dash geometry for the predicted path, in metres.
DASH_ON_M = 2.4
DASH_OFF_M = 2.6

#: A lane boundary is never this far to the side. Samples beyond it are the
#: quadratic diverging, not the road bending, and drawing them puts a confident
#: red line across scenery. Wide enough to keep genuine adjacent boundaries.
MAX_LATERAL_M = 9.0

#: Width drawn for a branch lane at a fork, in metres. The branch's far edge is
#: rarely marked and never measured here, so it is drawn one standard lane wide
#: -- because a branch *is* a lane -- rather than tinted out to the frame edge.
#: The opposing carriageway of a divided road is different and does run to the
#: edge: there the claim really is "everything past this line is theirs".
BRANCH_WIDTH_M = 3.6

#: Furthest the corridor is ever drawn, in metres.
MAX_RENDER_M = 110.0

#: Drawing stops this many pixels short of the vanishing line. Exactly on it the
#: projection is singular and a metre of road maps to no pixels at all; close to
#: it, a centimetre of lateral error maps to a visible swing.
HORIZON_MARGIN_PX = 22.0


@dataclass
class SceneStyle:
    """Sizes and weights. Colours are module constants; these vary by use."""

    boundary_thickness: int = 11
    path_thickness: int = 7
    fill_alpha: float = 0.30
    forbidden_alpha: float = 0.34
    samples: int = 200
    #: Extension past the fitted range, in metres, when the parallel model was
    #: adopted. Kept small and drawn thinner: even a shared curvature is a
    #: quadratic, and a quadratic run far past its evidence still diverges.
    extend_far_m: float = 6.0
    #: Deliberately zero. Extending *toward* the camera was the worst artefact
    #: in the first version of this view: near the vehicle the ground-to-image
    #: projection is at its most sensitive, so a metre of harmless lateral error
    #: in the polynomial swings the drawn line clear across the bonnet. The road
    #: immediately ahead is instead covered by a straight continuation in image
    #: space -- see ``_extend_to_bottom`` -- which cannot diverge.
    extend_near_m: float = 0.0
    label_scale: float = 0.62
    panel_scale: float = 0.58


def trusted_range(lanes: LaneSet, style: SceneStyle | None = None
                  ) -> tuple[float, float] | None:
    """Forward range over which the ego corridor may be drawn.

    With the parallel model adopted, both boundaries carry the union of the
    pair's support and that union is the answer. Without it, only the range both
    boundaries independently cover can be drawn as a corridor -- beyond it one
    edge would be measured and the other invented, and a filled region between
    them would look equally solid either way.
    """
    left, right = lanes.ego_left, lanes.ego_right
    if left is None or right is None:
        return None
    if lanes.parallel_model:
        lo = min(left.y_range_m[0], right.y_range_m[0])
        hi = max(left.y_range_m[1], right.y_range_m[1])
    else:
        lo = max(left.y_range_m[0], right.y_range_m[0])
        hi = min(left.y_range_m[1], right.y_range_m[1])
    return (lo, hi) if hi - lo > 0.5 else None


def corridor_range(lanes: LaneSet, camera: CameraModel) -> tuple[float, float] | None:
    """The trusted range, carried onward to where the road actually ends.

    ``trusted_range`` says how far the *pixels* reach. This says how far the
    *road* reaches, which is what a lane overlay should cover: a boundary drawn
    only as far as its paint survived leaves a stub floating in the middle of
    the image, and the road plainly continues past it.
    """
    rng = trusted_range(lanes)
    if rng is None:
        return None
    lo, hi = rng
    width = lanes.lane_width_m
    far = min(
        far_limit(lanes.ego_left, camera, width, lanes.ego_right),
        far_limit(lanes.ego_right, camera, width, lanes.ego_left),
    )
    return (lo, max(hi, far))


def far_limit(fit: LaneFit, camera: CameraModel, width_ref: float | None = None,
              partner: LaneFit | None = None) -> float:
    """How far out this boundary may be drawn before it stops meaning anything.

    Walks outward from the fitted range and stops at the first of:

    * the **vertex of the parabola**, where ``dx/dy`` changes sign. Past it the
      fitted curve turns back on itself, which produced a visible hook at the
      top of every long boundary. No road does that, so the vertex is not a
      threshold to be tuned -- it is the exact point beyond which the model has
      stopped describing a road at all;
    * the projected point reaching the horizon -- there is no more road to draw;
    * the curve passing ``MAX_LATERAL_M``, which is the quadratic diverging
      rather than the road bending;
    * the corridor width departing from the measured width by more than half,
      which is the pair fanning open. Under the parallel model width is constant
      by construction so this never fires, which is the point: a jointly-fitted
      pair earns the long reach and an independently-fitted one has to prove it
      metre by metre.

    This is what makes the drawn lane run from the car to the end of the road
    instead of stopping wherever the paint happened to run out.
    """
    lo, hi = fit.y_range_m
    limit = hi

    # The parabola's turning point, if it lies ahead of the fitted range.
    a, b, _ = fit.coeffs
    ceiling = MAX_RENDER_M
    if abs(a) > 1e-9:
        vertex = -b / (2.0 * a)
        if vertex > hi:
            ceiling = min(ceiling, vertex)

    for y in np.arange(hi, ceiling, 1.0):
        x = float(fit.x_at(float(y))[0])
        if abs(x) > MAX_LATERAL_M:
            break
        pt = np.asarray(camera.ground_to_image(np.array([[x, float(y)]]))).reshape(-1)
        if not np.all(np.isfinite(pt)):
            break
        v_h = float(np.asarray(camera.horizon_v(float(pt[0]))).reshape(-1)[0])
        if np.isfinite(v_h) and pt[1] - v_h < HORIZON_MARGIN_PX:
            break
        if partner is not None and width_ref:
            w = abs(float(partner.x_at(float(y))[0]) - x)
            if abs(w - width_ref) > 0.5 * width_ref:
                break
        limit = float(y)
    return max(limit, hi)


def _project(fit: LaneFit, camera: CameraModel, rng: tuple[float, float],
             n: int) -> np.ndarray:
    """Project a fitted boundary, dropping samples the model cannot support.

    A quadratic evaluated past its evidence eventually heads for the horizon
    sideways. Truncating at a plausible lateral offset turns that from a
    confident wrong line into a line that simply stops, which is the honest
    rendering of "the fit ran out".
    """
    lo, hi = rng
    if hi <= lo:
        return np.zeros((0, 2), dtype=np.float64)
    ys = np.linspace(lo, hi, max(2, n))
    xs = fit.x_at(ys)
    ok = np.abs(xs) <= MAX_LATERAL_M
    if not ok.any():
        return np.zeros((0, 2), dtype=np.float64)
    # Keep only the run of valid samples that starts at the near end, so the
    # curve is truncated rather than broken into disconnected pieces.
    first_bad = np.argmax(~ok) if (~ok).any() else len(ok)
    if first_bad == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ys = ys[:first_bad]

    pts = camera.ground_to_image(np.column_stack([fit.x_at(ys), ys]))
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts[np.all(np.isfinite(pts), axis=1)]


#: The bonnet continuation may cover at most this fraction of the frame height.
#: A fit whose near end is already high in the image starts far away, and
#: continuing it all the way down would invent more road than it measured.
#: Generous, because the continuation is a straight segment in image space and
#: cannot diverge: the cap guards against absurdity, not against error.
MAX_BOTTOM_EXTENSION_FRAC = 0.46


def _extend_to_bottom(pts: np.ndarray, height: int) -> np.ndarray:
    """Continue a projected curve straight down to the bottom of the frame.

    The near end of a fit sits several metres ahead, leaving a gap between the
    corridor and the bonnet that reads as the system not seeing the road it is
    on. The continuation is **linear in image space** on purpose: the road
    directly ahead is certainly the ego lane, the direction is taken from the
    curve's own near end, and a straight segment cannot fly off the way an
    extrapolated quadratic does.
    """
    if len(pts) < 2:
        return pts
    p0, p1 = pts[0], pts[1]
    dv = p0[1] - p1[1]
    if dv <= 1e-6 or p0[1] >= height - 1:
        return pts
    if (height - 1 - p0[1]) > MAX_BOTTOM_EXTENSION_FRAC * height:
        return pts
    t = (height - 1 - p0[1]) / dv
    tip = np.array([p0[0] + t * (p0[0] - p1[0]), float(height - 1)])
    return np.vstack([tip, pts])


def _extend_pair_to_bottom(
    lp: np.ndarray, rp: np.ndarray, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extend both corridor edges, but only while they stay in order.

    Extended independently, the two continuations can cross before they reach
    the bonnet, and a filled polygon whose edges cross renders as a bowtie --
    an hourglass of colour that is not a lane and not anything else. Where that
    would happen the extension is simply dropped for both: a corridor that stops
    a few metres short is a smaller lie than one that folds through itself.
    """
    le, re = _extend_to_bottom(lp, height), _extend_to_bottom(rp, height)
    if len(le) and len(re) and le[0][0] >= re[0][0] - 1.0:
        return lp, rp
    return le, re


def draw_drivable_area(
    image: np.ndarray, lanes: LaneSet, camera: CameraModel,
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Fill the ego corridor between the two boundaries."""
    style = style or SceneStyle()
    rng = corridor_range(lanes, camera)
    if rng is None:
        return image
    lp = _project(lanes.ego_left, camera, rng, style.samples)
    rp = _project(lanes.ego_right, camera, rng, style.samples)
    if len(lp) < 2 or len(rp) < 2:
        return image
    lp, rp = _extend_pair_to_bottom(lp, rp, image.shape[0])
    vis = image.copy()
    overlay = vis.copy()
    cv2.fillPoly(overlay, [np.vstack([lp, rp[::-1]]).astype(np.int32)], EGO_FILL)
    cv2.addWeighted(overlay, style.fill_alpha, vis, 1 - style.fill_alpha, 0, vis)
    return vis


def draw_boundaries(
    image: np.ndarray, lanes: LaneSet, camera: CameraModel,
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Left boundary green, right boundary red, both long and smooth.

    Each is drawn over its own fitted range at full weight and, where the
    parallel model earned it, a little further at half weight. The weight change
    is the honest part: a thinner line where the evidence thins out.
    """
    style = style or SceneStyle()
    vis = image.copy()
    rng = corridor_range(lanes, camera)
    for fit, partner, colour in ((lanes.ego_left, lanes.ego_right, LEFT_BOUNDARY),
                                 (lanes.ego_right, lanes.ego_left, RIGHT_BOUNDARY)):
        if fit is None:
            continue
        if rng is not None:
            lo, hi = rng
        else:
            lo, hi = fit.y_range_m
            hi = far_limit(fit, camera, lanes.lane_width_m, partner)
        pts = _project(fit, camera, (lo, hi), style.samples)
        if len(pts) < 2:
            continue
        # One polyline, bonnet to horizon. Drawn as a single stroke rather than
        # a measured part plus an extension: a line broken into pieces of
        # different weight reads as two separate detections, and the road it
        # describes is continuous.
        pts = _extend_to_bottom(pts, vis.shape[0])
        cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                      colour, style.boundary_thickness, cv2.LINE_AA)
    return vis


def draw_predicted_path(
    image: np.ndarray, lanes: LaneSet, camera: CameraModel,
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Dashed centreline down the middle of the ego corridor.

    Dashes are laid out in ground metres and then projected, so they shorten
    with distance like real paint instead of marching evenly up the screen.
    """
    style = style or SceneStyle()
    rng = corridor_range(lanes, camera)
    if rng is None or lanes.ego_left is None or lanes.ego_right is None:
        return image
    centre = LaneFit(
        coeffs=(lanes.ego_left.coeffs + lanes.ego_right.coeffs) / 2.0,
        role="ego_centre", n_pixels=0, y_range_m=rng,
    )
    vis = image.copy()
    lo, hi = rng
    y = lo
    period = DASH_ON_M + DASH_OFF_M
    while y < hi:
        seg_hi = min(y + DASH_ON_M, hi)
        pts = _project(centre, camera, (y, seg_hi), 10)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                          PATH_DASH, style.path_thickness, cv2.LINE_AA)
        y += period
    return vis


def _forbidden_band(
    image: np.ndarray, edge: LaneFit, camera: CameraModel, side: str,
    style: SceneStyle, width_m: float = BRANCH_WIDTH_M,
) -> np.ndarray:
    """Tint a lane-width band running alongside ``edge``, on ``side``.

    Used for a fork's branch, which is a lane and should look like one. Its
    outer edge is the fitted boundary offset by one lane width -- an assumption,
    stated here rather than hidden: the branch's real width was not measured.
    """
    sign = -1.0 if side == "left" else 1.0
    outer = LaneFit(
        coeffs=edge.coeffs + np.array([0.0, 0.0, sign * width_m]),
        role=edge.role + "_outer", n_pixels=edge.n_pixels,
        y_range_m=edge.y_range_m,
    )
    inner_pts = _project(edge, camera, edge.y_range_m, style.samples)
    outer_pts = _project(outer, camera, outer.y_range_m, style.samples)
    if len(inner_pts) < 2 or len(outer_pts) < 2:
        return image
    inner_pts, outer_pts = _extend_pair_to_bottom(
        *((inner_pts, outer_pts) if side == "right" else (outer_pts, inner_pts)),
        image.shape[0])
    if side == "left":
        inner_pts, outer_pts = outer_pts, inner_pts

    vis = image.copy()
    overlay = vis.copy()
    cv2.fillPoly(overlay, [np.vstack([inner_pts, outer_pts[::-1]]).astype(np.int32)],
                 FORBIDDEN_FILL)
    cv2.addWeighted(overlay, style.forbidden_alpha, vis,
                    1 - style.forbidden_alpha, 0, vis)
    cv2.polylines(vis, [inner_pts.reshape(-1, 1, 2).astype(np.int32)], False,
                  RIGHT_BOUNDARY, style.boundary_thickness, cv2.LINE_AA)
    return vis


def _forbidden_wedge(
    image: np.ndarray, edge: LaneFit, camera: CameraModel, side: str,
    style: SceneStyle, extend_to_bottom: bool = True,
) -> np.ndarray:
    """Tint everything beyond ``edge`` on ``side`` out to the frame edge.

    Bounded above by the horizon, because nothing above the vanishing line is
    road and so nothing above it can be the region in question. Its outer edge
    is left at the frame border rather than given a width: on a real frame that
    edge is usually unmarked or out of shot, and drawing a specific width would
    be drawing a measurement nobody took.
    """
    h, w = image.shape[:2]
    pts = _project(edge, camera, edge.y_range_m, style.samples)
    if len(pts) < 2:
        return image
    edge_x = 0 if side == "left" else w - 1
    v_h = float(np.asarray(camera.horizon_v(float(edge_x))).reshape(-1)[0])
    if not np.isfinite(v_h):
        v_h = float(pts[-1][1])
    v_top = float(np.clip(max(v_h, 0.0), 0, h - 1))
    bottom = h - 1 if extend_to_bottom else float(pts[0][1])
    pts = _extend_to_bottom(pts, h)
    poly = np.vstack([pts, [[edge_x, v_top]], [[edge_x, bottom]],
                      [[pts[0][0], bottom]]]).astype(np.int32)
    vis = image.copy()
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly], FORBIDDEN_FILL)
    cv2.addWeighted(overlay, style.forbidden_alpha, vis,
                    1 - style.forbidden_alpha, 0, vis)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                  RIGHT_BOUNDARY, style.boundary_thickness, cv2.LINE_AA)
    return vis


def draw_lane_options(
    image: np.ndarray, road_lanes, camera: CameraModel,
    style: SceneStyle | None = None, label: bool = True,
) -> np.ndarray:
    """Tint the free same-direction lanes orange: a change is available there.

    Only ``LaneKind.FREE`` is painted. An occupied neighbour is left alone
    rather than given a colour of its own -- the absence of orange already says
    "not available", and a fourth region colour would dilute the three that
    carry meaning.
    """
    from models.lane_seg.neighbours import LaneKind

    style = style or SceneStyle()
    vis = image
    for lane in road_lanes or ():
        if lane.kind is not LaneKind.FREE:
            continue
        lo, hi = lane.y_range_m
        hi = far_limit(lane.left, camera, lane.width_m, lane.right)
        lp = _project(lane.left, camera, (lo, hi), style.samples)
        rp = _project(lane.right, camera, (lo, hi), style.samples)
        if len(lp) < 2 or len(rp) < 2:
            continue
        lp, rp = _extend_pair_to_bottom(lp, rp, vis.shape[0])
        vis = vis.copy()
        overlay = vis.copy()
        cv2.fillPoly(overlay, [np.vstack([lp, rp[::-1]]).astype(np.int32)],
                     CHANGE_FILL)
        cv2.addWeighted(overlay, style.fill_alpha, vis, 1 - style.fill_alpha, 0, vis)
        if label:
            y_tag = lo + 0.3 * (hi - lo)
            pt = camera.ground_to_image(
                np.array([[lane.centre_x_at(y_tag), y_tag]]))[0]
            if np.all(np.isfinite(pt)):
                arrow = "<-" if lane.side == "left" else "->"
                _tag(vis, f"{arrow} Lane change {lane.side}",
                     (int(pt[0]), int(pt[1])), CHANGE_FILL, style.label_scale)
    return vis


def draw_branch(
    image: np.ndarray, split, lanes: LaneSet, camera: CameraModel,
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Tint the branch that peels away at a fork, in the do-not-enter colour."""
    if split is None:
        return image
    style = style or SceneStyle()
    side = getattr(split.side, "value", str(split.side))
    if side not in ("left", "right"):
        return image
    role = split.left_role if side == "left" else split.right_role
    edge = next((f for f in lanes.all_fits() if f.role == role), None)
    if edge is None:
        return image
    return _forbidden_band(image, edge, camera, side, style)


def draw_oncoming(
    image: np.ndarray, carriageway, lanes: LaneSet, camera: CameraModel,
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Tint the opposing carriageway of a divided road."""
    if carriageway is None:
        return image
    style = style or SceneStyle()
    edge = next((f for f in lanes.all_fits()
                 if f.role == carriageway.divider_role), None)
    if edge is None:
        return image
    return _forbidden_wedge(image, edge, camera, carriageway.divider_side, style)


# --------------------------------------------------------------------------- #
# Chrome: panel, legend, badge, labels
# --------------------------------------------------------------------------- #

def _rounded_panel(image: np.ndarray, x: int, y: int, w: int, h: int,
                   alpha: float = 0.82) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_perception_panel(
    image: np.ndarray, rows: list[tuple[str, str]], title: str = "ADAS PERCEPTION",
    style: SceneStyle | None = None,
) -> np.ndarray:
    """Top-left status panel: one ``label: value`` per row."""
    style = style or SceneStyle()
    if not rows:
        return image
    vis = image.copy()
    sc = style.panel_scale
    texts = [f"{k}: {v}" for k, v in rows]
    widths = [cv2.getTextSize(t, FONT, sc, 1)[0][0] for t in texts]
    tw_title = cv2.getTextSize(title, FONT, sc * 1.15, 2)[0][0]
    pad, line_h = 14, int(26 * max(sc, 0.5) / 0.58)
    w = max(max(widths), tw_title) + 2 * pad
    h = pad + int(line_h * 1.25) + line_h * len(texts) + pad // 2
    _rounded_panel(vis, 10, 10, w, h)
    cv2.putText(vis, title, (10 + pad, 10 + pad + int(line_h * 0.72)),
                FONT, sc * 1.15, PANEL_FG, 2, cv2.LINE_AA)
    for i, t in enumerate(texts):
        y = 10 + pad + int(line_h * 1.25) + line_h * i + int(line_h * 0.72)
        cv2.putText(vis, t, (10 + pad, y), FONT, sc, PANEL_FG, 1, cv2.LINE_AA)
    return vis


def draw_legend(image: np.ndarray, style: SceneStyle | None = None) -> np.ndarray:
    """Bottom-left key. Without it the colours are a private code."""
    style = style or SceneStyle()
    entries = [
        (LEFT_BOUNDARY, "Left lane boundary", False),
        (RIGHT_BOUNDARY, "Right lane boundary", False),
        (PATH_DASH, "Lane centre (predicted path)", True),
        (CHANGE_FILL, "Lane change available", False),
        (FORBIDDEN_FILL, "Do not enter (exit / oncoming)", False),
    ]
    vis = image.copy()
    sc = style.panel_scale * 0.95
    line_h, pad, swatch = 26, 12, 46
    tw = max(cv2.getTextSize(t, FONT, sc, 1)[0][0] for _, t, _ in entries)
    w = pad + swatch + 10 + tw + pad
    h = pad + line_h * len(entries) + pad // 2
    x0, y0 = 10, image.shape[0] - h - 10
    _rounded_panel(vis, x0, y0, w, h)
    for i, (colour, text, dashed) in enumerate(entries):
        cy = y0 + pad + line_h * i + line_h // 2
        if dashed:
            for k in range(3):
                sx = x0 + pad + k * 17
                cv2.line(vis, (sx, cy), (sx + 11, cy), colour, 5, cv2.LINE_AA)
        else:
            cv2.line(vis, (x0 + pad, cy), (x0 + pad + swatch, cy), colour, 5,
                     cv2.LINE_AA)
        cv2.putText(vis, text, (x0 + pad + swatch + 10, cy + 5), FONT, sc,
                    PANEL_FG, 1, cv2.LINE_AA)
    return vis


def draw_split_badge(image: np.ndarray, split, style: SceneStyle | None = None
                     ) -> np.ndarray:
    """Top-right badge: the warning, the distance, and a schematic of the fork.

    The schematic exists because "ROAD SPLIT AHEAD, right in 40 m" is a sentence
    that has to be read, and a driver glancing at a display has not got time to
    read a sentence. Two arrows and a dividing line are recognised, not read.
    """
    if split is None:
        return image
    style = style or SceneStyle()
    side = getattr(split.side, "value", str(split.side))
    vis = image.copy()
    w_box, h_box = 250, 190
    x0, y0 = image.shape[1] - w_box - 10, 10
    _rounded_panel(vis, x0, y0, w_box, h_box)

    cv2.putText(vis, "ROAD SPLIT AHEAD", (x0 + 14, y0 + 28), FONT, 0.55,
                PANEL_FG, 1, cv2.LINE_AA)
    cv2.putText(vis, f"{side.capitalize()} in {split.gore_distance_m:.0f} m",
                (x0 + 14, y0 + 58), FONT, 0.78, ACCENT, 2, cv2.LINE_AA)

    # Schematic: straight-on arrow, dashed divider, branch arrow.
    cx, base, top = x0 + w_box // 2, y0 + h_box - 22, y0 + 82
    cv2.arrowedLine(vis, (cx - 44, base), (cx - 44, top), PANEL_FG, 4,
                    cv2.LINE_AA, tipLength=0.28)
    for yy in range(top + 6, base, 16):
        cv2.line(vis, (cx, yy), (cx, min(yy + 8, base)), PANEL_FG, 2, cv2.LINE_AA)
    dx = 1 if side == "right" else -1
    curve = np.array([[cx + dx * 18, base], [cx + dx * 26, base - 34],
                      [cx + dx * 48, top + 16], [cx + dx * 62, top + 6]], np.int32)
    cv2.polylines(vis, [curve.reshape(-1, 1, 2)], False, RIGHT_BOUNDARY, 4,
                  cv2.LINE_AA)
    cv2.arrowedLine(vis, tuple(curve[-2]), tuple(curve[-1]), RIGHT_BOUNDARY, 4,
                    cv2.LINE_AA, tipLength=0.9)
    return vis


def _tag(image: np.ndarray, text: str, anchor: tuple[int, int],
         colour: tuple[int, int, int], scale: float) -> None:
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = anchor
    x = int(np.clip(x - tw // 2, 6, image.shape[1] - tw - 12))
    y = int(np.clip(y, th + 12, image.shape[0] - 12))
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 9, y - th - base - 5), (x + tw + 9, y + base + 2),
                  PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    cv2.rectangle(image, (x - 9, y - th - base - 5), (x + tw + 9, y + base + 2),
                  colour, 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, scale, PANEL_FG, 1, cv2.LINE_AA)


def draw_lane_labels(
    image: np.ndarray, lanes: LaneSet, camera: CameraModel, split=None,
    carriageway=None, style: SceneStyle | None = None,
) -> np.ndarray:
    """Name the regions on the frame itself."""
    style = style or SceneStyle()
    rng = trusted_range(lanes, style)
    if rng is None:
        return image
    vis = image.copy()
    y_tag = rng[0] + 0.28 * (rng[1] - rng[0])

    centre_x = 0.5 * (float(lanes.ego_left.x_at(y_tag)[0])
                      + float(lanes.ego_right.x_at(y_tag)[0]))
    pt = camera.ground_to_image(np.array([[centre_x, y_tag]]))[0]
    if np.all(np.isfinite(pt)):
        _tag(vis, "EGO LANE", (int(pt[0]), int(pt[1])), LEFT_BOUNDARY,
             style.label_scale)

    if split is not None:
        side = getattr(split.side, "value", str(split.side))
        if side in ("left", "right"):
            off = -6.0 if side == "left" else 6.0
            pt = camera.ground_to_image(np.array([[centre_x + off, y_tag]]))[0]
            if np.all(np.isfinite(pt)):
                _tag(vis, f"Exit / split {side}", (int(pt[0]), int(pt[1])),
                     RIGHT_BOUNDARY, style.label_scale)
    elif carriageway is not None:
        off = -6.0 if carriageway.divider_side == "left" else 6.0
        pt = camera.ground_to_image(np.array([[centre_x + off, y_tag]]))[0]
        if np.all(np.isfinite(pt)):
            _tag(vis, "Oncoming traffic", (int(pt[0]), int(pt[1])),
                 RIGHT_BOUNDARY, style.label_scale)
    return vis


def draw_scene(
    image: np.ndarray,
    lanes: LaneSet,
    camera: CameraModel | None,
    cfg: OverlayConfig,
    *,
    split=None,
    carriageway=None,
    road_lanes=None,
    status_rows: list[tuple[str, str]] | None = None,
    style: SceneStyle | None = None,
    show_panel: bool = True,
    show_legend: bool = True,
    show_labels: bool = True,
    show_badge: bool = True,
) -> np.ndarray:
    """Compose the full scene view in one call.

    Painting order matters and is not arbitrary: region fills first so the
    boundaries sit on top of them rather than being washed out, the predicted
    path above the fills it runs through, and the chrome last so a panel is
    never half-covered by a lane curve that happens to reach the corner.
    """
    style = style or SceneStyle()
    vis = image
    if camera is not None and lanes.status is not LaneStatus.NOT_DETECTED:
        vis = draw_drivable_area(vis, lanes, camera, style)
        # Options before prohibitions: where the two overlap -- a free lane the
        # carriageway logic then rules oncoming -- red must win, and painting it
        # second is what makes it win.
        vis = draw_lane_options(vis, road_lanes, camera, style)
        vis = draw_branch(vis, split, lanes, camera, style)
        if split is None:
            # A fork and a divided road are different claims; showing both
            # tints at once says the same region is two things.
            vis = draw_oncoming(vis, carriageway, lanes, camera, style)
        vis = draw_boundaries(vis, lanes, camera, style)
        vis = draw_predicted_path(vis, lanes, camera, style)
        if show_labels:
            vis = draw_lane_labels(vis, lanes, camera, split, carriageway, style)
    if show_panel and status_rows:
        vis = draw_perception_panel(vis, status_rows, style=style)
    if show_badge:
        vis = draw_split_badge(vis, split, style)
    if show_legend:
        vis = draw_legend(vis, style)
    return vis


def scene_status_rows(
    lanes: LaneSet, split=None, carriageway=None,
    weather: str | None = None, visibility: str | None = None,
    speed_kmh: float | None = None, road_lanes=None,
) -> list[tuple[str, str]]:
    """Rows for the perception panel, skipping anything not actually known.

    An unknown value is omitted rather than printed as "unknown". A panel full
    of "unknown" trains the reader to stop looking at the panel.
    """
    rows: list[tuple[str, str]] = []
    if speed_kmh is not None:
        rows.append(("Speed", f"{speed_kmh:.0f} km/h"))
    if lanes.status is LaneStatus.OK and lanes.ego_offset_m is not None:
        side = "right" if lanes.ego_offset_m > 0 else "left"
        rows.append(("Lane", f"centre {abs(lanes.ego_offset_m):.2f} m {side}"))
    else:
        rows.append(("Lane", lanes.status.value.replace("_", " ")))
    if split is not None:
        side = getattr(split.side, "value", str(split.side))
        rows.append(("Road", f"split ahead ({side})"))
    elif carriageway is not None:
        rows.append(("Road", f"divided, oncoming {carriageway.divider_side}"))
    if road_lanes:
        from models.lane_seg.neighbours import change_options

        opts = change_options(road_lanes)
        rows.append(("Lane change",
                     ", ".join(sorted({o.side for o in opts})) if opts else "none"))
    if lanes.lane_width_m is not None:
        rows.append(("Lane width", f"{lanes.lane_width_m:.2f} m"))
    if lanes.curvature_radius_m is not None:
        r = lanes.curvature_radius_m
        rows.append(("Curvature",
                     "straight" if not np.isfinite(r) or abs(r) > 1000
                     else f"{abs(r):.0f} m"))
    if weather and weather != "unknown":
        rows.append(("Weather", weather))
    if visibility and visibility != "unknown":
        rows.append(("Visibility", visibility))
    return rows
