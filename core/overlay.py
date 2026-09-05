"""Annotation rendering: masks, boxes, fitted lane curves, a text HUD, and MP4.

Everything here consumes :mod:`core.types` values in **image pixel** coordinates
at the canonical frame size, and draws on a copy -- no function mutates the frame
it is handed. Lane polynomials arrive in bird's-eye metres and are reprojected
through the :class:`~core.calibrate.CameraModel`, which is precisely the visual
check that calibration and fitting agree: the reprojected curve should lie along
the lane pixels it was fitted to.

Colours are BGR, matching OpenCV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from core.calibrate import CameraModel
from core.config import OverlayConfig, VideoConfig
from models.lane_seg.fitting import fit_to_image_points
from core.logging_utils import get_logger, log_kv
from core.types import FrameResult, LaneSet, LaneStatus, class_name

_log = get_logger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Road splits are drawn in their own colour, deliberately unlike any lane role.
#: A fork changes what the geometry means, so it must not read as one more
#: boundary among the others.
SPLIT_COLOR = (0, 200, 255)

#: Per-class box colours (BGR). Unlisted classes fall back to LABEL_COLOR_DEFAULT.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 200, 255),
    "person": (60, 220, 60),
    "bicycle": (255, 200, 80),
    "motorcycle": (255, 180, 0),
}
LABEL_COLOR_DEFAULT: tuple[int, int, int] = (200, 200, 200)

#: Per-role lane curve colours (BGR).
LANE_COLORS: dict[str, tuple[int, int, int]] = {
    "ego_left": (0, 255, 255),
    "ego_right": (0, 255, 255),
    "ego_centre": (255, 255, 255),
    "adjacent_left": (255, 160, 0),
    "adjacent_right": (255, 160, 0),
}
LANE_COLOR_DEFAULT: tuple[int, int, int] = (255, 255, 255)

DRIVABLE_COLOR: tuple[int, int, int] = (0, 180, 0)
#: Magenta, not red: the lane mask is drawn over the green drivable fill, and
#: red blends with green into an orange that reads as a third category.
LANE_MASK_COLOR: tuple[int, int, int] = (255, 0, 255)


def label_color(label: str) -> tuple[int, int, int]:
    """BGR colour for a detection class."""
    return LABEL_COLORS.get(label, LABEL_COLOR_DEFAULT)


def lane_color(role: str) -> tuple[int, int, int]:
    """BGR colour for a lane boundary role."""
    return LANE_COLORS.get(role, LANE_COLOR_DEFAULT)


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: OverlayConfig,
    color: tuple[int, int, int] = DRIVABLE_COLOR,
    alpha: float | None = None,
) -> np.ndarray:
    """Blend a translucent binary mask over the frame.

    Args:
        image: BGR frame; not modified.
        mask: 2-D array, non-zero where the overlay applies. Must match the
            frame's height and width.
        cfg: Supplies the default ``mask_alpha``.
        color: BGR fill colour.
        alpha: Overrides ``cfg.mask_alpha`` for this one layer.

    Raises:
        ValueError: if the mask shape does not match the frame.
    """
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} does not match frame {image.shape[:2]}"
        )
    vis = image.copy()
    selected = mask.astype(bool)
    if not selected.any():
        return vis
    blend = cfg.mask_alpha if alpha is None else alpha
    layer = np.zeros_like(vis)
    layer[selected] = color
    blended = cv2.addWeighted(vis, 1.0, layer, blend, 0.0)
    vis[selected] = blended[selected]
    return vis


def draw_boxes(
    image: np.ndarray,
    result: FrameResult,
    cfg: OverlayConfig,
    distances_m: dict[int, float] | None = None,
    track_ids: dict[int, int] | None = None,
) -> np.ndarray:
    """Draw detection boxes with class, confidence, distance and track id.

    Args:
        image: BGR frame; not modified.
        result: Detections in image pixels at this frame's size.
        cfg: Appearance settings.
        distances_m: Optional map from detection index to forward distance.
        track_ids: Optional map from detection index to persistent track id.
            Drawn as ``#id`` ahead of the class name, because when watching a
            clip the question is almost always "is this the same object as last
            frame", and an id buried after the confidence is unreadable at speed.
    """
    vis = image.copy()
    for i in range(len(result)):
        x1, y1, x2, y2 = (int(round(v)) for v in result.boxes_px[i])
        name = class_name(int(result.class_ids[i]))
        colour = label_color(name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, cfg.box_thickness)

        parts = []
        if track_ids is not None and i in track_ids:
            parts.append(f"#{track_ids[i]}")
        parts += [name, f"{result.scores[i]:.2f}"]
        if distances_m is not None and i in distances_m:
            parts.append(f"{distances_m[i]:.1f}m")
        text = " ".join(parts)

        (tw, th), baseline = cv2.getTextSize(text, FONT, cfg.font_scale, 1)
        # Keep the label inside the frame when the box touches the top edge.
        label_top = max(0, y1 - th - baseline - 2)
        cv2.rectangle(
            vis, (x1, label_top), (x1 + tw + 4, label_top + th + baseline + 2), colour, -1
        )
        cv2.putText(
            vis, text, (x1 + 2, label_top + th), FONT, cfg.font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Mark the ground contact point: the only part of a 2-D box on the road.
        cu, cv_ = 0.5 * (x1 + x2), y2
        cv2.drawMarker(
            vis, (int(round(cu)), int(round(cv_))), colour, cv2.MARKER_TRIANGLE_UP, 10, 2
        )
    return vis


def draw_lane_fits(
    image: np.ndarray,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
) -> np.ndarray:
    """Reproject fitted lane polynomials from bird's-eye metres into the image.

    Drawn thick and bright on purpose: laid over the raw lane mask, a curve that
    tracks the mask confirms calibration and fitting agree, and one that drifts
    away is the fastest possible signal that they do not.
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    for fit in lanes.all_fits():
        pts = fit_to_image_points(fit, camera, n_samples=60)
        if len(pts) < 2:
            continue
        # Keep only points within a sane margin of the frame; a boundary fitted
        # to 40 m can project far outside it near the horizon.
        inside = (
            (pts[:, 0] > -w) & (pts[:, 0] < 2 * w) & (pts[:, 1] > -h) & (pts[:, 1] < 2 * h)
        )
        pts = pts[inside]
        if len(pts) < 2:
            continue
        colour = lane_color(fit.role)
        poly = np.int32(np.round(pts)).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], False, (0, 0, 0), cfg.lane_thickness + 4, cv2.LINE_AA)
        cv2.polylines(vis, [poly], False, colour, cfg.lane_thickness, cv2.LINE_AA)
        cv2.putText(
            vis, fit.role, tuple(np.int32(pts[0]) + np.array([8, -8])),
            FONT, cfg.font_scale, colour, 2, cv2.LINE_AA,
        )
    return vis


def draw_hud(
    image: np.ndarray, lines: list[str], cfg: OverlayConfig, warnings: list[str] | None = None
) -> np.ndarray:
    """Draw a semi-transparent text panel in the top-left corner.

    Args:
        image: BGR frame; not modified.
        lines: Status rows, rendered in order.
        cfg: Supplies panel width, font scale and blend alpha.
        warnings: Optional rows rendered below the status block in red.
    """
    vis = image.copy()
    if not cfg.hud:
        return vis
    warnings = warnings or []
    row_h = int(round(24 * max(cfg.font_scale / 0.5, 1.0)))
    pad = 10
    height = pad * 2 + row_h * (len(lines) + len(warnings))
    width = min(cfg.hud_width_px, vis.shape[1] - 2 * pad)

    panel = vis[pad : pad + height, pad : pad + width]
    if panel.size:
        dark = np.zeros_like(panel)
        vis[pad : pad + height, pad : pad + width] = cv2.addWeighted(panel, 0.35, dark, 0.65, 0)
        cv2.rectangle(vis, (pad, pad), (pad + width, pad + height), (90, 90, 90), 1)

    y = pad + row_h
    for text in lines:
        cv2.putText(
            vis, text, (pad + 10, y), FONT, cfg.font_scale, (235, 235, 235), 1, cv2.LINE_AA
        )
        y += row_h
    for text in warnings:
        cv2.putText(vis, text, (pad + 10, y), FONT, cfg.font_scale, (60, 60, 255), 2, cv2.LINE_AA)
        y += row_h
    return vis


#: The filled-lane rendering. Green for the drivable ego lane, blue for the left
#: boundary, red for the right -- the convention from the classic lane-finding
#: visualisation, chosen because it is instantly readable: a driver sees the
#: corridor they are in, not two abstract curves.
LANE_AREA_COLOR = (90, 230, 90)
LANE_LEFT_COLOR = (235, 90, 60)     # BGR: blue
LANE_RIGHT_COLOR = (60, 60, 235)    # BGR: red


def draw_lane_area(
    image: np.ndarray,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
    alpha: float = 0.32,
    boundary_thickness: int = 14,
    show_metrics: bool = True,
) -> np.ndarray:
    """Fill the ego lane and draw its boundaries thickly.

    Replaces two thin curves with the corridor itself. The difference is not
    decoration: a filled polygon shows at a glance whether the fit encloses the
    road the vehicle is actually on, which two separate curves do not -- a fit
    that has latched onto a kerb on one side looks perfectly reasonable as a
    curve and obviously wrong as a filled region.

    Falls back to drawing whichever boundaries exist when the lane is not fully
    fitted, because a single boundary is still worth seeing; only the fill needs
    both.
    """
    left, right = lanes.ego_left, lanes.ego_right
    if left is None and right is None:
        return image
    from models.lane_seg.fitting import fit_to_image_points

    vis = image.copy()

    # Shared range, so the polygon does not extrapolate one side past the other.
    if left is not None and right is not None:
        lo = max(left.y_range_m[0], right.y_range_m[0])
        hi = min(left.y_range_m[1], right.y_range_m[1])
        if hi > lo:
            lp = fit_to_image_points(left, camera, (lo, hi))
            rp = fit_to_image_points(right, camera, (lo, hi))
            if len(lp) >= 2 and len(rp) >= 2:
                poly = np.vstack([lp, rp[::-1]]).astype(np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [poly], LANE_AREA_COLOR)
                cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    for fit, colour in ((left, LANE_LEFT_COLOR), (right, LANE_RIGHT_COLOR)):
        if fit is None:
            continue
        pts = fit_to_image_points(fit, camera)
        if len(pts) < 2:
            continue
        cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                      colour, boundary_thickness, cv2.LINE_AA)

    if show_metrics:
        rows = []
        if lanes.curvature_radius_m is not None:
            r = lanes.curvature_radius_m
            # A straight road fits an enormous radius; printing "412739 m" is
            # noise, so anything beyond a kilometre reads as straight.
            rows.append("Curvature: straight" if not np.isfinite(r) or abs(r) > 1000
                        else f"Curvature radius: {abs(r):.0f} m")
        if lanes.ego_offset_m is not None:
            side = "right" if lanes.ego_offset_m > 0 else "left"
            rows.append(f"Offset from centre: {abs(lanes.ego_offset_m):.2f} m {side}")
        if lanes.lane_width_m is not None:
            rows.append(f"Lane width: {lanes.lane_width_m:.2f} m")
        for i, text in enumerate(rows):
            y = 34 + i * 30
            (tw, th), base = cv2.getTextSize(text, FONT, 0.62, 2)
            x = vis.shape[1] - tw - 18
            cv2.rectangle(vis, (x - 8, y - th - base - 2), (x + tw + 8, y + base),
                          (26, 26, 26), -1)
            cv2.putText(vis, text, (x, y), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_debug_strip(
    image: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    height: int = 150,
) -> np.ndarray:
    """Inset small diagnostic views along the top of the frame.

    The intermediate stages are what explain a bad lane fit -- a mask full of
    zebra stripes, or a bird's-eye view where the markings are not parallel.
    Keeping them beside the result means a failure can be diagnosed from the
    output image alone, without re-running anything.
    """
    if not panels:
        return image
    vis = image.copy()
    x = 10
    for label, panel in panels:
        if panel is None or panel.size == 0:
            continue
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        w = int(panel.shape[1] * height / panel.shape[0])
        small = cv2.resize(panel, (w, height))
        if x + w > vis.shape[1] - 10:
            break
        cv2.rectangle(small, (0, 0), (w - 1, height - 1), (200, 200, 200), 1)
        cv2.putText(small, label, (6, 16), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        vis[10:10 + height, x:x + w] = small
        x += w + 8
    return vis


def draw_distance_lines(
    image: np.ndarray,
    objects,
    camera: CameraModel,
    cfg: OverlayConfig,
    max_lines: int = 4,
    max_distance_m: float = 60.0,
    in_path_only: bool = False,
) -> np.ndarray:
    """Draw a line from the ego vehicle to each detected object, labelled in metres.

    The line runs from the camera's own ground projection -- the point directly
    beneath the vehicle's nose -- to the object's ground-contact point, which is
    the only part of a 2-D box that lies on the road and therefore the only part
    the homography can place. Drawing to the box centre instead would look
    tidier and be wrong by however tall the object is.

    Two limits, both about legibility rather than correctness. Only the nearest
    ``max_lines`` are drawn: a busy junction has twenty tracked objects and
    twenty lines is an unreadable fan. And nothing beyond ``max_distance_m``,
    because monocular distance error grows with the square of range and a
    confidently-labelled 90 m line implies a precision that does not exist.
    """
    usable = [
        o for o in objects
        if getattr(o, "distance_m", None) is not None
        and np.isfinite(o.distance_m)
        and 0 < o.distance_m <= max_distance_m
        and (o.in_path if in_path_only else True)
    ]
    if not usable:
        return image
    usable.sort(key=lambda o: o.distance_m)
    usable = usable[:max_lines]

    vis = image.copy()
    # Ego origin: the camera's ground projection, at zero lateral offset and a
    # short distance ahead. Exactly zero is on the horizon-free singular row, so
    # a small positive offset is used.
    origin = np.asarray(camera.ground_to_image(np.array([[0.0, 0.5]]))).reshape(-1, 2)
    if not len(origin) or not np.isfinite(origin[0]).all():
        ego_px = (vis.shape[1] // 2, vis.shape[0] - 1)
    else:
        ego_px = (int(round(origin[0, 0])), min(vis.shape[0] - 1, int(round(origin[0, 1]))))

    for obj in usable:
        box = np.asarray(obj.box_px).reshape(4)
        contact = (int(round(0.5 * (box[0] + box[2]))), int(round(box[3])))
        colour = label_color(obj.class_name)

        cv2.line(vis, ego_px, contact, colour, 2, cv2.LINE_AA)
        cv2.circle(vis, contact, 4, colour, -1, cv2.LINE_AA)

        text = f"{obj.distance_m:.1f} m"
        if getattr(obj, "ttc", None) is not None and np.isfinite(obj.ttc.ttc_s):
            text += f"  {obj.ttc.ttc_s:.1f}s"
        mid = ((ego_px[0] + contact[0]) // 2, (ego_px[1] + contact[1]) // 2)
        (tw, th), base = cv2.getTextSize(text, FONT, cfg.font_scale, 1)
        x = int(np.clip(mid[0] - tw // 2, 0, vis.shape[1] - tw - 6))
        y = int(np.clip(mid[1], th + 4, vis.shape[0] - 4))
        cv2.rectangle(vis, (x - 3, y - th - base), (x + tw + 3, y + base), (24, 24, 24), -1)
        cv2.putText(vis, text, (x, y), FONT, cfg.font_scale, colour, 1, cv2.LINE_AA)
    return vis


#: Smooth split rendering. Green, matching the convention for "this is the road
#: geometry" rather than "this is an object".
SPLIT_CURVE_COLOR = (80, 235, 80)


def draw_smooth_curve(
    image: np.ndarray,
    fit,
    camera: CameraModel,
    colour: tuple[int, int, int],
    thickness: int = 9,
    extend_near_m: float = 3.0,
    extend_far_m: float = 8.0,
    samples: int = 160,
) -> np.ndarray:
    """Draw one boundary as a long, smooth curve.

    Two differences from the plain fit rendering, both about legibility.

    **Sample density.** 160 points instead of 40, so the polyline reads as a
    curve rather than a chain of visible segments once it is drawn thickly.

    **Controlled extension.** The fitted range is only as long as the pixels that
    supported it, which on dashed markings is often a short stub in the middle of
    the image. The curve is extended a little past that range so it reaches the
    road it describes.

    The extension is deliberately small and the *extrapolated* part is drawn
    thinner. A quadratic run far beyond its support diverges fast, and a thick
    confident curve out there would be an assertion the fit cannot back. Near and
    far are extended by different amounts because the near end is bounded by the
    bonnet while the far end runs to the horizon.
    """
    from models.lane_seg.fitting import fit_to_image_points

    lo, hi = fit.y_range_m
    inner = fit_to_image_points(fit, camera, (lo, hi), n_samples=samples)
    if len(inner) >= 2:
        cv2.polylines(image, [inner.reshape(-1, 1, 2).astype(np.int32)], False,
                      colour, thickness, cv2.LINE_AA)

    for a, b in ((max(0.5, lo - extend_near_m), lo), (hi, hi + extend_far_m)):
        if b <= a:
            continue
        pts = fit_to_image_points(fit, camera, (a, b), n_samples=max(24, samples // 4))
        if len(pts) >= 2:
            cv2.polylines(image, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                          colour, max(2, thickness // 2), cv2.LINE_AA)
    return image


def draw_split_smooth(
    image: np.ndarray,
    split,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
    thickness: int = 9,
    show_label: bool = True,
) -> np.ndarray:
    """Render a road split as smooth curves along each diverging boundary.

    Replaces the earlier amber-highlight-plus-diamond treatment. That marked
    *where* the split was but drew the boundaries as short segments, so the fork
    itself -- two curves parting company -- was not visible. Long smooth curves
    show the divergence directly, which is the whole point of detecting it.
    """
    if split is None:
        return image
    vis = image.copy()
    diverging = {split.left_role, split.right_role}
    drawn = 0
    for fit in lanes.all_fits():
        if fit.role not in diverging:
            continue
        vis = draw_smooth_curve(vis, fit, camera, SPLIT_CURVE_COLOR, thickness)
        drawn += 1
    if not drawn:
        return image

    if show_label:
        text = f"ROAD SPLIT  {split.side.value}  {split.gore_distance_m:.0f} m"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.66, 2)
        x, y = 16, vis.shape[0] - 18
        cv2.rectangle(vis, (x - 9, y - th - base - 4), (x + tw + 9, y + base + 2),
                      (24, 24, 24), -1)
        cv2.putText(vis, text, (x, y), FONT, 0.66, SPLIT_CURVE_COLOR, 2, cv2.LINE_AA)
    return vis


def draw_split(
    image: np.ndarray,
    split,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
) -> np.ndarray:
    """Mark a detected road split: the gore nose and the diverging boundaries.

    Drawn distinctly from the ordinary lane curves. A fork changes what the lane
    geometry *means* -- across a gore the widening gap is not a wide lane -- so
    it needs to read as a different kind of thing, not as another boundary.
    """
    if split is None:
        return image
    from models.lane_seg.fitting import fit_to_image_points

    vis = image.copy()
    diverging = {split.left_role, split.right_role}
    for fit in lanes.all_fits():
        if fit.role not in diverging:
            continue
        pts = fit_to_image_points(fit, camera)
        if pts is None or len(pts) < 2:
            continue
        cv2.polylines(vis, [pts.reshape(-1, 1, 2).astype(np.int32)], False,
                      SPLIT_COLOR, cfg.lane_thickness + 3, cv2.LINE_AA)

    # A marker at the gore nose, where the road actually begins to divide.
    ground = np.array([[0.0, split.gore_distance_m]])
    px = np.asarray(camera.ground_to_image(ground)).reshape(-1, 2)
    if len(px) and np.isfinite(px[0]).all():
        x, y = int(round(px[0, 0])), int(round(px[0, 1]))
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.drawMarker(vis, (x, y), SPLIT_COLOR, cv2.MARKER_DIAMOND, 22, 3)
            text = f"SPLIT {split.side.value} {split.gore_distance_m:.0f}m"
            cv2.putText(vis, text, (x + 14, y), FONT, cfg.font_scale * 1.1,
                        SPLIT_COLOR, 2, cv2.LINE_AA)
    return vis


def draw_status_corner(
    image: np.ndarray,
    cfg: OverlayConfig,
    weather: str | None = None,
    road_condition: str | None = None,
    extra: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Draw per-frame classifications in the top-right corner.

    Kept away from the top-left HUD deliberately. The HUD carries continuously
    changing geometry that a reader scans; these are slow-changing scene labels
    that a reader only glances at, and mixing the two makes both harder to read.

    An absent classifier renders nothing rather than "unknown" -- a corner full
    of "unknown" trains the eye to skip the whole region, including the times it
    says something.
    """
    # "unknown" is what an absent or unconfident classifier reports, and it is
    # not worth a row: a corner permanently reading "unknown" trains the eye to
    # skip the whole region, including the frames where it says something real.
    def _known(v: str | None) -> bool:
        return bool(v) and str(v).lower() not in {"unknown", "none", "undefined"}

    rows: list[tuple[str, str]] = []
    if _known(weather):
        rows.append(("weather", str(weather)))
    if _known(road_condition):
        rows.append(("road", str(road_condition)))
    rows.extend(extra or [])
    if not rows:
        return image

    vis = image.copy()
    pad, line_h = 8, int(26 * max(0.6, cfg.font_scale / 0.5))
    scale = cfg.font_scale * 1.05
    width = max(
        cv2.getTextSize(f"{k}: {v}", FONT, scale, 1)[0][0] for k, v in rows
    ) + 2 * pad
    height = line_h * len(rows) + pad
    x0 = vis.shape[1] - width - 10
    y0 = 10

    panel = vis[y0:y0 + height, x0:x0 + width]
    if panel.size:
        dark = np.zeros_like(panel)
        cv2.addWeighted(dark, 0.55, panel, 0.45, 0, panel)
    for i, (k, v) in enumerate(rows):
        y = y0 + pad + line_h * i + int(line_h * 0.6)
        cv2.putText(vis, f"{k}: ", (x0 + pad, y), FONT, scale,
                    (170, 170, 170), 1, cv2.LINE_AA)
        kw = cv2.getTextSize(f"{k}: ", FONT, scale, 1)[0][0]
        cv2.putText(vis, v, (x0 + pad + kw, y), FONT, scale,
                    label_color(v), 1, cv2.LINE_AA)
    return vis


def lane_hud_lines(lanes: LaneSet) -> list[str]:
    """HUD rows describing lane geometry, honest about degraded states."""
    if lanes.status is LaneStatus.OK:
        radius = lanes.curvature_radius_m or float("inf")
        radius_text = "straight" if not np.isfinite(radius) or radius > 3000 else f"{radius:7.0f} m"
        return [
            f"lane   : OK  width {lanes.lane_width_m:.2f} m",
            f"curve  : {radius_text}  @ {lanes.eval_distance_m:.0f} m",
            f"offset : {lanes.ego_offset_m:+.2f} m from centre",
        ]
    if lanes.status is LaneStatus.SINGLE_BOUNDARY:
        side = "left" if lanes.ego_left is not None else "right"
        return [f"lane   : {side} boundary only", "curve  : n/a", "offset : n/a"]
    return [f"lane   : {lanes.status.value}", "curve  : n/a", "offset : n/a"]


def annotate(
    image: np.ndarray,
    result: FrameResult,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
    hud_lines: list[str] | None = None,
    warnings: list[str] | None = None,
    track_ids: dict[int, int] | None = None,
    distances_m: dict[int, float] | None = None,
    weather: str | None = None,
    road_condition: str | None = None,
) -> np.ndarray:
    """Compose the full annotation for one frame, in draw order.

    Masks first so boxes and curves sit on top of them; the fitted curves last of
    the geometry so they remain readable over the lane mask they are checked
    against.
    """
    vis = draw_mask(image, result.drivable_mask, cfg, DRIVABLE_COLOR)
    if cfg.show_lane_mask:
        vis = draw_mask(vis, result.lane_mask, cfg, LANE_MASK_COLOR, alpha=cfg.lane_mask_alpha)
    vis = draw_lane_fits(vis, lanes, camera, cfg)
    vis = draw_boxes(vis, result, cfg, distances_m=distances_m, track_ids=track_ids)
    vis = draw_status_corner(vis, cfg, weather, road_condition)
    return draw_hud(vis, hud_lines or [], cfg, warnings)


class VideoWriter:
    """Encode annotated frames to an MP4. Use as a context manager.

    The frame size is fixed by the first frame written; a later frame of a
    different size raises rather than being silently letterboxed or stretched.
    """

    def __init__(self, path: str | Path, fps: float, cfg: VideoConfig) -> None:
        if fps <= 0:
            raise ValueError(f"output fps must be > 0, got {fps}")
        self.path = Path(path)
        self.fps = float(cfg.fps or fps)
        self.cfg = cfg
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames_written = 0

    def __enter__(self) -> VideoWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, image: np.ndarray) -> None:
        """Append one BGR frame."""
        h, w = image.shape[:2]
        if self._writer is None:
            self._size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.fourcc)
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self._size)
            if not self._writer.isOpened():
                raise OSError(
                    f"could not open a video writer for {self.path} with fourcc "
                    f"'{self.cfg.fourcc}' at {w}x{h}. Try a different video.fourcc."
                )
            log_kv(
                _log, logging.INFO, "video.open",
                path=str(self.path), size=f"{w}x{h}",
                fps=f"{self.fps:.3f}", fourcc=self.cfg.fourcc,
            )
        elif (w, h) != self._size:
            raise ValueError(
                f"frame size changed mid-video: {self._size} -> {(w, h)}. The writer "
                f"is fixed at its first frame's size."
            )
        self._writer.write(image)
        self.frames_written += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()
        log_kv(
            _log, logging.INFO, "video.close",
            path=str(self.path), frames=self.frames_written,
            status="error" if exc_type else "ok",
        )


# A divided highway is rendered as two half-roads, not two curves. The ego
# carriageway keeps the familiar green; the opposing one is amber, the colour
# used everywhere else in this project for "traffic that is not yours". The
# divider itself is drawn in the yellow it is painted in on the road.
CARRIAGEWAY_EGO_COLOR = (90, 230, 90)        # BGR: green
CARRIAGEWAY_ONCOMING_COLOR = (60, 170, 255)  # BGR: amber
CARRIAGEWAY_DIVIDER_COLOR = (40, 215, 245)   # BGR: road yellow


def draw_carriageway(
    image: np.ndarray,
    carriageway,
    lanes: LaneSet,
    camera: CameraModel,
    cfg: OverlayConfig,
    alpha: float = 0.28,
    show_label: bool = True,
) -> np.ndarray:
    """Render a divided highway: ego side, divider, oncoming side.

    The two halves are tinted rather than outlined, because the useful fact is
    not where the line is but which region is which. A driver -- or a reviewer
    checking the output -- reads "the amber half is coming towards me" instantly
    and reads a line annotated ``divider`` not at all.

    The oncoming region is filled out to the frame edge beyond the divider. That
    is an honest depiction of what is known: the far carriageway's outer edge is
    usually unmarked or out of frame, so claiming a specific width for it would
    be drawing a measurement that was never taken.
    """
    if carriageway is None:
        return image
    from models.lane_seg.fitting import fit_to_image_points

    divider = next((f for f in lanes.all_fits()
                    if f.role == carriageway.divider_role), None)
    if divider is None:
        return image

    lo, hi = divider.y_range_m
    div_pts = fit_to_image_points(divider, camera, (lo, hi), n_samples=160)
    if len(div_pts) < 2:
        return image

    h, w = image.shape[:2]
    vis = image.copy()
    overlay = vis.copy()

    # The oncoming half: everything between the divider and the frame edge on
    # the divider's side. The region runs down to the bottom of the frame rather
    # than stopping at the near end of the fit -- the road does not end where
    # the fitted pixels do, and a horizontal stripe floating in the middle of
    # the image reads as a rendering artefact rather than as half a road.
    # Its upper edge follows the horizon rather than running flat across at the
    # divider's far row, which tinted whatever buildings happened to be there.
    # Nothing above the vanishing line is road, so nothing above it can be the
    # other carriageway.
    edge_x = 0 if carriageway.divider_side == "left" else w - 1
    v_horizon = float(np.asarray(camera.horizon_v(float(edge_x))).reshape(-1)[0])
    if not np.isfinite(v_horizon):
        v_horizon = float(div_pts[-1][1])
    v_top = float(np.clip(max(v_horizon, 0.0), 0, h - 1))
    poly = np.vstack([
        div_pts,                            # near to far along the divider
        [[edge_x, v_top]],                  # out to the frame edge at the horizon
        [[edge_x, h - 1]],                  # down the frame edge
        [[div_pts[0][0], h - 1]],           # back under the near end
    ]).astype(np.int32)
    cv2.fillPoly(overlay, [poly], CARRIAGEWAY_ONCOMING_COLOR)

    # The ego half: between the divider and whichever ego boundary is on the
    # other side of the vehicle. Skipped when that boundary was not fitted --
    # a tint with one edge invented would misreport the lane's width.
    other = (lanes.ego_right if carriageway.divider_side == "left"
             else lanes.ego_left)
    if other is not None and other.role != divider.role:
        olo = max(lo, other.y_range_m[0])
        ohi = min(hi, other.y_range_m[1])
        if ohi > olo:
            a = fit_to_image_points(divider, camera, (olo, ohi), n_samples=160)
            b = fit_to_image_points(other, camera, (olo, ohi), n_samples=160)
            if len(a) >= 2 and len(b) >= 2:
                cv2.fillPoly(overlay, [np.vstack([a, b[::-1]]).astype(np.int32)],
                             CARRIAGEWAY_EGO_COLOR)

    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # The divider drawn as a double line, which is what a real one usually is,
    # and which distinguishes it at a glance from an ordinary lane boundary.
    for shift in (-5, 5):
        shifted = div_pts.copy()
        shifted[:, 0] += shift
        cv2.polylines(vis, [shifted.reshape(-1, 1, 2).astype(np.int32)], False,
                      CARRIAGEWAY_DIVIDER_COLOR, 5, cv2.LINE_AA)

    if show_label:
        onc_side = "left" if carriageway.divider_side == "left" else "right"
        lines = [
            f"DIVIDED HIGHWAY  ({carriageway.kind.value.replace('_', ' ')})",
            f"oncoming traffic on the {onc_side}  |  "
            f"{carriageway.ego_lanes} ego / {carriageway.oncoming_lanes}+ opposing",
        ]
        if carriageway.median_width_m is not None:
            lines.append(f"median {carriageway.median_width_m:.1f} m wide")
        y0 = vis.shape[0] - 18 - 26 * (len(lines) - 1)
        for i, text in enumerate(lines):
            scale = 0.66 if i == 0 else 0.54
            colour = CARRIAGEWAY_DIVIDER_COLOR if i == 0 else (235, 235, 235)
            (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)
            x, y = 16, y0 + 26 * i
            cv2.rectangle(vis, (x - 9, y - th - base - 3), (x + tw + 9, y + base + 1),
                          (24, 24, 24), -1)
            cv2.putText(vis, text, (x, y), FONT, scale, colour, 2, cv2.LINE_AA)
    return vis


def carriageway_hud_lines(carriageway) -> list[str]:
    """One-line summary for the HUD stack."""
    if carriageway is None:
        return []
    return [f"road   : divided, oncoming {carriageway.divider_side}, "
            f"{carriageway.ego_lanes}+{carriageway.oncoming_lanes} lanes "
            f"(conf {carriageway.confidence:.2f})"]
