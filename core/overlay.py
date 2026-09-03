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
) -> np.ndarray:
    """Draw detection boxes with class, confidence and optional distance.

    Args:
        image: BGR frame; not modified.
        result: Detections in image pixels at this frame's size.
        cfg: Appearance settings.
        distances_m: Optional map from detection index to forward distance.
    """
    vis = image.copy()
    for i in range(len(result)):
        x1, y1, x2, y2 = (int(round(v)) for v in result.boxes_px[i])
        name = class_name(int(result.class_ids[i]))
        colour = label_color(name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, cfg.box_thickness)

        parts = [name, f"{result.scores[i]:.2f}"]
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
    vis = draw_boxes(vis, result, cfg)
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
