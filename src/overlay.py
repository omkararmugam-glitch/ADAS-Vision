"""Annotation rendering: boxes, masks, polylines, a text HUD, and MP4 output.

Everything here consumes :mod:`src.types` values in **image pixel** coordinates
at the canonical frame size, and draws on a copy -- no function mutates the frame
it is handed. Any metric text (distances, TTC) arrives pre-computed as strings or
floats; this module performs no geometry of its own.

Colours are BGR, matching OpenCV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from src.config import OverlayConfig, VideoConfig
from src.logging_utils import get_logger, log_kv
from src.types import Detection, LaneLine

_log = get_logger(__name__)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Per-class box colours (BGR). Unlisted classes fall back to LABEL_COLOR_DEFAULT.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 255),
    "truck": (0, 140, 255),
    "bus": (0, 100, 220),
    "motorcycle": (255, 180, 0),
    "bicycle": (255, 200, 80),
    "person": (60, 220, 60),
}
LABEL_COLOR_DEFAULT: tuple[int, int, int] = (200, 200, 200)

#: Per-role lane colours (BGR).
LANE_COLORS: dict[str, tuple[int, int, int]] = {
    "ego_left": (0, 255, 255),
    "ego_right": (0, 255, 255),
    "adjacent_left": (255, 160, 0),
    "adjacent_right": (255, 160, 0),
}
LANE_COLOR_DEFAULT: tuple[int, int, int] = (255, 255, 255)


def label_color(label: str) -> tuple[int, int, int]:
    """BGR colour for a detection class."""
    return LABEL_COLORS.get(label, LABEL_COLOR_DEFAULT)


def lane_color(role: str) -> tuple[int, int, int]:
    """BGR colour for a lane boundary role."""
    return LANE_COLORS.get(role, LANE_COLOR_DEFAULT)


def draw_boxes(
    image: np.ndarray,
    detections: list[Detection],
    cfg: OverlayConfig,
    distances_m: dict[int, float] | None = None,
) -> np.ndarray:
    """Draw detection boxes with class, score, track id and optional distance.

    Args:
        image: BGR frame; not modified.
        detections: Boxes in image pixels at this frame's size.
        cfg: Appearance settings.
        distances_m: Optional map from a detection's position in the list to a
            forward distance in metres, rendered in the label.

    Returns:
        A new annotated BGR image.
    """
    vis = image.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_px)
        colour = label_color(det.label)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, cfg.box_thickness)

        parts = [det.label]
        if det.track_id is not None:
            parts.append(f"#{det.track_id}")
        parts.append(f"{det.score:.2f}")
        if distances_m is not None and i in distances_m:
            parts.append(f"{distances_m[i]:.1f}m")
        text = " ".join(parts)

        (tw, th), baseline = cv2.getTextSize(text, FONT, cfg.font_scale, 1)
        # Keep the label inside the frame when the box touches the top edge.
        label_top = max(0, y1 - th - baseline - 2)
        cv2.rectangle(vis, (x1, label_top), (x1 + tw + 4, label_top + th + baseline + 2), colour, -1)
        cv2.putText(
            vis, text, (x1 + 2, label_top + th), FONT, cfg.font_scale, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Mark the ground contact point: the only part of a 2-D box on the road.
        cu, cv_ = det.bottom_centre_px
        cv2.drawMarker(
            vis, (int(round(cu)), int(round(cv_))), colour, cv2.MARKER_TRIANGLE_UP, 10, 2
        )
    return vis


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: OverlayConfig,
    color: tuple[int, int, int] = (0, 180, 0),
) -> np.ndarray:
    """Blend a translucent binary mask (e.g. drivable area) over the frame.

    Args:
        image: BGR frame; not modified.
        mask: 2-D array, non-zero where the overlay applies. Must match the
            frame's height and width.
        cfg: Supplies ``mask_alpha``.
        color: BGR fill colour.

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
    layer = np.zeros_like(vis)
    layer[selected] = color
    blended = cv2.addWeighted(vis, 1.0, layer, cfg.mask_alpha, 0.0)
    vis[selected] = blended[selected]
    return vis


def draw_polylines(
    image: np.ndarray, lanes: list[LaneLine], cfg: OverlayConfig
) -> np.ndarray:
    """Draw lane boundaries, dashing the ones detected as dashed markings."""
    vis = image.copy()
    for lane in lanes:
        pts = np.int32(np.round(lane.points_px))
        colour = lane_color(lane.role)
        if lane.kind == "dashed":
            # Render alternating segments so the marking type is visible at a glance.
            for i in range(0, len(pts) - 1, 2):
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), colour, cfg.lane_thickness)
        else:
            cv2.polylines(vis, [pts], False, colour, cfg.lane_thickness)
        if len(pts):
            cv2.putText(
                vis, f"{lane.role}/{lane.kind}", tuple(pts[0] + np.array([6, -6])),
                FONT, cfg.font_scale, colour, 1, cv2.LINE_AA,
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
        warnings: Optional rows rendered below the status block in red, for
            active events.
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
