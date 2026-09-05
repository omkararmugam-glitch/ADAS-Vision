"""Render each module into its own panel, instead of one crowded frame.

    python eval/panel_preview.py --images data/raw/images --n 4

``eval/full_preview.py`` composites everything onto a single image. That proves
the modules run together, but on a busy junction the result is unreadable: boxes,
lane curves, distance lines, the gore marker and the HUD all overlap, and a wrong
lane fit is indistinguishable from a right one under the clutter.

This lays the same frame out as a grid, one module per panel:

    original          |  lane pixels        |  lane fit
    detections        |  distance/tracking  |  road split ahead
    divided highway   |  everything composited

Each panel carries only its own ink, so a bad lane fit is obvious *as a bad lane
fit* rather than as general mess. The composite is kept as the last panel,
because the modules genuinely do interact and the combined view is what the
pipeline actually produces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.autocalib import estimate as autocalib_estimate
from core.calibrate import CameraModel
from core.config import load_config
from core.overlay import (
    annotate,
    draw_carriageway,
    draw_boxes,
    draw_distance_lines,
    draw_lane_fits,
    draw_split,
    lane_hud_lines,
)
from core.types import CLASS_IDS, FrameResult, LaneSet, LaneStatus, class_name
from fusion.engine import DecisionEngine
from models.lane_seg.classical import ClassicalLaneSegmenter
from models.lane_seg.fitting import LaneFitter
from models.lane_seg.carriageway import detect_carriageway
from models.lane_seg.split import detect_split
from models.perception.factory import build_detector
from models.perception.traffic_light import classify_lights
from tracking.bytetrack import ByteTracker, iou_matrix
from tracking.collision import CollisionEstimator

LANE_PIXEL_COLOUR = (255, 90, 200)
HEADER_H = 30


def _panel(img: np.ndarray, title: str, note: str = "",
           note_colour=(150, 220, 150)) -> np.ndarray:
    """One titled panel. The note carries the module's own verdict."""
    bar = np.full((HEADER_H, img.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if note:
        (tw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, note, (img.shape[1] - tw - 12, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, note_colour, 1, cv2.LINE_AA)
    out = np.vstack([bar, img])
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (60, 60, 60), 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/panels")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--lane-weights", type=Path, default=None,
                    help="learned lane segmenter; falls back to classical")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--panel-width", type=int, default=560)
    args = ap.parse_args(argv)

    paths = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    h, w = frames[0].shape[:2]
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(PROJECT / "configs/default.yaml")
    det_cfg = cfg.detector.model_copy(
        update={"kind": "yolov8", "conf_threshold": args.conf})
    if args.weights:
        det_cfg = det_cfg.model_copy(
            update={"yolov8": det_cfg.yolov8.model_copy(update={"weights": args.weights})})
    detector = build_detector(det_cfg)

    segmenter = ClassicalLaneSegmenter()
    lane_source = "classical"
    if args.lane_weights and Path(args.lane_weights).is_file():
        try:
            from models.lane_seg.segmenter import LearnedLaneSegmenter
            segmenter = LearnedLaneSegmenter(args.lane_weights)
            lane_source = "learned"
        except Exception as exc:                      # noqa: BLE001
            print(f"learned segmenter unavailable ({exc}); using classical")
    print(f"lane source: {lane_source}")

    tracker = ByteTracker(cfg.tracker.model_copy(update={"n_init": 1}))
    collisions = CollisionEstimator(cfg.ttc, None)
    engine = DecisionEngine(cfg.fusion)
    light_id = CLASS_IDS["traffic_light"]

    for idx, (path, frame) in enumerate(zip(paths, frames)):
        est = autocalib_estimate([frame])
        cam = (CameraModel(H_image_to_ground=est.H_image_to_ground, image_size=(w, h))
               if est is not None else None)
        collisions.camera = cam

        result = detector.infer(frame)
        lane = segmenter.infer_detailed(frame)
        result = FrameResult(
            boxes_px=result.boxes_px, class_ids=result.class_ids,
            scores=result.scores, drivable_mask=result.drivable_mask,
            lane_mask=lane.mask,
        )
        lanes = (LaneFitter(cam, cfg.bev, cfg.lanes).fit(lane.mask) if cam
                 else LaneSet(status=LaneStatus.NOT_DETECTED))
        tracker.reset()
        tracks = tracker.update(result, idx, idx / 30.0)
        objects = collisions.update(tracks, idx / 30.0)

        li = np.flatnonzero(np.asarray(result.class_ids) == light_id)
        boxes_l = np.asarray(result.boxes_px)[li] if li.size else np.zeros((0, 4))
        states = classify_lights(frame, boxes_l) if li.size else []
        split = detect_split(
            lanes,
            min_divergence_ratio=cfg.fusion.lane_change.split_min_ratio,
            min_widening_m=cfg.fusion.lane_change.split_min_widening_m)
        carriageway = detect_carriageway(
            lanes, cam, lane.yellow_mask,
            exclude_boxes_px=np.asarray(result.boxes_px).reshape(-1, 4))
        decision = engine.update(idx, idx / 30.0, objects, lanes)

        counts: dict[str, int] = {}
        for c in result.class_ids:
            n = class_name(int(c))
            counts[n] = counts.get(n, 0) + 1

        # ---- panels -----------------------------------------------------
        p_original = _panel(frame.copy(), "1. input frame", path.name, (170, 170, 170))

        lane_vis = frame.copy()
        lane_vis[lane.mask > 0] = LANE_PIXEL_COLOUR
        p_mask = _panel(
            lane_vis, f"2. lane pixels ({lane_source})",
            f"{lane.coverage * 100:.2f}% of frame"
            + ("" if lane.is_plausible else "  IMPLAUSIBLE"),
            (150, 220, 150) if lane.is_plausible else (110, 140, 240))

        fit_vis = frame.copy()
        if cam is not None:
            fit_vis = draw_lane_fits(fit_vis, lanes, cam, cfg.overlay)
        p_fit = _panel(fit_vis, "3. lane fit",
                       lanes.status.value + (f"  width {lanes.lane_width_m:.2f} m"
                                             if lanes.lane_width_m else ""),
                       (150, 220, 150) if lanes.status is LaneStatus.OK
                       else (110, 140, 240))

        det_vis = draw_boxes(frame.copy(), result, cfg.overlay)
        p_det = _panel(det_vis, "4. detections",
                       ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none")

        dist_vis = frame.copy()
        if cam is not None and objects:
            dist_vis = draw_distance_lines(dist_vis, objects, cam, cfg.overlay,
                                           max_lines=6)
        nearest = min((o.distance_m for o in objects if o.distance_m is not None),
                      default=None)
        p_dist = _panel(dist_vis, "5. distance and tracking",
                        f"{len(objects)} tracked"
                        + (f", nearest {nearest:.1f} m" if nearest else ""))

        split_vis = frame.copy()
        if split is not None and cam is not None:
            split_vis = draw_lane_fits(split_vis, lanes, cam, cfg.overlay)
            split_vis = draw_split(split_vis, split, lanes, cam, cfg.overlay)
        p_split = _panel(
            split_vis, "6. road split",
            (f"{split.side.value} @ {split.gore_distance_m:.0f} m "
             f"(x{split.divergence_ratio:.2f})") if split else "no split",
            (90, 190, 255) if split else (150, 150, 150))

        cw_vis = frame.copy()
        if carriageway is not None and cam is not None:
            cw_vis = draw_carriageway(cw_vis, carriageway, lanes, cam, cfg.overlay,
                                      show_label=False)
        p_cw = _panel(
            cw_vis, "7. divided highway",
            (f"oncoming {carriageway.divider_side}, "
             f"{carriageway.ego_lanes}+{carriageway.oncoming_lanes} lanes")
            if carriageway else "not divided",
            (40, 215, 245) if carriageway else (150, 150, 150))

        hud = [f"{path.name}",
               "objects: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
                              or "none"),
               *lane_hud_lines(lanes)]
        if states:
            lit = ", ".join(s.colour for s in states if s.colour != "unknown")
            if lit:
                hud.append(f"lights : {lit}")
        warnings = [("!! " if a.severity.value == "critical" else "!  ")
                    + a.message.upper() for a in decision.alerts]
        comp = annotate(frame, result, lanes,
                        cam or CameraModel(H_image_to_ground=np.eye(3), image_size=(w, h)),
                        cfg.overlay, hud, warnings)
        if cam is not None and objects:
            comp = draw_distance_lines(comp, objects, cam, cfg.overlay, max_lines=4)
        p_all = _panel(comp, "8. everything composited",
                       f"{len(decision.alerts)} alerts",
                       (110, 140, 240) if decision.alerts else (150, 150, 150))

        panels = [p_original, p_mask, p_fit, p_det, p_dist, p_split,
                  p_cw, p_all]
        scale = args.panel_width / panels[0].shape[1]
        panels = [cv2.resize(p, (args.panel_width, int(p.shape[0] * scale)))
                  for p in panels]
        blank = np.full_like(panels[0], 18)
        while len(panels) % 3:
            panels.append(blank)
        rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
        sheet = np.vstack(rows)

        dst = args.out / f"{idx:02d}_{path.stem}_panels.jpg"
        cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {dst.name}: lane={lanes.status.value}, "
              f"{len(objects)} tracked, split={'yes' if split else 'no'}, "
              f"divided={'yes' if carriageway else 'no'}, "
              f"{len(decision.alerts)} alerts")

    detector.close()
    print(f"\npanel sheets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
