"""Annotate frames with EVERY module, not just the detector.

    python eval/full_preview.py --images data/raw/images --n 8

``eval/preview.py`` renders detector output alone, which is right for watching a
detector train but wrong as a picture of the system: it shows boxes and nothing
else, so lanes, signals and tracking are invisible. This runs the whole stack
over a sequence and composites the result:

  lane markings      classical segmentation + the metric fit on top
  vehicles / people  detector boxes, class and confidence
  traffic lights     box plus the colour read from its pixels
  track ids          persistent #id per object, from ByteTrack
  distance           metres, where calibration is available
  weather / road     top-right corner
  fusion alerts      top-left, over the geometry HUD

Frames are processed **in order and as a sequence**, because three of those
outputs do not exist for a single image: a track id needs a previous frame, a
time-to-collision needs a rate of change, and a lane-departure event needs a
history. Running this over shuffled stills would silently produce ids that never
persist and TTCs that never converge.
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
from core.overlay import annotate, lane_hud_lines
from core.types import CLASS_IDS, FrameResult, LaneStatus, class_name
from fusion.engine import DecisionEngine
from models.lane_seg.classical import ClassicalLaneSegmenter
from models.lane_seg.fitting import LaneFitter
from models.perception.factory import build_detector
from models.perception.traffic_light import classify_lights
from tracking.bytetrack import ByteTracker, iou_matrix
from tracking.collision import CollisionEstimator


def _sequence(image_dir: Path, n: int) -> list[Path]:
    """Consecutive frames from one BDD100K video where possible.

    BDD100K filenames carry the video id as their first field, so frames from
    one clip share a prefix. Tracking a sequence of *unrelated* stills would
    produce a new id for every object on every frame, which looks like a
    tracking bug and is really a sampling one.
    """
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        return []
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(p.name.split("-")[0], []).append(p)
    best = max(groups.values(), key=len)
    return (best if len(best) >= n else paths)[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/full_previews")
    ap.add_argument("--weights", type=Path, default=None,
                    help="detector checkpoint; defaults to the configured one")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args(argv)

    frames_paths = _sequence(args.images, args.n)
    if not frames_paths:
        raise SystemExit(f"no images under {args.images}")
    frames = [cv2.imread(str(p)) for p in frames_paths]
    frames = [f for f in frames if f is not None]
    h, w = frames[0].shape[:2]
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(PROJECT / "configs/default.yaml")
    det_cfg = cfg.detector.model_copy(update={"kind": "yolov8", "conf_threshold": args.conf})
    if args.weights:
        det_cfg = det_cfg.model_copy(
            update={"yolov8": det_cfg.yolov8.model_copy(update={"weights": args.weights})})
    detector = build_detector(det_cfg)
    segmenter = ClassicalLaneSegmenter()

    # Calibration from the frames themselves. Without it there are no metres,
    # so no distance, no TTC and no lane geometry -- the overlay would fall back
    # to boxes alone, which is the very gap this script exists to close.
    est = autocalib_estimate(frames[: min(8, len(frames))])
    camera = None
    if est is not None:
        camera = CameraModel(H_image_to_ground=est.H_image_to_ground, image_size=(w, h))
        print(f"auto-calibrated: horizon {est.horizon_v:.0f} px, "
              f"pitch {np.rad2deg(est.pitch_rad):+.1f} deg")
    else:
        print("! auto-calibration refused on these frames; "
              "lane geometry and distance will be unavailable")

    fitter = LaneFitter(camera, cfg.bev, cfg.lanes) if camera else None
    tracker = ByteTracker(cfg.tracker)
    collisions = CollisionEstimator(cfg.ttc, camera)
    engine = DecisionEngine(cfg.fusion)
    light_id = CLASS_IDS["traffic_light"]

    written: list[Path] = []
    for i, (path, frame) in enumerate(zip(frames_paths, frames)):
        result = detector.infer(frame)

        # Lane mask from classical segmentation, folded into the FrameResult so
        # the overlay and the fitter see it exactly as they would a learned one.
        mask_result = segmenter.infer_detailed(frame)
        result = FrameResult(
            boxes_px=result.boxes_px, class_ids=result.class_ids,
            scores=result.scores, drivable_mask=result.drivable_mask,
            lane_mask=mask_result.mask,
        )
        lanes = fitter.fit(mask_result.mask) if fitter else None
        if lanes is None:
            from core.types import LaneSet
            lanes = LaneSet(status=LaneStatus.NOT_DETECTED)

        timestamp = i / 30.0
        tracks = tracker.update(result, i, timestamp)
        objects = collisions.update(tracks, timestamp)

        # Traffic-light colours, attached to the matching tracked object.
        li = np.flatnonzero(np.asarray(result.class_ids) == light_id)
        light_boxes = np.asarray(result.boxes_px)[li] if li.size else np.zeros((0, 4))
        states = classify_lights(frame, light_boxes) if li.size else []
        if states:
            from dataclasses import replace
            obj_boxes = np.asarray([o.box_px for o in objects]).reshape(-1, 4)
            if len(obj_boxes):
                ious = iou_matrix(light_boxes, obj_boxes)
                for k, st in enumerate(states):
                    j = int(np.argmax(ious[k]))
                    if ious[k, j] >= 0.3 and objects[j].class_name == "traffic_light":
                        objects[j] = replace(objects[j], light_state=st)

        decision = engine.update(i, timestamp, objects, lanes)

        # Map detections to track ids and distances by overlap, not index.
        track_ids, distances = {}, {}
        if objects:
            det_boxes = np.asarray(result.boxes_px).reshape(-1, 4)
            obj_boxes = np.asarray([o.box_px for o in objects]).reshape(-1, 4)
            ious = iou_matrix(det_boxes, obj_boxes)
            for d in range(len(det_boxes)):
                j = int(np.argmax(ious[d]))
                if ious[d, j] < 0.5:
                    continue
                track_ids[d] = objects[j].track_id
                if objects[j].distance_m is not None:
                    distances[d] = float(objects[j].distance_m)

        counts: dict[str, int] = {}
        for cid in result.class_ids:
            n = class_name(int(cid))
            counts[n] = counts.get(n, 0) + 1
        lights_txt = ", ".join(f"{s.colour}" for s in states if s.colour != "unknown")

        hud = [
            f"frame {i:>3}   {path.name}",
            "objects: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none"),
            *lane_hud_lines(lanes),
            f"lane px: {mask_result.coverage * 100:.2f}% of road ROI",
        ]
        if lights_txt:
            hud.append(f"lights : {lights_txt}")
        if objects:
            nearest = min((o for o in objects if o.distance_m is not None),
                          key=lambda o: o.distance_m, default=None)
            if nearest is not None:
                ttc = nearest.ttc.ttc_s
                hud.append(f"lead   : #{nearest.track_id} {nearest.class_name} "
                           f"{nearest.distance_m:.1f}m "
                           f"ttc {ttc:.1f}s" if np.isfinite(ttc) else
                           f"lead   : #{nearest.track_id} {nearest.class_name} "
                           f"{nearest.distance_m:.1f}m")

        warnings = [("!! " if a.severity.value == "critical" else "!  ") + a.message.upper()
                    for a in decision.alerts]

        vis = annotate(
            frame, result, lanes, camera or CameraModel(
                H_image_to_ground=np.eye(3), image_size=(w, h)),
            cfg.overlay, hud, warnings,
            track_ids=track_ids, distances_m=distances,
            weather=None, road_condition=None,
        )
        dst = args.out / f"{i:03d}_{path.stem}.jpg"
        cv2.imwrite(str(dst), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(dst)
        print(f"  {dst.name}: {len(result)} objects, "
              f"{len(track_ids)} tracked, lane {lanes.status.value}, "
              f"{len(decision.alerts)} alerts")

    detector.close()
    print(f"\n{len(written)} fully annotated frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
