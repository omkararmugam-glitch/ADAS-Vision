"""Organise every detection capability into its own section of outputs/.

    python demo/build_sections.py --n 6

Produces ``outputs/sections/`` with one folder per capability::

    01_lane_detection/    lane pixels and the fitted boundaries, nothing else
    02_lane_split/        frames where a fork or exit ramp was found
    03_vehicles/          vehicle boxes only
    04_pedestrians/       people, cyclists, motorcyclists only
    05_traffic_control/   lights with the colour read from their pixels, and signs
    06_distance_ttc/      distance lines from the ego to each object
    07_tracking/          persistent track ids
    08_full_pipeline/     everything composited, as the pipeline really outputs it
    index.html            gallery linking all of them

One capability per image, deliberately. The composite proves the modules run
together, but on a busy junction every overlay lands on top of every other and a
*wrong* lane fit is indistinguishable from a right one. Isolated, each module's
output can be judged on its own -- which is how the zebra-crossing problem in the
lane segmenter became visible at all.

This is a generator, not a one-off tidy: rerunning it rebuilds every section from
the current code, so the folders cannot drift away from what the pipeline
actually does.
"""

from __future__ import annotations

import argparse
import html
import shutil
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
from models.lane_seg.split import detect_split
from models.perception.factory import build_detector
from models.perception.traffic_light import classify_lights
from tracking.bytetrack import ByteTracker
from tracking.collision import CollisionEstimator

LANE_PIXEL_COLOUR = (255, 90, 200)
VRU = {"person", "bicycle", "motorcycle"}
CONTROL = {"traffic_light", "traffic_sign"}

SECTIONS: list[tuple[str, str]] = [
    ("01_lane_detection", "Lane detection — segmented pixels and the fitted boundaries"),
    ("02_lane_split", "Road splits — forks, exit ramps, dividing carriageways"),
    ("03_vehicles", "Vehicle detection"),
    ("04_pedestrians", "Pedestrians, cyclists and motorcyclists"),
    ("05_traffic_control", "Traffic lights (with colour) and signs"),
    ("06_distance_ttc", "Distance from the ego vehicle, and time-to-collision"),
    ("07_tracking", "Multi-object tracking with persistent ids"),
    ("08_full_pipeline", "Everything composited, as the pipeline outputs it"),
]


def _titled(img: np.ndarray, title: str, note: str = "",
            note_colour=(150, 220, 150)) -> np.ndarray:
    bar = np.full((32, img.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    if note:
        (tw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(bar, note, (img.shape[1] - tw - 12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, note_colour, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _subset(result: FrameResult, keep_names: set[str]) -> FrameResult:
    """A FrameResult carrying only the named classes, so a panel shows one thing."""
    ids = np.asarray(result.class_ids)
    mask = np.array([class_name(int(c)) in keep_names for c in ids], dtype=bool)
    blank = np.zeros(result.lane_mask.shape, np.uint8)
    return FrameResult(
        boxes_px=np.asarray(result.boxes_px)[mask].astype(np.float32).reshape(-1, 4),
        class_ids=ids[mask].astype(np.int32),
        scores=np.asarray(result.scores)[mask].astype(np.float32),
        drivable_mask=blank, lane_mask=blank.copy(),
    )


def build_index(root: Path, counts: dict[str, int]) -> Path:
    cards = []
    for name, desc in SECTIONS:
        d = root / name
        shots = sorted(p.name for p in d.glob("*.jpg")) if d.is_dir() else []
        thumb = f"{name}/{shots[0]}" if shots else ""
        img = (f'<a href="{thumb}" target="_blank">'
               f'<img src="{thumb}" loading="lazy" alt="{html.escape(desc)}"></a>'
               if thumb else '<div class="empty">no frames</div>')
        cards.append(
            f'<figure>{img}<figcaption><b>{name.split("_", 1)[1].replace("_", " ")}</b>'
            f'<span>{html.escape(desc)}</span>'
            f'<em>{counts.get(name, 0)} image(s) &middot; {name}/</em>'
            f'</figcaption></figure>')

    return_html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADAS detection sections</title><style>
:root{{color-scheme:light;--bg:#f4f5f2;--surface:#fff;--sunk:#eceee8;--ink:#16191a;
--ink2:#454b4d;--muted:#6b7370;--line:#dcdfd8;--accent:#17864a;--warn:#a86a00}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--bg:#131614;--surface:#1b1f1c;
--sunk:#22271f;--ink:#ecefea;--ink2:#c2c9c1;--muted:#8e968d;--line:#2c322d;
--accent:#4ed07f;--warn:#e0a340}}}}
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1400px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:clamp(26px,4vw,40px);margin:0 0 10px;letter-spacing:-.02em}}
.sub{{color:var(--ink2);max-width:72ch;margin:0 0 22px}}
.banner{{padding:14px 18px;background:var(--sunk);border-left:3px solid var(--warn);
border-radius:0 4px 4px 0;margin:0 0 30px}}
.banner p{{margin:0;font-size:14.5px;color:var(--ink2);max-width:84ch}}
.banner strong{{color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:20px}}
figure{{margin:0;background:var(--surface);border:1px solid var(--line);
border-radius:6px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto}}
.empty{{padding:60px 0;text-align:center;color:var(--muted);background:var(--sunk)}}
figcaption{{padding:13px 15px;border-top:1px solid var(--line)}}
figcaption b{{display:block;font-size:16px;margin-bottom:4px;text-transform:capitalize}}
figcaption span{{display:block;font-size:14px;color:var(--ink2)}}
figcaption em{{display:block;margin-top:8px;font-style:normal;font-size:12.5px;
color:var(--muted);font-family:ui-monospace,Menlo,Consolas,monospace}}
a{{color:var(--accent)}}
</style></head><body><div class="wrap">
<h1>Detection sections</h1>
<p class="sub">Every capability the pipeline provides, each rendered on its own so
it can be judged separately. Click a thumbnail for full resolution; the folder
name under each card holds the rest.</p>
<div class="banner"><p><strong>One capability per image, deliberately.</strong>
The composite in section 8 proves the modules run together, but on a busy
junction every overlay lands on top of every other and a wrong lane fit looks
much like a right one. Isolated, each module can be judged on its own &mdash;
which is how the zebra-crossing weakness in the lane segmenter became visible.</p></div>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""
    out = root / "index.html"
    out.write_text(return_html, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/sections")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--lane-weights", type=Path, default=None)
    ap.add_argument("--n", type=int, default=6, help="frames to render per section")
    ap.add_argument("--scan", type=int, default=400, help="frames to scan for good examples")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--clean", action="store_true", help="wipe the sections first")
    args = ap.parse_args(argv)

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    for name, _ in SECTIONS:
        (args.out / name).mkdir(parents=True, exist_ok=True)

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
            print(f"learned lane segmenter unavailable ({exc}); using classical")
    print(f"lane source: {lane_source}")

    tracker = ByteTracker(cfg.tracker.model_copy(update={"n_init": 1}))
    collisions = CollisionEstimator(cfg.ttc, None)
    engine = DecisionEngine(cfg.fusion)
    light_id = CLASS_IDS["traffic_light"]

    paths = sorted(args.images.glob("*.jpg"))[: args.scan]
    counts = {name: 0 for name, _ in SECTIONS}
    scanned = 0

    for path in paths:
        if all(counts[n] >= args.n for n, _ in SECTIONS):
            break
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        scanned += 1
        h, w = frame.shape[:2]

        est = autocalib_estimate([frame])
        cam = (CameraModel(H_image_to_ground=est.H_image_to_ground, image_size=(w, h))
               if est is not None else None)
        collisions.camera = cam

        result = detector.infer(frame)
        lane = segmenter.infer_detailed(frame)
        result = FrameResult(
            boxes_px=result.boxes_px, class_ids=result.class_ids,
            scores=result.scores, drivable_mask=result.drivable_mask,
            lane_mask=lane.mask)
        lanes = (LaneFitter(cam, cfg.bev, cfg.lanes).fit(lane.mask) if cam
                 else LaneSet(status=LaneStatus.NOT_DETECTED))
        tracker.reset()
        tracks = tracker.update(result, 0, 0.0)
        objects = collisions.update(tracks, 0.0)
        split = detect_split(
            lanes,
            min_divergence_ratio=cfg.fusion.lane_change.split_min_ratio,
            min_widening_m=cfg.fusion.lane_change.split_min_widening_m)
        decision = engine.update(0, 0.0, objects, lanes)

        names = [class_name(int(c)) for c in result.class_ids]
        stem = path.stem

        # 01 lane detection
        if counts["01_lane_detection"] < args.n and lane.is_plausible:
            vis = frame.copy()
            vis[lane.mask > 0] = LANE_PIXEL_COLOUR
            if cam is not None:
                vis = draw_lane_fits(vis, lanes, cam, cfg.overlay)
            note = f"{lanes.status.value}"
            if lanes.lane_width_m:
                note += f"  width {lanes.lane_width_m:.2f} m"
            cv2.imwrite(str(args.out / "01_lane_detection" / f"{stem}.jpg"),
                        _titled(vis, f"lane detection ({lane_source})", note),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["01_lane_detection"] += 1

        # 02 lane split
        if counts["02_lane_split"] < args.n and split is not None and cam is not None:
            vis = draw_lane_fits(frame.copy(), lanes, cam, cfg.overlay)
            vis = draw_split(vis, split, lanes, cam, cfg.overlay)
            cv2.imwrite(str(args.out / "02_lane_split" / f"{stem}.jpg"),
                        _titled(vis, "road split",
                                f"{split.side.value} @ {split.gore_distance_m:.0f} m "
                                f"(x{split.divergence_ratio:.2f})", (90, 190, 255)),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["02_lane_split"] += 1

        # 03 vehicles
        n_veh = names.count("vehicle")
        if counts["03_vehicles"] < args.n and n_veh >= 3:
            vis = draw_boxes(frame.copy(), _subset(result, {"vehicle"}), cfg.overlay)
            cv2.imwrite(str(args.out / "03_vehicles" / f"{stem}.jpg"),
                        _titled(vis, "vehicles", f"{n_veh} detected"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["03_vehicles"] += 1

        # 04 pedestrians / VRU
        n_vru = sum(names.count(k) for k in VRU)
        if counts["04_pedestrians"] < args.n and n_vru >= 1:
            vis = draw_boxes(frame.copy(), _subset(result, VRU), cfg.overlay)
            cv2.imwrite(str(args.out / "04_pedestrians" / f"{stem}.jpg"),
                        _titled(vis, "pedestrians and cyclists", f"{n_vru} detected"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["04_pedestrians"] += 1

        # 05 traffic control
        n_ctrl = sum(names.count(k) for k in CONTROL)
        if counts["05_traffic_control"] < args.n and n_ctrl >= 1:
            vis = draw_boxes(frame.copy(), _subset(result, CONTROL), cfg.overlay)
            li = np.flatnonzero(np.asarray(result.class_ids) == light_id)
            states = (classify_lights(frame, np.asarray(result.boxes_px)[li])
                      if li.size else [])
            read = ", ".join(s.colour for s in states if s.colour != "unknown") or "unreadable"
            cv2.imwrite(str(args.out / "05_traffic_control" / f"{stem}.jpg"),
                        _titled(vis, "traffic lights and signs",
                                f"{n_ctrl} found; {read}"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["05_traffic_control"] += 1

        # 06 distance / TTC
        measured = [o for o in objects if o.distance_m is not None]
        if counts["06_distance_ttc"] < args.n and cam is not None and len(measured) >= 2:
            vis = draw_distance_lines(frame.copy(), objects, cam, cfg.overlay,
                                      max_lines=6)
            nearest = min(o.distance_m for o in measured)
            cv2.imwrite(str(args.out / "06_distance_ttc" / f"{stem}.jpg"),
                        _titled(vis, "distance to objects",
                                f"{len(measured)} measured; nearest {nearest:.1f} m"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["06_distance_ttc"] += 1

        # 07 tracking
        if counts["07_tracking"] < args.n and len(objects) >= 3:
            ids = {}
            det_boxes = np.asarray(result.boxes_px).reshape(-1, 4)
            from tracking.bytetrack import iou_matrix
            obj_boxes = np.asarray([o.box_px for o in objects]).reshape(-1, 4)
            ious = iou_matrix(det_boxes, obj_boxes)
            for d in range(len(det_boxes)):
                j = int(np.argmax(ious[d]))
                if ious[d, j] >= 0.5:
                    ids[d] = objects[j].track_id
            vis = draw_boxes(frame.copy(), result, cfg.overlay, track_ids=ids)
            cv2.imwrite(str(args.out / "07_tracking" / f"{stem}.jpg"),
                        _titled(vis, "tracking", f"{len(objects)} tracked ids"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["07_tracking"] += 1

        # 08 full pipeline
        if counts["08_full_pipeline"] < args.n and len(objects) >= 2:
            per_class: dict[str, int] = {}
            for nm in names:
                per_class[nm] = per_class.get(nm, 0) + 1
            hud = [stem,
                   "objects: " + ", ".join(f"{v} {k}" for k, v in sorted(per_class.items())),
                   *lane_hud_lines(lanes)]
            warns = [("!! " if a.severity.value == "critical" else "!  ")
                     + a.message.upper() for a in decision.alerts]
            vis = annotate(frame, result, lanes,
                           cam or CameraModel(H_image_to_ground=np.eye(3),
                                              image_size=(w, h)),
                           cfg.overlay, hud, warns)
            if cam is not None and objects:
                vis = draw_distance_lines(vis, objects, cam, cfg.overlay, max_lines=4)
            if split is not None and cam is not None:
                vis = draw_split(vis, split, lanes, cam, cfg.overlay)
            cv2.imwrite(str(args.out / "08_full_pipeline" / f"{stem}.jpg"),
                        _titled(vis, "full pipeline",
                                f"{len(decision.alerts)} alerts"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts["08_full_pipeline"] += 1

    detector.close()
    index = build_index(args.out, counts)

    print(f"scanned {scanned} frames")
    for name, desc in SECTIONS:
        print(f"  {name:22s} {counts[name]:2d} image(s)")
    print(f"\nindex -> {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
