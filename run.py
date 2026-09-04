"""Command-line entry point for the offline ADAS pipeline.

    python run.py calibrate ...              # intrinsics / homography / verify
    python run.py predict --source <media>   # run models, populate the cache
    python run.py analyse --source <media>   # cache -> tracks, TTC, alert log
    python run.py render  --source <media>   # cache -> annotated video

``predict`` and ``render`` are separate commands, not two halves of one, because
that separation is the point of the caching layer: ``render`` loads no model and
runs with the weights absent entirely. Restyling the overlay, or later retuning
event thresholds, must never re-run inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

#: Repo root, for placing deliverables outside the cache.
PROJECT_ROOT = Path(__file__).resolve().parent

from core.cache import RunCache
from core.calibrate import CameraModel, file_sha256_of_text
from core.config import Config, load_config
from models.perception import Detector, build_detector
from core.io import open_source
from models.lane_seg.fitting import LaneFitter
from core.logging_utils import StageTimer, get_logger, log_kv
from core.overlay import VideoWriter, annotate, lane_hud_lines
from core.types import CLASS_IDS, FrameRecord, FrameResult, LaneStatus, class_name
from fusion.engine import DecisionEngine
from models.perception.traffic_light import LightState, classify_lights
from models.weather_cls.classifier import WeatherClassifier
from fusion.log import AlertLog
from tracking.bytetrack import ByteTracker
from tracking.collision import CollisionEstimator

_log = get_logger("run")


def _load_camera(cfg: Config) -> CameraModel:
    """Load the homography. Raises if calibration is missing -- never defaults."""
    return CameraModel.load(cfg.calibration.homography_path)


def _open_cache(cfg: Config, source: str, camera: CameraModel, detector: Detector | None) -> RunCache:
    """Open the run cache for this config, calibration and model.

    When ``detector`` is None (the render path) the model identity is taken from
    the manifest already on disk, so that reading never requires loading weights.
    """
    calibration_hash = file_sha256_of_text(cfg.calibration.homography_path)
    if detector is not None:
        model_name, weights_hash = detector.name, detector.weights_hash
        class_names = detector.class_names
    else:
        run_id = cfg.cache.run_id or None
        probe = RunCache.open(
            cfg, source, calibration_hash, "", "", (), extra={}
        )
        existing = probe.existing_manifest()
        if existing is None:
            raise FileNotFoundError(
                f"no cache at {probe.run_dir}. Run `python run.py predict "
                f"--source {source}` first; render never loads a model."
            )
        model_name, weights_hash = existing.model_name, existing.weights_hash
        class_names = tuple(existing.class_names)
        _ = run_id
    return RunCache.open(
        cfg,
        source,
        calibration_hash=calibration_hash,
        model_name=model_name,
        weights_hash=weights_hash,
        class_names=class_names,
        extra={
            "homography_path": str(cfg.calibration.homography_path),
            "homography_residual_m": camera.residual_m,
            "undistort_applied": camera.undistort_applied,
            "calibrated_image_size": list(camera.image_size),
        },
    )


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #


def _light_readings(image, result: FrameResult) -> list[dict]:
    """Read the state of every traffic light in one frame.

    Returns a list of ``{box, colour, confidence}``, one per traffic-light
    detection, in detection order. The box travels with the reading so that
    `analyse` can attach a colour to the right track by overlap -- detection
    order alone would not survive tracking.
    """
    light_id = CLASS_IDS.get("traffic_light")
    if light_id is None or len(result) == 0:
        return []
    idx = np.flatnonzero(np.asarray(result.class_ids) == light_id)
    if idx.size == 0:
        return []
    boxes = np.asarray(result.boxes_px)[idx]
    states = classify_lights(image, boxes)
    return [
        {"box": [round(float(v), 1) for v in b],
         "colour": st.colour, "confidence": round(st.confidence, 3)}
        for b, st in zip(boxes, states)
    ]


def _build_weather(cfg: Config) -> WeatherClassifier | None:
    """Build the weather classifier, or None when it is off or unavailable.

    A missing checkpoint downgrades to None rather than raising: the fusion
    engine treats an unknown label as neutral, so losing weather means dry
    thresholds, not a dead pipeline.
    """
    if not cfg.weather.enabled:
        return None
    try:
        return WeatherClassifier(cfg.weather)
    except Exception as exc:                          # noqa: BLE001 - deliberate
        log_kv(_log, logging.WARNING, "weather.unavailable",
               error=str(exc)[:160],
               note="continuing with neutral (dry) collision thresholds")
        return None


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    camera = _load_camera(cfg)
    detector = build_detector(cfg.detector, camera)

    weather = _build_weather(cfg)
    cache = _open_cache(cfg, args.source, camera, detector)
    if args.refresh:
        cache.invalidate()

    if cache.is_reusable():
        n = cache.existing_manifest().frame_count
        log_kv(_log, logging.INFO, "predict.skipped", reason="cache reusable", frames=n)
        _print_summary(cache, rendered=None)
        return 0

    resume_after = 0 if args.refresh else cache.resumable_frames()
    # Weather changes over minutes, so it is classified every Nth frame and the
    # label carried forward. Cached per frame because it comes from a network
    # and `analyse` must not need one.
    last_weather = "unknown"
    frame_camera: CameraModel | None = None
    fitter: LaneFitter | None = None
    counts: dict[str, int] = {}

    with open_source(args.source, cfg.io) as src, cache.writer(resume_after) as writer:
        with StageTimer("predict", _log) as timer:
            batch: list = []

            def _set_last_weather(value: str) -> None:
                nonlocal last_weather
                last_weather = value

            def flush() -> None:
                nonlocal batch
                if not batch:
                    return
                results = detector.infer_batch([f.image for f in batch])
                nonlocal_weather = last_weather
                for frame, result in zip(batch, results):
                    lanes = fitter.fit(result.lane_mask)
                    if weather is not None and frame.index % cfg.weather.every_n_frames == 0:
                        nonlocal_weather = weather.infer_smoothed(frame.image).fusion_label
                    # Colour needs pixels, which only this stage has. Cached
                    # alongside the box so `analyse` never reopens the video.
                    lights = _light_readings(frame.image, result)
                    for cid in result.class_ids:
                        name = class_name(int(cid))
                        counts[name] = counts.get(name, 0) + 1
                    counts[f"lane:{lanes.status.value}"] = (
                        counts.get(f"lane:{lanes.status.value}", 0) + 1
                    )
                    writer.write(
                        FrameRecord(
                            frame_index=frame.index,
                            source_index=frame.source_index,
                            timestamp_s=frame.timestamp_s,
                            result=result,
                            lanes=lanes,
                            extras={"was_resized": frame.was_resized,
                                    "weather": nonlocal_weather,
                                    "lights": lights},
                        ),
                        result,
                    )
                    timer.tick()
                _set_last_weather(nonlocal_weather)
                batch = []

            for frame in src:
                if frame.index < resume_after:
                    continue
                if frame_camera is None:
                    frame_camera = camera.for_image_size(frame.size)
                    if frame_camera is not camera:
                        log_kv(
                            _log, logging.WARNING, "calibration.rescaled",
                            calibrated_for=f"{camera.image_size[0]}x{camera.image_size[1]}",
                            frames_are=f"{frame.size[0]}x{frame.size[1]}",
                            note="homography rescaled; metric accuracy degrades slightly",
                        )
                    fitter = LaneFitter(frame_camera, cfg.bev, cfg.lanes)
                    writer._manifest.extra["frame_size"] = list(frame.size)
                if hasattr(detector, "set_time"):
                    detector.set_time(frame.timestamp_s)
                    flush()  # the stub is time-dependent, so it cannot batch
                batch.append(frame)
                if len(batch) >= cfg.detector.batch_size:
                    flush()
            flush()

    detector.close()
    if weather is not None:
        weather.close()
    log_kv(_log, logging.INFO, "predict.counts", **{k: v for k, v in sorted(counts.items())})
    _print_summary(cache, rendered=None, counts=counts)
    return 0


# --------------------------------------------------------------------------- #
# analyse
# --------------------------------------------------------------------------- #


def analyse_cache(cfg: Config, cache: RunCache, camera: CameraModel) -> AlertLog:
    """Run tracking, collision estimation and the decision engine over a cache.

    Loads no model: this is post-processing over detections that already exist,
    which is why the tracker, TTC and fusion config sections are excluded from
    the cache key. Retuning an alert threshold costs seconds, not an inference
    run.
    """
    tracker = ByteTracker(cfg.tracker)
    collisions = CollisionEstimator(cfg.ttc, camera)
    engine = DecisionEngine(cfg.fusion)
    log = AlertLog()

    for record in cache.read(with_masks=False):
        engine.set_weather(record.extras.get("weather", "unknown"))
        tracks = tracker.update(record.result, record.frame_index, record.timestamp_s)
        objects = collisions.update(tracks, record.timestamp_s)
        objects = _attach_light_states(objects, record.extras.get("lights", []))
        log.add(engine.update(
            record.frame_index, record.timestamp_s, objects, record.lanes
        ))
    return log


def _attach_light_states(objects: list, lights: list[dict]) -> list:
    """Attach cached colour readings to the traffic-light objects they belong to.

    Matched by box overlap rather than by index: the tracker reorders objects and
    drops unconfirmed ones, so detection order does not survive it. An unmatched
    light keeps ``light_state=None``, which the engine reads as "not readable"
    and never as green.
    """
    from dataclasses import replace

    from models.perception.traffic_light import LightState
    from tracking.bytetrack import iou_matrix

    if not lights:
        return objects
    boxes = np.asarray([l["box"] for l in lights], dtype=np.float64)
    out = []
    for obj in objects:
        if obj.class_name != "traffic_light":
            out.append(obj)
            continue
        ious = iou_matrix(np.asarray(obj.box_px).reshape(1, 4), boxes)[0]
        best = int(np.argmax(ious))
        if ious[best] < 0.3:
            out.append(obj)
            continue
        rec = lights[best]
        out.append(replace(obj, light_state=LightState(
            colour=rec["colour"], confidence=float(rec["confidence"]))))
    return out


def cmd_analyse(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    camera = _load_camera(cfg)
    cache = _open_cache(cfg, args.source, camera, detector=None)

    reason = cache.staleness_reason()
    if reason is not None:
        raise ValueError(
            f"cannot analyse the cache at {cache.run_dir}: {reason}. "
            f"Re-run `predict` to rebuild it."
        )

    frame_size = tuple(cache.existing_manifest().extra.get("frame_size", (0, 0)))
    if frame_size != (0, 0):
        camera = camera.for_image_size(frame_size)

    with StageTimer("analyse", _log) as timer:
        log = analyse_cache(cfg, cache, camera)
        for _ in log.decisions:
            timer.tick()

    out_dir = Path(args.out_dir) if args.out_dir else cache.run_dir
    json_path = log.write_json(out_dir / "alerts.json", source=str(args.source))
    csv_path = log.write_csv(out_dir / "alerts.csv")
    ep_path = log.write_episodes_csv(out_dir / "episodes.csv")

    counts = log.counts_by_type()
    episodes = log.episodes()
    log_kv(_log, logging.INFO, "analyse.done",
           frames=len(log.decisions), alerts=len(log.alerts), episodes=len(episodes))

    print(chr(10) + "=" * 68)
    print(f"analysed        : {len(log.decisions)} frames from {cache.run_dir}")
    print(f"  alerts        : {len(log.alerts)} on individual frames")
    print(f"  episodes      : {len(episodes)} distinct events")
    if counts:
        for k, v in sorted(counts.items()):
            print(f"      {k:24s} {v:6d} frames")
    else:
        print("      (no alerts raised)")
    if episodes:
        print("  timeline      :")
        for e in episodes[:20]:
            tid = f" track {e.track_id}" if e.track_id is not None else ""
            print(f"      {e.start_s:7.2f}-{e.end_s:6.2f}s  {e.type.value:20s}"
                  f" {e.peak_severity.value:8s}{tid}")
        if len(episodes) > 20:
            print(f"      ... and {len(episodes) - 20} more")
    print(f"  json          : {json_path}")
    print(f"  csv           : {csv_path}")
    print(f"  episodes csv  : {ep_path}")
    print("=" * 68)
    return 0


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def cmd_render(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    camera = _load_camera(cfg)
    cache = _open_cache(cfg, args.source, camera, detector=None)

    reason = cache.staleness_reason()
    if reason is not None:
        raise ValueError(
            f"cannot render from the cache at {cache.run_dir}: {reason}. "
            f"Re-run `predict` to rebuild it."
        )

    records = {r.frame_index: r for r in cache.read(with_masks=False)}
    # Default into outputs/demo/ rather than beside the cache: the annotated
    # video is a deliverable someone looks at, not a cache artifact, and burying
    # it in a hashed run directory is why it kept getting overlooked.
    if args.out:
        out_path = Path(args.out)
    else:
        demo_dir = PROJECT_ROOT / "outputs/demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        out_path = demo_dir / f"{Path(str(args.source)).stem}_annotated.mp4"
    render_camera: CameraModel | None = None
    frame_size = tuple(cache.existing_manifest().extra.get("frame_size", (0, 0)))

    # Tracking, TTC and alerts are recomputed here rather than read from a file,
    # because they are post-processing over the same cache and cost seconds. That
    # keeps `render` self-contained: it never depends on `analyse` having been run,
    # and it still loads no model.
    decisions: dict[int, object] = {}
    objects_by_frame: dict[int, list] = {}
    if not args.no_alerts:
        alert_camera = camera.for_image_size(frame_size) if frame_size != (0, 0) else camera
        tracker = ByteTracker(cfg.tracker)
        collisions = CollisionEstimator(cfg.ttc, alert_camera)
        engine = DecisionEngine(cfg.fusion)
        for idx in sorted(records):
            rec = records[idx]
            engine.set_weather(rec.extras.get("weather", "unknown"))
            tracks = tracker.update(rec.result, rec.frame_index, rec.timestamp_s)
            objs = collisions.update(tracks, rec.timestamp_s)
            objects_by_frame[idx] = objs
            decisions[idx] = engine.update(
                rec.frame_index, rec.timestamp_s, objs, rec.lanes
            )

    with open_source(args.source, cfg.io) as src:
        fps = src.fps / cfg.io.stride
        with VideoWriter(out_path, fps, cfg.video) as writer, StageTimer("render", _log) as timer:
            for frame in src:
                record = records.get(frame.index)
                if record is None:
                    raise ValueError(
                        f"frame {frame.index} is missing from the cache at "
                        f"{cache.run_dir}. Re-run predict with --refresh."
                    )
                if render_camera is None:
                    render_camera = camera.for_image_size(frame.size)

                drivable, lane = cache.read_masks(frame.index, frame_size or frame.size)
                result = FrameResult(
                    boxes_px=record.result.boxes_px,
                    class_ids=record.result.class_ids,
                    scores=record.result.scores,
                    drivable_mask=drivable,
                    lane_mask=lane,
                )
                objs = objects_by_frame.get(frame.index, [])
                hud, warnings = _hud(cfg, record, result,
                                     decisions.get(frame.index), objs)
                track_ids, distances = _overlay_maps(result, objs)
                writer.write(annotate(
                    frame.image, result, record.lanes, render_camera, cfg.overlay,
                    hud, warnings,
                    track_ids=track_ids, distances_m=distances,
                    weather=record.extras.get("weather") or None,
                    road_condition=record.extras.get("road_condition") or None,
                ))
                timer.tick()
            rendered = timer.items

    _print_summary(cache, rendered=rendered, out_path=out_path)
    return 0


def _overlay_maps(result: FrameResult, objects: list) -> tuple[dict, dict]:
    """Map detection index -> track id and distance, for the box labels.

    Matched by box overlap rather than by index: tracking reorders objects and
    drops unconfirmed ones, so a positional correspondence with the frame's
    detections does not survive it.
    """
    from tracking.bytetrack import iou_matrix

    if not objects or len(result) == 0:
        return {}, {}
    det = np.asarray(result.boxes_px, dtype=np.float64).reshape(-1, 4)
    obj = np.asarray([o.box_px for o in objects], dtype=np.float64).reshape(-1, 4)
    ious = iou_matrix(det, obj)
    track_ids: dict[int, int] = {}
    distances: dict[int, float] = {}
    for i in range(len(det)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] < 0.5:
            continue
        track_ids[i] = objects[j].track_id
        if objects[j].distance_m is not None:
            distances[i] = float(objects[j].distance_m)
    return track_ids, distances


def _hud(
    cfg: Config,
    record: FrameRecord,
    result: FrameResult,
    decision: object | None = None,
    objects: list | None = None,
) -> tuple[list[str], list[str]]:
    """HUD rows and warning banners for one cached frame."""
    by_class: dict[str, int] = {}
    for cid in result.class_ids:
        name = class_name(int(cid))
        by_class[name] = by_class.get(name, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(by_class.items())) or "none"

    lines = [
        f"frame {record.frame_index:>5}   t={record.timestamp_s:6.2f}s",
        f"objects: {summary}",
        *lane_hud_lines(record.lanes),
    ]

    # The nearest in-path object is the one the driver needs a number for;
    # listing every track would bury it.
    nearest = None
    for o in objects or []:
        if not o.in_path or o.distance_m is None:
            continue
        if nearest is None or o.distance_m < nearest.distance_m:
            nearest = o
    if nearest is not None:
        ttc = nearest.ttc.ttc_s
        ttc_txt = f"{ttc:4.1f}s" if ttc != float("inf") else "  --"
        lines.append(
            f"lead   : #{nearest.track_id} {nearest.class_name} "
            f"{nearest.distance_m:5.1f}m  ttc {ttc_txt} ({nearest.ttc.source})"
        )

    lines.append(f"model  : {cfg.detector.kind}")

    warnings: list[str] = []
    if record.lanes.status is LaneStatus.NOT_DETECTED:
        warnings.append("! NO LANE")
    if decision is not None:
        for a in decision.alerts:                      # already worst-first
            prefix = "!! " if a.severity.value == "critical" else "!  "
            warnings.append(f"{prefix}{a.message.upper()}")
    return lines, warnings


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides onto the loaded config."""
    io_updates: dict[str, object] = {}
    if getattr(args, "max_frames", None) is not None:
        io_updates["max_frames"] = args.max_frames
    if getattr(args, "stride", None) is not None:
        io_updates["stride"] = args.stride
    if getattr(args, "start_frame", None) is not None:
        io_updates["start_frame"] = args.start_frame
    if io_updates:
        cfg = cfg.model_copy(update={"io": cfg.io.model_copy(update=io_updates)})
    det_updates: dict[str, object] = {}
    if getattr(args, "device", None) is not None:
        det_updates["device"] = args.device
    if getattr(args, "detector", None) is not None:
        det_updates["kind"] = args.detector
    if det_updates:
        cfg = cfg.model_copy(update={"detector": cfg.detector.model_copy(update=det_updates)})
    return cfg


def _print_summary(
    cache: RunCache, rendered: int | None, out_path: Path | None = None, counts: dict | None = None
) -> None:
    m = cache.existing_manifest()
    report = cache.size_report()
    print("\n" + "=" * 68)
    print(f"run id          : {m.run_id}")
    print(f"cache dir       : {cache.run_dir}")
    print(f"  frames        : {m.frame_count}" + (f", {rendered} rendered" if rendered else ""))
    print(f"  model         : {m.model_name}   weights {m.weights_hash[:12]}")
    print(f"  classes       : {', '.join(m.class_names)}")
    print(f"  config hash   : {m.config_hash}   calib {m.calibration_hash[:12]}")
    print(f"  code version  : {m.code_version}   schema v{m.schema_version}")
    print("  cache size    :")
    print(f"    masks       : {report['masks_bytes_per_frame']:9.0f} B/frame")
    print(f"    json        : {report['json_bytes_per_frame']:9.0f} B/frame")
    print(f"    TOTAL       : {report['total_bytes_per_frame']:9.0f} B/frame"
          f"   ({report['total_mb']:.1f} MB for {int(report['frames'])} frames)")
    if "raw_fullres_bytes_per_frame" in report:
        raw_full = report["raw_fullres_bytes_per_frame"]
        raw_native = report["raw_native_bytes_per_frame"]
        if raw_full:
            print(f"    (raw uint8 at full res would be {raw_full:.0f} B/frame; "
                  f"at native res {raw_native:.0f} B/frame)")
    if counts:
        print("  detections    : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if out_path is not None:
        print(f"annotated video : {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 68)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Forward everything after ``calibrate`` to the calibration CLI."""
    from core.calibrate import main as calibrate_main

    return calibrate_main(["--config", args.config, *args.rest])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Offline vision-based ADAS pipeline."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="camera intrinsics / homography / verification")
    p_cal.add_argument("rest", nargs=argparse.REMAINDER)
    p_cal.set_defaults(func=cmd_calibrate)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", required=True, help="video file or image folder")
        p.add_argument("--max-frames", type=int, default=None, help="override io.max_frames")
        p.add_argument("--stride", type=int, default=None, help="override io.stride")
        p.add_argument("--start-frame", type=int, default=None, help="override io.start_frame")

    p_pred = sub.add_parser("predict", help="run models and populate the cache")
    _common(p_pred)
    p_pred.add_argument("--refresh", action="store_true", help="discard any existing cache")
    p_pred.add_argument("--device", default=None, help="override detector.device")
    p_pred.add_argument("--detector", default=None, choices=["stub", "yolopv2", "yolov8"])
    p_pred.set_defaults(func=cmd_predict)

    p_ana = sub.add_parser(
        "analyse", help="cache -> tracks, TTC and alert log; loads no model")
    _common(p_ana)
    p_ana.add_argument("--out-dir", default=None,
                       help="where to write alerts.json/csv (default: the cache dir)")
    p_ana.set_defaults(func=cmd_analyse)

    p_rend = sub.add_parser("render", help="annotate from the cache; loads no model")
    _common(p_rend)
    p_rend.add_argument("--out", default=None, help="annotated MP4 path")
    p_rend.add_argument("--no-alerts", action="store_true",
                        help="draw detections and lanes only, without running fusion")
    p_rend.add_argument("--detector", default=None, choices=["stub", "yolopv2", "yolov8"])
    p_rend.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
