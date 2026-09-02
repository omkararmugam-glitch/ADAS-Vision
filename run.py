"""Command-line entry point for the offline ADAS pipeline.

    python run.py calibrate ...              # intrinsics / homography / verify
    python run.py predict --source <media>   # run models, populate the cache
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

from src.cache import RunCache
from src.calibrate import CameraModel, file_sha256_of_text
from src.config import Config, load_config
from src.detect import Detector, build_detector
from src.io import open_source
from src.lanes import LaneFitter
from src.logging_utils import StageTimer, get_logger, log_kv
from src.overlay import VideoWriter, annotate, lane_hud_lines
from src.types import FrameRecord, FrameResult, LaneStatus, class_name

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


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(load_config(args.config), args)
    camera = _load_camera(cfg)
    detector = build_detector(cfg.detector, camera)

    cache = _open_cache(cfg, args.source, camera, detector)
    if args.refresh:
        cache.invalidate()

    if cache.is_reusable():
        n = cache.existing_manifest().frame_count
        log_kv(_log, logging.INFO, "predict.skipped", reason="cache reusable", frames=n)
        _print_summary(cache, rendered=None)
        return 0

    resume_after = 0 if args.refresh else cache.resumable_frames()
    frame_camera: CameraModel | None = None
    fitter: LaneFitter | None = None
    counts: dict[str, int] = {}

    with open_source(args.source, cfg.io) as src, cache.writer(resume_after) as writer:
        with StageTimer("predict", _log) as timer:
            batch: list = []

            def flush() -> None:
                nonlocal batch
                if not batch:
                    return
                results = detector.infer_batch([f.image for f in batch])
                for frame, result in zip(batch, results):
                    lanes = fitter.fit(result.lane_mask)
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
                            extras={"was_resized": frame.was_resized},
                        ),
                        result,
                    )
                    timer.tick()
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
    log_kv(_log, logging.INFO, "predict.counts", **{k: v for k, v in sorted(counts.items())})
    _print_summary(cache, rendered=None, counts=counts)
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
    out_path = Path(args.out) if args.out else cache.run_dir / "annotated.mp4"
    render_camera: CameraModel | None = None
    frame_size = tuple(cache.existing_manifest().extra.get("frame_size", (0, 0)))

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
                hud, warnings = _hud(cfg, record, result)
                writer.write(
                    annotate(frame.image, result, record.lanes, render_camera, cfg.overlay, hud, warnings)
                )
                timer.tick()
            rendered = timer.items

    _print_summary(cache, rendered=rendered, out_path=out_path)
    return 0


def _hud(
    cfg: Config, record: FrameRecord, result: FrameResult
) -> tuple[list[str], list[str]]:
    """HUD rows for one cached frame.

    Object counts and lane geometry only. Distances, tracking and events are
    Stage 3; nothing here computes them.
    """
    by_class: dict[str, int] = {}
    for cid in result.class_ids:
        name = class_name(int(cid))
        by_class[name] = by_class.get(name, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(by_class.items())) or "none"

    lines = [
        f"frame {record.frame_index:>5}   t={record.timestamp_s:6.2f}s",
        f"objects: {summary}",
        *lane_hud_lines(record.lanes),
        f"model  : {cfg.detector.kind}",
    ]
    warnings: list[str] = []
    if record.lanes.status is LaneStatus.NOT_DETECTED:
        warnings.append("! NO LANE")
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
    from src.calibrate import main as calibrate_main

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
    p_pred.add_argument("--detector", default=None, choices=["stub", "yolopv2"])
    p_pred.set_defaults(func=cmd_predict)

    p_rend = sub.add_parser("render", help="annotate from the cache; loads no model")
    _common(p_rend)
    p_rend.add_argument("--out", default=None, help="annotated MP4 path")
    p_rend.add_argument("--detector", default=None, choices=["stub", "yolopv2"])
    p_rend.set_defaults(func=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
