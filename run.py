"""Command-line entry point for the offline ADAS pipeline.

Two subcommands exist at this stage::

    python run.py calibrate ...            # forwards to src/calibrate.py
    python run.py process --source <video> # cache per-frame results, write an MP4

``process`` is deliberately split in two passes over the same data, which is the
whole point of the caching layer: pass one runs the expensive per-frame work and
writes it to ``outputs/<run_id>/``; pass two reads that cache back and renders
the annotated video. Re-rendering with different overlay settings reuses the
cache, because overlay settings are excluded from the cache key. Changing
anything that affects the results themselves invalidates it, loudly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.cache import RunCache
from src.calibrate import CameraModel
from src.config import Config, load_config
from src.io import open_source
from src.logging_utils import StageTimer, get_logger, log_kv
from src.overlay import VideoWriter, draw_boxes, draw_hud, draw_mask, draw_polylines
from src.stubs import StubDetector
from src.types import FrameResult

_log = get_logger("run")


def _emitted_fps(source_fps: float, stride: int) -> float:
    """Frame rate of the emitted sequence after striding."""
    return source_fps / stride


def compute_pass(cfg: Config, source_path: str, cache: RunCache, camera: CameraModel) -> int:
    """Run per-frame inference and write it to the cache.

    Returns:
        Number of frames written.
    """
    detector_camera: CameraModel | None = None
    with open_source(source_path, cfg.io) as src, cache.writer() as writer:
        with StageTimer("compute", _log) as timer:
            for frame in src:
                if detector_camera is None:
                    # Bind the calibration to the canonical frame size once the
                    # first frame reveals it. Rescaling the homography is the
                    # only sanctioned way to read frames at another resolution.
                    detector_camera = camera.for_image_size(frame.size)
                    if detector_camera is not camera:
                        log_kv(
                            _log, logging.WARNING, "calibration.rescaled",
                            calibrated_for=f"{camera.image_size[0]}x{camera.image_size[1]}",
                            frames_are=f"{frame.size[0]}x{frame.size[1]}",
                            note="homography rescaled; metric accuracy degrades slightly",
                        )
                    detector = StubDetector(detector_camera, cfg.detector)
                result = detector.detect(frame)
                mask = detector.drivable_mask(frame.image.shape[0], frame.image.shape[1])
                writer.write(result, arrays={"drivable": mask})
                timer.tick()
            return timer.items


def _hud_lines(cfg: Config, result: FrameResult, camera: CameraModel) -> tuple[list[str], list[str]]:
    """Build the HUD status rows and warning rows for one cached frame.

    Distances are recomputed here from the cached box geometry, deliberately: it
    exercises the image -> ground path on the read side, and demonstrates that
    derived quantities can be recomputed from the cache without re-running any
    model. That is exactly what Layer 2 and Layer 3 will do.
    """
    lines = [
        f"frame {result.frame_index:>5}  t={result.timestamp_s:6.2f}s",
        f"objects {len(result.detections)}   lanes {len(result.lanes)}",
    ]
    warnings: list[str] = []
    for det in result.detections:
        contact = [det.bottom_centre_px]
        if not camera.is_on_ground(contact)[0]:
            lines.append(f"{det.label} #{det.track_id}: above horizon, no range")
            continue
        ground = camera.image_to_ground(contact)[0]
        lines.append(
            f"{det.label} #{det.track_id}: {ground[1]:5.1f} m fwd, {ground[0]:+5.2f} m lat"
        )
        truth = result.extras.get("stub_vehicle_distance_m")
        if truth is not None:
            # The stub placed the vehicle at a known range; recovering it through
            # the homography round-trip is the end-to-end geometry check.
            lines.append(f"  round-trip err: {abs(ground[1] - truth) * 100:.1f} cm")
        if ground[1] < 15.0:
            warnings.append(f"! CLOSE: {ground[1]:.1f} m")
    lines.append(f"model {cfg.cache.model_version}")
    return lines, warnings


def render_pass(cfg: Config, source_path: str, cache: RunCache, camera: CameraModel, out_path: Path) -> int:
    """Read cached results back and write the annotated MP4.

    Returns:
        Number of frames rendered.
    """
    results = {r.frame_index: r for r in cache.read()}
    render_camera: CameraModel | None = None
    with open_source(source_path, cfg.io) as src:
        fps = _emitted_fps(src.fps, cfg.io.stride)
        with VideoWriter(out_path, fps, cfg.video) as writer, StageTimer("render", _log) as timer:
            for frame in src:
                result = results.get(frame.index)
                if result is None:
                    raise ValueError(
                        f"frame {frame.index} is missing from the cache at "
                        f"{cache.run_dir}. The cache and the source disagree; "
                        f"rebuild with --refresh."
                    )
                if render_camera is None:
                    render_camera = camera.for_image_size(frame.size)

                vis = frame.image
                arrays = cache.read_arrays(frame.index)
                if "drivable" in arrays:
                    vis = draw_mask(vis, arrays["drivable"], cfg.overlay)
                vis = draw_polylines(vis, result.lanes, cfg.overlay)
                vis = draw_boxes(vis, result.detections, cfg.overlay)
                lines, warnings = _hud_lines(cfg, result, render_camera)
                vis = draw_hud(vis, lines, cfg.overlay, warnings)
                writer.write(vis)
                timer.tick()
            return timer.items


def cmd_process(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.max_frames is not None:
        cfg = cfg.model_copy(
            update={"io": cfg.io.model_copy(update={"max_frames": args.max_frames})}
        )
    if args.stride is not None:
        cfg = cfg.model_copy(update={"io": cfg.io.model_copy(update={"stride": args.stride})})

    camera = CameraModel.load(cfg.calibration.homography_path)
    cache = RunCache.open(
        cfg,
        args.source,
        extra={
            "homography_path": str(cfg.calibration.homography_path),
            "homography_residual_m": camera.residual_m,
            "undistort_applied": camera.undistort_applied,
            "calibrated_image_size": list(camera.image_size),
        },
    )

    if args.refresh:
        cache.invalidate()
    if cache.is_reusable():
        n_cached = cache.existing_manifest().frame_count
        log_kv(_log, logging.INFO, "compute.skipped", reason="cache reusable", frames=n_cached)
    else:
        compute_pass(cfg, args.source, cache, camera)

    out_path = Path(args.out) if args.out else cache.run_dir / "annotated.mp4"
    n_rendered = render_pass(cfg, args.source, cache, camera, out_path)

    manifest = cache.existing_manifest()
    print("\n" + "=" * 62)
    print(f"run id        : {cache.wanted.run_id}")
    print(f"cache dir     : {cache.run_dir}")
    print(f"  manifest    : {cache.manifest_path.name}  (config_hash={manifest.config_hash})")
    print(f"  frames      : {manifest.frame_count} cached, {n_rendered} rendered")
    print(f"  model       : {manifest.model_version}   schema v{manifest.schema_version}")
    print(f"annotated mp4 : {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 62)
    return 0


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

    p_cal = sub.add_parser(
        "calibrate",
        help="camera intrinsics / homography / verification (see `calibrate --help`)",
    )
    p_cal.add_argument("rest", nargs=argparse.REMAINDER)
    p_cal.set_defaults(func=cmd_calibrate)

    p_proc = sub.add_parser("process", help="cache per-frame results and render an MP4")
    p_proc.add_argument("--source", required=True, help="video file or image folder")
    p_proc.add_argument("--out", default=None, help="annotated MP4 path")
    p_proc.add_argument("--refresh", action="store_true", help="discard any existing cache")
    p_proc.add_argument("--max-frames", type=int, default=None, help="override io.max_frames")
    p_proc.add_argument("--stride", type=int, default=None, help="override io.stride")
    p_proc.set_defaults(func=cmd_process)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
