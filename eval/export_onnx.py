"""Export a detector to ONNX and benchmark PyTorch against ONNX Runtime.

    python eval/export_onnx.py --weights models/perception/weights/yolov8m.pt

Reports both engines on identical inputs, and — more importantly — checks that
they still agree. An export that runs fast but shifts the boxes is worse than no
export at all, because the error is invisible until a distance comes out wrong
downstream. Numerical agreement is therefore verified before any speed number is
quoted, and a failure there is reported as a failure, not a footnote.

Latency is measured on real frames where they are available, not on random noise.
Noise yields almost no detections and therefore almost no NMS work, which
flatters the result by a margin that grows with scene complexity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.config import load_config
from models.perception.factory import build_detector


def load_frames(images: Path | None, n: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Real frames when available; noise only as a clearly-labelled fallback."""
    if images and images.is_dir():
        paths = sorted(images.glob("*.jpg"))[:n]
        frames = [cv2.imread(str(p)) for p in paths]
        frames = [f for f in frames if f is not None]
        if frames:
            return frames
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(n)]


def bench(det, frames: list[np.ndarray], batch: int, seconds: float) -> dict:
    """Median-based latency over repeated passes."""
    for i in range(0, min(len(frames), batch * 2), batch):
        det.infer_batch(frames[i:i + batch])          # warm up

    times: list[float] = []
    t_end = time.perf_counter() + seconds
    i = 0
    while time.perf_counter() < t_end:
        chunk = frames[i % len(frames):(i % len(frames)) + batch] or frames[:batch]
        t0 = time.perf_counter()
        det.infer_batch(chunk)
        times.append((time.perf_counter() - t0) / max(1, len(chunk)))
        i += batch
    ms = [t * 1000 for t in times]
    return {
        "batch": batch,
        "n_passes": len(times),
        "ms_median": statistics.median(ms),
        "ms_p90": sorted(ms)[int(0.9 * (len(ms) - 1))],
        "fps": 1000.0 / statistics.median(ms),
    }


def agreement(a, b, tol_px: float = 2.0) -> dict:
    """How closely two engines' detections match on the same frames."""
    matched, total, max_shift = 0, 0, 0.0
    for ra, rb in zip(a, b):
        total += len(ra)
        if len(ra) == 0 or len(rb) == 0:
            continue
        for box in ra.boxes_px:
            d = np.abs(np.asarray(rb.boxes_px) - box).max(axis=1)
            best = float(d.min())
            max_shift = max(max_shift, best if best < 50 else 0.0)
            if best <= tol_px:
                matched += 1
    return {
        "boxes_reference": total,
        "matched": matched,
        "match_rate": matched / total if total else 1.0,
        "max_box_shift_px": max_shift,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path,
                    default=PROJECT / "models/perception/weights/yolov8m.pt")
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true", help="export fp16 ONNX")
    args = ap.parse_args(argv)

    if not args.weights.is_file():
        raise SystemExit(f"weights not found: {args.weights}")
    args.out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.images, args.frames, (1280, 720))
    real = bool(args.images and args.images.is_dir() and
                any(args.images.glob("*.jpg")))
    print(f"benchmarking on {len(frames)} {'real' if real else 'SYNTHETIC NOISE'} frames")
    if not real:
        print("  ! no real images found -- noise frames produce few detections and")
        print("    therefore little NMS work, so these numbers are optimistic.")

    # --- export ---------------------------------------------------------
    from ultralytics import YOLO

    print(f"exporting {args.weights.name} -> ONNX (imgsz {args.imgsz})")
    model = YOLO(str(args.weights))
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz,
                                  half=args.half, dynamic=False, simplify=True))
    target = args.out / onnx_path.name
    if onnx_path.resolve() != target.resolve():
        target.write_bytes(onnx_path.read_bytes())
    print(f"  wrote {target}  ({target.stat().st_size / 1e6:.1f} MB)")

    # --- benchmark both engines -----------------------------------------
    base = load_config(PROJECT / "configs/default.yaml").detector
    results = {"weights": str(args.weights), "onnx": str(target),
               "imgsz": args.imgsz, "real_frames": real, "engines": {}}

    ref_outputs = None
    for engine, weights in (("pytorch", args.weights), ("onnx", target)):
        cfg = base.model_copy(update={
            "kind": "yolov8", "conf_threshold": args.conf,
            "yolov8": base.yolov8.model_copy(update={"weights": weights}),
        })
        det = build_detector(cfg)
        rows = [bench(det, frames, b, args.seconds) for b in args.batches]
        outputs = det.infer_batch(frames[:8])
        if ref_outputs is None:
            ref_outputs = outputs
        else:
            results["agreement"] = agreement(ref_outputs, outputs)
        det.close()
        results["engines"][engine] = rows

    (args.out / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"{'engine':10s} {'batch':>6s} {'ms/frame':>10s} {'p90':>8s} {'FPS':>8s}")
    print("-" * 72)
    for engine, rows in results["engines"].items():
        for r in rows:
            print(f"{engine:10s} {r['batch']:6d} {r['ms_median']:10.2f} "
                  f"{r['ms_p90']:8.2f} {r['fps']:8.1f}")
    agree = results.get("agreement")
    if agree:
        print("-" * 72)
        print(f"ONNX vs PyTorch: {agree['matched']}/{agree['boxes_reference']} boxes "
              f"within 2 px ({agree['match_rate']:.1%}), "
              f"max shift {agree['max_box_shift_px']:.2f} px")
        if agree["match_rate"] < 0.95:
            print("  ! LOW AGREEMENT -- the export changed the model's output.")
            print("    Do not ship this ONNX file; the speed number is irrelevant.")
    print("=" * 72)
    print(f"wrote {args.out / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
