"""Compare detectors side by side on the same frames.

    python eval/compare_detectors.py --models yolov8n.pt <trained>.pt yolov8m.pt

Renders each frame once per model, stacked in a column, so a difference is
visible rather than inferred from two numbers. Also reports speed and detection
counts per model.

Intended use is the distillation check: stock student, trained student, teacher.
If the trained student sits closer to the teacher than the stock student does,
distillation worked -- and that is the honest claim available without ground
truth. It says the student learned to imitate; it does not say either model is
correct.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.config import load_config
from core.types import class_name
from models.perception.factory import build_detector
from tracking.bytetrack import iou_matrix

COLOUR = {"vehicle": (70, 220, 70), "person": (0, 170, 255),
          "bicycle": (255, 190, 0), "motorcycle": (255, 110, 210)}


def annotate(img, result, title, subtitle=""):
    vis = img.copy()
    for b, cid, s in zip(result.boxes_px, result.class_ids, result.scores):
        name = class_name(int(cid))
        c = COLOUR.get(name, (200, 200, 200))
        x0, y0, x1, y1 = (int(v) for v in b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2)
        lab = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ly = y0 - 4 if y0 - th - 6 >= 0 else y1 + th + 6
        cv2.rectangle(vis, (x0, ly - th - 4), (x0 + tw + 5, ly + 2), c, -1)
        cv2.putText(vis, lab, (x0 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
    bar = np.full((32, vis.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(bar, subtitle, (330, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 220, 120), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def agreement(a, b, iou_thr: float = 0.5) -> float:
    """Fraction of b's boxes matched by one of a's, at the given IoU.

    A crude but honest similarity: it says how much of the reference's output the
    candidate reproduces, ignoring class. Used only to compare two models to a
    third, never as a correctness measure.
    """
    if len(b) == 0:
        return 1.0 if len(a) == 0 else 0.0
    if len(a) == 0:
        return 0.0
    m = iou_matrix(np.asarray(a.boxes_px), np.asarray(b.boxes_px))
    return float((m.max(axis=0) >= iou_thr).mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True, help="weight paths, in column order")
    ap.add_argument("--labels", nargs="*", default=None, help="display names")
    ap.add_argument("--images", type=Path,
                    default=PROJECT / "data/processed/bdd100k_pseudo/images/val")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/compare")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--reference", type=int, default=-1,
                    help="index of the model others are compared against (default: last)")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args(argv)

    paths = sorted(Path(args.images).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    rng = random.Random(args.seed)
    sample = rng.sample(paths, min(args.n, len(paths)))
    labels = args.labels or [Path(m).stem for m in args.models]
    if len(labels) != len(args.models):
        raise SystemExit("--labels must match --models in length")

    args.out.mkdir(parents=True, exist_ok=True)
    base = load_config(PROJECT / "configs/default.yaml").detector

    imgs = [cv2.imread(str(p)) for p in sample]
    imgs = [im for im in imgs if im is not None]

    per_model_results, stats = [], []
    for weights, label in zip(args.models, labels):
        cfg = base.model_copy(update={
            "kind": "yolov8", "conf_threshold": args.conf,
            "yolov8": base.yolov8.model_copy(update={"weights": Path(weights)}),
        })
        det = build_detector(cfg)
        for _ in range(2):
            det.infer_batch(imgs[:2])                     # warm up
        t0 = time.perf_counter()
        results = det.infer_batch(imgs)
        ms = 1000 * (time.perf_counter() - t0) / max(1, len(imgs))
        counts: Counter = Counter()
        for r in results:
            counts.update(class_name(int(c)) for c in r.class_ids)
        det.close()
        per_model_results.append(results)
        stats.append({"label": label, "weights": weights, "ms": ms,
                      "counts": counts, "total": sum(counts.values())})

    ref = args.reference % len(args.models)
    for s, results in zip(stats, per_model_results):
        s["agreement"] = float(np.mean([
            agreement(r, ref_r) for r, ref_r in zip(results, per_model_results[ref])
        ]))

    # one stacked column per frame
    for i, im in enumerate(imgs):
        col = []
        for s, results in zip(stats, per_model_results):
            n = len(results[i])
            col.append(annotate(im, results[i], s["label"], f"{n} objects"))
        sheet = np.vstack(col)
        scale = min(1.0, 1400 / sheet.shape[1])
        if scale < 1.0:
            sheet = cv2.resize(sheet, None, fx=scale, fy=scale)
        cv2.imwrite(str(args.out / f"compare_{i:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("=" * 74)
    print(f"{'model':22s} {'objects':>8s} {'ms/frame':>9s} {'agree w/ ' + labels[ref]:>18s}")
    print("-" * 74)
    for s in stats:
        print(f"{s['label']:22s} {s['total']:8d} {s['ms']:9.2f} {s['agreement']:17.1%}")
    print("=" * 74)
    for s in stats:
        print(f"  {s['label']:20s} " + ", ".join(f"{k}={v}" for k, v in s["counts"].most_common()))
    print(f"\nwrote {len(imgs)} comparisons -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
