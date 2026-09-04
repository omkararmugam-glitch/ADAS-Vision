"""Plot training curves from an Ultralytics results.csv.

Drawn with OpenCV rather than matplotlib to avoid adding a dependency for one
figure, and to keep the styling consistent with the rest of the project's
generated imagery.

Ultralytics writes its own ``results.png``; this exists because that plot shows
eight panels at thumbnail size, and the two curves anyone actually reads --
mAP@0.5 and mAP@0.5:0.95 against epoch -- deserve to be legible on their own.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

BG = (24, 26, 24)
GRID = (52, 58, 52)
INK = (238, 240, 236)
MUTED = (150, 158, 150)
SERIES = [
    ("metrics/mAP50(B)", "mAP@0.5", (110, 210, 80)),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95", (230, 160, 60)),
    ("metrics/precision(B)", "precision", (200, 120, 220)),
    ("metrics/recall(B)", "recall", (90, 175, 235)),
]


def read_results(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                key = k.strip()
                try:
                    cols.setdefault(key, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    return cols


def plot(
    cols: dict[str, list[float]], title: str, subtitle: str,
    width: int = 1100, height: int = 560,
) -> np.ndarray:
    img = np.full((height, width, 3), BG, np.uint8)
    L, R, T, B = 78, width - 210, 92, height - 62

    epochs = cols.get("epoch") or list(range(1, len(next(iter(cols.values()))) + 1))
    n = len(epochs)
    x_max = max(epochs) if n else 1

    def sx(e: float) -> int:
        return int(L + (e - 1) / max(1e-9, x_max - 1) * (R - L)) if x_max > 1 else L

    def sy(v: float) -> int:
        return int(B - v * (B - T))

    # grid + y labels
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = sy(v)
        cv2.line(img, (L, y), (R, y), GRID, 1)
        cv2.putText(img, f"{v:.1f}", (L - 46, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, MUTED, 1, cv2.LINE_AA)
    # x ticks
    step = max(1, int(round(x_max / 8)))
    for e in range(1, int(x_max) + 1, step):
        x = sx(e)
        cv2.line(img, (x, T), (x, B), GRID, 1)
        cv2.putText(img, str(e), (x - 6, B + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, MUTED, 1, cv2.LINE_AA)
    cv2.line(img, (L, B), (R, B), (90, 96, 90), 2)

    legend_y = T + 8
    for key, label, colour in SERIES:
        ys = cols.get(key)
        if not ys:
            continue
        pts = [(sx(epochs[i]), sy(ys[i])) for i in range(min(n, len(ys)))]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, colour, -1, cv2.LINE_AA)
        # direct label at the final point, so the legend is not load-bearing
        cv2.circle(img, (R + 18, legend_y), 5, colour, -1, cv2.LINE_AA)
        cv2.putText(img, f"{label}", (R + 30, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, INK, 1, cv2.LINE_AA)
        cv2.putText(img, f"{ys[-1]:.3f}", (R + 30, legend_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        legend_y += 52

    cv2.putText(img, title, (L - 40, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                INK, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (L - 40, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (130, 180, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "epoch", ((L + R) // 2 - 22, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path,
                    default=PROJECT / "outputs/train/distill_v8n/results.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--title", default="YOLOv8n distillation")
    ap.add_argument("--subtitle",
                    default="measured against PSEUDO-labels: agreement with the "
                            "teacher, NOT correctness")
    args = ap.parse_args(argv)

    if not args.results.is_file():
        raise SystemExit(f"not found: {args.results}")
    cols = read_results(args.results)
    if not cols.get("metrics/mAP50(B)"):
        raise SystemExit(f"{args.results} has no mAP column; found {list(cols)[:8]}")

    img = plot(cols, args.title, args.subtitle)
    out = args.out or args.results.parent / "curves.jpg"
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    n = len(cols["metrics/mAP50(B)"])
    best = max(cols["metrics/mAP50(B)"])
    print(f"{n} epochs, best mAP@0.5 {best:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
