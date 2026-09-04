"""Save annotated predictions to disk, for training and evaluation alike.

The gap this fills: the project reported loss curves and mAP but never wrote out
what the model was actually drawing on a frame. A falling loss and a rising mAP
are consistent with a model that has learned to put boxes in plausible-looking
wrong places, and no scalar shows that. Previews do.

Three entry points, one implementation:

* :func:`save_training_previews` -- a *fixed* set of validation frames, saved
  every N epochs. Fixed on purpose: the whole value is flicking between
  ``epoch_005/`` and ``epoch_030/`` for the same image and seeing what changed.
  Resampling each time would make the comparison meaningless.
* :func:`save_eval_previews` -- the validation split, or an evenly-spaced sample
  when it is large.
* :func:`contact_sheet` -- a grid of the above, for a quick look without opening
  a folder.

Ultralytics' own ``results.plot()`` is used for plain detection output, per the
brief: it is maintained, it matches what everyone expects a YOLO prediction to
look like, and hand-rolling it would be duplicated code with subtly different
colours. Custom drawing is used only where ``plot()`` cannot help -- fusion
alerts, lane geometry, track ids and the classification corner, none of which
Ultralytics knows about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

#: Where each stage writes. Documented in ARCHITECTURE.md; kept here as the
#: single source of truth so the two cannot drift apart.
TRAIN_PREVIEW_ROOT = PROJECT / "outputs/train_previews"
EVAL_PREVIEW_ROOT = PROJECT / "outputs/eval_previews"
DEMO_ROOT = PROJECT / "outputs/demo"


def pick_fixed_frames(image_dir: Path, n: int = 8) -> list[Path]:
    """Choose a stable, evenly-spaced sample of images.

    Evenly spaced over the sorted listing rather than random: it is
    reproducible without carrying a seed around, and it spreads across whatever
    ordering the dataset has (which for BDD100K is by image id, so effectively
    arbitrary with respect to content).
    """
    paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    step = len(paths) / n
    return [paths[int(i * step)] for i in range(n)]


def _banner(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    h = 34 if not sub else 52
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(bar, sub, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def save_yolo_previews(
    weights: Path,
    images: list[Path],
    out_dir: Path,
    conf: float = 0.25,
    title: str = "",
    subtitle: str = "",
) -> list[Path]:
    """Annotate images with a YOLO checkpoint, using Ultralytics' own plotting.

    ``results.plot()`` rather than custom drawing, deliberately: it is the
    library's maintained renderer, so the previews look like what anyone
    familiar with YOLO expects, and there is no second copy of box-drawing code
    to drift.
    """
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    written: list[Path] = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = model.predict(source=frame, conf=conf, verbose=False)[0]
        vis = result.plot()                       # Ultralytics' own renderer
        n = 0 if result.boxes is None else len(result.boxes)
        if title:
            vis = _banner(vis, f"{title}   {path.name}   {n} detections", subtitle)
        dst = out_dir / f"{path.stem}.jpg"
        cv2.imwrite(str(dst), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(dst)
    return written


def save_training_previews(
    weights: Path,
    image_dir: Path,
    epoch: int,
    n: int = 8,
    conf: float = 0.25,
    root: Path = TRAIN_PREVIEW_ROOT,
    label_source: str = "",
) -> list[Path]:
    """Annotate a fixed validation sample and write it under ``epoch_<N>/``."""
    images = pick_fixed_frames(image_dir, n)
    if not images:
        return []
    out = Path(root) / f"epoch_{epoch:03d}"
    sub = ("targets are PSEUDO-labels: agreement with the teacher, not correctness"
           if label_source == "pseudo" else "")
    return save_yolo_previews(weights, images, out, conf,
                              title=f"epoch {epoch}", subtitle=sub)


def save_eval_previews(
    weights: Path,
    image_dir: Path,
    out_dir: Path = EVAL_PREVIEW_ROOT,
    max_images: int = 200,
    conf: float = 0.25,
    label_source: str = "",
) -> list[Path]:
    """Annotate the validation split, sampling evenly when it is large."""
    images = pick_fixed_frames(image_dir, max_images)
    if not images:
        return []
    sub = ("targets are PSEUDO-labels: agreement with the teacher, not correctness"
           if label_source == "pseudo" else "")
    return save_yolo_previews(weights, images, Path(out_dir), conf,
                              title="eval", subtitle=sub)


def contact_sheet(
    paths: list[Path], out_path: Path, cols: int = 4, width: int = 480,
    title: str = "",
) -> Path | None:
    """Tile annotated frames into one image, for a glance without a file browser."""
    if not paths:
        return None
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = width / img.shape[1]
        tiles.append(cv2.resize(img, (width, int(img.shape[0] * scale))))
    if not tiles:
        return None
    height = max(t.shape[0] for t in tiles)
    tiles = [
        np.vstack([t, np.full((height - t.shape[0], width, 3), 24, np.uint8)])
        if t.shape[0] < height else t
        for t in tiles
    ]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full((height, width, 3), 24, np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    if title:
        sheet = _banner(sheet, title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path
