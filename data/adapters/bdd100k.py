"""Convert BDD100K detection labels into this project's YOLO format.

    python data/adapters/bdd100k.py \
        --images data/raw/images \
        --labels data/raw/det_train.json \
        --out data/processed/bdd100k

BDD100K ships images and labels as separate downloads, and the images alone
carry no supervision at all. This adapter needs the label JSON
(``det_train.json`` from ``bdd100k_det_20_labels_trainval.zip``, or
``bdd100k_labels_images_train.json`` from the 2018 release -- both are read).

Beyond format conversion it does **curation**, because 70,000 frames is far more
than is needed and an unfiltered subset wastes the budget on frames that teach
nothing. Selection is by measured content: night fraction, required classes,
minimum object size. See ``--help`` and ``docs/DATASET_FORMAT.md``.

Class mapping is deliberate and lossy:

  car, bus, truck  -> vehicle      one class, matching what YOLOPv2 emits, so
                                   the two backbones stay comparable
  person, rider    -> person       a rider is a person; the bike beneath them is
                                   labelled separately
  bike             -> bicycle
  motor            -> motorcycle
  traffic light    -> traffic_light (optional; colour kept as an attribute)
  traffic sign     -> traffic_sign  (optional)
  train            -> dropped      rare, and not a road vehicle for our purposes

Only the ``train`` split of the images exists locally, so the validation split is
carved out of it here, by image, with a fixed seed. Splitting by image rather
than at random within a clip matters: BDD100K frames come from distinct videos,
and leaking frames of the same scene across the split would inflate every metric.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ADAPTER_VERSION = 1

#: BDD100K category -> this project's class name. Absent categories are dropped.
CATEGORY_MAP: dict[str, str] = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "person": "person",
    "rider": "person",
    "bike": "bicycle",
    "motor": "motorcycle",
}

#: Added only when --with-traffic-control is passed, because they extend the
#: pipeline's class vocabulary and therefore its cache schema.
TRAFFIC_MAP: dict[str, str] = {
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}

BASE_CLASSES = ["vehicle", "person", "bicycle", "motorcycle"]
TRAFFIC_CLASSES = ["traffic_light", "traffic_sign"]


@dataclass
class Record:
    """One image's labels, already mapped to project classes."""

    name: str
    weather: str = "undefined"
    timeofday: str = "undefined"
    scene: str = "undefined"
    #: ``(class_name, x1, y1, x2, y2)`` in pixels.
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    #: Traffic-light colours seen, for the colour-classifier question.
    light_colours: Counter = field(default_factory=Counter)

    @property
    def is_night(self) -> bool:
        return self.timeofday == "night"

    def counts(self) -> Counter:
        return Counter(b[0] for b in self.boxes)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #


def load_labels(path: Path, class_map: dict[str, str]) -> list[Record]:
    """Read a BDD100K label JSON into records.

    Handles both the 2018 release and the det_20 format; they differ only in
    incidental fields, not in the parts used here.
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):                    # some mirrors wrap the list
        raw = raw.get("frames") or raw.get("data") or list(raw.values())[0]
    if not isinstance(raw, list):
        raise ValueError(f"{path} does not contain a list of image records")

    out: list[Record] = []
    for item in raw:
        attrs = item.get("attributes") or {}
        rec = Record(
            name=item["name"],
            weather=attrs.get("weather", "undefined"),
            timeofday=attrs.get("timeofday", "undefined"),
            scene=attrs.get("scene", "undefined"),
        )
        for lab in item.get("labels") or []:
            box = lab.get("box2d")
            if not box:                          # segmentation-only entries
                continue
            mapped = class_map.get(lab.get("category", ""))
            if mapped is None:
                continue
            rec.boxes.append((
                mapped,
                float(box["x1"]), float(box["y1"]),
                float(box["x2"]), float(box["y2"]),
            ))
            if lab.get("category") == "traffic light":
                colour = (lab.get("attributes") or {}).get("trafficLightColor", "none")
                rec.light_colours[colour] += 1
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# curation
# --------------------------------------------------------------------------- #


def curate(
    records: list[Record],
    available: set[str],
    max_images: int | None,
    night_fraction: float | None,
    require: list[str],
    min_box_px: float,
    min_objects: int,
    seed: int,
) -> list[Record]:
    """Select the subset worth training on.

    Filtering happens before balancing, so ``night_fraction`` is honoured over
    images that already passed the content filters rather than over the raw set.
    """
    rng = random.Random(seed)

    kept: list[Record] = []
    for r in records:
        if r.name not in available:
            continue
        # Tiny boxes carry almost no signal and dominate the loss by count.
        r.boxes = [b for b in r.boxes if (b[4] - b[2]) >= min_box_px]
        if len(r.boxes) < min_objects:
            continue
        if require:
            present = set(r.counts())
            if not present.issuperset(require):
                continue
        kept.append(r)

    if max_images is None or len(kept) <= max_images:
        rng.shuffle(kept)
        return kept

    if night_fraction is None:
        rng.shuffle(kept)
        return kept[:max_images]

    night = [r for r in kept if r.is_night]
    day = [r for r in kept if not r.is_night]
    rng.shuffle(night)
    rng.shuffle(day)
    want_night = min(len(night), int(round(max_images * night_fraction)))
    want_day = min(len(day), max_images - want_night)
    # If one pool is short, spend the remainder on the other rather than
    # silently returning fewer images than asked for.
    want_night = min(len(night), max_images - want_day)
    out = night[:want_night] + day[:want_day]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def write_split(
    records: list[Record], images_dir: Path, out_dir: Path, split: str,
    classes: list[str], link: bool,
) -> Counter:
    """Write one split's images and YOLO label files."""
    img_out = out_dir / "images" / split
    lab_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    index = {c: i for i, c in enumerate(classes)}
    # A class the caller did not declare would otherwise surface as a bare
    # KeyError deep in the write loop, after all the expensive work is done.
    unknown = {b[0] for r in records for b in r.boxes} - set(index)
    if unknown:
        raise ValueError(
            f"records contain classes not in the declared list: {sorted(unknown)}. "
            f"Declared: {classes}. Pass the full vocabulary (core.types.CLASS_NAMES) "
            f"rather than a hand-listed subset."
        )
    counts: Counter = Counter()

    for r in records:
        src = images_dir / r.name
        dst = img_out / r.name
        if not dst.exists():
            if link:
                try:
                    dst.hardlink_to(src)
                except (OSError, AttributeError):
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)

        # BDD100K is uniformly 1280x720; the boxes are in those pixels.
        w, h = 1280.0, 720.0
        lines = []
        for name, x1, y1, x2, y2 in r.boxes:
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            lines.append(f"{index[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts[name] += 1
        # An image with no objects gets an EMPTY file, never a missing one:
        # missing means "never annotated", and conflating the two teaches the
        # detector false negatives.
        (lab_out / f"{Path(r.name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    return counts


def write_dataset_yaml(
    out_dir: Path, classes: list[str], splits: dict[str, int],
    stats: dict[str, Any], args: argparse.Namespace,
) -> None:
    lines = [
        "# Generated by data/adapters/bdd100k.py -- do not edit by hand.",
        "name: bdd100k-det",
        "task: detection",
        "source_url: https://bdd-data.berkeley.edu/",
        "adapter: data/adapters/bdd100k.py",
        f"adapter_version: {ADAPTER_VERSION}",
        f"created: {__import__('datetime').date.today().isoformat()}",
        "",
        "# Ultralytics reads these three keys directly.",
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    lines += ["", "splits:"]
    lines += [f"  {k}: {v}" for k, v in splits.items()]
    lines += [
        "",
        "notes: |",
        f"  Curated subset of the BDD100K train split ({stats['n_source']} labelled",
        f"  images available). Selection: max_images={args.max_images},",
        f"  night_fraction={args.night_fraction}, require={args.require or 'none'},",
        f"  min_box_px={args.min_box_px}, min_objects={args.min_objects}, seed={args.seed}.",
        "",
        "  Only BDD100K's train split exists locally, so val is carved out of it",
        f"  by image with seed {args.seed}. Frames from one video never span the",
        "  split boundary at random -- they are split by image id, which is what",
        "  BDD100K's own naming makes possible.",
        "",
        f"  Night images in the selection: {stats['n_night']} "
        f"({stats['pct_night']:.1f}%).",
        "  car/bus/truck are merged into one 'vehicle' class, matching what",
        "  YOLOPv2 emits so the two backbones stay directly comparable.",
        "  'rider' maps to person; the bike beneath is labelled separately.",
    ]
    (out_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert and curate BDD100K detection labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", type=Path, default=Path("data/raw/images"),
                   help="directory of BDD100K jpgs")
    p.add_argument("--labels", type=Path, required=True,
                   help="det_train.json (det_20) or bdd100k_labels_images_train.json")
    p.add_argument("--out", type=Path, default=Path("data/processed/bdd100k"))
    p.add_argument("--max-images", type=int, default=12000,
                   help="cap on curated images; None uses everything")
    p.add_argument("--night-fraction", type=float, default=0.45,
                   help="target share of night images, since night is the weak case")
    p.add_argument("--require", nargs="*", default=[],
                   help="only keep images containing ALL of these classes")
    p.add_argument("--min-box-px", type=float, default=12.0,
                   help="drop boxes shorter than this; they carry little signal")
    p.add_argument("--min-objects", type=int, default=1,
                   help="drop images with fewer than this many surviving boxes")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--with-traffic-control", action="store_true",
                   help="also emit traffic_light and traffic_sign classes")
    p.add_argument("--copy", action="store_true",
                   help="copy images instead of hard-linking (uses real disk space)")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.labels.is_file():
        raise SystemExit(
            f"label file not found: {args.labels}\n\n"
            f"BDD100K ships images and labels separately. The images alone carry\n"
            f"no supervision. Download the detection labels from\n"
            f"  https://bdd-data.berkeley.edu/  ->  'Detection 2020 Labels'\n"
            f"and point --labels at det_train.json inside the archive."
        )
    if not args.images.is_dir():
        raise SystemExit(f"images directory not found: {args.images}")

    class_map = dict(CATEGORY_MAP)
    classes = list(BASE_CLASSES)
    if args.with_traffic_control:
        class_map.update(TRAFFIC_MAP)
        classes += TRAFFIC_CLASSES

    print(f"reading {args.labels} ...")
    records = load_labels(args.labels, class_map)
    print(f"  {len(records):,} labelled images in the label file")

    available = {p.name for p in args.images.glob("*.jpg")}
    print(f"  {len(available):,} images present on disk")

    selected = curate(
        records, available, args.max_images, args.night_fraction,
        args.require, args.min_box_px, args.min_objects, args.seed,
    )
    if not selected:
        raise SystemExit("curation selected 0 images; loosen the filters")

    n_night = sum(1 for r in selected if r.is_night)
    n_val = max(1, int(round(len(selected) * args.val_fraction)))
    val, train = selected[:n_val], selected[n_val:]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"writing {len(train):,} train / {len(val):,} val -> {args.out}")
    c_train = write_split(train, args.images, args.out, "train", classes, not args.copy)
    c_val = write_split(val, args.images, args.out, "val", classes, not args.copy)

    stats = {
        "n_source": len(records),
        "n_night": n_night,
        "pct_night": 100.0 * n_night / max(1, len(selected)),
    }
    write_dataset_yaml(args.out, classes,
                       {"train": len(train), "val": len(val)}, stats, args)

    total = c_train + c_val
    weather = Counter(r.weather for r in selected)
    tod = Counter(r.timeofday for r in selected)
    lights = Counter()
    for r in selected:
        lights.update(r.light_colours)

    print("=" * 62)
    print(f"selected {len(selected):,} of {len(records):,} labelled images")
    print(f"  night        {n_night:,} ({stats['pct_night']:.1f}%)")
    print("  time of day  " + ", ".join(f"{k}={v}" for k, v in tod.most_common()))
    print("  weather      " + ", ".join(f"{k}={v}" for k, v in weather.most_common()))
    print("  boxes        " + ", ".join(f"{k}={v:,}" for k, v in total.most_common()))
    if lights:
        print("  light colour " + ", ".join(f"{k}={v:,}" for k, v in lights.most_common()))
    print(f"  dataset.yaml {args.out / 'dataset.yaml'}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
