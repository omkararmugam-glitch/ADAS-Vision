"""Build the weather-classification dataset from BDD100K attributes.

    python data/adapters/bdd100k_weather.py --labels path/to/det_train.json

BDD100K records ``attributes.weather`` per image, so the labels already exist in
the same JSON the detection adapter reads -- no extra annotation needed. This
writes the folder-per-class layout ``models/weather_cls/train.py`` expects.

Two decisions worth stating. Images labelled ``undefined`` are **dropped**, not
folded into ``clear``: a wrong label is worse than a missing one, and there are
plenty of frames. And the split is by image id with a fixed seed, matching the
detection adapter, so a frame cannot appear in train for one task and val for
the other.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

#: Dropped rather than guessed at.
SKIP = {"undefined", "", None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="det_train.json")
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "data/processed/weather")
    ap.add_argument("--max-per-class", type=int, default=2500,
                    help="cap per class, so `clear` cannot dominate outright")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-linking")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.labels.is_file():
        raise SystemExit(
            f"label file not found: {args.labels}\n"
            f"BDD100K ships images and labels separately; weather lives in the "
            f"same det_train.json the detection adapter reads.\n"
            f"Get it from https://bdd-data.berkeley.edu/ -> 'Detection 2020 Labels'."
        )

    records = json.loads(args.labels.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("frames") or list(records.values())[0]
    available = {p.name for p in args.images.glob("*.jpg")}

    by_class: dict[str, list[str]] = {}
    skipped = 0
    for item in records:
        name = item.get("name")
        weather = (item.get("attributes") or {}).get("weather")
        if weather in SKIP:
            skipped += 1
            continue
        if name not in available:
            continue
        by_class.setdefault(str(weather), []).append(name)

    if not by_class:
        raise SystemExit("no usable weather labels found")

    rng = random.Random(args.seed)
    counts, split_counts = Counter(), Counter()
    for weather, names in sorted(by_class.items()):
        rng.shuffle(names)
        names = names[: args.max_per_class]
        n_val = max(1, int(round(len(names) * args.val_fraction)))
        for split, subset in (("val", names[:n_val]), ("train", names[n_val:])):
            d = args.out / split / weather.replace(" ", "_")
            d.mkdir(parents=True, exist_ok=True)
            for n in subset:
                dst = d / n
                if dst.exists():
                    continue
                src = args.images / n
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        dst.hardlink_to(src)
                    except (OSError, AttributeError):
                        shutil.copy2(src, dst)
            split_counts[split] += len(subset)
        counts[weather] = len(names)

    print("=" * 60)
    print(f"weather dataset -> {args.out}")
    for w, n in counts.most_common():
        print(f"  {w:16s} {n:6d}")
    print(f"  {'TRAIN':16s} {split_counts['train']:6d}")
    print(f"  {'VAL':16s} {split_counts['val']:6d}")
    print(f"  dropped 'undefined': {skipped}")
    print("=" * 60)
    print("Train with:  python models/weather_cls/train.py --data", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
