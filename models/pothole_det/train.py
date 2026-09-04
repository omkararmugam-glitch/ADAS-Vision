"""Train a road-hazard model, in whichever of the two modes is being compared.

    python models/pothole_det/train.py --mode detector   --data <dataset.yaml>
    python models/pothole_det/train.py --mode classifier --data <folder-per-class>

Detector mode delegates to eval/train_detector.py, so a pothole detector is
trained by exactly the same code path as the vehicle detector -- one trainer to
maintain, and identical metric reporting.

Classifier mode delegates to models/weather_cls/train.py, which is already a
class-weighted image classifier with macro-F1 reporting; a binary damaged/sound
task is the same problem with two classes.

Reusing both rather than writing a third trainer is deliberate: the point of this
module is to *compare* the two approaches on the supplied data, and a comparison
is only meaningful if the training and evaluation code is shared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["detector", "classifier"], default="detector")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "models/pothole_det/weights/pothole.pt")
    args, rest = ap.parse_known_args(argv)

    if not args.data.exists():
        raise SystemExit(
            f"dataset not found: {args.data}\n"
            f"Pothole data must be supplied separately -- BDD100K carries no road-damage "
            f"labels. See docs/DATASET_FORMAT.md for the expected layout:\n"
            f"  detector   -> YOLO layout with a dataset.yaml (single class: pothole)\n"
            f"  classifier -> train/{{damaged,sound}}/ and val/{{damaged,sound}}/"
        )

    if args.mode == "detector":
        from eval.train_detector import main as train_detector
        print("training a single-class pothole detector via eval/train_detector.py")
        return train_detector([
            "--data", str(args.data), "--epochs", str(args.epochs),
            "--batch", str(args.batch), "--name", "pothole_det", *rest,
        ])

    from models.weather_cls.train import main as train_classifier
    print("training a binary damaged/sound patch classifier")
    return train_classifier([
        "--data", str(args.data), "--epochs", str(args.epochs),
        "--batch", str(args.batch), "--out", str(args.out), *rest,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
