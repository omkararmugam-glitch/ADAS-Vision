"""Score fusion alerts against labelled event intervals.

    python eval/score_alerts.py --predicted run/episodes.csv --truth events.csv

Both files are episode lists: ``type,start_s,end_s`` (see ``DATASET_FORMAT.md``
§4). Matching is by temporal overlap, not by frame, because an alert that holds
for 90 frames is one event a driver experienced. Scoring per frame would let a
single long warning dominate every metric.

**Lead time is reported alongside precision and recall, and matters more.** A
forward-collision warning that is correct but fires 0.3 s before impact has no
practical value; the same warning 2.5 s ahead is actionable. Precision and recall
are identical in both cases, which is exactly why they are not enough on their
own.

Lead time is measured from the start of the predicted alert to the start of the
labelled event — how much warning the driver actually received before the thing
being warned about began.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


@dataclass(frozen=True)
class Interval:
    type: str
    start_s: float
    end_s: float

    def overlaps(self, other: Interval, tolerance_s: float = 0.0) -> bool:
        return (self.start_s - tolerance_s) <= other.end_s and \
               (self.end_s + tolerance_s) >= other.start_s


def read_intervals(path: Path, type_field="type",
                   start_field="start_s", end_field="end_s") -> list[Interval]:
    """Read an episode CSV. Tolerates the extra columns our own logs carry."""
    out: list[Interval] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if type_field not in row:
                raise SystemExit(
                    f"{path} has no '{type_field}' column; found {list(row)}"
                )
            out.append(Interval(
                type=row[type_field].strip(),
                start_s=float(row[start_field]),
                end_s=float(row[end_field]),
            ))
    return out


def score(
    predicted: list[Interval], truth: list[Interval], tolerance_s: float = 1.0
) -> dict:
    """Match predictions to ground truth per alert type.

    A prediction is a true positive if it overlaps a labelled event of the same
    type, within ``tolerance_s``. Each labelled event is claimed at most once, so
    a detector that fires ten times over one event scores one TP and nine FPs
    rather than ten TPs -- which is the behaviour a driver would experience.
    """
    types = sorted({i.type for i in predicted} | {i.type for i in truth})
    per_type: dict[str, dict] = {}

    for t in types:
        preds = sorted([i for i in predicted if i.type == t], key=lambda i: i.start_s)
        gts = sorted([i for i in truth if i.type == t], key=lambda i: i.start_s)
        claimed: set[int] = set()
        tp, fp = 0, 0
        lead_times: list[float] = []

        for p in preds:
            hit = None
            for gi, g in enumerate(gts):
                if gi in claimed:
                    continue
                if p.overlaps(g, tolerance_s):
                    hit = gi
                    break
            if hit is None:
                fp += 1
            else:
                claimed.add(hit)
                tp += 1
                lead_times.append(gts[hit].start_s - p.start_s)

        fn = len(gts) - len(claimed)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_type[t] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_lead_s": (sum(lead_times) / len(lead_times)) if lead_times else None,
            "min_lead_s": min(lead_times) if lead_times else None,
            "n_predicted": len(preds), "n_truth": len(gts),
        }

    tp = sum(v["tp"] for v in per_type.values())
    fp = sum(v["fp"] for v in per_type.values())
    fn = sum(v["fn"] for v in per_type.values())
    return {
        "tolerance_s": tolerance_s,
        "per_type": per_type,
        "overall": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predicted", type=Path, required=True,
                    help="episodes.csv from `run.py analyse`")
    ap.add_argument("--truth", type=Path, required=True,
                    help="labelled events CSV: type,start_s,end_s")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="seconds of slack when matching an alert to an event")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    for p in (args.predicted, args.truth):
        if not p.is_file():
            raise SystemExit(f"not found: {p}")

    result = score(read_intervals(args.predicted), read_intervals(args.truth),
                   args.tolerance)

    print("=" * 78)
    print(f"{'alert':22s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>7s} "
          f"{'recall':>7s} {'F1':>6s} {'lead s':>8s}")
    print("-" * 78)
    for t, v in result["per_type"].items():
        lead = f"{v['mean_lead_s']:.2f}" if v["mean_lead_s"] is not None else "-"
        print(f"{t:22s} {v['tp']:4d} {v['fp']:4d} {v['fn']:4d} "
              f"{v['precision']:7.3f} {v['recall']:7.3f} {v['f1']:6.3f} {lead:>8s}")
    o = result["overall"]
    print("-" * 78)
    print(f"{'OVERALL':22s} {o['tp']:4d} {o['fp']:4d} {o['fn']:4d} "
          f"{o['precision']:7.3f} {o['recall']:7.3f}")
    print("=" * 78)
    print("Lead time is the number to judge collision alerts by: a correct warning")
    print("0.3 s before impact and one 2.5 s before score identically on precision.")

    out = args.out or args.predicted.parent / "alert_scores.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
