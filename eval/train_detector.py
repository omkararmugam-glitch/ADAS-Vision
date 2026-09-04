"""Fine-tune a YOLO detector on a prepared dataset.

    python eval/train_detector.py --data data/processed/bdd100k_pseudo/dataset.yaml

Works with any dataset in the layout ``docs/DATASET_FORMAT.md`` describes, which
means the same command trains against real BDD100K annotations once
``det_train.json`` is available and against pseudo-labels before then. The only
difference is what the resulting numbers mean, and that difference is large:

  * against **ground truth**, mAP measures correctness
  * against **pseudo-labels**, mAP measures agreement with the teacher

The script reads ``label_source`` from the dataset.yaml and states which case it
is in, in the console and in the run summary, so a number can never be quoted
later without its provenance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def _per_class(meta: dict, metrics) -> dict[str, float]:
    """Per-class mAP@0.5, keyed by class name.

    Reported because a single averaged mAP hides exactly what matters here: a
    class with a handful of training examples drags nothing down visibly while
    being useless in practice.
    """
    import numpy as np

    ap = getattr(getattr(metrics, "box", None), "ap50", None)
    if ap is None:
        return {}
    values = np.atleast_1d(np.asarray(ap, dtype=float))
    names = list(meta.get("names", {}).values())
    return {n: float(v) for n, v in zip(names, values)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=PROJECT / "outputs/train")
    ap.add_argument("--name", default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    # 0 by default on Windows: ultralytics' dataloader workers use spawn there,
    # and they routinely deadlock before the first batch. Loading in-process is
    # slightly slower per epoch and reliably starts.
    ap.add_argument("--workers", type=int,
                    default=0 if sys.platform == "win32" else 8)
    ap.add_argument("--preview-every", type=int, default=5,
                    help="save annotated validation frames every N epochs; 0 disables")
    ap.add_argument("--preview-images", type=int, default=8,
                    help="how many fixed validation frames to annotate each time")
    args = ap.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"dataset.yaml not found: {args.data}")
    meta = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
    source = meta.get("label_source", "ground_truth")
    pseudo = source == "pseudo"

    name = args.name or (
        f"{Path(args.model).stem}_{'distill' if pseudo else 'ft'}_"
        f"{datetime.now():%Y%m%d_%H%M}"
    )

    print("=" * 68)
    print(f"training {args.model} on {meta.get('name', args.data.parent.name)}")
    print(f"  classes     : {', '.join(meta.get('names', {}).values())}")
    print(f"  epochs      : {args.epochs}   imgsz {args.imgsz}   batch {args.batch}")
    if pseudo:
        print(f"  LABEL SOURCE: PSEUDO (teacher {meta.get('teacher', '?')} "
              f"@ conf {meta.get('teacher_conf', '?')})")
        print("  -> mAP will measure AGREEMENT WITH THE TEACHER, not correctness.")
        print("  -> the student cannot exceed the teacher, and will inherit its errors.")
    else:
        print("  label source: ground truth -> mAP measures correctness")
    print("=" * 68)

    from ultralytics import YOLO

    from eval.preview import (
        TRAIN_PREVIEW_ROOT,
        contact_sheet,
        save_training_previews,
    )

    val_images = args.data.parent / str(meta.get("val", "images/val"))

    model = YOLO(args.model)

    # Previews are written from an on_train_epoch_end callback rather than after
    # training, because the entire point is to watch the model change. A loss
    # curve that falls while the boxes stay in the wrong place looks identical to
    # one that falls because the model is learning; only the frames distinguish
    # them.
    if args.preview_every > 0 and val_images.is_dir():
        preview_root = TRAIN_PREVIEW_ROOT / name

        def _on_epoch_end(trainer) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            if epoch % args.preview_every and epoch != args.epochs:
                return
            ckpt = Path(trainer.save_dir) / "weights" / "last.pt"
            if not ckpt.is_file():
                return
            try:
                written = save_training_previews(
                    ckpt, val_images, epoch, n=args.preview_images,
                    root=preview_root, label_source=source,
                )
                if written:
                    contact_sheet(
                        written, preview_root / f"epoch_{epoch:03d}_sheet.jpg",
                        title=f"{name} - epoch {epoch}",
                    )
                    print(f"  previews -> {written[0].parent}  ({len(written)} frames)")
            except Exception as exc:                  # noqa: BLE001 - deliberate
                # A preview failure must never abort a training run that is
                # otherwise fine; the checkpoint matters more than the picture.
                print(f"  preview failed at epoch {epoch}: {exc}")

        model.add_callback("on_train_epoch_end", _on_epoch_end)
    elif args.preview_every > 0:
        print(f"  ! no validation images at {val_images}; previews disabled")

    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=name,
        patience=args.patience,
        seed=args.seed,
        plots=True,
        val=True,
        verbose=True,
        workers=args.workers,
    )

    run_dir = args.project / name
    metrics = model.val(data=str(args.data.resolve()), split="val",
                        project=str(args.project), name=f"{name}_val", plots=True)

    summary = {
        "model": args.model,
        "dataset": str(args.data),
        "label_source": source,
        "label_source_warning": (
            "mAP measures agreement with the teacher, not correctness"
            if pseudo else "mAP measures correctness against ground truth"
        ),
        "teacher": meta.get("teacher"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "classes": list(meta.get("names", {}).values()),
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        # `array or []` raises on a numpy array -- truth value is ambiguous.
        # Convert explicitly, and tolerate the attribute being absent.
        "per_class_mAP50": _per_class(meta, metrics),
        "run_dir": str(run_dir),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    # Full evaluation previews once training is done.
    from eval.preview import EVAL_PREVIEW_ROOT, save_eval_previews

    eval_dir = EVAL_PREVIEW_ROOT / name
    best = run_dir / "weights" / "best.pt"
    if best.is_file() and val_images.is_dir():
        shots = save_eval_previews(best, val_images, eval_dir,
                                   max_images=args.preview_images * 6,
                                   label_source=source)
        if shots:
            contact_sheet(shots[:12], eval_dir / "contact_sheet.jpg",
                          title=f"{name} - validation predictions")
            summary["eval_previews"] = str(eval_dir)
            print(f"eval previews -> {eval_dir}  ({len(shots)} frames)")

    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"mAP@0.5      {summary['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 {summary['metrics']['mAP50_95']:.4f}")
    print(f"precision    {summary['metrics']['precision']:.4f}")
    print(f"recall       {summary['metrics']['recall']:.4f}")
    if pseudo:
        print("(agreement with the teacher -- NOT correctness)")
    print(f"artifacts    {run_dir}")
    print(f"previews     {TRAIN_PREVIEW_ROOT / name}  (per-epoch)")
    print(f"summary      {out}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
