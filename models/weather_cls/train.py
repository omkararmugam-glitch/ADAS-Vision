"""Train the weather classifier.

    python models/weather_cls/train.py --data data/processed/weather

Expects the folder-per-class layout documented in ``docs/DATASET_FORMAT.md``::

    data/processed/weather/
      train/clear/*.jpg   train/rainy/*.jpg   ...
      val/clear/*.jpg     val/rainy/*.jpg     ...

``data/adapters/bdd100k_weather.py`` produces exactly that from BDD100K's
``attributes.weather``.

Two choices here are driven by what this classifier is actually for. Its output
selects one of a handful of stopping-distance multipliers, so:

* **Class imbalance is handled explicitly.** Driving datasets are overwhelmingly
  ``clear``; BDD100K is roughly 50% clear and about 5% each for snow and fog. A
  model trained on that raw distribution reaches high accuracy by never
  predicting the classes that actually change the braking threshold. Loss is
  therefore class-weighted, and **macro F1 is the reported metric**, because
  plain accuracy would hide precisely this failure.
* **Augmentation excludes horizontal flips of the colour statistics.** Weather
  lives in global colour and contrast, not in shape, so heavy geometric
  augmentation trains away the signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True,
                    help="root containing train/<class>/ and val/<class>/")
    ap.add_argument("--architecture", default="resnet18",
                    choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "models/weather_cls/weights/weather.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0 if sys.platform == "win32" else 4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    train_dir, val_dir = args.data / "train", args.data / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise SystemExit(
                f"missing {d}.\n"
                f"Expected data/processed/weather/{{train,val}}/<class>/*.jpg -- see "
                f"docs/DATASET_FORMAT.md. Build it with "
                f"data/adapters/bdd100k_weather.py once det_train.json is available."
            )

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from models.weather_cls.classifier import build_model

    # Colour jitter is mild and geometric augmentation is limited to a
    # horizontal flip: weather is carried by global colour and contrast, and
    # aggressive jitter would erase the very thing being classified.
    train_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=eval_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(
            f"train and val disagree on classes:\n  train {train_ds.classes}\n"
            f"  val   {val_ds.classes}\nA class present in one split only makes "
            f"every per-class metric meaningless."
        )
    classes = tuple(train_ds.classes)

    counts = Counter(y for _, y in train_ds.samples)
    print(f"{len(train_ds)} train / {len(val_ds)} val over {len(classes)} classes")
    for i, c in enumerate(classes):
        print(f"  {c:16s} {counts.get(i, 0):6d}")
    if min(counts.values()) * 8 < max(counts.values()):
        print("  ! heavily imbalanced -- macro F1 is the metric that matters here,\n"
              "    since accuracy can be high while a rare class is never predicted")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, len(classes), pretrained=True).to(device)

    # Inverse-frequency weighting, so a rare but braking-relevant class such as
    # snow is not simply ignored by the optimiser.
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(classes) * max(1, counts.get(i, 0))) for i in range(len(classes))],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    best_f1, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            running += float(loss) * x.size(0)
        sched.step()

        model.eval()
        confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu()
                for t, p in zip(y, pred):
                    confusion[int(t), int(p)] += 1

        correct = int(confusion.diag().sum())
        accuracy = correct / max(1, int(confusion.sum()))
        f1s = []
        for i in range(len(classes)):
            tp = int(confusion[i, i])
            fp = int(confusion[:, i].sum()) - tp
            fn = int(confusion[i, :].sum()) - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        macro_f1 = sum(f1s) / len(f1s)
        history.append({"epoch": epoch, "loss": running / max(1, len(train_ds)),
                        "accuracy": accuracy, "macro_f1": macro_f1,
                        "per_class_f1": dict(zip(classes, f1s))})
        print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  "
              f"acc {accuracy:.4f}  macro F1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": list(classes),
                        "architecture": args.architecture,
                        "input_size": args.input_size,
                        "macro_f1": macro_f1, "accuracy": accuracy}, args.out)

    summary = {
        "architecture": args.architecture,
        "classes": list(classes),
        "epochs": args.epochs,
        "best_macro_f1": best_f1,
        "final": history[-1] if history else None,
        "history": history,
        "weights": str(args.out),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "note": "macro F1 is the headline; accuracy is misleading under the "
                "class imbalance typical of driving datasets",
    }
    (args.out.parent / "weather_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nbest macro F1 {best_f1:.4f}  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
