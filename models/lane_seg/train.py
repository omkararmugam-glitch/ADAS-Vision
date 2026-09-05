"""Train the lane segmenter on masks distilled from the classical detector.

    python models/lane_seg/train.py --n 6000 --epochs 12

No lane annotations were supplied, so the targets come from
:mod:`models.lane_seg.classical`. That makes this weak supervision, and the
metric reported is **agreement with the classical detector**, not correctness --
the same caveat that applies to the distilled object detector.

The one thing worth stating plainly: the student can beat its teacher here, and
that is not wishful thinking. The classical detector's *successes* are
consistent (painted markings look alike in every frame) while its *failures* are
not (a particular bonnet reflection or sunlit wall occurs once). Averaged over
thousands of frames, gradient descent fits the repeated pattern and washes out
the one-offs. So agreement of, say, 0.75 IoU is not a ceiling on quality -- a
good part of the disagreement is the student declining to copy noise.

Only frames whose classical mask passes ``is_plausible`` are used as targets.
Training on a mask that covers the sky teaches exactly the wrong thing, and
there are far more frames available than needed.

Loss is Dice plus weighted cross-entropy. Lane pixels are about 2% of a frame,
so plain cross-entropy is minimised almost perfectly by predicting "no lane"
everywhere -- a model with 98% pixel accuracy and no output at all.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from models.lane_seg.classical import ClassicalLaneSegmenter
from models.lane_seg.segmenter import INPUT_SIZE, build_segmentation_model


def build_weak_dataset(
    image_dir: Path, n: int, out_dir: Path, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Generate classical masks and keep only the plausible ones."""
    seg = ClassicalLaneSegmenter()
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no images under {image_dir}")
    sample = random.Random(seed).sample(paths, min(n, len(paths)))

    img_out, msk_out = out_dir / "images", out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    kept, rejected = [], 0
    for i, p in enumerate(sample):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        res = seg.infer_detailed(frame)
        if not res.is_plausible:
            rejected += 1
            continue
        small_img = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        small_msk = cv2.resize(res.mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(img_out / f"{p.stem}.jpg"), small_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(msk_out / f"{p.stem}.png"), small_msk)
        kept.append(p)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(sample)} scanned, {len(kept)} kept", flush=True)

    print(f"weak labels: {len(kept)} kept, {rejected} rejected as implausible")
    imgs = sorted(img_out.glob("*.jpg"))
    msks = [msk_out / f"{p.stem}.png" for p in imgs]
    return imgs, msks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--work", type=Path, default=PROJECT / "data/processed/lane_weak")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "models/lane_seg/weights/lane_seg.pt")
    ap.add_argument("--n", type=int, default=6000, help="frames to scan for targets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--rebuild", action="store_true", help="regenerate the weak labels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.rebuild or not (args.work / "images").is_dir():
        build_weak_dataset(args.images, args.n, args.work, args.seed)
    imgs = sorted((args.work / "images").glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no weak labels under {args.work}; run with --rebuild")
    msks = [args.work / "masks" / f"{p.stem}.png" for p in imgs]

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    class LaneSet(Dataset):
        def __init__(self, images, masks, augment: bool):
            self.images, self.masks, self.augment = images, masks, augment

        def __len__(self):
            return len(self.images)

        def __getitem__(self, i):
            img = cv2.imread(str(self.images[i]))
            msk = cv2.imread(str(self.masks[i]), cv2.IMREAD_GRAYSCALE)
            if self.augment and random.random() < 0.5:
                # Horizontal flip only. A road scene mirrors legitimately; a
                # vertical flip or a large rotation produces a view no dashcam
                # ever sees, and the network would waste capacity on it.
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            arr = (img[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
            return (torch.from_numpy(arr.transpose(2, 0, 1)),
                    torch.from_numpy((msk > 127).astype(np.int64)))

    rng = random.Random(args.seed)
    order = list(range(len(imgs)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * args.val_fraction)))
    val_idx, train_idx = set(order[:n_val]), order[n_val:]
    train_ds = LaneSet([imgs[i] for i in train_idx], [msks[i] for i in train_idx], True)
    val_ds = LaneSet([imgs[i] for i in sorted(val_idx)],
                     [msks[i] for i in sorted(val_idx)], False)
    print(f"{len(train_ds)} train / {len(val_ds)} val at {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")

    workers = 0 if sys.platform == "win32" else 4
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(2, pretrained_backbone=True).to(device)

    # Lane pixels are ~2% of a frame. Unweighted cross-entropy is minimised by
    # predicting background everywhere -- 98% pixel accuracy, zero output.
    frac = np.mean([
        (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127).mean()
        for m in msks[: min(200, len(msks))]
    ])
    pos_weight = float(np.clip((1 - frac) / max(frac, 1e-4), 1.0, 40.0))
    print(f"lane pixels {frac * 100:.2f}% of frame -> class weight {pos_weight:.1f}")
    ce_weight = torch.tensor([1.0, pos_weight], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def dice_loss(logits, target):
        """Dice directly optimises overlap, which is what IoU measures.

        Cross-entropy alone is dominated by the vast background even when
        weighted; Dice sees only the small positive region and keeps the thin
        structures from being smoothed away.
        """
        prob = torch.softmax(logits, dim=1)[:, 1]
        t = target.float()
        inter = (prob * t).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        return 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()

    best_iou, history = 0.0, []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)["out"]
            loss = F.cross_entropy(out, y, weight=ce_weight) + dice_loss(out, y)
            loss.backward()
            opt.step()
            total += float(loss) * x.size(0)
        sched.step()

        model.eval()
        inter = union = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)["out"].argmax(1)
                inter += int(((pred == 1) & (y == 1)).sum())
                union += int(((pred == 1) | (y == 1)).sum())
        iou = inter / max(1, union)
        history.append({"epoch": epoch, "loss": total / max(1, len(train_ds)),
                        "lane_iou": iou})
        print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  lane IoU {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save({"model": model.state_dict(), "input_size": list(INPUT_SIZE),
                        "lane_iou": iou, "label_source": "classical-weak"}, args.out)

    (args.out.parent / "lane_seg_summary.json").write_text(json.dumps({
        "best_lane_iou": best_iou,
        "label_source": "classical-weak",
        "warning": "IoU measures AGREEMENT with the classical detector, not "
                   "correctness. Part of the disagreement is the student "
                   "declining to copy the teacher's noise.",
        "epochs": args.epochs, "train": len(train_ds), "val": len(val_ds),
        "history": history,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"\nbest lane IoU {best_iou:.4f} (vs classical)  ->  {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
