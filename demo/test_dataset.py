"""Run the detector over a sample of a dataset and write browsable results.

    python demo/test_dataset.py --images data/raw/images --n 120

Produces, under ``outputs/tested/``:

  ``images/``      one annotated JPEG per frame, named so it sorts by condition
  ``results.csv``  one row per image: counts, confidences, brightness
  ``index.html``   a gallery, openable with file://

The sample is balanced across day, dusk and night rather than taken at random.
An unbalanced sample of BDD100K is about 43% night, and a reader skimming a
random gallery would draw conclusions about night performance from whichever
frames happened to land in it. Balancing makes the comparison deliberate.

This is inference, not evaluation: the dataset shipped without labels, so there
is nothing to score against and no mAP is produced. The counts are the model's
own output.
"""

from __future__ import annotations

import argparse
import csv
import html
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.config import load_config
from core.types import class_name
from models.perception.factory import build_detector

COLOUR = {"vehicle": (70, 220, 70), "person": (0, 170, 255),
          "bicycle": (255, 190, 0), "motorcycle": (255, 110, 210)}

#: Mean-pixel thresholds separating the lighting bands. Crude but honest: the
#: dataset's own timeofday labels live in the label file, which is absent.
NIGHT_MAX, DUSK_MAX = 55.0, 95.0


@dataclass
class Result:
    name: str
    band: str
    brightness: float
    counts: Counter
    scores: list[float]
    heights: list[float]

    @property
    def n(self) -> int:
        return sum(self.counts.values())


def band_of(brightness: float) -> str:
    if brightness < NIGHT_MAX:
        return "night"
    if brightness < DUSK_MAX:
        return "dusk"
    return "day"


def annotate(img: np.ndarray, result) -> np.ndarray:
    vis = img.copy()
    for b, cid, s in zip(result.boxes_px, result.class_ids, result.scores):
        name = class_name(int(cid))
        c = COLOUR.get(name, (200, 200, 200))
        x0, y0, x1, y1 = (int(v) for v in b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2)
        label = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ly = y0 - 4 if y0 - th - 6 >= 0 else y1 + th + 6
        cv2.rectangle(vis, (x0, ly - th - 4), (x0 + tw + 5, ly + 2), c, -1)
        cv2.putText(vis, label, (x0 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def banner(vis: np.ndarray, text: str, summary: str) -> np.ndarray:
    bar = np.full((30, vis.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, summary, (300, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (110, 230, 110), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


#: Shown verbatim in the gallery banner. Keyed on whether the checkpoint was
#: trained here, because "COCO-pretrained" is simply false for a distilled one.
BANNER_COCO = ("These weights are COCO-pretrained and have not been fine-tuned "
               "on this dataset.")
BANNER_DISTILLED = ("These weights were trained here by distillation against a "
                    "larger model&rsquo;s predictions, so they reproduce that "
                    "teacher&rsquo;s behaviour &mdash; including its mistakes.")


def gallery_html(results: list[Result], model: str, conf: float,
                 banner: str = BANNER_COCO) -> str:
    cards = []
    for r in results:
        summary = ", ".join(f"{v}&nbsp;{k}" for k, v in sorted(r.counts.items())) or "nothing found"
        cards.append(
            f'<figure class="{r.band}">'
            f'<a href="images/{html.escape(r.name)}" target="_blank">'
            f'<img src="images/{html.escape(r.name)}" loading="lazy" alt="{html.escape(r.name)}"></a>'
            f'<figcaption><b>{r.band}</b> &middot; brightness {r.brightness:.0f}'
            f'<span>{summary}</span></figcaption></figure>'
        )
    by_band = Counter(r.band for r in results)
    totals = Counter()
    for r in results:
        totals.update(r.counts)
    rows = "".join(
        f"<tr><td>{b}</td><td>{by_band[b]}</td>"
        f"<td>{sum(r.n for r in results if r.band == b) / max(1, by_band[b]):.1f}</td>"
        f"<td>{sum(1 for r in results if r.band == b and r.n == 0)}</td></tr>"
        for b in ("day", "dusk", "night") if by_band[b]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector output on the dataset</title>
<style>
:root{{color-scheme:light;--bg:#f4f5f2;--surface:#fff;--sunk:#eceee8;--ink:#16191a;
--ink2:#454b4d;--muted:#6b7370;--line:#dcdfd8;--accent:#17864a;--warn:#a86a00}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--bg:#131614;--surface:#1b1f1c;
--sunk:#22271f;--ink:#ecefea;--ink2:#c2c9c1;--muted:#8e968d;--line:#2c322d;
--accent:#4ed07f;--warn:#e0a340}}}}
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:1500px;margin:0 auto;padding:36px 22px 70px}}
h1{{font-size:clamp(24px,3.6vw,36px);margin:0 0 12px;letter-spacing:-.02em}}
.sub{{color:var(--ink2);max-width:72ch;margin:0 0 18px}}
.banner{{padding:13px 17px;background:var(--sunk);border-left:3px solid var(--warn);
border-radius:0 4px 4px 0;margin:0 0 26px}}
.banner p{{margin:0;font-size:14.5px;color:var(--ink2);max-width:82ch}}
.banner strong{{color:var(--ink)}}
table{{border-collapse:collapse;font-size:14.5px;margin:0 0 26px;min-width:380px}}
th{{text-align:left;font:500 11px/1.4 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.09em;
text-transform:uppercase;color:var(--muted);padding:0 18px 8px 0;border-bottom:1px solid var(--line)}}
td{{padding:9px 18px 9px 0;border-bottom:1px solid var(--line);color:var(--ink2)}}
td:first-child{{color:var(--ink)}}
.filters{{display:flex;gap:9px;flex-wrap:wrap;margin:0 0 22px}}
.filters button{{font:inherit;font-size:14px;padding:7px 15px;border-radius:20px;
border:1px solid var(--line);background:var(--surface);color:var(--ink2);cursor:pointer}}
.filters button[aria-pressed="true"]{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}}
figure{{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:6px;
overflow:hidden}}
figure img{{display:block;width:100%;height:auto}}
figure[hidden]{{display:none}}
figcaption{{padding:10px 13px 12px;font-size:13px;color:var(--muted);border-top:1px solid var(--line)}}
figcaption b{{color:var(--ink);text-transform:capitalize}}
figcaption span{{display:block;margin-top:3px;color:var(--accent);font-size:13.5px}}
a{{color:var(--accent)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style></head><body><div class="wrap">
<h1>Detector output on the dataset</h1>
<p class="sub">{len(results)} frames from the supplied BDD100K images, sampled evenly
across lighting conditions. Model <code>{model}</code>, confidence &ge; {conf}.
Click any frame for full resolution.</p>
<div class="banner"><p><strong>Inference, not evaluation.</strong> {banner}
The counts are the model&rsquo;s own output &mdash; the dataset shipped without its
label file, so there is no ground truth here to score against and no mAP is
quoted.</p></div>
<table><thead><tr><th>Condition</th><th>Frames</th><th>Objects / frame</th><th>Empty</th></tr></thead>
<tbody>{rows}</tbody></table>
<div class="filters">
  <button aria-pressed="true" data-band="all">All ({len(results)})</button>
  <button aria-pressed="false" data-band="day">Day ({by_band['day']})</button>
  <button aria-pressed="false" data-band="dusk">Dusk ({by_band['dusk']})</button>
  <button aria-pressed="false" data-band="night">Night ({by_band['night']})</button>
</div>
<div class="grid">{"".join(cards)}</div>
</div>
<script>
const buttons = document.querySelectorAll('.filters button');
const figures = document.querySelectorAll('.grid figure');
buttons.forEach(b => b.addEventListener('click', () => {{
  const band = b.dataset.band;
  buttons.forEach(o => o.setAttribute('aria-pressed', String(o === b)));
  figures.forEach(f => {{ f.hidden = !(band === 'all' || f.classList.contains(band)); }});
}}));
</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=Path, default=PROJECT / "data/raw/images")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/tested")
    ap.add_argument("--n", type=int, default=120, help="frames to keep, balanced across bands")
    ap.add_argument("--scan", type=int, default=900, help="frames to scan when balancing")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", type=Path, default=None,
                    help="checkpoint to run; defaults to the configured detector")
    ap.add_argument("--tag", default=None,
                    help="label shown on each frame and in the gallery title")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg files under {args.images}")
    rng = random.Random(args.seed)
    scan = rng.sample(paths, min(args.scan, len(paths)))
    print(f"{len(paths):,} images available; scanning {len(scan)} to balance bands")

    cfg = load_config(PROJECT / "configs/default.yaml").detector
    update = {"kind": "yolov8", "conf_threshold": args.conf}
    if args.weights is not None:
        update["yolov8"] = cfg.yolov8.model_copy(update={"weights": args.weights})
    cfg = cfg.model_copy(update=update)
    det = build_detector(cfg)
    model = args.tag or Path(cfg.yolov8.weights).stem

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    per_band = args.n // 3
    kept: dict[str, list[Result]] = {"day": [], "dusk": [], "night": []}
    for i in range(0, len(scan), args.batch):
        if all(len(v) >= per_band for v in kept.values()):
            break
        chunk = scan[i:i + args.batch]
        imgs, keep = [], []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im); keep.append(p)
        if not imgs:
            continue
        for p, im, r in zip(keep, imgs, det.infer_batch(imgs)):
            br = float(np.asarray(im).mean())
            band = band_of(br)
            if len(kept[band]) >= per_band:
                continue
            counts = Counter(class_name(int(c)) for c in r.class_ids)
            res = Result(
                name=f"{band}_{p.stem}.jpg", band=band, brightness=br, counts=counts,
                scores=[float(s) for s in r.scores],
                heights=[float(b[3] - b[1]) for b in r.boxes_px],
            )
            summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing found"
            vis = banner(annotate(im, r), f"{band}  {p.name}  brightness {br:.0f}", summary)
            cv2.imwrite(str(img_dir / res.name), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
            kept[band].append(res)
    det.close()

    results = kept["day"] + kept["dusk"] + kept["night"]
    if not results:
        raise SystemExit("no frames processed")

    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "band", "brightness", "n_objects", "vehicle", "person",
                    "bicycle", "motorcycle", "mean_conf", "median_box_h_px"])
        for r in results:
            w.writerow([
                r.name, r.band, f"{r.brightness:.1f}", r.n,
                r.counts.get("vehicle", 0), r.counts.get("person", 0),
                r.counts.get("bicycle", 0), r.counts.get("motorcycle", 0),
                f"{np.mean(r.scores):.3f}" if r.scores else "",
                f"{np.median(r.heights):.1f}" if r.heights else "",
            ])

    banner = BANNER_DISTILLED if args.weights is not None else BANNER_COCO
    (args.out / "index.html").write_text(
        gallery_html(results, model, args.conf, banner), encoding="utf-8")

    print("=" * 66)
    print(f"{'band':8s} {'frames':>7s} {'obj/frame':>10s} {'empty':>6s} "
          f"{'mean conf':>10s} {'med box h':>10s}")
    for band in ("day", "dusk", "night"):
        rs = kept[band]
        if not rs:
            continue
        confs = [s for r in rs for s in r.scores]
        hs = [h for r in rs for h in r.heights]
        print(f"{band:8s} {len(rs):7d} {sum(r.n for r in rs)/len(rs):10.1f} "
              f"{sum(1 for r in rs if r.n == 0):6d} "
              f"{np.mean(confs) if confs else 0:10.3f} "
              f"{np.median(hs) if hs else 0:10.1f}")
    print("=" * 66)
    print(f"images  : {img_dir}  ({len(results)} files)")
    print(f"csv     : {args.out / 'results.csv'}")
    print(f"gallery : file:///{(args.out / 'index.html').resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
