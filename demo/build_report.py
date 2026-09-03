"""Generate a standalone HTML results report you can open in any browser.

    python demo/build_report.py            # -> outputs/report/index.html
    python demo/build_report.py --embed    # single self-contained file

Two modes, because they suit different uses:

*Linked* (default) writes ``index.html`` beside an ``assets/`` folder. The page
stays small and the images stay browsable as ordinary files. Open it directly
with ``file://`` -- no server needed.

*Embedded* (``--embed``) inlines every image as a data URI, producing one file
that can be emailed or dropped on any host with nothing else alongside it. It is
several megabytes, which is the trade.

Deliberately plain HTML and CSS: no framework, no build step, no CDN. The report
has to open years from now on a machine with no network, which rules out
anything fetched at load time.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import shutil
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT / "outputs" / "report"


@dataclass
class Figure:
    """One image and the claim it supports."""

    file: str
    title: str
    caption: str
    note: str = ""


@dataclass
class Section:
    heading: str
    body: list[str] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    table: tuple[list[str], list[list[str]]] | None = None


def build_sections() -> list[Section]:
    """The report's content. Edit here, not in the HTML."""
    return [
        Section(
            heading="Status",
            body=[
                "The perception and decision pipeline is built and tested end to end. "
                "<strong>No model has been fine-tuned on the BDD100K dataset</strong>, because "
                "the 70,000 images were supplied without their label file. Every detection "
                "shown below is inference using COCO-pretrained YOLOv8m weights.",
                "Getting to a trained model needs one download: <code>det_train.json</code> from "
                "&ldquo;Detection 2020 Labels&rdquo; at "
                "<a href=\"https://bdd-data.berkeley.edu/\">bdd-data.berkeley.edu</a>. The "
                "adapter that consumes it is written and tested.",
            ],
            table=(
                ["Component", "State"],
                [
                    ["Camera calibration &amp; bird's-eye geometry", "working"],
                    ["Content-addressed cache, predict/render split", "working"],
                    ["YOLOPv2 multi-task backbone", "working"],
                    ["YOLOv8 baseline (COCO weights)", "working"],
                    ["Lane segmentation &amp; metric fitting", "working"],
                    ["ByteTrack multi-object tracking", "working"],
                    ["Time-to-collision, lane-change detection", "working"],
                    ["Fusion / alert engine", "working"],
                    ["BDD100K adapter", "written, waiting on labels"],
                    ["Detector fine-tuning", "blocked: no labels"],
                    ["Weather classifier, pothole detector", "blocked: no labels"],
                    ["MiDaS depth, ONNX export", "not started"],
                ],
            ),
        ),
        Section(
            heading="Detection on the BDD100K frames",
            body=[
                "Eight frames drawn at random from the supplied 70,000, spanning daylight, "
                "dusk and night. Counts are the model's own output; there is no ground truth "
                "to score against, so no mAP is quoted.",
            ],
            figures=[
                Figure(
                    "bdd100k_inference.jpg",
                    "YOLOv8m inference on real BDD100K frames",
                    "Daylight frames yield 10&ndash;23 objects each. The night frames yield "
                    "16&ndash;18 vehicles &mdash; better than an earlier test suggested.",
                    "The large box across the bottom of the &ldquo;23 vehicle&rdquo; panel is a "
                    "false positive on the ego car&rsquo;s own hood reflection. Fine-tuning on "
                    "this dataset would remove that error class, since every frame contains that "
                    "hood and none of them label it.",
                ),
                Figure(
                    "bdd100k_montage.jpg",
                    "Twelve random frames, unannotated",
                    "The dataset spans day, night, rain, snow, urban and highway. A survey of "
                    "500 random frames measured 42.8% night, 96.2% containing vehicles and "
                    "26.8% containing pedestrians.",
                ),
            ],
        ),
        Section(
            heading="Recall is governed by object size, not darkness",
            body=[
                "Tested on the five-scene collage supplied earlier. Every detection is magnified "
                "from the original pixels and labelled with its confidence and box height.",
                "Above roughly 20&nbsp;px of object height, confidence sits at 0.45&ndash;0.85 "
                "regardless of weather &mdash; the rain panel performs as well as the clear one. "
                "Below about 15&nbsp;px it collapses under 0.30. The night failures in that test "
                "were distant tail lights 6&ndash;16&nbsp;px tall, not darkness as such; the "
                "BDD100K night frames above contain close, streetlit vehicles and detect well.",
            ],
            figures=[
                Figure(
                    "collage_sheet.jpg",
                    "Per-scene detection with magnified crops",
                    "yolov8m at confidence 0.10. At 8&ndash;13&nbsp;px a car is a few red pixels "
                    "of tail light, which is why confidence never rises enough to separate it "
                    "from noise. No threshold setting fixes that &mdash; only training data with "
                    "distant night vehicles in it.",
                ),
                Figure(
                    "car_detection_perpanel.jpg",
                    "The same collage, boxes at default settings",
                    "Each panel detected separately. Detecting the whole collage in one pass "
                    "finds fewer objects, because letterboxing 1672&times;941 down to 640 shrinks "
                    "every panel to a third of native size.",
                ),
            ],
        ),
        Section(
            heading="Tracking, time-to-collision and alerts",
            body=[
                "Validated against a synthetic scenario with an analytically known answer: a "
                "lead vehicle at 45&nbsp;m closing at 8&nbsp;m/s, and an ego lane change at "
                "2.2&nbsp;s.",
            ],
            table=(
                ["Quantity", "Error vs. analytic truth"],
                [
                    ["Ground-plane distance", "0.038 m mean, 0.10 m max"],
                    ["Time-to-collision", "0.008 s mean, 0.016 s max"],
                    ["Lane change detected at", "2.20 s (truth 2.20 s)"],
                    ["Collision warning fired at", "3.16 s (truth 3.125 s &mdash; within one frame)"],
                ],
            ),
            figures=[
                Figure(
                    "collision_alert.png",
                    "Forward collision warning, rendered",
                    "The HUD reports the tracked lead object, its distance in metres, the "
                    "time-to-collision and which estimator produced it.",
                    "These figures come from exact synthetic boxes on perfectly flat ground. "
                    "They validate the geometry and the rules, and say nothing about real-world "
                    "accuracy, which is dominated by road curvature, vehicle pitch and pixel "
                    "quantisation &mdash; and degrades with the square of distance.",
                ),
            ],
        ),
        Section(
            heading="Performance",
            body=[
                "Measured on the target machine: RTX 5050 Laptop (8&nbsp;GB), fp16, "
                "1280&times;720 letterboxed to 640.",
            ],
            table=(
                ["Configuration", "FPS", "ms/frame", "GPU memory"],
                [
                    ["YOLOv8n, batch 1", "134", "7.45", "65 MiB"],
                    ["YOLOv8n, batch 8", "455", "2.20", "65 MiB"],
                    ["YOLOv8m, batch 1", "57", "17.6", "65 MiB"],
                    ["Tracking + TTC + fusion (CPU)", "~4,200", "0.24", "&mdash;"],
                ],
            ),
        ),
    ]


CSS = """
:root{
  color-scheme:light;
  --bg:#f4f5f2; --surface:#fff; --sunk:#eceee8;
  --ink:#16191a; --ink2:#454b4d; --muted:#6b7370;
  --line:#dcdfd8; --accent:#17864a; --warn:#a86a00;
  --shadow:0 1px 2px rgba(20,25,20,.05),0 6px 18px -10px rgba(20,25,20,.16);
}
@media (prefers-color-scheme:dark){
  :root{
    color-scheme:dark;
    --bg:#131614; --surface:#1b1f1c; --sunk:#22271f;
    --ink:#ecefea; --ink2:#c2c9c1; --muted:#8e968d;
    --line:#2c322d; --accent:#4ed07f; --warn:#e0a340;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 26px -14px rgba(0,0,0,.7);
  }
}
*,*::before,*::after{box-sizing:border-box}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:20px;margin-bottom:36px}
.eyebrow{font:500 11.5px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h1{font-size:clamp(28px,4.5vw,44px);line-height:1.05;letter-spacing:-.02em;margin:0 0 14px;
  text-wrap:balance;max-width:20ch}
.standfirst{font-size:17.5px;color:var(--ink2);margin:0;max-width:66ch}
.banner{margin:22px 0 0;padding:14px 18px;background:var(--sunk);
  border-left:3px solid var(--warn);border-radius:0 4px 4px 0}
.banner p{margin:0;font-size:15px;color:var(--ink2);max-width:80ch}
.banner strong{color:var(--ink)}
section{margin:0 0 52px}
h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
  margin:0 0 16px;padding-bottom:8px;border-bottom:1px solid var(--line);font-weight:700}
section p{max-width:70ch;color:var(--ink2);margin:0 0 14px}
section p strong{color:var(--ink)}
figure{margin:24px 0 0;background:var(--surface);border:1px solid var(--line);
  border-radius:6px;overflow:hidden;box-shadow:var(--shadow)}
figure img{display:block;width:100%;height:auto}
figcaption{padding:14px 18px 16px;border-top:1px solid var(--line)}
figcaption b{display:block;font-size:15px;margin-bottom:5px;color:var(--ink)}
figcaption span{font-size:14px;color:var(--ink2)}
figcaption .note{display:block;margin-top:10px;padding-left:11px;
  border-left:2px solid var(--warn);font-size:13.5px;color:var(--muted)}
.tbl{overflow-x:auto;margin:6px 0 4px}
table{border-collapse:collapse;width:100%;font-size:14.5px;min-width:420px}
th{text-align:left;font:500 11px/1.4 ui-monospace,Menlo,Consolas,monospace;
  letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  padding:0 16px 9px 0;border-bottom:1px solid var(--line)}
td{padding:10px 16px 10px 0;border-bottom:1px solid var(--line);color:var(--ink2)}
td:first-child{color:var(--ink)}
tr:last-child td{border-bottom:none}
code{font:13.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--sunk);padding:1.5px 5px;border-radius:3px}
a{color:var(--accent)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  font-size:13.5px;color:var(--muted)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def img_src(fig: Figure, assets: Path, embed: bool) -> str:
    if not embed:
        return f"assets/{fig.file}"
    p = assets / fig.file
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def render(sections: list[Section], assets: Path, embed: bool) -> str:
    parts: list[str] = []
    for sec in sections:
        parts.append(f"<section>\n<h2>{sec.heading}</h2>")
        for para in sec.body:
            parts.append(f"<p>{para}</p>")
        if sec.table:
            head, rows = sec.table
            parts.append('<div class="tbl"><table><thead><tr>')
            parts += [f"<th>{h}</th>" for h in head]
            parts.append("</tr></thead><tbody>")
            for row in rows:
                parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
            parts.append("</tbody></table></div>")
        for fig in sec.figures:
            src = img_src(fig, assets, embed)
            note = f'<span class="note">{fig.note}</span>' if fig.note else ""
            parts.append(
                f'<figure><img src="{src}" alt="{html.escape(fig.title)}" loading="lazy">'
                f"<figcaption><b>{fig.title}</b><span>{fig.caption}</span>{note}</figcaption>"
                f"</figure>"
            )
        parts.append("</section>")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADAS Perception &mdash; Results</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="eyebrow">ADAS perception &middot; monocular camera only</p>
  <h1>Pipeline results</h1>
  <p class="standfirst">Lane detection, object detection, multi-object tracking,
  time-to-collision and a rules-based alert engine, running on recorded dashcam
  video. Generated {date.today().isoformat()}.</p>
  <div class="banner"><p><strong>No model has been fine-tuned on the BDD100K
  dataset.</strong> The 70,000 images were supplied without their label file, so
  every detection here is inference with COCO-pretrained weights. No mAP is
  quoted anywhere, because scoring requires ground-truth boxes that are not
  present.</p></div>
</header>
{"".join(parts)}
<footer>Generated by <code>demo/build_report.py</code>. Open directly in a
browser &mdash; no server required.</footer>
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--embed", action="store_true",
                    help="inline images as data URIs -> one self-contained file")
    args = ap.parse_args(argv)

    assets = args.out / "assets"
    if not assets.is_dir():
        raise SystemExit(
            f"no assets directory at {assets}.\n"
            f"Populate it with the result images first."
        )
    sections = build_sections()

    missing = [f.file for s in sections for f in s.figures if not (assets / f.file).is_file()]
    if missing:
        raise SystemExit(f"missing images in {assets}: {', '.join(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    page = render(sections, assets, args.embed)
    name = "report_standalone.html" if args.embed else "index.html"
    path = args.out / name
    path.write_text(page, encoding="utf-8")

    size = path.stat().st_size / 1e6
    print(f"wrote {path}  ({size:.2f} MB)")
    print(f"open it with:  file:///{path.resolve().as_posix()}")
    if not args.embed:
        print(f"images stay in {assets} -- keep the folder beside the html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
