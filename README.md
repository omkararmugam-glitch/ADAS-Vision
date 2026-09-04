# ADAS — monocular multi-module perception

Offline ADAS perception over recorded dashcam video. Takes a video in, produces an
annotated video and a structured per-frame log of detections and alerts.

Camera only — no LiDAR, no radar. Distances are metric via a calibrated
ground-plane homography, with documented limits (see
[ARCHITECTURE.md](docs/ARCHITECTURE.md#distance-estimation--and-why-not-midas-alone)).

## Capabilities

| | Capability | State |
|---|---|---|
| ✅ | Lane detection (segmentation + geometric fit in metres) | working |
| ✅ | Vehicle detection | working |
| ✅ | Pedestrian / cyclist detection | working |
| ✅ | Multi-object tracking (ByteTrack) | working |
| ✅ | Lane departure & lane-change detection | working |
| ✅ | Collision detection / time-to-collision | working |
| ✅ | Fusion / alert engine (JSON + CSV logs) | working |
| ✅ | Monocular depth (MiDaS), scale-aligned | working, off by default |
| ⛔ | Traffic light & sign detection | needs BDD100K labels |
| ⛔ | Weather classification | needs BDD100K labels |
| ⛔ | Pothole / road damage | needs a pothole dataset |

⛔ items are blocked on data, not on code. See [Datasets](#datasets).

## Setup

Install torch **first**, from the CUDA 12.8 index. Default PyPI wheels carry no
sm_120 kernels and fail at runtime on Blackwell GPUs, not at install time:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

Verified on Python 3.14 + torch 2.11.0+cu128 + RTX 5050 Laptop (sm_120).

## Usage

```bash
# 1. Calibrate the camera. Required before any distance is meaningful.
python run.py calibrate intrinsics --source calib_chessboard.mp4
python run.py calibrate homography --source drive.mp4
python run.py calibrate verify     --source drive.mp4

# 2. Run perception. Fills a content-addressed cache; resumable.
python run.py predict --source drive.mp4

# 3. Tracks, TTC and alerts from the cache. Loads no model.
python run.py analyse --source drive.mp4

# 4. Render the annotated video from the cache. Loads no model.
python run.py render --source drive.mp4 --out annotated.mp4
```

`analyse` writes `alerts.json` (every frame plus an episode summary),
`alerts.csv` (one row per alert) and `episodes.csv` beside the cache. It runs at
roughly 4,000 frames/second, because tracking, TTC and fusion are excluded from
the cache key — retuning an alert threshold re-derives every event across a clip
in under a second.

`predict` and `render` are separate on purpose — restyling the overlay or
retuning a threshold never re-runs inference. `render` works with the weights
file absent entirely.

Try it without any model at all:

```bash
python run.py predict --source drive.mp4 --detector stub
```

Switch backbones with one config value — `detector.kind` is `yolopv2`
(multi-task), `yolov8` (single-task baseline, step 3), or `stub`.

## Layout

```
core/       config, IO, calibration, cache, typed records, overlay  (no torch)
models/     perception/ lane_seg/ weather_cls/ pothole_det/
tracking/   ByteTrack                          depth/    MiDaS
fusion/     rules-based alert engine           eval/     metrics
demo/       dashboard, ONNX benchmark          data/     raw/ processed/ adapters/
configs/    YAML — every tunable value lives here, never in a module body
docs/       ARCHITECTURE.md  DATASET_FORMAT.md  EVAL_RESULTS.md
```

## Tests

```bash
.venv/Scripts/python -m pytest
```

216 tests, no GPU or weights required. They cover the letterbox round-trip,
calibration geometry, cache invalidation and resume, lane fitting against
synthetic ground truth, and the CLI end-to-end via the stub detector.

## Current state

Stages 1–2 are working: calibration, IO, caching, the YOLOPv2 multi-task backbone
with a separate pedestrian model, lane fitting, and annotated output. Stages 3–8
are scaffolded placeholders.

Two things need attention before the next stage — see
[ARCHITECTURE.md](docs/ARCHITECTURE.md):

- **Camera calibration must be regenerated.** `calib/` was gitignored and its
  contents were lost; every metric output depends on it.
- **Datasets are needed** to build the adapters. The expected format is specified
  in [DATASET_FORMAT.md](docs/DATASET_FORMAT.md), which ends with the open
  questions I need answered first.


## Datasets

The data layer is dataset-agnostic; see [DATASET_FORMAT.md](docs/DATASET_FORMAT.md)
for the expected layout and `data/adapters/` for the converters.

**BDD100K.** The adapter both converts *and* curates — 70,000 frames is far more
than fine-tuning needs, and an unfiltered slice spends the budget on frames that
teach nothing:

```bash
python data/adapters/bdd100k.py --labels path/to/det_train.json
```

Defaults select 12,000 images at 45% night, filtered by minimum box height.
Night is deliberately over-represented because it is the measured weak case.

> **Images alone are not enough.** BDD100K ships images and labels as separate
> downloads. Without `det_train.json` there is nothing to train against and no
> ground truth to score with — which is why several capabilities above are marked
> blocked rather than merely unfinished.

## Training and evaluation

```bash
# fine-tune a detector on any dataset in the documented layout
python eval/train_detector.py --data data/processed/bdd100k/dataset.yaml

# score alerts against labelled event intervals (reports LEAD TIME, not just P/R)
python eval/score_alerts.py --predicted run/episodes.csv --truth events.csv

# export to ONNX and benchmark — verifies both engines agree before quoting speed
python eval/export_onnx.py --weights models/perception/weights/yolov8m.pt
```

Without ground truth, `eval/train_detector.py` can still run against pseudo-labels
produced by a larger teacher (`data/adapters/pseudo_label.py`). That is genuine
distillation and produces a genuine checkpoint — but **mAP against pseudo-labels
measures agreement with the teacher, not correctness**, and the tooling says so
in its own output rather than leaving it to be remembered.

## Demo

```bash
python demo/test_dataset.py --n 120     # annotate a dataset sample + gallery
python demo/build_report.py             # standalone HTML results report
```

Both write to `outputs/` and open with `file://` — no server, no framework, no CDN.

## Tests

```bash
python -m pytest
```

412 tests. The ML models are not the interesting part to test; the geometry and
the rules are. TTC is verified against analytically-known scenarios, the letterbox
transform is round-trip tested, and every fusion rule is exercised with no video,
no weights and no GPU.
