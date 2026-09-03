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
| 🔧 | Lane-change detection | step 4 |
| 🔧 | Collision detection / TTC | step 4 |
| 🔧 | Traffic light & sign detection | step 3 |
| 🔧 | Weather classification | step 3 |
| 🔧 | Pothole / road damage | step 3 |
| 🔧 | Fusion / alert engine | step 5 |

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

# 3. Render the annotated video from the cache. Loads no model.
python run.py render --source drive.mp4 --out annotated.mp4
```

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
