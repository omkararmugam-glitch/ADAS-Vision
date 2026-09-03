# Evaluation results

**Only §7 (inference speed) has measurements so far.** Everything else is
accuracy, and accuracy needs a dataset that has not been supplied yet.

The empty tables are deliberate and are written first: agreeing the metrics
before any model is trained removes the temptation to pick a favourable one
afterwards. They get filled in at build step 3 (per-module) and step 7
(end-to-end).

Populated by scripts in `eval/`. Every row must record the exact checkpoint and
the dataset split it was measured on — a number without both is not a result.

---

## Protocol

- Metrics are reported on the **test** split, never on val. Val is for choosing
  checkpoints; reusing it as a headline number is self-report.
- Every run records: checkpoint hash, dataset name and `adapter_version`, split
  sizes, and the config hash. These come from the cache manifest automatically.
- Latency is measured on the target hardware (RTX 5050 Laptop, 8 GB) at the
  resolution the pipeline actually runs, not at a favourable one.

---

## 1. Object detection — mAP@0.5

| Model | Dataset / split | vehicle | person | bicycle | motorcycle | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| YOLOv8 baseline | — | — | — | — | — | — | — |
| YOLOPv2 (vehicles only) | — | — | n/a | n/a | n/a | — | — |
| VRU model | — | n/a | — | — | — | — | — |

## 2. Lane segmentation — mIoU

| Model | Dataset / split | IoU lane | IoU drivable | mIoU | Pixel acc. |
|---|---|---|---|---|---|
| YOLOPv2 lane head | — | — | — | — | — |
| Separate lane model (baseline) | — | — | — | — | — |

Segmentation IoU alone does not say whether the *geometry* is right. Also report
lane-fit quality, which is what the pipeline actually consumes:

| Metric | Value |
|---|---|
| Frames with `LaneStatus.OK` | — |
| Mean lane width error vs. ground truth (m) | — |
| Lateral offset RMSE at 10 m (m) | — |

## 3. Weather classification — accuracy / F1

| Model | Dataset / split | Accuracy | Macro F1 | Per-class F1 |
|---|---|---|---|---|
| ResNet18 | — | — | — | — |
| EfficientNet-B0 | — | — | — | — |

Confusion matrix goes here. The pair that matters is rain vs. fog — they call for
different threshold adjustments and look similar at low resolution.

## 4. Pothole / road damage

Detector **or** binary classifier — the choice gets made on these numbers, not in
advance. See the open question in `ARCHITECTURE.md`.

| Approach | Dataset / split | Precision | Recall | mAP@0.5 | Notes |
|---|---|---|---|---|---|
| YOLOv8 class | — | — | — | — | — |
| Binary classifier on road patches | — | — | — | n/a | — |

Precision is weighted above recall here. A hazard warning that fires on every
tar seam gets ignored, and an ignored warning is worse than no warning.

## 5. Fusion engine — alert quality

Scored against the event intervals in the video datasets (see
`DATASET_FORMAT.md` §4). An alert counts as a true positive if it fires inside
the labelled interval.

| Alert | TP | FP | FN | Precision | Recall | Mean lead time (s) |
|---|---|---|---|---|---|---|
| Lane departure | — | — | — | — | — | — |
| Forward collision | — | — | — | — | — | — |
| Pedestrian in path | — | — | — | — | — | — |
| Red-light running | — | — | — | — | — | — |
| Road hazard | — | — | — | — | — | — |

**Lead time is the number that matters** for the collision and pedestrian alerts.
A warning that is correct but fires 0.3 s before impact has no practical value;
one that fires 2.5 s ahead is actionable. Precision and recall alone hide this
entirely.

## 6. Distance and TTC accuracy

The honest one. Reported per distance band, because monocular error grows with
the square of distance and a single aggregate number hides that completely.

| Range | n | Mean abs. error (m) | Rel. error | TTC abs. error (s) |
|---|---|---|---|---|
| 0–10 m | — | — | — | — |
| 10–20 m | — | — | — | — |
| 20–35 m | — | — | — | — |
| 35–50 m | — | — | — | — |
| > 50 m | — | — | — | — |

Also to be reported: homography-vs-MiDaS disagreement rate, and how often the
flat-ground assumption is violated badly enough to matter.

## 7. Inference speed

Measured on the target machine: RTX 5050 Laptop (8 GB), fp16, 1280x720 input
letterboxed to 640 long side.

| Configuration | Batch | FPS | ms/frame | GPU mem | Notes |
|---|---|---|---|---|---|
| YOLOv8n (detection only) | 1 | **134** | 7.45 | 65 MiB | measured 2026-09-03 |
| YOLOv8n (detection only) | 4 | **378** | 2.65 | 65 MiB | measured 2026-09-03 |
| YOLOv8n (detection only) | 8 | **455** | 2.20 | 65 MiB | measured 2026-09-03 |
| YOLOPv2 only | — | — | — | — | — |
| YOLOPv2 + VRU | — | — | — | — | — |
| YOLOv8 baseline + lane model | — | — | — | — | — |
| Full pipeline (+ tracking + depth + fusion) | — | — | — | — | — |
| Full pipeline, ONNX | — | — | — | — | — |

Two caveats on the YOLOv8n rows, so they are not read as more than they are.
They were measured on **random-noise frames**, which yield almost no detections
and therefore almost no NMS work; a busy real scene will be slower. And this is
the **nano** model — the smallest of the family, chosen as a starting point, not
as the accuracy-optimal choice. The 65 MiB figure is why: it leaves essentially
the whole 8 GB budget for MiDaS and the tracker, which is the real constraint.

The baseline-vs-multi-task comparison is the point of build step 6. Expect the
full pipeline to be memory-bound on 8 GB rather than compute-bound.
