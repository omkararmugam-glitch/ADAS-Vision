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

## 0. Distillation run — a note on what it does and does not show

The BDD100K images arrived without their label file, so before any ground truth
exists the only training possible is **distillation**: YOLOv8m labels the frames,
and YOLOv8n is trained to reproduce those labels.

**Any mAP in this section measures agreement with the teacher, not correctness.**
The student cannot detect what the teacher misses and faithfully inherits its
mistakes — including the false positives on the ego vehicle's own hood. It is a
real training run producing a real checkpoint, and it validates the whole
training pipeline end to end, but it is not an evaluation.

| | |
|---|---|
| Teacher | yolov8m.pt (COCO-pretrained), conf ≥ 0.4 |
| Student | yolov8n.pt |
| Images | 2,500 pseudo-labelled — 2,000 train / 500 val |
| Boxes | 14,279 (vehicle 12,958 · person 1,210 · bicycle 69 · motorcycle 42) |
| Empty frames | 155 (6.2%) |

| Metric (vs. pseudo-labels) | Value |
|---|---|
| mAP@0.5 | **0.6097** |
| mAP@0.5:0.95 | **0.4095** |
| Precision | 0.6306 |
| Recall | 0.5852 |

Per class, which is where the story actually is:

| Class | Train boxes | Val instances | mAP@0.5 |
|---|---|---|---|
| vehicle | 12,958 | 2,535 | **0.910** |
| person | 1,210 | 251 | 0.663 |
| bicycle | 69 | 16 | 0.591 |
| motorcycle | 42 | 8 | **0.275** |

The prediction made before training — that 42 motorcycle examples would teach the
model nothing — is confirmed: motorcycle recall is 0.125, i.e. it finds one in
eight. The headline 0.61 conceals that completely, which is precisely why the
per-class table is here.

Agreement with the teacher on held-out frames, as a distillation check:

| Model | Objects found | Agreement with teacher |
|---|---|---|
| yolov8n, stock COCO | 36 | 59.9% |
| yolov8n, distilled | 38 | **65.8%** |
| yolov8m, teacher | 53 | 100% (reference) |

### Data-scale ablation: 50 training images

Same student, same teacher, same held-out 500-image val split — only the training
set size changed. This makes the comparison clean.

| | 2,000 train images | **50 train images** |
|---|---|---|
| Train boxes | ~11,400 | 319 |
| Optimiser steps | ~1,875 | ~160 |
| mAP@0.5 | 0.610 | **0.095** |
| mAP@0.5:0.95 | 0.410 | 0.063 |
| Recall | 0.585 | **0.042** |
| vehicle mAP@0.5 | 0.910 | 0.381 |
| person mAP@0.5 | 0.663 | **0.000** |
| bicycle mAP@0.5 | 0.591 | **0.000** |
| motorcycle mAP@0.5 | 0.275 | **0.000** |

Recall 0.042 means it finds **one object in twenty-four**. Three of the four
classes score exactly zero.

The cause is not only "too little data". Fine-tuning to a different class count
**replaces the detection head**, which must then be relearned from scratch — and
50 images gives roughly 160 optimiser steps to do it. A confidence sweep confirms
the model is not blind but profoundly unconfident: its highest score on any frame
is 0.456, and below conf 0.01 it still emits only `vehicle`.

This is the same lesson the motorcycle class taught at 42 examples in the full
run, applied to every class at once.

Distillation moved the student ~6 points closer to the teacher. It also
**amplified the teacher's ego-hood false positive** rather than fixing it — the
student's hood box is larger and more confident than the stock model's. That is
the "inherits the teacher's mistakes" caveat, observed rather than predicted.

The class imbalance is worth noting on its own: 91% of the boxes are `vehicle`,
and `motorcycle` has 42 examples. Nothing trained on this distribution will learn
motorcycles, and a headline mAP averaged over four classes would hide that
entirely — which is why per-class numbers are reported below rather than a single
figure.

Replacing the pseudo-labels with `det_train.json` turns every table below from
"agreement" into "correctness" with no other change to the pipeline.

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
advance. **Both are implemented** (`models/pothole_det/`) behind one interface,
sharing the same trainers as the vehicle detector and the weather classifier, so
the comparison is like-for-like. Awaiting a road-damage dataset; BDD100K carries
no such labels.

The trade-off the numbers will settle:

| | Localises | Label cost | Weakness |
|---|---|---|---|
| Detection | yes — enables the ego-corridor test | box labels | pothole boundaries are ill-defined, so annotator disagreement caps achievable IoU |
| Patch classifier | no — grid cell only | image-level, far cheaper and more consistent | no position, so every hazard reads as equally relevant |

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

Measured on **real BDD100K frames**, which is the number that counts. The
earlier figures in this table were taken on random-noise frames; noise yields
almost no detections and therefore almost no NMS work, so it flattered the
result. Both are kept so the difference is visible.

| Configuration | Batch | FPS | ms/frame | Objects / 8 frames |
|---|---|---|---|---|
| YOLOv8n (COCO) | 1 | **145** | 6.89 | 19 |
| YOLOv8n (COCO) | 8 | **363** | 2.76 | 19 |
| YOLOv8m (COCO) | 1 | **92** | 10.84 | 31 |
| YOLOv8m (COCO) | 8 | **171** | 5.83 | 31 |
| YOLOv8n distilled | 1 | **146** | 6.86 | 19 |
| YOLOv8n distilled | 8 | **369** | 2.71 | 19 |

The distilled student is indistinguishable from stock `n` in speed — same
architecture — while finding the same number of objects and agreeing more closely
with the teacher. `m` costs about 1.6x the time for 63% more objects found.

**YOLOPv2 is configured but its weights are absent**, so build step 6's
multi-task-versus-baseline comparison is set up and cannot be run. Restoring
`models/perception/weights/yolopv2.pt` completes it with no code change.

Earlier noise-frame figures, retained for contrast:

| Configuration | Batch | FPS | ms/frame | GPU mem | Notes |
|---|---|---|---|---|---|
| YOLOv8n (noise frames) | 1 | 134 | 7.45 | 65 MiB | optimistic |
| YOLOv8n (noise frames) | 4 | 378 | 2.65 | 65 MiB | optimistic |
| YOLOv8n (noise frames) | 8 | 455 | 2.20 | 65 MiB | optimistic |
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
