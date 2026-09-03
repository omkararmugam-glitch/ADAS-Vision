# Unified dataset format

Every module in this project reads one of the four layouts below. Raw third-party
datasets are never read directly by training or evaluation code — an **adapter**
in `data/adapters/` converts each one first. That boundary is deliberate: it keeps
each dataset's quirks (BDD100K's nested JSON, CULane's `.lines.txt`, Cityscapes'
`labelIds` palette) contained in one write-once script instead of leaking into the
training loop as special cases.

All adapters write into `data/processed/<dataset-name>/`. Raw downloads live in
`data/raw/<dataset-name>/` and are never modified.

---

## Conventions shared by all four layouts

| Rule | Value |
|---|---|
| Splits | `train/`, `val/`, `test/` — directory-level, never a CSV of filenames |
| Image format | `.jpg` or `.png`, RGB, any resolution |
| Stem matching | A label file's stem **must** equal its image's stem (`000123.jpg` ↔ `000123.txt`) |
| Coordinates | Normalised to `[0, 1]` by image width/height, so resizing never invalidates a label |
| Missing labels | An image with no objects gets an **empty** label file, not a missing one |
| Metadata | Each dataset root carries a `dataset.yaml` (below) |

A missing label file and an empty one mean different things: empty is "verified to
contain nothing", missing is "never annotated". Conflating them silently poisons a
detector with false negatives, so the loaders treat a missing file as an error.

### `dataset.yaml`

```yaml
name: bdd100k-det
task: detection            # detection | segmentation | classification
source_url: https://...
adapter: data/adapters/bdd100k.py
adapter_version: 1
created: 2026-09-03
classes: [vehicle, person, bicycle, motorcycle]   # index = class id
splits: {train: 52341, val: 7218, test: 0}
notes: |
  Free-text. Record anything that will surprise you in three weeks — occlusion
  policy, whether crowds are labelled, resolution changes mid-dataset.
```

---

## 1. Object detection — YOLO format

Used by: `models/perception/` (YOLOv8 baseline), `models/pothole_det/`.

```
data/processed/<name>/
├── dataset.yaml
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt
```

One line per object, space-separated:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four geometry values normalised to `[0, 1]`. This is Ultralytics' native
layout, so the YOLOv8 baseline trains on it with no conversion step.

**Class ids must match `core/types.py:CLASS_NAMES`** for anything feeding the main
pipeline: `0 vehicle`, `1 person`, `2 bicycle`, `3 motorcycle`. Traffic lights,
signs and potholes extend this vocabulary — see the open question at the end.

## 2. Lane / drivable segmentation — indexed masks

Used by: `models/lane_seg/`.

```
data/processed/<name>/
├── dataset.yaml
├── images/{train,val,test}/*.jpg
└── masks/{train,val,test}/*.png
```

Masks are **single-channel 8-bit PNG**, same width/height as their image, where
each pixel holds a class index — *not* a colour:

| Value | Meaning |
|---|---|
| 0 | background |
| 1 | lane marking |
| 2 | drivable area (optional; omit if the dataset lacks it) |

Indexed rather than RGB because a palette PNG saved by one tool and read by
another silently changes colours; an index cannot drift. Masks must be read and
resized with **nearest-neighbour** interpolation only — interpolating between
class 0 and class 1 invents boundary values that mean nothing.

Datasets that ship lanes as polylines (CULane, TuSimple) are rasterised by their
adapter at a documented stroke width, recorded in `dataset.yaml:notes`.

## 3. Whole-frame classification — folder-per-class

Used by: `models/weather_cls/`.

```
data/processed/<name>/
└── {train,val,test}/<class_name>/*.jpg
```

The directory name *is* the label. Class order is fixed by the sorted list in
`dataset.yaml:classes`, so a class appearing later never renumbers the others and
invalidates a trained checkpoint.

Suggested weather classes — **confirm before I build the adapter**:
`clear`, `rain`, `snow`, `fog`, `night`. Note that `night` is an illumination
state, not a weather condition, and can co-occur with the others; if your dataset
labels both, this should be two heads rather than one 5-way softmax.

## 4. Video + event timeline — for end-to-end fusion evaluation

Used by: `eval/` to score the fusion engine (lane-change and alert timing).

```
data/processed/<name>/
├── dataset.yaml
├── videos/*.mp4
└── events/*.json        # stem matches the video
```

```json
{
  "video": "clip_004.mp4",
  "fps": 30.0,
  "events": [
    {"t_start": 12.40, "t_end": 14.10, "type": "lane_change", "direction": "left"},
    {"t_start": 31.05, "t_end": 33.60, "type": "forward_collision_risk"}
  ]
}
```

Events are **intervals, not instants**. A lane change takes about 1.5 s and no two
annotators will agree on a single frame for it, so scoring against an instant
measures annotator jitter rather than model quality. `eval/` scores an alert as a
true positive when it fires anywhere inside the interval.

---

## Adapter contract

Each adapter is a standalone script:

```bash
python data/adapters/<name>.py --raw data/raw/<name> --out data/processed/<name>
```

It must: be **idempotent** (re-running overwrites, never appends), write
`dataset.yaml` including its own `adapter_version`, refuse to write a partial
dataset if any input is malformed, and print a summary of counts per split and
class. It must **not** resize, augment, or rebalance — those are training-time
decisions, and baking them into the data makes them impossible to change later.

---

## Open questions — I need your input before finalising

1. **Class vocabulary.** The pipeline currently knows four classes. Traffic
   lights and signs, and potholes, need to be added. Signs especially: one
   `traffic_sign` class, or per-type (`stop`, `speed_limit`, `yield`, …)? Per-type
   is far more useful for the decision engine but needs many more labels.
2. **Traffic light state.** Red/amber/green must be recoverable. As separate
   detection classes (`light_red`, `light_green`, …) or as one `traffic_light`
   box plus a small colour classifier on the crop? The second is more robust at
   distance, where a light is only a few pixels across.
3. **Weather labels.** Per-frame or per-clip? Per-clip is much cheaper to
   annotate and weather rarely changes mid-clip, but it leaks between train/val
   if frames from one clip land in both splits — splits must then be by clip.
4. **Which dataset first?** Send me a sample — a handful of images plus their
   annotation files, and the directory tree (`tree -L 3`) — and I will write that
   adapter and validate it against this spec.
