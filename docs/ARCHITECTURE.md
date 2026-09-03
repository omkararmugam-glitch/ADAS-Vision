# Architecture

## Status

| Stage | Capability | State |
|---|---|---|
| 1 | Repo scaffold, typed config, CLI | **done** |
| 1 | Camera calibration (intrinsics + ground homography) | **done** — needs re-running, see below |
| 1 | Frame IO, canonical sizing, container timestamps | **done** |
| 1 | Content-addressed run cache, `predict`/`render` split | **done** |
| 2 | YOLOPv2 multi-task backbone (boxes + drivable + lane) | **done** |
| 2 | VRU detector (pedestrian/cyclist) | **done** |
| 2 | Lane fitting in bird's-eye metres | **done** |
| 2 | Overlay + annotated video | **done** |
| 3 | Dataset adapters | spec written, blocked on a dataset sample |
| 3 | YOLOv8 baseline | **working** on COCO-pretrained weights; not fine-tuned |
| 3 | Weather classifier | placeholder |
| 3 | Pothole detector | placeholder |
| 4 | ByteTrack tracking | placeholder |
| 4 | MiDaS depth | placeholder |
| 4 | Lane-change detection, TTC | not started |
| 5 | Fusion / decision engine | placeholder |
| 7 | Evaluation scripts | placeholder |
| 8 | ONNX export, FPS benchmark, dashboard | not started |

> **Calibration must be regenerated.** `calib/*.yaml` was excluded by
> `.gitignore` and was lost when the working tree was cleared. Distance, TTC and
> lane geometry are all metric and all depend on it. Run
> `python run.py calibrate` before trusting any distance the pipeline reports.

---

## Layout

```
core/       shared infrastructure - imports no model, no torch
models/     one subpackage per capability: wrapper code + its weights/
tracking/   ByteTrack
depth/      MiDaS
fusion/     rules-based decision engine
eval/       metrics
demo/       dashboard, ONNX benchmark
data/       raw/, processed/, adapters/
```

`core/` exists because the prescribed structure has no home for config, IO,
calibration, caching and the typed records that every module passes around.
Putting them in `models/` would have made the model packages import each other.

**`core/` never imports torch.** This is enforced by what lives where, and it is
what makes `python run.py render` work with no weights and no deep-learning stack
installed. `LetterboxTransform` sits in `core/letterbox.py` for exactly this
reason — it is pure geometry, and the render path needs it to restore cached
masks.

---

## The `predict` / `render` split

These are two commands, not two halves of one:

```bash
python run.py predict --source drive.mp4   # runs models, fills the cache
python run.py render  --source drive.mp4   # cache -> annotated.mp4, loads no model
```

Inference is the expensive, deterministic part; presentation is the cheap part
that gets iterated on twenty times. Restyling an overlay, or later retuning an
alert threshold, must never re-run a network. `render` is tested with the weights
file physically absent, so the separation cannot rot.

The cache key covers the config sections that change *results* (`io`,
`calibration`, `detector`) plus the source fingerprint, the calibration file hash,
and the model's name and weights digest. It deliberately excludes `overlay` and
`video`, so changing those re-renders from the existing cache. Swapping weights
changes the digest and invalidates automatically — model identity comes from the
detector, never from a config string that can lie.

Writes are resumable: an interrupted 20-minute run continues from the last
completed frame rather than starting over.

---

## Backbone: baseline vs multi-task

Both live behind `detector.kind`, so switching is a one-line config change and the
benchmark in `EVAL_RESULTS.md` is honest.

**`yolopv2`** — one forward pass yields vehicle boxes, drivable-area segmentation
and lane-marking segmentation. Shares a backbone across three heads, so it is
substantially cheaper than running two networks. Its cost is coupling: the tasks
share weights, so retraining for a new class risks the lane head.

**`yolov8`** — separate detection model plus a separate lane segmentation model.
More compute, but each task trains, evaluates and swaps independently. This is
the baseline the build order asks for and the fallback if fine-tuning YOLOPv2 on
your data goes badly.

**Why a second model for pedestrians.** YOLOPv2 has no pedestrian class, and no
released multi-task network in that family does — HybridNets ships
`obj_list: ['car']`. Vulnerable road users therefore come from a separate COCO
model (`models/perception/vru.py`) which emits **only** person/bicycle/motorcycle;
vehicles come from YOLOPv2 alone. Taking vehicles from both would double-count
every car in the frame. `CompositeDetector` merges them: masks from the primary,
boxes concatenated. Under `yolov8` this second model is not attached, because
YOLOv8 already detects people and stacking one on would double-count pedestrians.

---

## Distance estimation — and why not MiDaS alone

This is the most important accuracy decision in the project, because every
collision warning depends on it.

**MiDaS predicts relative inverse depth, not metres.** Its output is defined only
up to an unknown scale *and* shift, both of which drift between frames. You cannot
convert it to a distance in metres without fitting it to some metric reference,
and a TTC computed from unfitted MiDaS output is not a physical quantity.

The calibrated **ground-plane homography** is a better metric reference for the
common case. For any object whose contact point with the road is visible, the
bottom-centre of its box back-projects through the homography to a real ground
position in metres. The error sources are understood and bounded:

| Source | Effect |
|---|---|
| Non-flat road (crest, dip, camber) | The dominant error. A 1° pitch change moves a 30 m estimate by several metres |
| Vehicle pitch under braking | Systematic, and worst exactly when a collision warning matters |
| Box bottom is not ground contact | Occluded wheels, or a box clipped at the frame edge, read as *further away* |
| Pixel quantisation | Negligible near, dominant far — error grows as the square of distance |

That last row is why **accuracy degrades sharply with distance**. Near 10 m the
estimate is good to a few tens of centimetres; at 50 m a one-pixel error is
metres. Alert thresholds are set with this in mind, and the design intent is to
report a confidence band rather than a single number.

**The intended division of labour:**

- **Homography** — primary, for objects touching the road.
- **MiDaS** — fallback where ground contact is occluded or the object is
  off-plane, and as a cross-check that *flags disagreement* rather than silently
  averaging two estimates that disagree for good reason.

Fitting MiDaS's scale and shift per frame against the homography's readings for
ground-contact objects would give a metric depth map for the rest of the frame.
That is the plan for step 4, and it is a genuinely better use of MiDaS than
trusting it raw.

---

## Lane fitting

Lane polynomials are `x = a*y^2 + b*y + c` in **bird's-eye metres**, lateral
position as a function of forward distance. Fitting lateral-on-forward rather
than the reverse is what makes a near-vertical lane boundary representable at all
— the reverse form has infinite slope exactly where lanes usually are.

Fitting happens in the bird's-eye view, not the image, because lane width and
curvature are only meaningful in metres. Sliding windows walk up from the ego
vehicle; the search half-width must reach past the adjacent lane's outer edge
(for a 3.5 m lane that is 5.25 m, so the default is 7.0 m — a 5 m search finds no
adjacent lanes at all).

**A fit is never faked.** When evidence does not support a lane model, `LaneStatus`
says which way it failed (`not_detected`, `mask_too_sparse`, `single_boundary`)
and the coefficients are absent. Returning a garbage polynomial that downstream
code treats as real is the failure mode this design exists to prevent.

---

## Tracking — ByteTrack over DeepSORT

Two downstream capabilities cannot be built without persistent IDs: TTC needs the
*same* object's box across consecutive frames to measure scale growth, and
lane-change detection needs lateral offset tracked over time to distinguish a real
crossing from a one-frame lane-fit wobble.

ByteTrack associates using detection confidence alone — including low-confidence
boxes in a second association pass, which is what makes it robust through partial
occlusion. DeepSORT needs an appearance-embedding network, i.e. a second model on
the critical path. On an 8 GB laptop GPU already hosting the backbone and MiDaS,
that cost is not justified by the benefit for road scenes, where objects are
mostly well-separated and motion is predictable.

---

## Fusion

Rules, not a learned policy. Every alert has to be explainable after the fact and
tunable without retraining, and there is no labelled alert dataset to learn from.
The rules are pure functions over typed records, which makes them unit-testable
with no video, no GPU and no weights — the TTC arithmetic and the lane-crossing
state machine get tested directly.

Weather feeds back into thresholds: stopping distance on wet tarmac is roughly
double the dry figure, so a TTC threshold that is right in the dry is too late in
the rain.

---

## Hardware

Target is an **RTX 5050 Laptop (Blackwell, sm_120, 8 GB)**. Blackwell needs CUDA
12.8 wheels; default PyPI torch has no sm_120 kernels and fails at runtime, not at
install. Verified: torch 2.11.0+cu128 on Python 3.14.

8 GB is the real constraint on step 4. Backbone + VRU model + MiDaS + tracker
running together will not all fit comfortably at full resolution, so expect to
either run MiDaS at reduced resolution, run it every Nth frame, or accept a lower
frame rate. This is flagged now rather than discovered at integration time.
