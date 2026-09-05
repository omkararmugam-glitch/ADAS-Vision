# Architecture

## Status

| Stage | Capability | State |
|---|---|---|
| 1 | Repo scaffold, typed config, CLI | **done** |
| 1 | Camera calibration (intrinsics + ground homography) | **done** |
| 1 | Target-free auto-calibration from road geometry | **done** |
| 5 | Traffic light + sign detection, red-light warning | **done** |
| 1 | Frame IO, canonical sizing, container timestamps | **done** |
| 1 | Content-addressed run cache, `predict`/`render` split | **done** |
| 2 | YOLOPv2 multi-task backbone (boxes + drivable + lane) | **done** |
| 2 | VRU detector (pedestrian/cyclist) | **done** |
| 2 | Lane fitting in bird's-eye metres | **done** |
| 2 | Overlay + annotated video | **done** |
| 3 | Dataset adapters | **BDD100K adapter done**, blocked on the label file |
| 3 | YOLOv8 baseline | **working** on COCO-pretrained weights; not fine-tuned |
| 3 | Weather classifier | **code complete** — trainer + adapter written, awaiting labels |
| 3 | Pothole detector | **code complete** — both approaches implemented, awaiting a dataset |
| 4 | ByteTrack tracking | **done** |
| 4 | MiDaS depth | **done** |
| 4 | Lane-change detection, TTC | **done** |
| 5 | Fusion / decision engine | **done** |
| 7 | Evaluation scripts | **done** — detection mAP, alert scoring, FPS benchmark |
| 8 | ONNX export + FPS benchmark | **done** |
| 8 | Demo: HTML report + result gallery | **done** |

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

## Annotated outputs — where every stage writes pictures

A falling loss and a rising mAP are both consistent with a model that has learned
to put boxes in confidently wrong places. No scalar distinguishes that from real
progress; a frame does. So **every stage that produces predictions also writes
what those predictions look like**, and the locations are fixed:

| Stage | Written to | Contents | Controlled by |
|---|---|---|---|
| Training | `outputs/train_previews/<run>/epoch_<N>/` | A **fixed** validation sample, annotated, plus `epoch_<N>_sheet.jpg` | `--preview-every N`, `--preview-images` |
| Evaluation | `outputs/eval_previews/<run>/` | Validation split (evenly sampled if large) + `contact_sheet.jpg` | automatic after training |
| Full pipeline still | `outputs/full_previews/` | **Every** module composited: lanes, boxes, signals, track ids, alerts | `eval/full_preview.py` |
| One module per panel | `outputs/panels/` | The same frame as a grid, one module per titled panel | `eval/panel_preview.py` |
| Per-capability sections | `outputs/sections/<NN>_<name>/` + `index.html` | One folder per capability, each showing only that capability | `demo/build_sections.py` |
| Demo pipeline | `outputs/demo/<source>_annotated.mp4` | Every module composited on one frame | `run.py render` |
| Dataset spot-check | `outputs/tested*/` + `index.html` | Per-frame annotations, filterable gallery | `demo/test_dataset.py` |
| Model comparison | `outputs/compare*/` | Same frame through several checkpoints | `eval/compare_detectors.py` |

The ten sections, in the order the index lists them:

    01_lane_detection/    lane pixels and the fitted boundaries, nothing else
    02_lane_split/        a fork or exit ramp opening up ahead
    03_divided_highway/   a road already split into opposing directions
    04_lane_change/       neighbouring lanes that are free to move into
    05_vehicles/          vehicle boxes only
    06_pedestrians/       people, cyclists, motorcyclists only
    07_traffic_control/   lights with the colour read from their pixels, and signs
    08_distance_ttc/      distance lines from the ego to each object
    09_tracking/          persistent track ids
    10_full_pipeline/     everything composited, as the pipeline really outputs it

`build_sections.py` deletes section folders no longer listed in `SECTIONS`, so a
renumbering cannot leave stale ones behind for the index to disagree with.

`eval/preview.py` holds those paths as module constants, so this table and the
code cannot drift apart.

**Two preview paths, deliberately.** `eval/preview.py` renders the detector
alone, which is the right picture while a *detector* is training -- adding lanes
and tracking would only obscure whether the boxes are improving.
`eval/full_preview.py` renders the whole stack, which is the right picture of the
*system*. Asking one script to do both produced the complaint that started this:
previews that showed vehicles and nothing else.

Full previews run frames **in order, as a sequence**, because three outputs do
not exist for an isolated still: a track id needs a previous frame, a TTC needs a
rate of change, and a lane-departure event needs a history.

**The training sample is fixed, not resampled.** The entire value is flicking
between `epoch_005/` and `epoch_030/` for the *same* image and seeing what
changed; resampling each time would make the comparison meaningless. It is
evenly spaced over the sorted listing rather than random, so it is reproducible
without carrying a seed around.

**Previews never abort a run.** The callback is wrapped: a checkpoint matters
more than a picture, so a preview failure prints and continues.

### What is drawn, and by whom

| Element | Drawn by |
|---|---|
| Boxes, class, confidence | Ultralytics `results.plot()` for plain detection previews; `core/overlay.draw_boxes` in the pipeline |
| Track id (`#7`) | `core/overlay.draw_boxes`, ahead of the class name |
| Distance in metres | `core/overlay.draw_boxes`, when calibration is available |
| Distance **line** ego to object | `core/overlay.draw_distance_lines` |
| Lane fits and masks | `core/overlay.draw_lane_fits`, `draw_mask` |
| Weather / road condition | `core/overlay.draw_status_corner`, top-right |
| Fusion alerts | `core/overlay.draw_hud` warning rows, top-left |

Ultralytics' own renderer is used for detection-only previews per the brief: it
is maintained, it matches what anyone familiar with YOLO expects, and a second
copy of box-drawing code would drift. Custom drawing is used only where
`plot()` cannot help — lane geometry, track ids, the classification corner and
fusion alerts, none of which Ultralytics knows about.

**Distance lines run to the ground-contact point**, not the box centre. The
contact point is the only part of a 2-D box that lies on the road and therefore
the only part the homography can place; drawing to the centre would look tidier
and be wrong by however tall the object is. Only the nearest few are drawn, and
nothing beyond 60 m — monocular error grows with the square of range, and a
confidently-labelled 90 m line implies a precision that does not exist.

**Previews composite lanes even when no objects are detected.** An empty road
with its markings drawn is a correct and informative result; a blank frame reads
as "the system found nothing" when the truth is "the system was only asked about
vehicles". An *implausible* lane mask is still not painted — sky and foliage in
lane colour looks like a confident detection and is read as one.

**The status corner suppresses `unknown`.** A corner permanently reading
"weather: unknown" trains the eye to skip the whole region, including the frames
where it says something real.

**Track ids and distances are matched to detections by box overlap, not index.**
Tracking reorders objects and drops unconfirmed ones, so positional
correspondence with the frame's detections does not survive it.

## Post-processing is not inference

`tracker`, `ttc` and `fusion` are deliberately **excluded from the cache key**.
They run over cached detections at roughly 4,000 frames/second, so retuning an
alert threshold and re-deriving every event across a clip costs well under a
second — no network re-run. This is the same principle as the `predict`/`render`
split, extended to the decision layer:

```bash
python run.py predict --source drive.mp4   # expensive, cached
python run.py analyse --source drive.mp4   # tracks, TTC, alerts. No model.
python run.py render  --source drive.mp4   # annotated video. No model.
```

`analyse` writes three files beside the cache: `alerts.json` (every frame, plus
an episode summary), `alerts.csv` (one row per alert), and `episodes.csv`.

**Episodes, not frames.** A collision warning that holds for 90 frames is one
event a driver experienced, not 90. Scoring per frame would let a single long
warning dominate the metrics, so `eval/` compares episodes against the labelled
intervals in `DATASET_FORMAT.md`.

## Time-to-collision

Two independent estimators, because they fail differently and disagreement is
itself information.

**Scale expansion** needs no calibration. Under a pinhole camera `h = f·H/Z`, so
`1/h` is proportional to distance and — at constant closing speed — falls
*linearly* to zero at contact. The fit is therefore done in inverse-height space,
where a straight line is the exact model rather than an approximation.
Differentiating a noisy height directly gives a wildly unstable TTC; this does
not.

**Ground-plane range** back-projects the box's bottom-centre through the
homography and fits distance against time. Gives metres, but inherits the
flat-road assumption.

Ground-plane range is preferred within 30 m, scale expansion beyond it. The two
are **never averaged**: when they disagree it is because an assumption has
broken, and averaging would hide exactly the case worth flagging, so the gap is
recorded as `disagreement_s` instead.

Validated against an analytic scenario (lead vehicle at 45 m closing at 8 m/s):
distance error 0.038 m mean / 0.10 m max, TTC error 0.008 s mean / 0.016 s max,
and the warning fired within one frame of the analytically correct moment. Those
figures are for exact synthetic boxes on perfectly flat ground — they validate
the *geometry*, and say nothing about real-world accuracy, which is dominated by
the error sources tabulated above.

## Panelled previews

`eval/panel_preview.py` renders each module into its own titled panel — input,
lane pixels, lane fit, detections, distance/tracking, road split ahead, divided
highway, then the composite — rather than stacking every overlay on one frame.

The composite proves the modules run together, but on a busy junction it is
unreadable: boxes, lane curves, distance lines, the gore marker and the HUD all
overlap, and a *wrong* lane fit is indistinguishable from a right one under the
clutter. Each panel carrying only its own ink makes a bad lane fit obvious **as a
bad lane fit**. The composite is kept as the final panel, because the modules do
genuinely interact.

### Known weakness: pedestrian crossings

The classical segmenter takes zebra crossings as lane markings at junctions.
Three filters were tried:

1. **Thinness** (`minAreaRect` minor axis) — crossing bars are thin too, so it
   passes them.
2. **Flat-angle rejection** — under perspective a crossing's bars are not
   horizontal in the image; they run diagonally, steeply enough to survive any
   threshold loose enough to keep genuine markings near the horizon.
3. **Vanishing-point alignment** — correct in principle, and it does reject
   isolated transverse stripes. It fails here for a specific reason: adjacent
   crossing bars **merge into one connected component**, and the merged diagonal
   chain points along the road, so it aligns with the vanishing point and is
   kept. Measured deltas on a real junction were 1.6–10.9°, well inside the
   22° gate.

The filters are retained because each rejects real cases, but none solves this
one. The promising route is the **learned segmenter**
(`models/lane_seg/segmenter.py`), which has the spatial context a per-pixel rule
lacks and can learn that a crossing is not a lane — something no threshold on an
individual component can express.

## The scene view — one colour language

`core/scene.py`. `core/overlay.py` holds the drawing primitives; this composes
them into the single annotated view the system presents, with a fixed palette so
the same fact always looks the same whichever script rendered the frame.

| | Means |
|---|---|
| **Green** | your side: the ego corridor and its left boundary |
| **Red** | do not enter: the right boundary, an exit branch, the oncoming carriageway |
| **Orange** | a free neighbouring lane: a lane change is available |
| **Blue, dashed** | the predicted path down the lane centre — a *path*, deliberately not an edge |

Orange sits between the green of "yours" and the red of "not yours" because
that is exactly what it is: not yours, but you may take it.

Red carries two apparent meanings — "edge of your lane" and "region that is not
yours" — but one idea: crossing it is the thing not to do. That is worth more
than a palette in which every region has a hue and none of them mean anything.

Dashes are spaced in **ground metres**, not pixels, then projected, so they
foreshorten like real paint. A pixel-spaced dash reads as a dotted line drawn on
the screen; a metre-spaced one reads as markings lying on the road.

### Making the far field honest

The reference behaviour asked for is a line that stays on the marking from the
bonnet to the vanishing point. Four things were in the way, and each fix is a
constraint rather than a nudge:

**1. Boundaries were fitted independently.** Each was then extrapolated on its
own pixels, so a boundary supported by a short stub of dashed paint gave a
quadratic that was fine over the stub and left the road beyond it — precisely
the part of the image a driver looks at. `fit_parallel_pair` solves both
boundaries in one least-squares system with a **shared curvature** and two
offsets: four unknowns instead of six. A boundary with support to 35 m now lends
its curvature to one that runs out of paint at 12 m, and the weak boundary's far
end becomes a measurement taken on the other side of the lane. Lane width is
also constant by construction, so the corridor cannot silently fan open.

It is not applied blindly — a shared curvature *cannot* represent a fork. The
parallel model is adopted only when its residual is within
`_PARALLEL_TOLERANCE` (1.25×, plus 4 cm slack) of the independent fit; a
diverging pair blows past that and keeps its independent fits, which is what
`detect_split` needs to see. `LaneSet.parallel_model` records which was used,
and `scene.trusted_range` reads it: the union of supports when parallel, only
the overlap when not.

Measured on 300 random BDD100K frames, of the fits that reach status OK the
parallel model is adopted on **84%**, and the *weaker* boundary's reach goes from
**22.7 m to 37.4 m median**.

**2. Curves were extended toward the camera.** Near the vehicle the
ground-to-image projection is at its most sensitive, so a metre of harmless
lateral error swung the drawn line clear across the bonnet. Near extension is
now zero; the road immediately ahead is covered by `_extend_to_bottom`, a
**straight continuation in image space**, which cannot diverge. It is capped at
28% of frame height — a fit whose near end is already high in the image starts
far away, and continuing it would invent more road than it measured.

**3. Diverging quadratics were drawn in full.** `_project` truncates at
`MAX_LATERAL_M` (9 m), so a curve that leaves the road *stops* instead of
carrying on confidently across scenery.

**4. Curves ran past the parabola's vertex.** Every long boundary ended in a
visible hook where the fitted quadratic turned back on itself. `far_limit` now
stops at the vertex, where `dx/dy` changes sign. That is not a threshold to
tune — no road reverses direction, so the vertex is the exact point beyond
which the model has stopped describing a road at all.

**5. The two corridor edges were extended separately** and could cross before
reaching the bonnet, filling a bowtie. `_extend_pair_to_bottom` drops the
extension for both when that would happen: a corridor stopping a few metres
short is a smaller lie than one folding through itself.

### Lane-change options

`models/lane_seg/neighbours.py`. The fitter says where the boundaries are;
`enumerate_lanes` answers the question a driver actually has — **which of these
lanes could I move into.** Every adjacent pair of boundaries is a candidate
lane, filtered on plausible and roughly constant width (a gap that opens is a
gore; one six metres wide is a median), then classified:

| | |
|---|---|
| `EGO` | the lane being driven |
| `FREE` | same direction, nothing in it — **orange**, a change is available |
| `OCCUPIED` | same direction, but a vehicle is in it |
| `ONCOMING` | beyond the divider of a divided road; never an option |
| `BRANCH` | the exit lane of a fork; an option, but a different journey |

`FREE` versus `ONCOMING` is the distinction that carries the weight, because
geometrically they are identical — an empty strip of road alongside. Only
`carriageway.py` can separate them, so this consumes its answer rather than
guessing. Inviting a lane change into head-on traffic is the worst output this
system could produce, so classification is ordered most-restrictive-first and
oncoming always beats free.

Occupancy is decided at the **ground contact point**, the bottom edge of the
box — the only part of a 2-D detection actually on the road plane. A box centre
floats at bumper height and back-projects well beyond the vehicle, which on a
three-lane road puts a lorry in the wrong lane.

Three bugs here were found by rendering frames and looking at them, and each is
now a test:

* **The lane was painted to the horizon but checked for occupants only over the
  few metres its paint survived** — orange laid across three visible cars.
* **The same at the near end.** A fit starting at 5.6 m ignored a car at 4.3 m:
  directly alongside, and the single most important vehicle in the frame for a
  lane change.
* **A traffic light's box back-projected to the road**, putting a phantom
  vehicle in whichever lane the gantry crossed. Occupants are now filtered to
  `ROAD_USER_CLASSES`.

Occupancy is therefore checked over a *wider* range than the boundaries were
fitted, extrapolating them to do it. Acceptable in this direction only: a
wrongly-included vehicle downgrades a lane to occupied, and being conservative
about an invitation costs nothing.

What it does not do is decide whether a change is **safe**. Free means
unoccupied *ahead*. There is nothing behind — a vehicle closing from the rear is
invisible to a forward camera, and no geometry here can supply it.

### Regions

A fork's branch is drawn as a **lane-width band** (`BRANCH_WIDTH_M`, 3.6 m) —
a branch is a lane and should look like one; its true width was not measured, and
that assumption is stated rather than hidden. The opposing carriageway of a
divided road *is* drawn out to the frame edge, bounded above by the horizon,
because there the claim really is "everything past this line is theirs".

A fork and a divided road are never tinted at once: they are different claims
about the same region.

`annotate()` delegates its lane geometry here, so the composite and the
per-capability sections share both the language and the guards.

## Road splits — forks and exit ramps ahead

The lane fitter assumes one continuous road with roughly parallel boundaries.
Where the road actually divides, that assumption fails **silently and in the
worst way**: the gore between the two carriageways reads as one very wide lane,
`lane_width_m` grows past anything plausible, and the ego offset the
lane-departure rule depends on stops meaning anything. Nothing raises. The
system simply believes it is centred in a nine-metre lane.

Detection works on the **fitted boundaries**, not the mask. Two boundaries
belonging to one road stay roughly parallel — their separation at 30 m is close
to their separation at 10 m. Two boundaries either side of a fork do not; the gap
opens with distance, which is what a driver sees as the road dividing.

The test is a **ratio**, not an absolute width, so it holds for a narrow urban
lane and a wide motorway alike; a fixed threshold would fire on one and miss the
other. An absolute widening is required as well, because the ratio alone fires on
boundaries that start very close together.

Two deliberate limits:

* **It never claims a split from one frame.** A single bad lane fit — a kerb
  taken as a boundary, a shadow edge — produces exactly a fork's signature.
  `SplitTracker` requires agreement across several frames in a window, which
  costs a few frames against a fork that stays visible for seconds.
* **It does not guess which branch the vehicle will take.** That needs indicator
  state or a route, neither of which exists here, and a confident wrong answer
  is worse than "the road divides ahead", which is both true and useful.

Rendered in its own colour, distinct from every lane role, with a marker at the
gore nose. A fork changes what the lane geometry *means*, so it must not read as
one more boundary among the others.

## Divided roads — the two directions of travel

`models/lane_seg/carriageway.py`. Adjacent to road splits and answering a
different question, which is worth stating plainly because the two get confused:

* **Road split** (`split.py`) — the road is *about to divide*. A fork, an exit
  ramp. The gap between two boundaries **opens** with distance. An event the
  driver is approaching.
* **Divided road** (`carriageway.py`) — the road is *already divided*, one
  carriageway each way. The gap **stays constant**. A standing property of the
  road, true for miles.

The distinction drives behaviour, not taxonomy. Everything beyond the divider is
**oncoming traffic**: closing at the sum of both speeds, never a lane the ego may
change into, and liable to leave frame fast. Reading a divided road as one wide
road gets all three wrong.

### Two cues

**Yellow paint.** Under the US marking standard — which is what BDD100K shows —
yellow separates *opposing directions* and white separates *same-direction*
lanes. A yellow boundary is therefore the direction divider by definition, not by
inference. `models/lane_seg/classical.py` already thresholded yellow separately
and then merged it away with `colour = white | yellow`; `LaneMaskResult` now
carries `yellow_mask` alongside the merged one. Colour is assigned **per
connected component by majority**, not per pixel, because a marking's colour is a
property of the marking and a few desaturated pixels along a worn edge should not
split one line into two differently-coloured fragments.

**A median gap.** Two boundaries ≥ 5 m apart whose separation holds constant are
not one very wide lane; they are the inner edges of two carriageways. Constancy
is what separates this from a fork — and it is why `detect_split` skips pairs
already that far apart. The two detectors partition the cases rather than
overlapping, which `TestDistinctFromFork` pins down in both directions.

### What the real frames forced

Measured on 300 random BDD100K frames (129 auto-calibrate, 123 get a lane fit).
A first version fired on **33** of those 123, and rendering them showed most were
wrong. Three fixes, each aimed at an observed failure rather than guessed at:

| Failure seen | Fix | Why it is not a hack |
|---|---|---|
| Yellow **kerb/edge lines** on the ego's right read as dividers | `drive_side` **gates** the yellow cue instead of discounting it | Under the same standard that makes yellow meaningful, a yellow line to the driver's right marks the roadway edge. Accepting one there reads the standard backwards. |
| A **yellow taxi** filling the projected curve scored a yellow fraction of 1.0 | also require the yellow to span ≥ 60% of the fit's own range | A thing at a place occupies a band of range; paint runs the length of the road. Relative, not absolute — fits on dashed markings are routinely 7–8 m long and an absolute 12 m floor rejected correct ones. |
| Taxis again, when they happen to lie along the fit | `exclude_boxes_px`: yellow inside a detected object box is discounted | The one place the detector feeds back into lane understanding. No colour or shape test separates a taxi's flank from paint; a box around it says so directly. |

Detections fell 33 → 14, every survivor on the correct side, and spot-rendering
the survivors put most of them right. **This is a heuristic on a colour cue, not
a measurement**: it is regional (the yellow convention does not hold in the UK,
Japan, Australia or much of Europe), it depends on a lane fit that is itself weak
on cluttered urban scenes, and no ground truth for it was available, so the
precision above is my own reading of rendered frames, not a scored metric.

### State, not event

`CarriagewayTracker` is deliberately **not** shaped like `SplitTracker`. A fork
is announced once and then passed. A divided road stays true, so once confirmed
it is reported on *every* frame and decays only after the evidence has been
absent for a while. Announcing it once and forgetting would leave the fusion
layer with no idea, ten seconds later, whether the traffic to its left is
oncoming. Both trackers pick the divider's side by **majority** across the
window: the side is the one thing that must not flicker, since reversing it
inverts the meaning of every object on half the road.

### Rendering

`core.overlay.draw_carriageway` tints the two halves rather than outlining them
— the useful fact is which region is which, not where the line is. Ego side
green, oncoming side amber (the colour used throughout for "traffic that is not
yours"), divider drawn as a double line in road yellow. The oncoming region is
filled out to the frame edge and bounded above by the horizon: its outer edge is
usually unmarked or out of frame, so claiming a specific width for it would be
drawing a measurement that was never taken.

## Lane departure and lane change

Both from one signal: lateral offset normalised by half the lane width, so
`|u| = 1` is a boundary whether the lane is 3.7 m or 2.6 m wide.

They are different kinds of thing and are detected differently. A **departure**
is a sustained state — `|u|` high for several consecutive frames. A **lane
change** is an event, and it announces itself by a *sign flip*: as the vehicle
crosses, the fitter re-locks onto the new lane and the offset jumps from "far
right of the old lane" to "left of the new one". Requiring a preceding sustained
departure in the matching direction is what stops a single frame's fit wobble
from reading as a lane change.

Hysteresis (`recover_threshold` below `departure_threshold`) stops a vehicle
riding near the limit from re-triggering every few frames.

## Monocular depth — shape from MiDaS, scale from the homography

MiDaS predicts **relative inverse depth**. The map is correct up to an unknown
scale and shift, so a raw MiDaS value is not a distance and is not even
comparable between two frames. That is the fact most integrations gloss over.

The standard relation is affine in inverse depth:

```
1/Z  ≈  a·d + b
```

Fitting `(a, b)` needs points whose true `Z` is known — and this pipeline already
has them. Every tracked object whose ground-contact point the homography accepted
is a pixel with a known metric distance. **The anchors come from the scene
itself**: no extra sensor, and no assumption beyond the one the homography
already makes.

That division of labour is the whole design:

| | Coverage | Scale |
|---|---|---|
| Homography | objects on flat road only | metric, accurate |
| MiDaS | every pixel | none |
| Fitted together | every pixel | metric, inherited |

So **MiDaS is never the source of scale, only of shape.** Where the homography
can place an object it wins outright. Depth earns its cost only on what the
homography cannot see at all: an occluded ground-contact point, a vehicle on a
crest or in a dip, the back of a truck whose wheels are hidden. It is therefore
`enabled: false` by default — it is a fallback, not an upgrade.

**The fit is rejected more often than it is used, deliberately.** A bad scale
produces confident, wrong metres, which is the worst failure available to a
collision warning. Rejection triggers on: fewer than `min_anchors` points, a
slope the anchors do not constrain, an inverted depth ordering (the network and
the ground plane describing different scenes), and — the one that matters most —
a median metric error above `max_residual_m`. That last guard exists because
`r²` can look healthy while the metres are badly wrong: the fit lives in inverse
space, where distant points contribute almost nothing to the residual. An invalid
scale returns `inf`, never a guess.

`ObjectState.distance_source` records `homography`, `depth` or `none`, so any
alert stays auditable back to which estimator produced its distance.

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
