"""Typed configuration loaded from YAML under ``configs/``.

Config objects carry no coordinate-frame semantics themselves; modules that
consume them state their own frames. Unknown keys are rejected rather than
ignored, so a typo in YAML fails loudly instead of silently taking a default.

Cache invalidation depends on :meth:`Config.cache_key`, which hashes only the
sections that actually influence cached per-frame results. Recolouring the
overlay must not throw away an hour of inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IOConfig(_Strict):
    """Frame-source behaviour.

    Attributes:
        image_extensions: Suffixes (lowercase, with dot) treated as frames when
            reading an image directory.
        image_dir_fps: Nominal capture rate for image-directory sources. There is
            no timing information in a folder of stills, so this must be stated
            explicitly; opening an image directory without it raises.
        canonical_size: ``[width, height]`` every emitted frame is resized to.
            None means "adopt the first frame's size". Frames arriving at any
            other size are resized to match, because the homography is defined
            for exactly one resolution.
        use_container_timestamps: For video, take each frame's time from the
            container's presentation timestamp rather than ``index / fps``. This
            is what makes variable-frame-rate recordings come out with correct
            timing. Falls back to ``index / fps`` when the container reports
            nothing usable.
        start_frame: Index of the first frame to emit, counted in source frames
            before ``stride`` is applied.
        stride: Emit every Nth source frame. 1 means every frame.
        max_frames: Stop after this many emitted frames. None means no limit.
    """

    image_extensions: list[str] = Field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    image_dir_fps: float | None = None
    canonical_size: tuple[int, int] | None = None
    use_container_timestamps: bool = True
    start_frame: int = 0
    stride: int = 1
    max_frames: int | None = None

    @field_validator("image_extensions")
    @classmethod
    def _lowercase_with_dot(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("image_extensions must not be empty")
        bad = [e for e in v if not e.startswith(".") or e != e.lower()]
        if bad:
            raise ValueError(f"image_extensions must be lowercase and dot-prefixed: {bad}")
        return v

    @field_validator("image_dir_fps")
    @classmethod
    def _positive_fps(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"image_dir_fps must be > 0, got {v}")
        return v

    @field_validator("canonical_size")
    @classmethod
    def _positive_size(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and (v[0] < 1 or v[1] < 1):
            raise ValueError(f"canonical_size must be positive, got {v}")
        return v

    @field_validator("start_frame")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"start_frame must be >= 0, got {v}")
        return v

    @field_validator("stride")
    @classmethod
    def _positive_stride(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"stride must be >= 1, got {v}")
        return v

    @field_validator("max_frames")
    @classmethod
    def _positive_max(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError(f"max_frames must be >= 1 or null, got {v}")
        return v


class ChessboardConfig(_Strict):
    """Calibration target geometry.

    Attributes:
        cols: Inner corners per row (9 for a standard 10x7 board).
        rows: Inner corners per column.
        square_size_m: Physical square edge length in metres. This scales the
            extrinsic translations only; the camera matrix and distortion
            coefficients are unaffected by it.
    """

    cols: int = 9
    rows: int = 6
    square_size_m: float = 0.024

    @field_validator("cols", "rows")
    @classmethod
    def _at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"chessboard needs >= 2 inner corners per side, got {v}")
        return v

    @field_validator("square_size_m")
    @classmethod
    def _positive_square(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"square_size_m must be > 0, got {v}")
        return v


class CalibrationConfig(_Strict):
    """Where calibration lives and how it is applied.

    Attributes:
        intrinsics_path: Output/input path for the chessboard calibration.
        homography_path: Output/input path for the ground-plane homography.
        undistort: Whether frames are undistorted before anything else. When
            False the homography absorbs whatever lens distortion is present,
            which is accurate near the image centre and degrades toward the
            edges -- exactly where distant vehicles and lane markings sit.
            Whichever way this is set, the choice is recorded in the homography
            file and in the cache manifest, so results are never ambiguous.
        chessboard: Calibration target geometry.
    """

    intrinsics_path: Path = Path("calib/intrinsics.yaml")
    homography_path: Path = Path("calib/homography.yaml")
    undistort: bool = False
    chessboard: ChessboardConfig = Field(default_factory=ChessboardConfig)


class BEVConfig(_Strict):
    """Bird's-eye rendering window, in ego-vehicle metres (X right, Y forward).

    Attributes:
        x_min_m, x_max_m: Lateral extent; 0 is the camera's ground projection.
        y_min_m, y_max_m: Longitudinal extent ahead of the camera. ``y_min_m``
            must be > 0: the ground plane meets the image only ahead of the car.
        px_per_m: Resolution of the rendered bird's-eye image.
        grid_step_m: Spacing of the metre grid drawn by ``calibrate verify``.
    """

    x_min_m: float = -6.0
    x_max_m: float = 6.0
    y_min_m: float = 3.0
    y_max_m: float = 40.0
    px_per_m: float = 20.0
    grid_step_m: float = 5.0

    @field_validator("px_per_m", "grid_step_m")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.x_max_m <= self.x_min_m:
            raise ValueError(f"x_max_m must exceed x_min_m, got {self.x_min_m}..{self.x_max_m}")
        if self.y_max_m <= self.y_min_m:
            raise ValueError(f"y_max_m must exceed y_min_m, got {self.y_min_m}..{self.y_max_m}")
        if self.y_min_m <= 0:
            raise ValueError(
                f"y_min_m must be > 0 (the road plane is only imaged ahead of the "
                f"camera), got {self.y_min_m}"
            )

    @property
    def size_px(self) -> tuple[int, int]:
        """``(width, height)`` of the rendered bird's-eye image."""
        w = int(round((self.x_max_m - self.x_min_m) * self.px_per_m))
        h = int(round((self.y_max_m - self.y_min_m) * self.px_per_m))
        return w, h


class CacheConfig(_Strict):
    """Two-stage cache location and identity.

    Attributes:
        root: Directory holding one subdirectory per run.
        run_id: Fixed run directory name, or None to derive one from the source
            name and the config hash (so identical settings reuse the cache).
    """

    root: Path = Path("outputs")
    run_id: str | None = None


class VRUConfig(_Strict):
    """Second detector covering vulnerable road users.

    YOLOPv2 has no pedestrian class and no released multi-task network in that
    family does, so people, bicycles and motorcycles need their own model. This
    one deliberately emits only those classes; vehicles come from YOLOPv2 and
    taking them from both would double-count every car.

    Attributes:
        enabled: Whether to run the second model at all.
        architecture: A torchvision ``models.detection`` builder name.
        conf_threshold: Minimum confidence for a kept detection.
    """

    enabled: bool = True
    architecture: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    conf_threshold: float = 0.5

    @field_validator("conf_threshold")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1], got {v}")
        return v


class YOLOv8Config(_Strict):
    """Single-task YOLOv8 baseline settings.

    Separate from the top-level ``weights`` because the two backbones take
    different checkpoints, and sharing one field would silently hand YOLOv8 the
    YOLOPv2 weights when ``kind`` is switched.

    Attributes:
        weights: Checkpoint path. A COCO-pretrained ``yolov8*.pt`` gives a
            working baseline with no training; a fine-tuned checkpoint replaces
            it without any other change.
        allow_download: Whether a missing *standard* pretrained checkpoint may be
            fetched by ultralytics. A missing checkpoint that is not a recognised
            pretrained name always raises -- silently training-from-scratch or
            substituting the wrong weights is never the right default.
    """

    weights: Path = Path("models/perception/weights/yolov8m.pt")
    allow_download: bool = True


class DetectorConfig(_Strict):
    """Perception model settings.

    Attributes:
        kind: ``"yolopv2"`` for the multi-task backbone, ``"yolov8"`` for the
            single-task baseline, ``"stub"`` for the synthetic detector that
            keeps pipeline tests fast and model-free.
        weights: TorchScript weights path for the real backbone.
        input_size: Network input size for the longer image side. The letterbox
            pads to a multiple of ``stride`` from there.
        stride: Network stride; the padded input is a multiple of this.
        conf_threshold: Minimum detection confidence.
        iou_threshold: NMS IoU threshold.
        max_detections: Cap on boxes kept per frame after NMS.
        lane_threshold: Probability above which a lane pixel is set.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or an explicit ``"cuda:N"``.
            ``auto`` warns loudly when it lands on CPU.
        half: Use fp16 on CUDA. Ignored on CPU, where fp16 is slower.
        batch_size: Frames per forward pass.
        score_threshold: Stub only -- fake detections below this are dropped.
        seed: Stub only -- makes the fake output deterministic.
        vru: Second-detector settings. Ignored when ``kind`` is ``"yolov8"``,
            which detects vulnerable road users itself.
        yolov8: Baseline-backbone settings, used only when ``kind`` is
            ``"yolov8"``.
    """

    kind: str = "yolopv2"
    weights: Path = Path("models/yolopv2.pt")
    input_size: int = 640
    stride: int = 32
    conf_threshold: float = 0.3
    iou_threshold: float = 0.45
    max_detections: int = 100
    lane_threshold: float = 0.5
    device: str = "auto"
    half: bool = True
    batch_size: int = 4
    score_threshold: float = 0.3
    seed: int = 0
    vru: VRUConfig = Field(default_factory=VRUConfig)
    yolov8: YOLOv8Config = Field(default_factory=YOLOv8Config)

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        known = ("stub", "yolopv2", "yolov8")
        if v not in known:
            raise ValueError(
                f"unknown detector kind '{v}'; expected one of {', '.join(known)}"
            )
        return v

    @field_validator("input_size", "stride", "batch_size", "max_detections")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("conf_threshold", "iou_threshold", "lane_threshold", "score_threshold")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v


class LaneFitConfig(_Strict):
    """Sliding-window lane fitting in the bird's-eye view.

    Attributes:
        n_windows: Vertical search windows per boundary.
        window_half_width_m: Lateral half-width of each search window.
        min_pixels_per_window: Pixels needed in a window before its centre is
            used to re-centre the next one.
        min_pixels_for_fit: Total pixels a boundary needs before a polynomial is
            fitted at all. Below this the status is MASK_TOO_SPARSE rather than
            a fit nobody should trust.
        min_mask_pixels: Lane pixels needed in the whole bird's-eye window before
            fitting is attempted.
        search_half_width_m: Lateral extent either side of the camera axis
            searched for boundaries. Must reach past the adjacent lane's far
            edge -- for a 3.5 m lane that boundary sits at 1.75 + 3.5 = 5.25 m,
            so a 5 m search silently finds no adjacent lanes at all.
        max_lane_width_m / min_lane_width_m: Plausibility bounds on the measured
            ego lane width. Outside these the pair is rejected as a mis-pairing.
        eval_distance_m: Forward distance at which curvature, offset and width
            are reported.
        adjacent_lanes: Whether to also fit boundaries outside the ego lane.
    """

    n_windows: int = 12
    window_half_width_m: float = 0.6
    min_pixels_per_window: int = 25
    min_pixels_for_fit: int = 150
    min_mask_pixels: int = 300
    search_half_width_m: float = 7.0
    min_lane_width_m: float = 2.2
    max_lane_width_m: float = 5.0
    eval_distance_m: float = 10.0
    adjacent_lanes: bool = True

    @field_validator("n_windows", "min_pixels_per_window", "min_pixels_for_fit", "min_mask_pixels")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v


class OverlayConfig(_Strict):
    """Annotation appearance. Excluded from the cache key by design.

    Attributes:
        box_thickness: Outline width in pixels for detection boxes.
        lane_thickness: Polyline width in pixels for lane boundaries.
        mask_alpha: Blend weight for the drivable-area fill, in ``[0, 1]``.
        show_lane_mask: Draw the raw lane segmentation under the fitted curves.
            This is what makes the fit checkable by eye.
        lane_mask_alpha: Blend weight for the raw lane mask.
        font_scale: OpenCV Hershey font scale for labels and HUD text.
        hud: Whether to draw the text HUD panel.
        hud_width_px: Width of the HUD panel.
    """

    box_thickness: int = 2
    lane_thickness: int = 4
    mask_alpha: float = 0.35
    show_lane_mask: bool = True
    lane_mask_alpha: float = 0.85
    font_scale: float = 0.5
    hud: bool = True
    hud_width_px: int = 380

    @field_validator("mask_alpha", "lane_mask_alpha")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {v}")
        return v


class VideoConfig(_Strict):
    """Annotated-output encoding. Excluded from the cache key by design.

    Attributes:
        fourcc: OpenCV FourCC code for the writer.
        fps: Output frame rate, or None to inherit the emitted frame rate
            (source fps divided by ``io.stride``).
    """

    fourcc: str = "mp4v"
    fps: float | None = None

    @field_validator("fourcc")
    @classmethod
    def _four_chars(cls, v: str) -> str:
        if len(v) != 4:
            raise ValueError(f"fourcc must be exactly 4 characters, got '{v}'")
        return v


class TrackerConfig(_Strict):
    """ByteTrack association and track lifecycle.

    Deliberately excluded from the cache key: tracking is post-processing over
    cached detections, so retuning it re-runs association in seconds rather than
    re-running the network.

    Attributes:
        enabled: Whether to assign persistent ids at all.
        high_threshold: Detections at or above this drive the first association
            pass and may start new tracks.
        low_threshold: Floor for the second pass. Detections between the two
            thresholds can only *continue* an existing track, never start one --
            a partly occluded object scores low, but so does noise.
        match_threshold: Minimum IoU for a first-pass match.
        match_threshold_low: Minimum IoU for a second-pass match. Looser,
            because a weak detection on a partly hidden object is also a
            poorly-placed one.
        new_track_threshold: Score needed to create a track.
        n_init: Matches before a track is confirmed and reported.
        max_age: Frames a confirmed track survives unmatched before removal. At
            25-30 fps, 30 frames is about a second -- long enough to ride out a
            vehicle passing behind a sign, short enough that an id is not still
            reserved when an unrelated object appears in that spot.
        use_mahalanobis_gate: Veto matches the motion model deems implausible,
            even when the boxes overlap.
    """

    enabled: bool = True
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    match_threshold: float = 0.2
    match_threshold_low: float = 0.4
    new_track_threshold: float = 0.6
    n_init: int = 3
    max_age: int = 30
    use_mahalanobis_gate: bool = True

    @field_validator(
        "high_threshold", "low_threshold", "match_threshold",
        "match_threshold_low", "new_track_threshold",
    )
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v

    @field_validator("n_init", "max_age")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be below "
                f"high_threshold ({self.high_threshold}); otherwise the "
                f"second association pass has no detections to work with"
            )
        if self.new_track_threshold < self.high_threshold:
            raise ValueError(
                f"new_track_threshold ({self.new_track_threshold}) must be at "
                f"least high_threshold ({self.high_threshold}): tracks are only "
                f"ever created from first-pass detections"
            )


class TTCConfig(_Strict):
    """Distance and time-to-collision estimation.

    Post-processing over cached detections, so excluded from the cache key.

    Attributes:
        fit_window: Most recent samples the linear fits use. Short enough to
            follow a real change in closing speed, long enough to average out
            per-frame box jitter -- about half a second at 25-30 fps.
        min_samples: Samples required before any estimate is made. Two points
            define a line through any noise; more is needed to mean anything.
        history_window: Cap on retained distance samples per track.
        min_fit_quality: R-squared below which the scale estimate is discarded.
            A poor fit means the object is not moving the way the constant-
            velocity model assumes, and the arithmetic would carry no meaning.
        prefer_range_within_m: Below this distance the calibrated ground-plane
            estimate is preferred; beyond it, scale expansion is. Pixel
            quantisation makes homography distance unreliable far out, while
            scale expansion needs no calibration at all.
        corridor_half_width_m: Half-width of the ego corridor at zero distance.
            Slightly wider than a half lane, since a vehicle straddling a line
            is still a hazard.
        corridor_widening: Extra corridor half-width per metre of distance. This
            compensates for lateral-estimate error growing with range, not for
            the road getting wider.
    """

    fit_window: int = 15
    min_samples: int = 4
    history_window: int = 60
    min_fit_quality: float = 0.5
    prefer_range_within_m: float = 30.0
    corridor_half_width_m: float = 1.6
    corridor_widening: float = 0.02

    @field_validator("fit_window", "min_samples", "history_window")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"must be >= 2, got {v}")
        return v

    @field_validator("min_fit_quality")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v

    @field_validator("prefer_range_within_m", "corridor_half_width_m")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.min_samples > self.fit_window:
            raise ValueError(
                f"min_samples ({self.min_samples}) exceeds fit_window "
                f"({self.fit_window}), so no fit could ever run"
            )
        if self.history_window < self.fit_window:
            raise ValueError(
                f"history_window ({self.history_window}) is smaller than "
                f"fit_window ({self.fit_window}); the fit would be starved"
            )


class LaneChangeConfig(_Strict):
    """Lane departure and lane-change detection.

    Thresholds are on the *normalised* offset -- offset divided by half the lane
    width -- so 1.0 is a boundary regardless of whether the lane is a 3.7 m
    motorway lane or a 2.6 m urban one.

    Attributes:
        departure_threshold: Normalised offset at which the vehicle counts as
            riding a boundary. 0.65 is roughly two-thirds of the way out.
        recover_threshold: Offset the vehicle must return inside before a new
            departure can be reported. Deliberately lower than
            ``departure_threshold``: without that hysteresis, a vehicle sitting
            near the limit would re-trigger every few frames.
        min_departure_frames: Consecutive frames above the threshold before a
            departure fires. Rejects single-frame fit wobble.
        cross_threshold: Magnitude both sides of the sign flip must reach for a
            crossing to be believed.
        cooldown_frames: Frames after a change during which no new lane event
            fires, while the fitter settles onto the new lane.
        max_gap_frames: How long a departure survives with no usable lane fit.
            Markings vanish briefly all the time; forgetting instantly would
            drop a departure that is still happening.
        history_window: Offsets retained, for the confidence figure.
        min_plausible_width_m, max_plausible_width_m: A fit outside this band is
            rejected as a mis-lock rather than trusted.
    """

    departure_threshold: float = 0.65
    recover_threshold: float = 0.45
    min_departure_frames: int = 3
    cross_threshold: float = 0.5
    cooldown_frames: int = 15
    max_gap_frames: int = 5
    history_window: int = 60
    min_plausible_width_m: float = 2.2
    max_plausible_width_m: float = 5.0

    @field_validator("departure_threshold", "recover_threshold", "cross_threshold")
    @classmethod
    def _positive_unit(cls, v: float) -> float:
        if not 0.0 < v <= 2.0:
            raise ValueError(f"must be in (0, 2], got {v}")
        return v

    @field_validator("min_departure_frames", "cooldown_frames", "max_gap_frames",
                     "history_window")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.recover_threshold >= self.departure_threshold:
            raise ValueError(
                f"recover_threshold ({self.recover_threshold}) must be below "
                f"departure_threshold ({self.departure_threshold}) to give the "
                f"state machine hysteresis; equal values re-trigger every frame"
            )
        if self.min_plausible_width_m >= self.max_plausible_width_m:
            raise ValueError("min_plausible_width_m must be below max_plausible_width_m")


class FusionConfig(_Strict):
    """Rules-based decision engine thresholds.

    Post-processing, so excluded from the cache key: retuning an alert threshold
    must never re-run inference.

    Attributes:
        ttc_warning_s: TTC at or below which a forward-collision warning fires.
            2.5 s is roughly driver reaction time plus a margin.
        ttc_critical_s: TTC for a critical warning -- too late to be comfortable,
            still time to brake.
        pedestrian_distance_m: A vulnerable road user in the corridor closer
            than this alerts on proximity alone, regardless of TTC. A pedestrian
            standing still has infinite TTC and is still a hazard.
        weather_ttc_factor: Multiplier on both TTC thresholds per weather label.
            Wet tarmac roughly doubles stopping distance, so the warning has to
            arrive proportionally earlier. Labels absent from this map use 1.0.
        warn_on_lane_loss: Emit an informational alert when no lane is fitted.
        warn_on_red_light: Warn when approaching a red light without slowing.
        red_light_distance_m: Ignore reds beyond this. Generous, because a
            traffic light sits *above* the road plane and the homography's
            flat-ground assumption does not hold for it, so its distance is
            indicative rather than precise.
        red_light_min_closing_mps: Minimum closing speed before a red warns.
            Approaching a red while braking is normal driving; warning about it
            is exactly the noise that makes a driver switch the system off.
        lane_change: Lane event detection settings.
    """

    ttc_warning_s: float = 2.5
    ttc_critical_s: float = 1.2
    pedestrian_distance_m: float = 12.0
    weather_ttc_factor: dict[str, float] = Field(
        default_factory=lambda: {
            "clear": 1.0, "cloudy": 1.0, "night": 1.2,
            "rain": 1.6, "snow": 2.0, "fog": 1.8,
        }
    )
    warn_on_lane_loss: bool = False
    warn_on_red_light: bool = True
    red_light_distance_m: float = 45.0
    red_light_min_closing_mps: float = 1.0
    lane_change: LaneChangeConfig = Field(default_factory=LaneChangeConfig)

    @field_validator("ttc_warning_s", "ttc_critical_s", "pedestrian_distance_m",
                     "red_light_distance_m", "red_light_min_closing_mps")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v

    @field_validator("weather_ttc_factor")
    @classmethod
    def _sane_factors(cls, v: dict[str, float]) -> dict[str, float]:
        for k, f in v.items():
            if f <= 0:
                raise ValueError(f"weather_ttc_factor['{k}'] must be > 0, got {f}")
            if f < 1.0:
                raise ValueError(
                    f"weather_ttc_factor['{k}'] is {f}: a factor below 1 would "
                    f"make the warning fire LATER in worse conditions"
                )
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.ttc_critical_s >= self.ttc_warning_s:
            raise ValueError(
                f"ttc_critical_s ({self.ttc_critical_s}) must be below "
                f"ttc_warning_s ({self.ttc_warning_s}); otherwise a critical "
                f"alert could never fire before the warning"
            )


class DepthConfig(_Strict):
    """Monocular depth estimation.

    Part of the cache key: unlike tracking and fusion, this runs a network per
    frame, so its output is expensive and belongs in the cache.

    Attributes:
        enabled: Whether depth runs at all. Off by default -- the calibrated
            homography is the better estimate for objects on flat road, and
            depth earns its cost only on the cases the homography cannot see.
        kind: ``midas`` for the real model, ``stub`` for the planar stand-in
            that needs no weights and no network.
        model: torch.hub entry point. The small variant is the deliberate
            choice on an 8 GB GPU already hosting the detector and tracker.
        device: ``auto``, ``cuda`` or ``cpu``.
        half: fp16 on CUDA.
        min_anchors: Ground-plane points needed before a metric scale is fitted.
        max_residual_m: Median anchor error above which the fit is rejected.
            The guard that matters -- a numerically healthy fit can still be
            metrically wrong, and a wrong scale yields confident false distances.
        min_r_squared: Minimum correlation, in inverse-depth space, for a fit.
        patch_px: Side of the median window sampled around each anchor point.
            Depth maps are noisiest at object boundaries, which is exactly where
            a box's bottom edge sits.
    """

    enabled: bool = False
    kind: str = "midas"
    model: str = "MiDaS_small"
    device: str = "auto"
    half: bool = True
    min_anchors: int = 4
    max_residual_m: float = 6.0
    min_r_squared: float = 0.5
    patch_px: int = 5

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, v: str) -> str:
        if v not in {"midas", "stub"}:
            raise ValueError(f"depth.kind must be 'midas' or 'stub', got '{v}'")
        return v

    @field_validator("min_anchors")
    @classmethod
    def _enough_anchors(cls, v: int) -> int:
        if v < 2:
            raise ValueError(
                f"min_anchors must be >= 2 (two points define the affine fit), got {v}"
            )
        return v

    @field_validator("patch_px")
    @classmethod
    def _odd_patch(cls, v: int) -> int:
        if v < 1 or v % 2 == 0:
            raise ValueError(f"patch_px must be a positive odd number, got {v}")
        return v


class WeatherConfig(_Strict):
    """Weather classification, used to scale the collision thresholds.

    In the cache key: it runs a network per frame.

    Attributes:
        enabled: Off until a model is trained. The fusion engine treats an
            unknown label as neutral (factor 1.0), so disabling it degrades to
            dry-weather thresholds rather than failing.
        architecture: ``resnet18`` or ``efficientnet_b0``. ResNet18 by default --
            both saturate on this task, and ResNet18 is faster, smaller and
            better behaved through ONNX export.
        weights: Trained checkpoint.
        input_size: Square input. Frames are resized, not cropped: weather
            evidence lives in the sky and the road surface, and a centre crop
            discards exactly those pixels.
        device: ``auto``, ``cuda`` or ``cpu``.
        smoothing_window: Frames in the majority-vote window. Weather does not
            change between frames, so per-frame flicker is model noise, and
            letting it through would modulate the collision threshold several
            times a second.
        switch_fraction: Share of the window a new label needs before it is
            adopted. Must exceed 0.5, or the label oscillates.
        every_n_frames: Classify only every Nth frame. Weather changes over
            minutes; classifying every frame is wasted compute.
    """

    enabled: bool = False
    architecture: str = "resnet18"
    weights: Path = Path("models/weather_cls/weights/weather.pt")
    input_size: int = 224
    device: str = "auto"
    smoothing_window: int = 30
    switch_fraction: float = 0.6
    every_n_frames: int = 5

    @field_validator("architecture")
    @classmethod
    def _known_arch(cls, v: str) -> str:
        if v not in {"resnet18", "efficientnet_b0"}:
            raise ValueError(
                f"weather.architecture must be 'resnet18' or 'efficientnet_b0', got '{v}'"
            )
        return v

    @field_validator("switch_fraction")
    @classmethod
    def _majority(cls, v: float) -> float:
        if not 0.5 < v <= 1.0:
            raise ValueError(
                f"switch_fraction must be in (0.5, 1.0]; at or below a majority "
                f"the smoothed label would oscillate, got {v}"
            )
        return v

    @field_validator("smoothing_window", "every_n_frames", "input_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v


class PotholeConfig(_Strict):
    """Road-hazard detection. In the cache key: it runs a network per frame.

    Attributes:
        enabled: Off until a model is trained.
        mode: ``detector`` (YOLOv8 on pothole boxes) or ``classifier`` (binary
            over road patches). Left configurable because the brief asks for
            whichever gives better *precision* on the supplied data, and that is
            a measurement, not a prior belief. See ``docs/EVAL_RESULTS.md`` s4.
        weights: Trained checkpoint for the selected mode.
        architecture: Backbone, classifier mode only.
        conf_threshold: Deliberately high. Precision is weighted above recall:
            a hazard warning that fires on every tar seam and manhole gets
            ignored, and an ignored warning is worse than no warning because it
            teaches the driver to dismiss the alerts that do matter.
        input_size: Detector input size.
        patch_px, patch_stride_px: Patch geometry, classifier mode only.
        roi_top: Fraction of frame height above which patches are never sampled.
            The road is in the lower frame; a classifier run over the whole image
            calls dark windows potholes, having never been shown a non-road.
        min_drivable_fraction: Share of a patch that must be drivable before it
            is classified, when a drivable mask is available.
        min_distance_m, max_distance_m: Hazards outside this band are dropped.
            Too close to react to, or too far to be worth reporting.
    """

    enabled: bool = False
    mode: str = "detector"
    weights: Path = Path("models/pothole_det/weights/pothole.pt")
    architecture: str = "resnet18"
    conf_threshold: float = 0.55
    input_size: int = 640
    patch_px: int = 96
    patch_stride_px: int = 64
    roi_top: float = 0.5
    min_drivable_fraction: float = 0.6
    min_distance_m: float = 3.0
    max_distance_m: float = 35.0

    @field_validator("mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in {"detector", "classifier"}:
            raise ValueError(
                f"pothole.mode must be 'detector' or 'classifier', got '{v}'")
        return v

    @field_validator("conf_threshold", "roi_top", "min_drivable_fraction")
    @classmethod
    def _unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v

    @field_validator("patch_px", "patch_stride_px", "input_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    def model_post_init(self, _ctx: Any) -> None:
        if self.patch_stride_px > self.patch_px:
            raise ValueError(
                f"patch_stride_px ({self.patch_stride_px}) exceeds patch_px "
                f"({self.patch_px}), which would leave gaps the sweep never looks at"
            )
        if self.min_distance_m >= self.max_distance_m:
            raise ValueError("min_distance_m must be below max_distance_m")


class Config(_Strict):
    """Root config. Sections are added as pipeline stages land."""

    io: IOConfig = Field(default_factory=IOConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    bev: BEVConfig = Field(default_factory=BEVConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    lanes: LaneFitConfig = Field(default_factory=LaneFitConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    pothole: PotholeConfig = Field(default_factory=PotholeConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    ttc: TTCConfig = Field(default_factory=TTCConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)

    #: Sections whose values change cached per-frame results. Two kinds of
    #: section are deliberately absent. Presentation-only ones, because
    #: restyling the overlay must not invalidate an expensive cache. And the
    #: post-processing ones -- `tracker`, and later TTC and fusion -- because
    #: they run over cached detections in seconds; retuning an alert threshold
    #: must never re-run a network.
    CACHE_KEY_SECTIONS: ClassVar[tuple[str, ...]] = (
        "io", "calibration", "bev", "detector", "lanes", "depth", "weather", "pothole",
    )

    def cache_key_payload(self) -> dict[str, Any]:
        """The config subset that cached results depend on."""
        dumped = self.model_dump(mode="json")
        return {k: dumped[k] for k in self.CACHE_KEY_SECTIONS}

    def cache_key(self) -> str:
        """Stable 16-hex-char digest of the cache-relevant config."""
        blob = json.dumps(self.cache_key_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a YAML mapping.
        pydantic.ValidationError: if any field is missing, mistyped or unknown.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}: {p}")
    return Config.model_validate(raw)
