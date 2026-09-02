"""Detection and segmentation: the letterbox transform, and the model wrappers.

Coordinate frames:
  * **original px** -- the frame as it came from :mod:`src.io`, at the canonical
    size. Everything a :class:`Detector` returns is in this frame.
  * **letterbox px** -- the network's padded input. Purely internal; nothing
    outside this module should ever see a coordinate in it.

The letterbox transform is the single most common source of silent misalignment
in this kind of pipeline: boxes and masks come back scaled and offset, and an
overlay that looks *almost* right hides a metric error that only surfaces
downstream as a wrong distance. :class:`LetterboxTransform` therefore owns both
directions of the mapping and is round-trip tested.

Failures are loud: missing weights raise, an unsupported device raises, and a
CPU fallback is reported at WARNING rather than being discovered later from the
frame rate.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import DetectorConfig, VRUConfig
from src.logging_utils import get_logger, log_kv
from src.types import CLASS_IDS, FrameResult, NativeMasks

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Letterbox
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LetterboxTransform:
    """Resize-and-pad mapping between original pixels and network input pixels.

    A frame is scaled by a single factor (preserving aspect ratio) and padded to
    a multiple of the network stride. Forward: ``px_letterbox = px_original *
    scale + pad``. Inverse: ``px_original = (px_letterbox - pad) / scale``.

    Attributes:
        original_size: ``(width, height)`` before the transform.
        scaled_size: ``(width, height)`` after scaling, before padding.
        padded_size: ``(width, height)`` handed to the network.
        scale: Single isotropic scale factor.
        pad: ``(left, top)`` padding in letterbox pixels.
    """

    original_size: tuple[int, int]
    scaled_size: tuple[int, int]
    padded_size: tuple[int, int]
    scale: float
    pad: tuple[float, float]

    @classmethod
    def compute(
        cls,
        original_size: tuple[int, int],
        target_long_side: int,
        stride: int = 32,
        allow_upscale: bool = False,
    ) -> LetterboxTransform:
        """Plan the transform for one frame size.

        Args:
            original_size: ``(width, height)`` of the incoming frame.
            target_long_side: Network input size for the longer image side.
            stride: Network stride; the padded size is a multiple of this.
            allow_upscale: Whether a smaller frame may be scaled up. Off by
                default, matching YOLO practice -- upscaling invents detail and
                costs accuracy.
        """
        w, h = original_size
        if w < 1 or h < 1:
            raise ValueError(f"invalid frame size {original_size}")
        scale = min(target_long_side / w, target_long_side / h)
        if not allow_upscale:
            scale = min(scale, 1.0)
        sw, sh = int(round(w * scale)), int(round(h * scale))
        pw = int(np.ceil(sw / stride) * stride)
        ph = int(np.ceil(sh / stride) * stride)
        return cls(
            original_size=(w, h),
            scaled_size=(sw, sh),
            padded_size=(pw, ph),
            scale=scale,
            pad=((pw - sw) / 2.0, (ph - sh) / 2.0),
        )

    def apply(self, image: np.ndarray, fill: int = 114) -> np.ndarray:
        """Scale and pad a BGR frame to the network's input size."""
        h, w = image.shape[:2]
        if (w, h) != self.original_size:
            raise ValueError(
                f"frame is {(w, h)} but this transform was computed for "
                f"{self.original_size}"
            )
        sw, sh = self.scaled_size
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        resized = image if (sw, sh) == (w, h) else cv2.resize(image, (sw, sh), interpolation=interp)
        left = int(round(self.pad[0] - 0.1))
        top = int(round(self.pad[1] - 0.1))
        right = self.padded_size[0] - sw - left
        bottom = self.padded_size[1] - sh - top
        if (left, top, right, bottom) == (0, 0, 0, 0):
            return resized
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(fill, fill, fill)
        )

    def invert_boxes(self, boxes: np.ndarray, clip: bool = True) -> np.ndarray:
        """Map ``(N, 4)`` xyxy boxes from letterbox pixels back to original pixels."""
        out = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        if not len(out):
            return out.astype(np.float32)
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad[0]) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad[1]) / self.scale
        if clip:
            w, h = self.original_size
            out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
            out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
        return out.astype(np.float32)

    def forward_points(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` original pixels into letterbox pixels. For tests."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return pts * self.scale + np.array(self.pad)

    def crop_mask(self, mask: np.ndarray) -> np.ndarray:
        """Strip the letterbox padding from a network-resolution mask.

        Leaves the mask at the resolution the network predicted it, which is what
        gets cached: upsampling adds interpolated pixels but no information.
        """
        mh, mw = mask.shape[:2]
        pw, ph = self.padded_size
        if (mw, mh) != (pw, ph):
            # Some heads emit at a fraction of the input size; scale the crop.
            fx, fy = mw / pw, mh / ph
        else:
            fx = fy = 1.0
        left = int(round(self.pad[0] * fx))
        top = int(round(self.pad[1] * fy))
        right = left + int(round(self.scaled_size[0] * fx))
        bottom = top + int(round(self.scaled_size[1] * fy))
        return mask[top:bottom, left:right]

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map a network-resolution mask back to the original frame size.

        Nearest-neighbour, because these are label masks: interpolating between
        class 0 and class 1 would invent boundary values that mean nothing.
        """
        cropped = self.crop_mask(mask)
        if cropped.size == 0:
            return np.zeros(self.original_size[::-1], dtype=mask.dtype)
        return cv2.resize(cropped, self.original_size, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def upsample_cached_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
        """Restore a cached (already cropped) mask to the original frame size."""
        if mask.shape[:2][::-1] == original_size:
            return mask
        return cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# Detector interface
# --------------------------------------------------------------------------- #


class Detector(ABC):
    """A per-frame perception model.

    The single required method is :meth:`infer`, so the stub and the real network
    are interchangeable everywhere in the pipeline. Everything returned is in
    original image pixels.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded in the cache manifest."""

    @property
    @abstractmethod
    def weights_hash(self) -> str:
        """Digest identifying the exact weights, for cache invalidation."""

    @property
    def class_names(self) -> tuple[str, ...]:
        """Classes this detector is capable of emitting.

        Recorded in the manifest so a cached run is never ambiguous about what it
        was *able* to see, as opposed to what it happened to find.
        """
        return tuple(CLASS_IDS)

    @abstractmethod
    def infer(self, frame: np.ndarray) -> FrameResult:
        """Run the model on one BGR frame and return results in original pixels."""

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        """Run on several frames. The default loops; subclasses may batch."""
        return [self.infer(f) for f in frames]

    def close(self) -> None:
        """Release any GPU resources."""


def resolve_device(requested: str) -> torch.device:
    """Pick a torch device, warning loudly when CUDA was wanted but is absent.

    A silent CPU fallback is the difference between a 20-minute run and an
    overnight one, so it is never silent.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log_kv(
            _log,
            logging.WARNING,
            "detect.cpu_fallback",
            requested="auto",
            reason="torch.cuda.is_available() is False",
            note="inference will be roughly an order of magnitude slower",
        )
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device '{requested}' was requested explicitly but CUDA is not "
            f"available. Set detector.device to 'cpu' or 'auto' to proceed."
        )
    if device.type == "cpu":
        log_kv(
            _log, logging.WARNING, "detect.cpu_selected",
            note="CPU inference was requested explicitly; this will be slow",
        )
    return device


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so 150 MB of weights is not read into RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# YOLOPv2
# --------------------------------------------------------------------------- #


def _make_grid(nx: int, ny: int, device: torch.device) -> torch.Tensor:
    """``(1, 1, ny, nx, 2)`` grid of cell indices for anchor decoding."""
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


def decode_yolo_head(
    pred: list[torch.Tensor], anchor_grid: list[torch.Tensor], strides: tuple[int, ...] = (8, 16, 32)
) -> torch.Tensor:
    """Decode YOLOPv2's three raw detection scales into ``(B, A, 5 + nc)`` boxes.

    Mirrors the reference ``split_for_trace_model``: sigmoid everything, then map
    xy through the cell grid and wh through the anchor priors. Boxes come out as
    ``(cx, cy, w, h)`` in letterbox pixels.
    """
    out = []
    for i, (p, anchors) in enumerate(zip(pred, anchor_grid)):
        bs, _, ny, nx = p.shape
        n_out = p.shape[1] // anchors.shape[1]
        y = p.view(bs, anchors.shape[1], n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous().sigmoid()
        grid = _make_grid(nx, ny, p.device).to(y.dtype)
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * strides[i]
        wh = (y[..., 2:4] * 2.0) ** 2 * anchors.to(y.dtype)
        y = torch.cat((xy, wh, y[..., 4:]), dim=-1)
        out.append(y.view(bs, -1, n_out))
    return torch.cat(out, 1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Centre-form to corner-form, preserving dtype and device."""
    out = torch.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


class YOLOPv2Detector(Detector):
    """The multi-task backbone: vehicles, drivable area and lane markings.

    One forward pass yields all three heads. Detection is vehicle-only -- the
    head has 80 class slots inherited from its COCO-shaped architecture, but the
    released weights are trained on BDD100K traffic objects, so pedestrians are
    not covered here. :class:`VulnerableRoadUserDetector` fills that gap.

    Segmentation heads emit at the network's *input* resolution, so masks are
    kept at that size for caching and upsampled for callers.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"YOLOPv2 weights not found: {weights}. Download the TorchScript "
                f"release to models/, or set detector.kind to 'stub'. There is no "
                f"default model."
            )
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.half = bool(cfg.half and self.device.type == "cuda")
        self._weights_path = weights
        self._weights_hash = file_sha256(weights)

        self.model = torch.jit.load(str(weights), map_location=self.device).eval()
        if self.half:
            self.model = self.model.half()
        self._transforms: dict[tuple[int, int], LetterboxTransform] = {}

        log_kv(
            _log, logging.INFO, "detect.loaded",
            model="yolopv2", weights=weights.name,
            sha256=self._weights_hash[:12], device=str(self.device),
            precision="fp16" if self.half else "fp32",
            input_long_side=cfg.input_size, batch_size=cfg.batch_size,
        )

    @property
    def name(self) -> str:
        return "yolopv2"

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    @property
    def class_names(self) -> tuple[str, ...]:
        return ("vehicle",)

    def _transform_for(self, size: tuple[int, int]) -> LetterboxTransform:
        """Cache the letterbox plan per frame size; it is fixed for a given size."""
        if size not in self._transforms:
            self._transforms[size] = LetterboxTransform.compute(
                size, self.cfg.input_size, self.cfg.stride
            )
        return self._transforms[size]

    def _preprocess(self, frames: list[np.ndarray]) -> tuple[torch.Tensor, LetterboxTransform]:
        h, w = frames[0].shape[:2]
        tf = self._transform_for((w, h))
        batch = np.stack([tf.apply(f) for f in frames])          # (B, H, W, 3) BGR
        batch = batch[..., ::-1].transpose(0, 3, 1, 2)            # to RGB, (B, 3, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        return tensor / 255.0, tf

    def _postprocess_detections(self, decoded: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Confidence-filter and NMS one image's decoded predictions.

        Returns ``(boxes_xyxy_letterbox, scores)``.
        """
        from torchvision.ops import nms

        obj_conf = decoded[:, 4]
        cls_scores = decoded[:, 5:]
        # Standard YOLO score: objectness times best class probability.
        best_cls = cls_scores.max(dim=1).values if cls_scores.shape[1] else torch.ones_like(obj_conf)
        conf = obj_conf * best_cls
        keep = conf > self.cfg.conf_threshold
        if not bool(keep.any()):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes = xywh_to_xyxy(decoded[keep, :4].float())
        conf = conf[keep].float()
        idx = nms(boxes, conf, self.cfg.iou_threshold)
        idx = idx[: self.cfg.max_detections]
        return boxes[idx].cpu().numpy(), conf[idx].cpu().numpy()

    def infer(self, frame: np.ndarray) -> FrameResult:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        """Run a batch through the network. All frames must share a size."""
        if not frames:
            return []
        sizes = {f.shape[:2] for f in frames}
        if len(sizes) != 1:
            raise ValueError(f"a batch must share one frame size, got {sizes}")

        tensor, tf = self._preprocess(frames)
        with torch.no_grad():
            (pred, anchor_grid), seg, ll = self.model(tensor)
            decoded = decode_yolo_head(list(pred), list(anchor_grid))
            # Drivable area is a 2-class argmax; lane is a single sigmoid channel.
            drivable = seg.argmax(dim=1).to(torch.uint8)
            lane = (ll[:, 0] > self.cfg.lane_threshold).to(torch.uint8)
        drivable_np = drivable.cpu().numpy()
        lane_np = lane.cpu().numpy()

        results: list[FrameResult] = []
        for i in range(len(frames)):
            boxes_lb, scores = self._postprocess_detections(decoded[i])
            boxes_px = tf.invert_boxes(boxes_lb)
            # Drop boxes that clipping collapsed to nothing at the frame edge.
            if len(boxes_px):
                valid = (boxes_px[:, 2] > boxes_px[:, 0] + 1) & (boxes_px[:, 3] > boxes_px[:, 1] + 1)
                boxes_px, scores = boxes_px[valid], scores[valid]
            native = NativeMasks(
                drivable=np.ascontiguousarray(tf.crop_mask(drivable_np[i])),
                lane=np.ascontiguousarray(tf.crop_mask(lane_np[i])),
                original_size=tf.original_size,
            )
            results.append(
                FrameResult(
                    boxes_px=boxes_px,
                    class_ids=np.full(len(boxes_px), CLASS_IDS["vehicle"], dtype=np.int32),
                    scores=scores,
                    drivable_mask=LetterboxTransform.upsample_cached_mask(
                        native.drivable, tf.original_size
                    ),
                    lane_mask=LetterboxTransform.upsample_cached_mask(
                        native.lane, tf.original_size
                    ),
                    native_masks=native,
                )
            )
        return results

    def close(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Vulnerable road users
# --------------------------------------------------------------------------- #


class VulnerableRoadUserDetector(Detector):
    """COCO-pretrained detector covering people, bicycles and motorcycles.

    YOLOPv2 has no pedestrian class, and no released multi-task network in that
    family does -- HybridNets ships ``obj_list: ['car']`` and YOLOP is the same.
    Getting vulnerable road users therefore requires a second model.

    This uses torchvision's COCO detector rather than adding an ultralytics
    dependency: torchvision is already a hard requirement, so this costs no new
    package. It deliberately emits *only* vulnerable road users -- vehicles come
    from YOLOPv2, and taking them from both would double-count every car.
    """

    #: COCO category id -> this project's class name.
    COCO_TO_CLASS = {1: "person", 2: "bicycle", 4: "motorcycle"}

    def __init__(self, cfg: VRUConfig, device: torch.device) -> None:
        from torchvision.models import detection as tvdet

        self.cfg = cfg
        self.device = device
        builder = getattr(tvdet, cfg.architecture, None)
        if builder is None:
            raise ValueError(
                f"unknown torchvision detection architecture '{cfg.architecture}'"
            )
        self.model = builder(weights="DEFAULT", box_score_thresh=cfg.conf_threshold)
        self.model.eval().to(device)
        self._weights_hash = self._resolve_weights_hash(cfg.architecture)
        log_kv(
            _log, logging.INFO, "detect.loaded",
            model="vru", architecture=cfg.architecture,
            sha256=self._weights_hash[:12], device=str(device),
            classes=",".join(self.class_names),
        )

    @staticmethod
    def _resolve_weights_hash(architecture: str) -> str:
        """Hash the cached checkpoint so swapping weights invalidates the cache."""
        hub = Path(torch.hub.get_dir()) / "checkpoints"
        candidates = sorted(hub.glob("*.pth")) if hub.is_dir() else []
        stem = architecture.replace("_", "")
        for path in candidates:
            if path.name.replace("_", "").lower().startswith(stem[:12].lower()):
                return file_sha256(path)
        # Not fatal: fall back to an identity string so the manifest still records
        # *something* that changes when the architecture does.
        return hashlib.sha256(architecture.encode()).hexdigest()

    @property
    def name(self) -> str:
        return f"vru:{self.cfg.architecture}"

    @property
    def weights_hash(self) -> str:
        return self._weights_hash

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.COCO_TO_CLASS.values())

    def infer(self, frame: np.ndarray) -> FrameResult:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        if not frames:
            return []
        tensors = [
            torch.from_numpy(np.ascontiguousarray(f[..., ::-1].transpose(2, 0, 1)))
            .to(self.device)
            .float()
            / 255.0
            for f in frames
        ]
        with torch.no_grad():
            outputs = self.model(tensors)

        results = []
        for frame, out in zip(frames, outputs):
            h, w = frame.shape[:2]
            labels = out["labels"].cpu().numpy()
            wanted = np.isin(labels, list(self.COCO_TO_CLASS))
            boxes = out["boxes"].cpu().numpy()[wanted]
            scores = out["scores"].cpu().numpy()[wanted]
            ids = np.array(
                [CLASS_IDS[self.COCO_TO_CLASS[int(c)]] for c in labels[wanted]], dtype=np.int32
            )
            blank = np.zeros((h, w), dtype=np.uint8)
            results.append(
                FrameResult(
                    boxes_px=boxes.astype(np.float32),
                    class_ids=ids,
                    scores=scores.astype(np.float32),
                    drivable_mask=blank,
                    lane_mask=blank.copy(),
                )
            )
        return results


class CompositeDetector(Detector):
    """Merges a segmentation-bearing primary detector with extra box sources.

    Masks always come from the primary; boxes are concatenated. Each member keeps
    its own class vocabulary, and the union is what the manifest records.
    """

    def __init__(self, primary: Detector, extras: list[Detector]) -> None:
        self.primary = primary
        self.extras = extras

    @property
    def name(self) -> str:
        return "+".join([self.primary.name, *(d.name for d in self.extras)])

    @property
    def weights_hash(self) -> str:
        """Digest over every member, so changing any of them invalidates caches."""
        h = hashlib.sha256()
        for d in [self.primary, *self.extras]:
            h.update(d.name.encode())
            h.update(d.weights_hash.encode())
        return h.hexdigest()

    @property
    def class_names(self) -> tuple[str, ...]:
        seen: list[str] = []
        for d in [self.primary, *self.extras]:
            seen.extend(n for n in d.class_names if n not in seen)
        return tuple(seen)

    def infer(self, frame: np.ndarray) -> FrameResult:
        return self.infer_batch([frame])[0]

    def infer_batch(self, frames: list[np.ndarray]) -> list[FrameResult]:
        base = self.primary.infer_batch(frames)
        for extra in self.extras:
            for i, extra_result in enumerate(extra.infer_batch(frames)):
                base[i] = FrameResult(
                    boxes_px=np.vstack([base[i].boxes_px, extra_result.boxes_px]),
                    class_ids=np.concatenate([base[i].class_ids, extra_result.class_ids]),
                    scores=np.concatenate([base[i].scores, extra_result.scores]),
                    drivable_mask=base[i].drivable_mask,
                    lane_mask=base[i].lane_mask,
                    native_masks=base[i].native_masks,
                )
        return base

    def close(self) -> None:
        self.primary.close()
        for d in self.extras:
            d.close()


def build_detector(cfg: DetectorConfig, camera: object | None = None) -> Detector:
    """Construct the detector named by the config.

    Args:
        cfg: Detector settings.
        camera: A :class:`~src.calibrate.CameraModel`, required only by the stub,
            which generates its fake output through the calibration.

    Raises:
        ValueError: for an unknown ``detector.kind``, or a stub without a camera.
        FileNotFoundError: if the weights are missing.
    """
    if cfg.kind == "stub":
        from src.stubs import StubDetector

        if camera is None:
            raise ValueError("the stub detector requires a CameraModel")
        return StubDetector(camera, cfg)  # type: ignore[arg-type]
    if cfg.kind == "yolopv2":
        primary: Detector = YOLOPv2Detector(cfg)
        if cfg.vru.enabled:
            return CompositeDetector(primary, [VRUDetectorFactory(cfg)])
        return primary
    raise ValueError(f"unknown detector kind '{cfg.kind}'")


def VRUDetectorFactory(cfg: DetectorConfig) -> Detector:  # noqa: N802 - factory alias
    """Build the vulnerable-road-user detector on the same device as the primary."""
    return VulnerableRoadUserDetector(cfg.vru, resolve_device(cfg.device))
