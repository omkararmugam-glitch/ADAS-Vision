"""YOLOPv2: the multi-task backbone.

One forward pass yields three heads -- vehicle boxes, drivable-area segmentation
and lane-marking segmentation -- which is why it is the default backbone here:
the baseline needs two separate networks to cover the same ground.

Its one gap is vulnerable road users. YOLOPv2 has no pedestrian class, and no
released multi-task network in this family does (HybridNets ships
``obj_list:['car']``). Pedestrians and cyclists therefore come from the separate
model in :mod:`models.perception.vru`, combined by
:class:`~models.perception.composite.CompositeDetector`.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from core.config import DetectorConfig
from core.letterbox import LetterboxTransform
from core.logging_utils import get_logger, log_kv
from core.types import CLASS_IDS, FrameResult, NativeMasks
from models.perception.base import Detector, file_sha256, resolve_device

_log = get_logger(__name__)


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

