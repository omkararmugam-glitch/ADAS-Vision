"""The resize-and-pad mapping between original frame pixels and network input.

This lives in :mod:`core` rather than beside the model wrappers on purpose. The
transform is pure geometry -- numpy and OpenCV, no torch -- and the ``render``
path needs it to restore cached masks while explicitly *not* loading a model.
Keeping it here is what lets ``python run.py render`` run in an environment with
no deep-learning stack installed at all.

Coordinate frames:
  * **original px** -- the frame as it came from :mod:`core.io`, at the canonical
    size. Everything a ``Detector`` returns is in this frame.
  * **letterbox px** -- the network's padded input. Purely internal; nothing
    outside the perception models should ever see a coordinate in it.

The letterbox transform is the single most common source of silent misalignment
in this kind of pipeline: boxes and masks come back scaled and offset, and an
overlay that looks *almost* right hides a metric error that only surfaces
downstream as a wrong distance. :class:`LetterboxTransform` therefore owns both
directions of the mapping and is round-trip tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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

