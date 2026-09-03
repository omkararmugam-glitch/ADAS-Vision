"""Combining a segmentation-bearing primary detector with extra box sources."""

from __future__ import annotations

import numpy as np

from core.types import FrameResult
from models.perception.base import Detector


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
