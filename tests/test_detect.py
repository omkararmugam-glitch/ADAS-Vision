"""Tests for the detector-side helpers: YOLO head decoding, device, weight hashes.

These need torch. The letterbox geometry that used to live here moved to
``test_letterbox.py`` when the transform moved into :mod:`core`, so that the
torch-free render path keeps a torch-free test.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.perception.base import file_sha256, resolve_device
from models.perception.yolopv2 import decode_yolo_head, xywh_to_xyxy


class TestYoloDecoding:
    def test_xywh_to_xyxy(self) -> None:
        import torch

        boxes = torch.tensor([[100.0, 200.0, 40.0, 60.0]])
        assert xywh_to_xyxy(boxes).tolist() == [[80.0, 170.0, 120.0, 230.0]]

    def test_decode_produces_one_row_per_anchor_cell(self) -> None:
        import torch

        na, nc = 3, 80
        pred = [
            torch.zeros(1, na * (5 + nc), 48, 80),
            torch.zeros(1, na * (5 + nc), 24, 40),
            torch.zeros(1, na * (5 + nc), 12, 20),
        ]
        anchors = [torch.ones(1, na, 1, 1, 2) for _ in range(3)]
        out = decode_yolo_head(pred, anchors)
        expected = na * (48 * 80 + 24 * 40 + 12 * 20)
        assert out.shape == (1, expected, 5 + nc)

    def test_decoded_boxes_land_inside_the_input_grid(self) -> None:
        """Cell centres must map into letterbox pixel space, not stay in cells."""
        import torch

        na, nc = 3, 80
        pred = [torch.zeros(1, na * (5 + nc), 12, 20)]
        anchors = [torch.full((1, na, 1, 1, 2), 10.0)]
        out = decode_yolo_head(pred, anchors, strides=(32,))
        xy = out[0, :, :2]
        # With zero logits, sigmoid gives 0.5 -> xy = (0.5*2 - 0.5 + cell) * stride
        assert float(xy[:, 0].min()) == pytest.approx(0.5 * 32)
        assert float(xy[:, 0].max()) == pytest.approx((0.5 + 19) * 32)


class TestDeviceAndHashing:
    def test_explicit_cpu_is_honoured(self) -> None:
        assert resolve_device("cpu").type == "cpu"

    def test_auto_selects_something_valid(self) -> None:
        assert resolve_device("auto").type in ("cuda", "cpu")

    def test_file_hash_is_stable_and_content_sensitive(self, tmp_path) -> None:
        a = tmp_path / "a.bin"
        a.write_bytes(b"weights" * 1000)
        first = file_sha256(a)
        assert first == file_sha256(a)
        a.write_bytes(b"different")
        assert file_sha256(a) != first


