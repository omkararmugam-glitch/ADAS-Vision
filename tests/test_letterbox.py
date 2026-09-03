"""Tests for the letterbox transform.

The round-trip is the important one. Scaling and padding is where box and mask
coordinates silently drift, and a misalignment of a few pixels near the bottom of
the frame becomes a metre of distance error after the homography -- which looks
plausible and is wrong. So the mapping is asserted against known rectangles at
several frame sizes, not just checked for absence of exceptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.letterbox import LetterboxTransform


class TestLetterboxGeometry:
    def test_720p_matches_the_reference_implementation(self) -> None:
        """1280x720 at long-side 640 must give 640x384 with 12 px of vertical pad.

        YOLOPv2's own post-processing hardcodes ``seg[:, :, 12:372, :]``, so this
        is the geometry the released weights were shipped against.
        """
        tf = LetterboxTransform.compute((1280, 720), 640, stride=32)
        assert tf.scale == pytest.approx(0.5)
        assert tf.scaled_size == (640, 360)
        assert tf.padded_size == (640, 384)
        assert tf.pad == pytest.approx((0.0, 12.0))

    def test_padded_size_is_a_multiple_of_stride(self) -> None:
        for size in [(1280, 720), (1920, 1080), (640, 480), (1000, 667), (333, 777)]:
            tf = LetterboxTransform.compute(size, 640, stride=32)
            assert tf.padded_size[0] % 32 == 0
            assert tf.padded_size[1] % 32 == 0

    def test_aspect_ratio_is_preserved(self) -> None:
        tf = LetterboxTransform.compute((1920, 1080), 640, stride=32)
        assert tf.scaled_size[0] / tf.scaled_size[1] == pytest.approx(1920 / 1080, abs=0.01)

    def test_small_frames_are_not_upscaled_by_default(self) -> None:
        tf = LetterboxTransform.compute((320, 240), 640, stride=32)
        assert tf.scale == 1.0
        assert tf.scaled_size == (320, 240)

    def test_upscaling_can_be_requested(self) -> None:
        tf = LetterboxTransform.compute((320, 240), 640, stride=32, allow_upscale=True)
        assert tf.scale > 1.0

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid frame size"):
            LetterboxTransform.compute((0, 720), 640)


class TestLetterboxRoundTrip:
    """A known rectangle must return to its original pixel coordinates."""

    @pytest.mark.parametrize(
        "size", [(1280, 720), (1920, 1080), (640, 480), (800, 600), (1000, 667)]
    )
    def test_rectangle_survives_the_round_trip(self, size: tuple[int, int]) -> None:
        w, h = size
        rect = np.array([[0.2 * w, 0.3 * h, 0.6 * w, 0.75 * h]])
        tf = LetterboxTransform.compute(size, 640, stride=32)

        corners = np.array([[rect[0, 0], rect[0, 1]], [rect[0, 2], rect[0, 3]]])
        lb = tf.forward_points(corners)
        boxes_lb = np.array([[lb[0, 0], lb[0, 1], lb[1, 0], lb[1, 1]]])
        back = tf.invert_boxes(boxes_lb)
        assert back == pytest.approx(rect, abs=1e-3)

    def test_round_trip_through_an_actual_rendered_image(self) -> None:
        """End to end: draw a rectangle, letterbox the image, recover the box.

        Uses the drawn pixels rather than the analytic transform, so an error in
        ``apply`` (wrong pad side, off-by-one) is caught, not just an error in
        the arithmetic.
        """
        w, h = 1280, 720
        image = np.zeros((h, w, 3), dtype=np.uint8)
        x1, y1, x2, y2 = 400, 300, 900, 600
        image[y1:y2, x1:x2] = 255

        tf = LetterboxTransform.compute((w, h), 640, stride=32)
        lb_image = tf.apply(image)
        assert lb_image.shape[:2][::-1] == tf.padded_size

        ys, xs = np.nonzero(lb_image[:, :, 0] > 128)
        boxes_lb = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float64)
        back = tf.invert_boxes(boxes_lb)[0]

        # One letterbox pixel is two original pixels at this scale.
        assert back == pytest.approx([x1, y1, x2, y2], abs=2.5)

    def test_mask_round_trip_returns_to_original_resolution(self) -> None:
        w, h = 1280, 720
        tf = LetterboxTransform.compute((w, h), 640, stride=32)
        # A mask at the network's input resolution, marking a known band.
        mask = np.zeros(tf.padded_size[::-1], dtype=np.uint8)
        mask[tf.crop_mask(mask).shape[0] // 2 + 12, :] = 1  # a row inside the content area
        restored = tf.invert_mask(mask)
        assert restored.shape == (h, w)

    def test_mask_crop_removes_exactly_the_padding(self) -> None:
        tf = LetterboxTransform.compute((1280, 720), 640, stride=32)
        mask = np.ones(tf.padded_size[::-1], dtype=np.uint8)
        cropped = tf.crop_mask(mask)
        assert cropped.shape == (360, 640), "crop should leave the 360 content rows"

    def test_padding_region_is_excluded_from_the_crop(self) -> None:
        """Content placed only in the pad must not survive cropping."""
        tf = LetterboxTransform.compute((1280, 720), 640, stride=32)
        mask = np.zeros(tf.padded_size[::-1], dtype=np.uint8)
        mask[0:12, :] = 1     # top padding only
        mask[372:384, :] = 1  # bottom padding only
        assert tf.crop_mask(mask).sum() == 0

    def test_corners_map_to_the_content_area_corners(self) -> None:
        tf = LetterboxTransform.compute((1280, 720), 640, stride=32)
        corners = np.array([[0.0, 0.0], [1280.0, 720.0]])
        lb = tf.forward_points(corners)
        assert lb[0] == pytest.approx([0.0, 12.0])
        assert lb[1] == pytest.approx([640.0, 372.0])

    def test_boxes_are_clipped_to_the_frame(self) -> None:
        tf = LetterboxTransform.compute((1280, 720), 640, stride=32)
        # A box extending into the letterbox padding maps outside the frame.
        out = tf.invert_boxes(np.array([[-50.0, 0.0, 700.0, 384.0]]))[0]
        assert out[0] >= 0 and out[1] >= 0
        assert out[2] <= 1280 and out[3] <= 720

    def test_empty_boxes_round_trip(self) -> None:
        tf = LetterboxTransform.compute((1280, 720), 640)
        assert tf.invert_boxes(np.zeros((0, 4))).shape == (0, 4)

    def test_apply_rejects_a_mismatched_frame(self) -> None:
        tf = LetterboxTransform.compute((1280, 720), 640)
        with pytest.raises(ValueError, match="computed for"):
            tf.apply(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_upsample_cached_mask_restores_size(self) -> None:
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[100:200, 200:400] = 1
        out = LetterboxTransform.upsample_cached_mask(mask, (1280, 720))
        assert out.shape == (720, 1280)
        assert set(np.unique(out)) <= {0, 1}, "nearest-neighbour must not blend labels"

