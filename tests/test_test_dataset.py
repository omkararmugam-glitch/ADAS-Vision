"""Tests for the dataset-testing tool's pure logic.

The banding decides how results are grouped and therefore what conclusion a
reader draws about night performance, so the thresholds are pinned. The gallery
markup is checked because a malformed page is silently unusable.
"""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser

import pytest

from demo.test_dataset import DUSK_MAX, NIGHT_MAX, Result, band_of, gallery_html


class TestBanding:
    @pytest.mark.parametrize("brightness,expected", [
        (0.0, "night"), (20.0, "night"), (54.9, "night"),
        (55.0, "dusk"), (80.0, "dusk"), (94.9, "dusk"),
        (95.0, "day"), (200.0, "day"), (255.0, "day"),
    ])
    def test_thresholds(self, brightness: float, expected: str) -> None:
        assert band_of(brightness) == expected

    def test_the_bands_are_contiguous_and_ordered(self) -> None:
        assert NIGHT_MAX < DUSK_MAX
        assert band_of(NIGHT_MAX - 0.01) == "night"
        assert band_of(NIGHT_MAX) == "dusk"
        assert band_of(DUSK_MAX) == "day"


def _result(band: str, counts: dict[str, int]) -> Result:
    return Result(name=f"{band}_x.jpg", band=band, brightness=50.0,
                  counts=Counter(counts), scores=[0.8] * sum(counts.values()),
                  heights=[40.0] * sum(counts.values()))


class _Parser(HTMLParser):
    VOID = {"meta", "img", "br", "hr", "link", "input", "source"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.imgs = 0

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.imgs += 1
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(tag)
        else:
            self.stack.pop()


class TestGallery:
    def test_markup_is_well_formed(self) -> None:
        rs = [_result("day", {"vehicle": 3}), _result("night", {})]
        p = _Parser()
        p.feed(gallery_html(rs, "yolov8m", 0.25))
        assert p.errors == []
        assert [t for t in p.stack if t != "html"] == []

    def test_one_figure_per_result(self) -> None:
        rs = [_result("day", {"vehicle": 1}) for _ in range(5)]
        p = _Parser()
        p.feed(gallery_html(rs, "yolov8m", 0.25))
        assert p.imgs == 5

    def test_an_empty_frame_says_so_rather_than_showing_nothing(self) -> None:
        out = gallery_html([_result("night", {})], "yolov8m", 0.25)
        assert "nothing found" in out

    def test_the_inference_only_disclaimer_is_present(self) -> None:
        """The gallery must never read as an evaluation. Load-bearing."""
        out = gallery_html([_result("day", {"vehicle": 1})], "yolov8m", 0.25)
        assert "Inference, not evaluation" in out
        assert "no mAP is quoted" in out

    def test_counts_reach_the_page(self) -> None:
        out = gallery_html([_result("day", {"vehicle": 3, "person": 2})], "yolov8m", 0.25)
        assert "3&nbsp;vehicle" in out and "2&nbsp;person" in out
