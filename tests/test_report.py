"""Tests for the standalone HTML report generator.

The report is the thing someone else actually looks at, so the checks are the
ones that make it unusable if wrong: well-formed markup, images that resolve,
and the embedded variant genuinely carrying its own pixels rather than silently
linking to files that will not travel with it.
"""

from __future__ import annotations

import base64
from html.parser import HTMLParser
from pathlib import Path

import pytest

from demo.build_report import Figure, Section, build_sections, render

VOID = {"meta", "img", "br", "hr", "link", "input", "source"}


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.img_srcs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.img_srcs.append(dict(attrs).get("src", ""))
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"mismatched </{tag}>")
        else:
            self.stack.pop()


def parse(markup: str) -> _Parser:
    p = _Parser()
    p.feed(markup)
    return p


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    d = tmp_path / "assets"
    d.mkdir()
    # A real one-pixel PNG, so base64 embedding is genuinely exercised
    # rather than passing on a placeholder that happens to be non-empty.
    (d / "a.png").write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ))
    return d


@pytest.fixture
def sections() -> list[Section]:
    return [Section(
        heading="Test section",
        body=["A paragraph."],
        table=(["A", "B"], [["1", "2"]]),
        figures=[Figure("a.png", "A title", "A caption", "A note")],
    )]


class TestMarkup:
    def test_output_is_well_formed(self, sections, assets: Path) -> None:
        p = parse(render(sections, assets, embed=False))
        assert p.errors == []
        assert [t for t in p.stack if t != "html"] == []

    def test_content_reaches_the_page(self, sections, assets: Path) -> None:
        out = render(sections, assets, embed=False)
        for expected in ("Test section", "A paragraph.", "A title", "A caption", "A note"):
            assert expected in out

    def test_the_table_is_rendered(self, sections, assets: Path) -> None:
        out = render(sections, assets, embed=False)
        assert "<th>A</th>" in out and "<td>1</td>" in out

    def test_the_no_training_banner_is_present(self, assets: Path) -> None:
        """The report must never imply a model was trained. This is load-bearing."""
        out = render(build_sections(), assets, embed=False)
        assert "No model has been fine-tuned" in out
        assert "no mAP" in out.lower() or "No mAP" in out


class TestImageModes:
    def test_linked_mode_points_at_the_assets_folder(self, sections, assets: Path) -> None:
        srcs = parse(render(sections, assets, embed=False)).img_srcs
        assert srcs == ["assets/a.png"]

    def test_embedded_mode_inlines_the_bytes(self, sections, assets: Path) -> None:
        """An embedded report has to survive being emailed on its own."""
        srcs = parse(render(sections, assets, embed=True)).img_srcs
        assert len(srcs) == 1
        assert srcs[0].startswith("data:image/png;base64,")
        assert len(srcs[0]) > 60

    def test_embedded_is_larger_than_linked(self, sections, assets: Path) -> None:
        assert len(render(sections, assets, True)) > len(render(sections, assets, False))


class TestRealSections:
    def test_every_declared_figure_names_a_file(self) -> None:
        """Catches a typo in build_sections before the report is generated."""
        for s in build_sections():
            for f in s.figures:
                assert f.file.endswith((".jpg", ".png"))
                assert f.title and f.caption

    def test_the_status_table_lists_the_blocked_work(self) -> None:
        head_rows = [s.table for s in build_sections() if s.table]
        flat = " ".join(str(r) for t in head_rows for r in t[1])
        assert "blocked" in flat.lower()
