"""Tests for the BDD100K adapter.

Built against synthetic fixtures in the real BDD100K schema, so the adapter is
verified before the 100 MB label file arrives rather than after.

The cases that matter are the ones a careless converter gets wrong: normalised
coordinates (a sign error here silently trains the detector on mirrored boxes),
the empty-vs-missing label file distinction, and the night balancing that is the
entire reason for curating rather than taking the first N images.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.adapters.bdd100k import (
    BASE_CLASSES,
    CATEGORY_MAP,
    TRAFFIC_MAP,
    Record,
    curate,
    load_labels,
    write_split,
)

W, H = 1280.0, 720.0


def bdd_image(name: str, timeofday="daytime", weather="clear", labels=None) -> dict:
    """One record in BDD100K's own JSON shape."""
    return {
        "name": name,
        "attributes": {"weather": weather, "timeofday": timeofday, "scene": "city street"},
        "timestamp": 10000,
        "labels": labels or [],
    }


def bdd_box(category: str, x1, y1, x2, y2, colour=None) -> dict:
    attrs = {"occluded": False, "truncated": False}
    if colour is not None:
        attrs["trafficLightColor"] = colour
    return {
        "id": "0", "category": category, "attributes": attrs,
        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


@pytest.fixture
def label_file(tmp_path: Path) -> Path:
    data = [
        bdd_image("a.jpg", "daytime", "clear", [
            bdd_box("car", 100, 200, 300, 400),
            bdd_box("person", 500, 300, 540, 420),
        ]),
        bdd_image("b.jpg", "night", "rainy", [
            bdd_box("bus", 0, 100, 200, 300),
            bdd_box("traffic light", 600, 100, 620, 150, colour="red"),
        ]),
        bdd_image("c.jpg", "night", "clear", []),           # legitimately empty
        bdd_image("d.jpg", "daytime", "snowy", [
            bdd_box("train", 0, 0, 100, 100),               # dropped category
            bdd_box("rider", 700, 300, 730, 400),
            bdd_box("bike", 700, 350, 740, 420),
        ]),
    ]
    p = tmp_path / "det_train.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestLoading:
    def test_all_records_are_read(self, label_file: Path) -> None:
        recs = load_labels(label_file, CATEGORY_MAP)
        assert [r.name for r in recs] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]

    def test_attributes_are_carried_through(self, label_file: Path) -> None:
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert recs["b.jpg"].timeofday == "night"
        assert recs["b.jpg"].weather == "rainy"
        assert recs["b.jpg"].is_night
        assert not recs["a.jpg"].is_night

    def test_vehicle_categories_are_merged(self, label_file: Path) -> None:
        """car, bus and truck must all become one class, matching YOLOPv2."""
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert recs["a.jpg"].counts()["vehicle"] == 1     # car
        assert recs["b.jpg"].counts()["vehicle"] == 1     # bus

    def test_rider_becomes_a_person(self, label_file: Path) -> None:
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert recs["d.jpg"].counts()["person"] == 1
        assert recs["d.jpg"].counts()["bicycle"] == 1

    def test_unmapped_categories_are_dropped(self, label_file: Path) -> None:
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert "train" not in recs["d.jpg"].counts()

    def test_traffic_control_is_excluded_by_default(self, label_file: Path) -> None:
        """These extend the pipeline vocabulary, so they are opt-in."""
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert "traffic_light" not in recs["b.jpg"].counts()

    def test_traffic_control_is_included_when_asked(self, label_file: Path) -> None:
        cmap = {**CATEGORY_MAP, **TRAFFIC_MAP}
        recs = {r.name: r for r in load_labels(label_file, cmap)}
        assert recs["b.jpg"].counts()["traffic_light"] == 1

    def test_traffic_light_colour_is_captured(self, label_file: Path) -> None:
        """BDD100K records the colour, which answers the red/green question."""
        cmap = {**CATEGORY_MAP, **TRAFFIC_MAP}
        recs = {r.name: r for r in load_labels(label_file, cmap)}
        assert recs["b.jpg"].light_colours["red"] == 1

    def test_an_image_with_no_labels_is_kept(self, label_file: Path) -> None:
        recs = {r.name: r for r in load_labels(label_file, CATEGORY_MAP)}
        assert recs["c.jpg"].boxes == []


class TestCuration:
    def _records(self, n_night: int, n_day: int) -> list[Record]:
        out = []
        for i in range(n_night):
            out.append(Record(name=f"n{i}.jpg", timeofday="night",
                              boxes=[("vehicle", 0, 0, 100, 100)]))
        for i in range(n_day):
            out.append(Record(name=f"d{i}.jpg", timeofday="daytime",
                              boxes=[("vehicle", 0, 0, 100, 100)]))
        return out

    def _all(self, recs) -> set[str]:
        return {r.name for r in recs}

    def test_images_missing_from_disk_are_skipped(self) -> None:
        recs = self._records(2, 2)
        got = curate(recs, {"n0.jpg"}, None, None, [], 0.0, 1, 0)
        assert [r.name for r in got] == ["n0.jpg"]

    def test_night_fraction_is_honoured(self) -> None:
        recs = self._records(500, 500)
        got = curate(recs, self._all(recs), 100, 0.45, [], 0.0, 1, 0)
        night = sum(1 for r in got if r.is_night)
        assert len(got) == 100
        assert night == pytest.approx(45, abs=2)

    def test_a_short_night_pool_is_backfilled_with_day(self) -> None:
        """Asking for more night than exists must not silently return fewer."""
        recs = self._records(10, 500)
        got = curate(recs, self._all(recs), 100, 0.45, [], 0.0, 1, 0)
        assert len(got) == 100

    def test_small_boxes_are_dropped(self) -> None:
        recs = [Record(name="a.jpg", boxes=[("vehicle", 0, 0, 50, 8),
                                            ("vehicle", 0, 0, 50, 60)])]
        got = curate(recs, {"a.jpg"}, None, None, [], min_box_px=12.0,
                     min_objects=1, seed=0)
        assert len(got[0].boxes) == 1

    def test_an_image_left_empty_by_filtering_is_dropped(self) -> None:
        recs = [Record(name="a.jpg", boxes=[("vehicle", 0, 0, 50, 5)])]
        assert curate(recs, {"a.jpg"}, None, None, [], 12.0, 1, 0) == []

    def test_require_filters_to_images_containing_the_class(self) -> None:
        recs = [
            Record(name="a.jpg", boxes=[("vehicle", 0, 0, 50, 50)]),
            Record(name="b.jpg", boxes=[("vehicle", 0, 0, 50, 50),
                                        ("person", 0, 0, 20, 60)]),
        ]
        got = curate(recs, self._all(recs), None, None, ["person"], 0.0, 1, 0)
        assert [r.name for r in got] == ["b.jpg"]

    def test_selection_is_deterministic_for_a_seed(self) -> None:
        recs = self._records(50, 50)
        a = curate(recs, self._all(recs), 20, 0.5, [], 0.0, 1, seed=7)
        b = curate(self._records(50, 50), self._all(recs), 20, 0.5, [], 0.0, 1, seed=7)
        assert [r.name for r in a] == [r.name for r in b]


class TestWriting:
    def _image_dir(self, tmp_path: Path, names: list[str]) -> Path:
        d = tmp_path / "images"
        d.mkdir()
        for n in names:
            (d / n).write_bytes(b"\xff\xd8\xff\xd9")     # a minimal stand-in
        return d

    def test_boxes_are_normalised_correctly(self, tmp_path: Path) -> None:
        """A sign or axis error here silently trains on mirrored boxes."""
        rec = Record(name="a.jpg", boxes=[("vehicle", 100, 200, 300, 400)])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        line = (out / "labels/train/a.txt").read_text().strip()
        cls, cx, cy, bw, bh = line.split()
        assert cls == "0"
        assert float(cx) == pytest.approx(200 / W)      # centre x
        assert float(cy) == pytest.approx(300 / H)      # centre y
        assert float(bw) == pytest.approx(200 / W)      # width
        assert float(bh) == pytest.approx(200 / H)      # height

    def test_all_values_are_within_the_unit_interval(self, tmp_path: Path) -> None:
        rec = Record(name="a.jpg", boxes=[("vehicle", 0, 0, 1280, 720)])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        vals = [float(v) for v in
                (out / "labels/train/a.txt").read_text().split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_an_empty_image_gets_an_empty_file_not_a_missing_one(
        self, tmp_path: Path
    ) -> None:
        """Missing means 'never annotated'; conflating them teaches false negatives."""
        rec = Record(name="a.jpg", boxes=[])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        p = out / "labels/train/a.txt"
        assert p.is_file()
        assert p.read_text() == ""

    def test_class_indices_follow_the_declared_order(self, tmp_path: Path) -> None:
        rec = Record(name="a.jpg", boxes=[("motorcycle", 10, 10, 60, 60)])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        assert (out / "labels/train/a.txt").read_text().split()[0] == "3"

    def test_the_image_is_placed_beside_its_label(self, tmp_path: Path) -> None:
        rec = Record(name="a.jpg", boxes=[("vehicle", 10, 10, 60, 60)])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        assert (out / "images/train/a.jpg").is_file()
        assert (out / "labels/train/a.txt").is_file()

    def test_writing_twice_is_idempotent(self, tmp_path: Path) -> None:
        rec = Record(name="a.jpg", boxes=[("vehicle", 10, 10, 60, 60)])
        images = self._image_dir(tmp_path, ["a.jpg"])
        out = tmp_path / "out"
        a = write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        b = write_split([rec], images, out, "train", BASE_CLASSES, link=False)
        assert a == b
        assert len((out / "labels/train/a.txt").read_text().strip().splitlines()) == 1
