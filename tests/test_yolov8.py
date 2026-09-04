"""Tests for the YOLOv8 baseline wrapper.

Most of these exercise logic that must hold without loading a network at all:
the COCO-to-project class mapping, and the weight-resolution rules. The mapping
is the part most likely to break silently -- a wrong index would relabel buses as
pedestrians and every downstream alert would still look plausible.

The one test that needs real weights skips when they are absent, so the suite
stays runnable on a machine with no checkpoints and no GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.config import DetectorConfig, YOLOv8Config
from core.types import CLASS_IDS, CLASS_NAMES
from models.perception.yolov8 import _PRETRAINED, YOLOv8Detector

pytest.importorskip("ultralytics", reason="ultralytics is optional until step 3")

WEIGHTS = Path("models/perception/weights/yolov8n.pt")


class TestClassMapping:
    def test_every_target_name_is_in_the_project_vocabulary(self) -> None:
        """A mapped name that CLASS_IDS does not know would KeyError at inference."""
        for name in YOLOv8Detector.COCO_TO_CLASS.values():
            assert name in CLASS_IDS, f"'{name}' is not in {CLASS_NAMES}"

    def test_the_three_coco_vehicle_classes_collapse_to_vehicle(self) -> None:
        """car, bus and truck must all read as 'vehicle', matching YOLOPv2."""
        for coco_id in (2, 5, 7):
            assert YOLOv8Detector.COCO_TO_CLASS[coco_id] == "vehicle"

    def test_vulnerable_road_users_keep_distinct_classes(self) -> None:
        assert YOLOv8Detector.COCO_TO_CLASS[0] == "person"
        assert YOLOv8Detector.COCO_TO_CLASS[1] == "bicycle"
        assert YOLOv8Detector.COCO_TO_CLASS[3] == "motorcycle"

    def test_class_names_deduplicates_vehicle(self) -> None:
        """Three COCO ids map to 'vehicle'; the manifest must list it once."""
        names = YOLOv8Detector.vocabulary()
        assert names.count("vehicle") == 1
        assert set(names) == {"person", "bicycle", "vehicle", "motorcycle",
                              "traffic_light", "traffic_sign"}

    def test_traffic_control_maps_to_its_own_classes(self) -> None:
        """COCO 9/11 now map to traffic control, added at schema v3.

        They must never fold into an existing class: a traffic light counted as
        a vehicle would enter the collision logic and warn about hitting a
        signal gantry.
        """
        assert YOLOv8Detector.COCO_TO_CLASS[9] == "traffic_light"
        assert YOLOv8Detector.COCO_TO_CLASS[11] == "traffic_sign"
        assert "vehicle" not in (YOLOv8Detector.COCO_TO_CLASS[9],
                                 YOLOv8Detector.COCO_TO_CLASS[11])

    def test_coco_has_only_one_kind_of_sign(self) -> None:
        """Worth pinning: COCO's only sign is the stop sign, so `traffic_sign`
        from this detector is narrower than the same class from a BDD100K model.
        """
        signs = [k for k, v in YOLOv8Detector.COCO_TO_CLASS.items()
                 if v == "traffic_sign"]
        assert signs == [11]


class TestPretrainedNamePattern:
    @pytest.mark.parametrize("name", ["yolov8n.pt", "yolov8x.pt", "yolo11s.pt", "yolo12m.pt"])
    def test_standard_names_are_downloadable(self, name: str) -> None:
        assert _PRETRAINED.match(name)

    @pytest.mark.parametrize(
        "name", ["best.pt", "yolov8n.onnx", "my-yolov8n.pt", "yolov8.pt", "yolov9n.pt"]
    )
    def test_everything_else_is_not(self, name: str) -> None:
        """A fine-tuned checkpoint must never be fetched from the internet."""
        assert not _PRETRAINED.match(name)


class TestWeightResolution:
    def test_an_existing_file_is_used_as_is(self, tmp_path: Path) -> None:
        w = tmp_path / "best.pt"
        w.write_bytes(b"not really a checkpoint")
        assert YOLOv8Detector._resolve_weights(YOLOv8Config(weights=w)) == w

    def test_a_missing_custom_checkpoint_raises(self, tmp_path: Path) -> None:
        cfg = YOLOv8Config(weights=tmp_path / "best.pt", allow_download=True)
        with pytest.raises(FileNotFoundError, match="standard pretrained"):
            YOLOv8Detector._resolve_weights(cfg)

    def test_download_disabled_raises_even_for_a_standard_name(self, tmp_path: Path) -> None:
        cfg = YOLOv8Config(weights=tmp_path / "yolov8n.pt", allow_download=False)
        with pytest.raises(FileNotFoundError):
            YOLOv8Detector._resolve_weights(cfg)


class TestConfig:
    def test_yolov8_is_an_accepted_kind(self) -> None:
        assert DetectorConfig(kind="yolov8").kind == "yolov8"

    def test_an_unknown_kind_still_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown detector kind"):
            DetectorConfig(kind="yolov9")

    def test_yolov8_weights_are_separate_from_the_top_level_weights(self) -> None:
        """Sharing one field would hand YOLOv8 the YOLOPv2 checkpoint."""
        cfg = DetectorConfig()
        assert cfg.yolov8.weights != cfg.weights


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="yolov8n.pt not downloaded")
class TestAgainstRealWeights:
    def test_a_blank_frame_yields_no_detections(self) -> None:
        d = YOLOv8Detector(DetectorConfig(kind="yolov8"))
        try:
            r = d.infer(np.zeros((480, 640, 3), np.uint8))
            assert len(r) == 0
            assert r.size == (640, 480)
        finally:
            d.close()

    def test_masks_are_empty_because_this_model_does_not_segment(self) -> None:
        """Lane data on this path must come from models/lane_seg, not from here."""
        d = YOLOv8Detector(DetectorConfig(kind="yolov8"))
        try:
            r = d.infer(np.zeros((480, 640, 3), np.uint8))
            assert r.drivable_mask.max() == 0
            assert r.lane_mask.max() == 0
        finally:
            d.close()

    def test_the_bundled_bus_image_gives_people_and_a_vehicle(self) -> None:
        """A real image, so the mapping is checked end to end and not just as a dict."""
        cv2 = pytest.importorskip("cv2")
        import ultralytics

        asset = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if not asset.is_file():
            pytest.skip("ultralytics assets not present")
        img = cv2.imread(str(asset))
        d = YOLOv8Detector(DetectorConfig(kind="yolov8"))
        try:
            r = d.infer(img)
        finally:
            d.close()
        names = {CLASS_NAMES[i] for i in r.class_ids}
        assert "person" in names
        assert "vehicle" in names, "the bus must map to 'vehicle'"


class TestCheckpointClassMapping:
    """The mapping must come from the checkpoint, never be assumed.

    A model fine-tuned on this project's vocabulary emits index 0 for `vehicle`.
    Read through the COCO table that is `person` -- silently wrong, and invisible
    until a pedestrian alert fires at a parked car.
    """

    class _FakeModel:
        def __init__(self, names):
            self.names = names

    COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign"}
    PROJECT = {0: "vehicle", 1: "person", 2: "bicycle", 3: "motorcycle"}

    def test_a_coco_checkpoint_uses_the_coco_table(self) -> None:
        from models.perception.yolov8 import YOLOv8Detector
        m = YOLOv8Detector._build_index_map(self._FakeModel(self.COCO))
        assert m[2] == "vehicle" and m[5] == "vehicle" and m[7] == "vehicle"
        assert m[0] == "person"
        assert m[9] == "traffic_light" and m[11] == "traffic_sign"
        assert 4 not in m, "airplane is outside the vocabulary and must be dropped"

    def test_a_project_trained_checkpoint_maps_identically(self) -> None:
        from models.perception.yolov8 import YOLOv8Detector
        m = YOLOv8Detector._build_index_map(self._FakeModel(self.PROJECT))
        assert m == {0: "vehicle", 1: "person", 2: "bicycle", 3: "motorcycle"}

    def test_index_zero_is_not_confused_between_the_two(self) -> None:
        """The exact bug this guards: class 0 means different things."""
        from models.perception.yolov8 import YOLOv8Detector
        coco = YOLOv8Detector._build_index_map(self._FakeModel(self.COCO))
        proj = YOLOv8Detector._build_index_map(self._FakeModel(self.PROJECT))
        assert coco[0] == "person"
        assert proj[0] == "vehicle"

    def test_an_unrecognised_vocabulary_raises_rather_than_guessing(self) -> None:
        from models.perception.yolov8 import YOLOv8Detector
        import pytest as _pytest
        weird = {0: "aeroplane", 1: "kangaroo", 2: "sausage"}
        with _pytest.raises(ValueError, match="cannot map"):
            YOLOv8Detector._build_index_map(self._FakeModel(weird))

    def test_a_reordered_eighty_class_model_is_not_assumed_to_be_coco(self) -> None:
        """Matched by name, not by class count."""
        from models.perception.yolov8 import YOLOv8Detector
        import pytest as _pytest
        shuffled = dict(self.COCO)
        shuffled[0], shuffled[2] = shuffled[2], shuffled[0]   # car <-> person
        with _pytest.raises(ValueError, match="cannot map"):
            YOLOv8Detector._build_index_map(self._FakeModel(shuffled))
