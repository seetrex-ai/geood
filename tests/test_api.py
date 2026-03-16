import geood
from geood.detector import Detector


def test_calibrate_is_importable():
    assert callable(geood.calibrate)


def test_load_is_importable():
    assert callable(geood.load)


def test_version():
    assert geood.__version__  # not empty
    parts = geood.__version__.split(".")
    assert len(parts) >= 2  # semver


def test_detection_result_importable():
    from geood import DetectionResult
    assert DetectionResult is not None


def test_detector_importable():
    from geood import Detector
    assert Detector is not None
