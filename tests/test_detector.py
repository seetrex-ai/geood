import os
import tempfile
import numpy as np
from geood.detector import Detector
from geood.result import DetectionResult


def _make_fake_hidden(n_samples, dim=64):
    rng = np.random.RandomState(42)
    layers = [0, 8, 16, 24, 31]
    hidden = {}
    for idx in layers:
        hidden[idx] = [rng.randn(dim) for _ in range(n_samples)]
    return hidden, layers


def test_calibrate_from_vectors():
    hidden, layers = _make_fake_hidden(50)
    detector = Detector.calibrate_from_vectors(hidden, layers)
    assert detector.layer_used in layers
    assert detector._centroid is not None
    assert detector._cov is not None
    assert detector._reference_dim > 0


def test_detect_from_vector():
    hidden, layers = _make_fake_hidden(50)
    detector = Detector.calibrate_from_vectors(hidden, layers)
    rng = np.random.RandomState(99)
    vec = rng.randn(64)
    result = detector._detect_from_vector(vec)
    assert isinstance(result, DetectionResult)
    assert isinstance(result.is_ood, bool)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_save_load_roundtrip():
    hidden, layers = _make_fake_hidden(50)
    detector = Detector.calibrate_from_vectors(hidden, layers)
    rng = np.random.RandomState(99)
    vec = rng.randn(64)
    result_before = detector._detect_from_vector(vec)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_detector")
        detector.save(path)
        # Verify file has .npz extension
        assert os.path.exists(path + ".npz")
        # Load with or without extension
        loaded = Detector.load(path)
        result_after = loaded._detect_from_vector(vec)
        assert result_before.is_ood == result_after.is_ood
        assert abs(result_before.score - result_after.score) < 1e-6
        assert result_before.intrinsic_dim == result_after.intrinsic_dim


def test_layer_used_inspectable():
    hidden, layers = _make_fake_hidden(50)
    detector = Detector.calibrate_from_vectors(hidden, layers)
    assert isinstance(detector.layer_used, int)


def test_repr():
    hidden, layers = _make_fake_hidden(50)
    detector = Detector.calibrate_from_vectors(hidden, layers)
    r = repr(detector)
    assert "layer=" in r
    assert "ref_dim=" in r


def test_uncalibrated_repr():
    detector = Detector()
    assert "uncalibrated" in repr(detector)
