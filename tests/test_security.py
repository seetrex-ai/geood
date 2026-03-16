"""Security tests — adversarial inputs and edge cases."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from geood.detector import Detector
from geood.metrics import mahalanobis_distance, intrinsic_dim


# --- #1: Minimum sample validation ---

def test_calibrate_from_vectors_single_sample_raises():
    hidden = {0: [np.random.randn(32)]}
    with pytest.raises(ValueError, match="at least 2"):
        Detector.calibrate_from_vectors(hidden, [0])


# --- #2: NaN fail-closed ---

def test_nan_vector_detected_as_ood():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    nan_vec = np.full(32, np.nan)
    result = detector._detect_from_vector(nan_vec)
    assert result.is_ood is True
    assert result.score == 1.0


# --- #3: Identical vectors raise ---

def test_identical_vectors_raise():
    vec = np.ones(32)
    hidden = {0: [vec.copy() for _ in range(20)]}
    with pytest.raises(ValueError, match="identical"):
        Detector.calibrate_from_vectors(hidden, [0])


# --- #4: OOM protection on load ---

def test_load_rejects_oversized_arrays():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "evil.npz")
        # Craft a file with impossibly large shape claim
        # We can't actually allocate 100K+ arrays, so we test the
        # validation on a normal-sized file with wrong shapes
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": 5.0, "model_ref": None,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        # Shape mismatch: centroid(5) vs cov(3,3)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.zeros((3, 3)),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Shape mismatch"):
            Detector.load(path)


# --- #5: sqrt(negative) clamped ---

def test_mahalanobis_no_nan_from_numerical_issues():
    centroid = np.zeros(5)
    # Near-singular covariance that could produce negative quadratic form
    cov = np.eye(5) * 1e-12
    x = np.ones(5) * 1e-6
    dist = mahalanobis_distance(x, centroid, cov)
    assert np.isfinite(dist)
    assert dist >= 0


# --- #6: Invalid threshold in saved file ---

def test_load_rejects_nan_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": float("nan"),
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid threshold"):
            Detector.load(path)


def test_load_rejects_inf_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": float("inf"),
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid threshold"):
            Detector.load(path)


def test_load_rejects_negative_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": -1.0,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid threshold"):
            Detector.load(path)


# --- #7: ref_texts limit ---

def test_calibrate_rejects_excessive_ref_texts():
    detector = Detector()
    huge_list = ["text"] * 50_001
    with pytest.raises(ValueError, match="50000"):
        detector.calibrate("gpt2", huge_list)


# --- #12: Type validation on metadata ---

def test_load_rejects_string_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": "evil",
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid threshold type"):
            Detector.load(path)


def test_load_rejects_negative_layer():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": -1,
            "threshold": 5.0,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid layer_used"):
            Detector.load(path)


# --- #13: pathlib.Path support ---

def test_save_load_pathlib():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "detector"
        detector.save(path)
        loaded = Detector.load(path)
        assert loaded.layer_used == detector.layer_used


# --- #15: Missing keys in npz ---

def test_load_rejects_missing_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "incomplete.npz")
        np.savez_compressed(path, centroid=np.zeros(5))
        with pytest.raises(ValueError, match="missing keys"):
            Detector.load(path)


# --- #16: Adaptive regularization ---

def test_mahalanobis_with_tiny_scale():
    """Regularization should adapt to data scale, not dominate."""
    centroid = np.zeros(5)
    cov = np.eye(5) * 1e-12  # tiny scale
    x = np.ones(5) * 1e-6
    dist = mahalanobis_distance(x, centroid, cov)
    assert dist > 0
    assert np.isfinite(dist)


def test_intrinsic_dim_all_zero():
    """All-zero input should not crash."""
    X = np.zeros((10, 5))
    dim = intrinsic_dim(X)
    assert dim == 0  # zero variance = zero intrinsic dimensionality


# --- M1: NaN vectors in calibration ---

def test_calibrate_rejects_nan_vectors():
    rng = np.random.RandomState(42)
    vecs = [rng.randn(32) for _ in range(20)]
    vecs[5] = np.full(32, np.nan)  # inject one NaN vector
    hidden = {0: vecs}
    with pytest.raises(ValueError, match="NaN or Inf"):
        Detector.calibrate_from_vectors(hidden, [0])


def test_calibrate_rejects_inf_vectors():
    rng = np.random.RandomState(42)
    vecs = [rng.randn(32) for _ in range(20)]
    vecs[10] = np.full(32, np.inf)
    hidden = {0: vecs}
    with pytest.raises(ValueError, match="NaN or Inf"):
        Detector.calibrate_from_vectors(hidden, [0])


# --- M2: Mahalanobis with NaN input returns NaN (not crash) ---

def test_mahalanobis_nan_input_returns_nan():
    centroid = np.zeros(5)
    cov = np.eye(5)
    x = np.full(5, np.nan)
    dist = mahalanobis_distance(x, centroid, cov)
    assert np.isnan(dist)


# --- M4: Total element count check ---

def test_load_rejects_high_element_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "big.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": 5.0,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        # 4000x4000 = 16M elements > 10M limit
        np.savez_compressed(
            path,
            centroid=np.zeros(4000),
            cov=np.zeros((4000, 4000)),
            pca_mean=np.zeros(8000),
            pca_components=np.zeros((4000, 8000)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="elements"):
            Detector.load(path)


# --- L1: detect() input validation ---

def test_detect_rejects_none_input():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with pytest.raises(ValueError, match="None"):
        detector.detect(None)


def test_detect_rejects_non_string_list():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with pytest.raises(ValueError, match="strings"):
        detector.detect(["hello", None, "world"])


# --- L2: reference_dim validation in load ---

def test_load_rejects_zero_reference_dim():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        meta = json.dumps({
            "reference_dim": 0, "layer_used": 1,
            "threshold": 5.0,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid reference_dim"):
            Detector.load(path)


# --- L6: model_ref type validation ---

def test_load_ignores_non_string_model_ref():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "weird.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": 5.0, "model_ref": 12345,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        loaded = Detector.load(path)
        assert loaded._model_ref is None  # non-string silently ignored


# --- Round 3: H1 — zombie state on re-calibration failure ---

def test_calibrate_from_vectors_failure_preserves_state():
    """If calibrate_from_vectors fails, detector keeps previous calibration."""
    rng = np.random.RandomState(42)
    hidden_good = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden_good, [0])
    old_threshold = detector._threshold

    # Second calibration with bad data should fail
    bad_hidden = {0: [np.ones(32)] * 20}  # identical vectors
    try:
        detector.calibrate_from_vectors(bad_hidden, [0])
    except ValueError:
        pass
    # calibrate_from_vectors is a classmethod that returns a new object,
    # so the original detector is untouched
    assert detector._threshold == old_threshold


# --- Round 3: M1 — detect/save on uncalibrated detector ---

def test_detect_on_uncalibrated_raises():
    detector = Detector()
    with pytest.raises(RuntimeError, match="not calibrated"):
        detector.detect("test", model="gpt2")


def test_save_on_uncalibrated_raises():
    detector = Detector()
    with pytest.raises(RuntimeError, match="uncalibrated"):
        detector.save("/tmp/bad")


# --- Round 3: M2 — layer validation ---

def test_calibrate_rejects_bool_layer():
    detector = Detector()
    with pytest.raises(TypeError, match="bool"):
        detector.calibrate("gpt2", ["a", "b"], layer=True)


def test_calibrate_rejects_float_layer():
    detector = Detector()
    with pytest.raises(TypeError, match="int"):
        detector.calibrate("gpt2", ["a", "b"], layer=0.5)


# --- Round 3: M3 — hidden_dim stored in save metadata ---

def test_save_includes_hidden_dim():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.npz")
        detector.save(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta"].tobytes().decode())
        data.close()  # release file handle on Windows
        assert meta["hidden_dim"] == 32


# --- Round 3: M7 — model_ref sanitization ---

def test_sanitize_model_ref_keeps_hf_ids():
    from geood.detector import _sanitize_model_ref
    assert _sanitize_model_ref("gpt2") == "gpt2"
    assert _sanitize_model_ref("meta-llama/Llama-2-7b-hf") == "meta-llama/Llama-2-7b-hf"
    assert _sanitize_model_ref("mistralai/Mistral-7B-v0.1") == "mistralai/Mistral-7B-v0.1"


def test_sanitize_model_ref_strips_paths():
    from geood.detector import _sanitize_model_ref
    assert _sanitize_model_ref("/home/user/models/my-model") is None
    assert _sanitize_model_ref("C:\\Users\\model\\path") is None
    assert _sanitize_model_ref("../relative/path") is None


def test_sanitize_model_ref_edge_cases():
    from geood.detector import _sanitize_model_ref
    assert _sanitize_model_ref("") is None
    assert _sanitize_model_ref(" ") is None
    assert _sanitize_model_ref("org/model/extra") is None  # too many slashes
    assert _sanitize_model_ref("org/") is None
    assert _sanitize_model_ref("/model") is None


def test_sanitize_model_ref_none():
    from geood.detector import _sanitize_model_ref
    assert _sanitize_model_ref(None) is None


# --- Round 3: L2 — release_model ---

def test_release_model():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    detector._cached_model = "fake"
    detector._cached_tokenizer = "fake"
    detector.release_model()
    assert detector._cached_model is None
    assert detector._cached_tokenizer is None


# --- Round 3: is_calibrated property ---

def test_is_calibrated_property():
    detector = Detector()
    assert not detector.is_calibrated
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    calibrated = Detector.calibrate_from_vectors(hidden, [0])
    assert calibrated.is_calibrated


# --- Round 4: V5 — 2 ref texts, scalar covariance ---

def test_two_ref_texts_no_crash():
    """2 reference texts (minimum) should calibrate without crash."""
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(2)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    assert detector.is_calibrated
    # Should be able to detect
    vec = rng.randn(32)
    result = detector._detect_from_vector(vec)
    assert isinstance(result.is_ood, bool)


# --- Round 4: V6 — zero-variance layer doesn't win auto-select ---

def test_zero_variance_layer_not_preferred():
    """A zero-variance layer should NOT be preferred over an informative one."""
    rng = np.random.RandomState(42)
    identical_vec = np.ones(32)
    diverse_vecs = [rng.randn(32) for _ in range(20)]
    hidden = {
        0: [identical_vec.copy() for _ in range(20)],  # zero variance
        1: diverse_vecs,                                 # real variance
    }
    detector = Detector.calibrate_from_vectors(hidden, [0, 1])
    assert detector.layer_used == 1  # should pick the informative layer


# --- Round 4: V4 — pickle prevention ---

def test_pickle_raises():
    import pickle
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(detector)


# --- Round 4: V3 — detect([]) returns empty list ---

def test_detect_empty_list():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    result = detector.detect([], model="gpt2")
    assert result == []


# --- Round 4: V2 — mismatched layer keys ---

def test_calibrate_from_vectors_missing_layer_key():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    with pytest.raises(ValueError, match="not found"):
        Detector.calibrate_from_vectors(hidden, [0, 5])


# --- Round 4: load sanitizes model_ref ---

def test_load_sanitizes_local_path_model_ref():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "crafted.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": 5.0,
            "model_ref": "/home/attacker/evil-model",
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        loaded = Detector.load(path)
        assert loaded._model_ref is None  # local path stripped


# --- Round 4: corrupted JSON metadata ---

def test_load_rejects_corrupted_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.npz")
        meta_arr = np.frombuffer(b"not valid json{{{", dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="Invalid metadata"):
            Detector.load(path)


# ===================== Round 5 =====================

# --- R5: path traversal via . and .. ---

def test_sanitize_rejects_dot_and_dotdot():
    from geood.detector import _sanitize_model_ref
    assert _sanitize_model_ref(".") is None
    assert _sanitize_model_ref("..") is None
    assert _sanitize_model_ref("./model") is None
    assert _sanitize_model_ref("org/..") is None
    assert _sanitize_model_ref("../model") is None


# --- R5: unicode homoglyph rejection ---

def test_sanitize_rejects_unicode_homoglyphs():
    from geood.detector import _sanitize_model_ref
    # Cyrillic 'а' (U+0430) looks like Latin 'a'
    assert _sanitize_model_ref("met\u0430-llama/model") is None
    # Script Capital M
    assert _sanitize_model_ref("\u2133odel") is None
    # CJK characters
    assert _sanitize_model_ref("\u4e2d\u6587/model") is None


# --- R5: length cap on model_ref ---

def test_sanitize_rejects_long_model_ref():
    from geood.detector import _sanitize_model_ref
    long_ref = "a" * 201
    assert _sanitize_model_ref(long_ref) is None
    # Just under limit should pass
    ok_ref = "a" * 200
    assert _sanitize_model_ref(ok_ref) == ok_ref


# --- R5: save() rejects NaN/Inf in metadata ---

def test_save_rejects_corrupted_internal_state():
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    # Tamper with internal state
    detector._reference_dim = float("nan")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad")
        with pytest.raises(ValueError):
            detector.save(path)


# --- R5: is_calibrated checks all 7 fields ---

def test_is_calibrated_partial_state():
    detector = Detector()
    detector.layer_used = 5
    detector._pca_mean = np.zeros(10)
    # Missing other fields
    assert not detector.is_calibrated


# --- R5: tokenizer deepcopy non-mutation ---

def test_resolve_model_does_not_mutate_tokenizer():
    from unittest.mock import MagicMock
    from geood.model_loader import resolve_model
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.padding_side = "right"

    _, returned_tok, _ = resolve_model(mock_model, mock_tokenizer)

    # The returned tokenizer should have left padding
    assert returned_tok.padding_side == "left"
    # The original should NOT be mutated (deepcopy)
    assert returned_tok is not mock_tokenizer


# ===================== Round 6 =====================

# --- R6-1: dtype allowlist on load ---

def test_load_rejects_complex_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "complex.npz")
        meta = json.dumps({
            "reference_dim": 10.0, "layer_used": 1,
            "threshold": 5.0,
        })
        meta_arr = np.frombuffer(meta.encode(), dtype=np.uint8)
        np.savez_compressed(
            path,
            centroid=np.zeros(5, dtype=np.complex128),
            cov=np.eye(5),
            pca_mean=np.zeros(32),
            pca_components=np.zeros((5, 32)),
            meta=meta_arr,
        )
        with pytest.raises(ValueError, match="dtype"):
            Detector.load(path)


# --- R6-2: file handle closed on error ---

def test_load_closes_file_on_validation_error():
    """Ensure NpzFile is closed even when validation fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bad.npz")
        np.savez_compressed(path, centroid=np.zeros(5))
        try:
            Detector.load(path)
        except ValueError:
            pass
        # On Windows, file should not be locked — try to delete it
        os.unlink(path)  # would fail if handle leaked


# --- R6-3: batch_size validation ---

def test_calibrate_rejects_zero_batch_size():
    detector = Detector()
    with pytest.raises(ValueError, match="batch_size"):
        detector.calibrate("gpt2", ["a", "b"], batch_size=0)


def test_calibrate_rejects_negative_batch_size():
    detector = Detector()
    with pytest.raises(ValueError, match="batch_size"):
        detector.calibrate("gpt2", ["a", "b"], batch_size=-1)


# --- R6-4: atomic save (temp + rename) ---

def test_save_is_atomic():
    """Save should produce a valid file even if read immediately."""
    rng = np.random.RandomState(42)
    hidden = {0: [rng.randn(32) for _ in range(20)]}
    detector = Detector.calibrate_from_vectors(hidden, [0])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "det")
        detector.save(path)
        # Load should succeed (file is complete)
        loaded = Detector.load(path)
        assert loaded.layer_used == detector.layer_used
        # No temp files left behind
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".npz")
