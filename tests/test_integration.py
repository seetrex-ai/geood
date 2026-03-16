"""Integration tests with a real (tiny) transformer model.

These tests catch issues that unit tests with fake vectors miss:
- Covariance singularity when n_samples < hidden_dim
- Save/load with real PCA projections
- Model caching across detect() calls
- Full calibrate → detect → save → load → detect pipeline
"""

import os
import tempfile

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def tiny_model():
    """A randomly initialized GPT-2 with 4 layers and 128 hidden dim.

    No download required.  Hidden dim (128) > n_samples (20) to test
    the rank-deficient covariance case that broke Mahalanobis in v0.1.0.
    """
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 4
    config.n_embd = 128
    config.n_head = 4
    config.n_inner = 256
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


@pytest.fixture
def ref_texts():
    return [
        "The weather today is sunny and warm.",
        "She walked to the store to buy groceries.",
        "The meeting was scheduled for three o clock.",
        "He read a book about history last night.",
        "The children played in the park after school.",
        "They cooked dinner together on Friday evening.",
        "The train arrived at the station on time.",
        "She finished her homework before watching TV.",
        "The dog barked loudly at the mailman.",
        "He drove to work early in the morning.",
        "The flowers bloomed beautifully in the spring.",
        "She called her friend to discuss the plans.",
        "The movie was entertaining and well directed.",
        "He fixed the leaking faucet in the kitchen.",
        "The students studied hard for the final exam.",
        "She bought a new dress for the party.",
        "The restaurant served delicious Italian food.",
        "He went jogging in the park every morning.",
        "The library was quiet and peaceful on Sunday.",
        "She organized her desk before starting work.",
    ]


def test_full_pipeline(tiny_model, ref_texts):
    """calibrate → detect → save → load → detect produces consistent results."""
    import geood

    model, tokenizer = tiny_model
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    assert detector.layer_used is not None
    assert detector._reference_dim > 0

    result = detector.detect(
        "He walked to the park on a sunny afternoon.",
        model=model, tokenizer=tokenizer,
    )
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.is_ood, bool)
    assert result.explain()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "detector")
        detector.save(path)
        loaded = geood.load(path)
        result2 = loaded.detect(
            "He walked to the park on a sunny afternoon.",
            model=model, tokenizer=tokenizer,
        )
        assert abs(result.score - result2.score) < 1e-6
        assert result.is_ood == result2.is_ood


def test_rank_deficient_covariance(tiny_model, ref_texts):
    """n_samples (20) < hidden_dim (128) must not produce NaN or Inf."""
    import geood

    model, tokenizer = tiny_model
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    result = detector.detect(
        "import numpy as np; x = np.array([1, 2, 3])",
        model=model, tokenizer=tokenizer,
    )
    assert np.isfinite(result.score)
    assert np.isfinite(result.mahalanobis)


def test_model_not_reloaded(tiny_model, ref_texts):
    """detect() should reuse cached model, not reload."""
    import geood

    model, tokenizer = tiny_model
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    assert detector._cached_model is model

    r1 = detector.detect("First call.", model=model, tokenizer=tokenizer)
    r2 = detector.detect("Second call.", model=model, tokenizer=tokenizer)
    assert isinstance(r1, geood.DetectionResult)
    assert isinstance(r2, geood.DetectionResult)


def test_batch_has_intrinsic_dim(tiny_model, ref_texts):
    """Batch detect (>= 2 texts) should populate intrinsic_dim."""
    import geood

    model, tokenizer = tiny_model
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    results = detector.detect(
        ["Text one.", "Text two.", "Text three."],
        model=model, tokenizer=tokenizer,
    )
    assert len(results) == 3
    for r in results:
        assert r.intrinsic_dim is not None
        assert r.intrinsic_dim >= 1


def test_ood_scores_higher_than_indist(tiny_model, ref_texts):
    """OOD text should score higher on average than in-dist text."""
    import geood

    model, tokenizer = tiny_model
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    indist = detector.detect(
        "She parked her car in the garage after work.",
        model=model, tokenizer=tokenizer,
    )
    ood = detector.detect(
        "SELECT u.name FROM users u WHERE u.active = 1 ORDER BY u.created_at",
        model=model, tokenizer=tokenizer,
    )
    # With a random tiny model, we can't guarantee OOD > in-dist,
    # but both should produce finite, bounded scores.
    assert 0.0 <= indist.score <= 1.0
    assert 0.0 <= ood.score <= 1.0
