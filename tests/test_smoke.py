"""Smoke test: simulates what a real user would do after pip install geood.

No mocks, no internals, no fixtures. Just the public API.
Must verify that results are MEANINGFUL, not just that the API doesn't crash.
"""

import os
import tempfile

import pytest


# ── Tiny model (fast, runs everywhere) ────────────────────────────


def _make_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 4
    config.n_embd = 128
    config.n_head = 4
    config.n_inner = 256
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


def test_user_journey_tiny():
    """Full API flow with tiny random model."""
    import geood

    model, tokenizer = _make_tiny_model()
    ref_texts = [f"This is reference sentence number {i}." for i in range(30)]
    detector = geood.calibrate(model, ref_texts, tokenizer=tokenizer)

    assert repr(detector).startswith("Detector(")
    assert detector.layer_used is not None
    assert detector.layer_used > 0

    result = detector.detect("Hello world.", model=model, tokenizer=tokenizer)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.explain(), str)

    # INVARIANT: calibration texts must be in-distribution
    ref_result = detector.detect(ref_texts[0], model=model, tokenizer=tokenizer)
    assert ref_result.score < 0.9, (
        f"Calibration text scored {ref_result.score:.3f} — detector is broken."
    )

    # Batch
    results = detector.detect(
        ["One text.", "Another text.", "Third text."],
        model=model, tokenizer=tokenizer,
    )
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.score <= 1.0
        assert r.intrinsic_dim is not None

    # Save/load roundtrip
    with tempfile.TemporaryDirectory() as d:
        detector.save(os.path.join(d, "det"))
        loaded = geood.load(os.path.join(d, "det"))
        r2 = loaded.detect("Hello world.", model=model, tokenizer=tokenizer)
        assert abs(result.score - r2.score) < 1e-6


def test_empty_ref_texts_raises():
    """Empty ref_texts should raise ValueError."""
    import geood

    model, tokenizer = _make_tiny_model()
    with pytest.raises(ValueError, match="at least 2 texts"):
        geood.calibrate(model, [], tokenizer=tokenizer)


# ── Real GPT-2 (slow, but catches real-world issues) ─────────────


def _make_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer


_ENGLISH_TEXTS = [
    "The weather today is sunny and warm.",
    "She walked to the store to buy groceries.",
    "The meeting was scheduled for three o'clock.",
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
    "The concert was sold out within minutes.",
    "He painted the bedroom walls a light blue.",
    "The baby slept through the entire night.",
    "She took the bus to the downtown area.",
    "The team celebrated their victory with dinner.",
    "He cleaned the garage over the weekend.",
    "The sunset was beautiful from the hilltop.",
    "She enrolled in an online cooking class.",
    "The neighbor mowed his lawn on Saturday.",
    "He returned the overdue books to the library.",
]

_CODE_TEXTS = [
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "import numpy as np; x = np.linspace(0, 1, 100); y = np.sin(x)",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
    "for i in range(len(arr)): arr[i] = arr[i] ** 2",
    "class Node: def __init__(self, val): self.val = val; self.next = None",
]


@pytest.mark.slow
def test_gpt2_calibration_texts_are_indist():
    """Calibration texts must score as in-distribution on real GPT-2."""
    import geood

    model, tokenizer = _make_gpt2()
    detector = geood.calibrate(model, _ENGLISH_TEXTS, tokenizer=tokenizer)

    for text in _ENGLISH_TEXTS[:5]:
        result = detector.detect(text, model=model, tokenizer=tokenizer)
        assert result.score < 0.9, (
            f"Calibration text scored {result.score:.3f}: {text[:50]}..."
        )


@pytest.mark.slow
def test_gpt2_scores_are_bounded():
    """All scores must be finite and in [0, 1] on real GPT-2.

    Note: GPT-2 is too small to produce strong OOD separation.
    This test verifies correctness, not separation quality.
    Use 7B+ models for meaningful OOD detection.
    """
    import geood
    import numpy as np

    model, tokenizer = _make_gpt2()
    detector = geood.calibrate(model, _ENGLISH_TEXTS, tokenizer=tokenizer)

    all_texts = _ENGLISH_TEXTS[:5] + _CODE_TEXTS
    for text in all_texts:
        result = detector.detect(text, model=model, tokenizer=tokenizer)
        assert np.isfinite(result.score), f"Non-finite score for: {text[:40]}"
        assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of [0,1]"
        assert np.isfinite(result.mahalanobis)


@pytest.mark.slow
def test_gpt2_save_load_consistent():
    """Save/load roundtrip with real GPT-2 must produce identical scores."""
    import geood

    model, tokenizer = _make_gpt2()
    detector = geood.calibrate(model, _ENGLISH_TEXTS, tokenizer=tokenizer)

    test_text = "The cat sat on the mat."
    result_before = detector.detect(test_text, model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as d:
        detector.save(os.path.join(d, "gpt2_det"))
        loaded = geood.load(os.path.join(d, "gpt2_det"))
        result_after = loaded.detect(test_text, model=model, tokenizer=tokenizer)

    assert abs(result_before.score - result_after.score) < 1e-6
    assert result_before.is_ood == result_after.is_ood
