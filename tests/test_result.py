from geood.result import DetectionResult


def test_detection_result_fields():
    r = DetectionResult(
        is_ood=True, score=0.95, mahalanobis=12.3,
        intrinsic_dim=4, reference_dim=33.8, layer=0
    )
    assert r.is_ood is True
    assert r.score == 0.95
    assert r.intrinsic_dim == 4
    assert r.layer == 0


def test_explain_ood():
    r = DetectionResult(
        is_ood=True, score=0.95, mahalanobis=12.3,
        intrinsic_dim=4, reference_dim=33.8, layer=0
    )
    explanation = r.explain()
    assert "OOD detected" in explanation
    assert "dim=4" in explanation
    assert "ref=33.8" in explanation
    assert "8.5x" in explanation
    assert "mahalanobis=12.3" in explanation


def test_explain_in_distribution():
    r = DetectionResult(
        is_ood=False, score=0.1, mahalanobis=1.2,
        intrinsic_dim=30, reference_dim=33.8, layer=0
    )
    explanation = r.explain()
    assert "In-distribution" in explanation


def test_explain_single_sample_no_dim():
    r = DetectionResult(
        is_ood=True, score=0.8, mahalanobis=15.0,
        intrinsic_dim=None, reference_dim=33.8, layer=0
    )
    explanation = r.explain()
    assert "OOD detected" in explanation
    assert "mahalanobis=15.0" in explanation
    assert "ref_dim=33.8" in explanation
