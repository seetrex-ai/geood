import numpy as np
from geood.metrics import intrinsic_dim, mahalanobis_distance, cosine_to_centroid


def test_intrinsic_dim_high_rank():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 50)
    dim = intrinsic_dim(X, threshold=0.95)
    assert dim >= 40


def test_intrinsic_dim_low_rank():
    rng = np.random.RandomState(42)
    basis = rng.randn(2, 50)
    coeffs = rng.randn(100, 2)
    X = coeffs @ basis
    dim = intrinsic_dim(X, threshold=0.95)
    assert dim == 2


def test_mahalanobis_in_dist_low():
    rng = np.random.RandomState(42)
    ref = rng.randn(100, 10)
    centroid = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False)
    sample = rng.randn(10)
    dist = mahalanobis_distance(sample, centroid, cov)
    assert dist < 10


def test_mahalanobis_ood_high():
    rng = np.random.RandomState(42)
    ref = rng.randn(100, 10)
    centroid = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False)
    ood = np.ones(10) * 100
    dist = mahalanobis_distance(ood, centroid, cov)
    assert dist > 50


def test_cosine_to_centroid():
    centroid = np.array([1.0, 0.0, 0.0])
    parallel = np.array([2.0, 0.0, 0.0])
    orthogonal = np.array([0.0, 1.0, 0.0])
    assert cosine_to_centroid(parallel, centroid) > 0.99
    assert abs(cosine_to_centroid(orthogonal, centroid)) < 0.01
