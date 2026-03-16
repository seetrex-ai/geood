"""Core geometric metrics for OOD detection (pure numpy, no torch)."""

from __future__ import annotations

import numpy as np

__all__ = ["intrinsic_dim", "mahalanobis_distance", "cosine_to_centroid"]


def intrinsic_dim(X: np.ndarray, threshold: float = 0.95) -> int:
    """Number of PCA components needed to explain *threshold* variance."""
    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or Inf values.")
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total == 0:
        return 0  # zero variance = zero intrinsic dimensionality
    cumvar = np.cumsum(eigenvalues) / total
    return int(np.searchsorted(cumvar, threshold) + 1)


def mahalanobis_distance(
    x: np.ndarray, centroid: np.ndarray, cov: np.ndarray,
) -> float:
    """Mahalanobis distance from *x* to *centroid* under *cov*."""
    # Scale-adaptive regularization: proportional to data magnitude
    reg_scale = max(float(np.abs(cov).max()) * 1e-6, 1e-10)
    reg_cov = cov + np.eye(cov.shape[0]) * reg_scale
    cov_inv = np.linalg.pinv(reg_cov)
    diff = x - centroid
    quadratic = float(diff @ cov_inv @ diff)
    if not np.isfinite(quadratic) or quadratic < 0:
        return float("nan") if not np.isfinite(quadratic) else 0.0
    return float(np.sqrt(quadratic))


def cosine_to_centroid(x: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity between *x* and *centroid*."""
    norm_x = np.linalg.norm(x)
    norm_c = np.linalg.norm(centroid)
    if norm_x < 1e-8 or norm_c < 1e-8:
        return 0.0
    return float(np.dot(x, centroid) / (norm_x * norm_c))
