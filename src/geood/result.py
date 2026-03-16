"""Detection result with human-readable explanation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of a single OOD detection call."""

    is_ood: bool
    score: float
    mahalanobis: float
    intrinsic_dim: int | None
    reference_dim: float
    layer: int

    def explain(self) -> str:
        """One-line human-readable explanation of the detection result."""
        if self.intrinsic_dim is not None:
            ratio = math.ceil(self.reference_dim / max(self.intrinsic_dim, 1) * 10) / 10
            dim_info = (
                f"dim={self.intrinsic_dim} vs ref={self.reference_dim:.1f} "
                f"({ratio:.1f}x collapse), "
            )
        else:
            dim_info = f"ref_dim={self.reference_dim:.1f}, "
        if self.is_ood:
            return f"OOD detected: {dim_info}mahalanobis={self.mahalanobis:.1f}"
        return f"In-distribution: {dim_info}mahalanobis={self.mahalanobis:.1f}"
