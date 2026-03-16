from __future__ import annotations

import os
from typing import Union

from geood.result import DetectionResult
from geood.detector import Detector

from importlib.metadata import version as _get_version

__version__ = _get_version("geood")


def calibrate(
    model: Union[str, object],
    ref_texts: list[str],
    tokenizer: object | None = None,
    layer: Union[str, int] = "auto",
    batch_size: int = 4,
    threshold: float = 0.5,
) -> Detector:
    """Calibrate an OOD detector from reference in-distribution texts.

    Args:
        model: HuggingFace model name (str) or loaded PreTrainedModel.
        ref_texts: List of in-distribution reference texts.
        tokenizer: Required when model is an object.
        layer: ``"auto"`` to auto-select best layer, or int for manual override.
        batch_size: Batch size for forward passes (default 4).
        threshold: Score threshold for ``is_ood`` (default 0.5).
            Scores are normalized so calibration texts fall in 0-0.5.
            Adjust based on your precision/recall requirements.

    Returns:
        Calibrated detector ready for ``detect()`` calls.
    """
    detector = Detector()
    detector.calibrate(
        model, ref_texts, tokenizer=tokenizer, layer=layer,
        batch_size=batch_size, threshold=threshold,
    )
    return detector


def load(path: "str | os.PathLike") -> Detector:
    """Load a previously saved detector.

    Args:
        path: Path to a saved detector (created by ``Detector.save()``).

    Returns:
        Loaded detector ready for ``detect()`` calls.
    """
    return Detector.load(path)


__all__ = ["calibrate", "load", "DetectionResult", "Detector"]
