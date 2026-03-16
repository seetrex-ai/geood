"""Core detector: calibrate once, detect many times."""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from typing import Union

import numpy as np

from geood.metrics import intrinsic_dim, mahalanobis_distance
from geood.result import DetectionResult

_NPZ_EXT = ".npz"
_MAX_MODEL_REF_LEN = 200
# ASCII-only HuggingFace model identifiers (org/model or just model)
# Rejects: unicode homoglyphs, path traversal (./.. ), local paths
_HF_MODEL_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)?$")


class Detector:
    """Geometric OOD detector calibrated on in-distribution hidden states."""

    def __init__(self) -> None:
        self._centroid: np.ndarray | None = None  # in PCA space
        self._cov: np.ndarray | None = None       # in PCA space
        self._pca_mean: np.ndarray | None = None   # original space mean
        self._pca_components: np.ndarray | None = None  # projection matrix
        self._reference_dim: float | None = None
        self.layer_used: int | None = None
        self._threshold: float | None = None  # kept for compat
        self._max_cal_dist: float | None = None
        self._ood_threshold: float = 0.5  # score threshold for is_ood
        self._model_ref: str | None = None
        self._cached_model: object | None = None
        self._cached_tokenizer: object | None = None

    def __repr__(self) -> str:
        if self.layer_used is None:
            return "Detector(uncalibrated)"
        return (
            f"Detector(layer={self.layer_used}, "
            f"ref_dim={self._reference_dim:.1f}, "
            f"ood_threshold={self._ood_threshold})"
        )

    @property
    def is_calibrated(self) -> bool:
        """Whether the detector has been fully calibrated."""
        return all(x is not None for x in [
            self.layer_used, self._pca_mean, self._pca_components,
            self._centroid, self._cov, self._reference_dim,
            self._threshold, self._max_cal_dist,
        ])

    def __reduce__(self):
        raise TypeError(
            "Detector objects cannot be pickled for security reasons. "
            "Use detector.save(path) and geood.load(path) instead."
        )

    @classmethod
    def calibrate_from_vectors(
        cls,
        hidden_states: dict[int, list[np.ndarray]],
        layer_indices: list[int],
    ) -> Detector:
        """Calibrate from pre-extracted hidden state vectors.

        Useful for testing or when you have already extracted hidden
        states yourself.

        Args:
            hidden_states: ``{layer_index: [vector_per_sample, ...]}``.
            layer_indices: Which layers to consider for auto-selection.

        Returns:
            Calibrated detector.
        """
        detector = cls()

        if not hidden_states or not layer_indices:
            raise ValueError("hidden_states and layer_indices must be non-empty.")

        # Validate layer_indices exist in hidden_states
        missing_layers = set(layer_indices) - set(hidden_states.keys())
        if missing_layers:
            raise ValueError(
                f"layer_indices {missing_layers} not found in hidden_states. "
                f"Available: {set(hidden_states.keys())}."
            )

        # Validate sample count before any computation
        first_key = layer_indices[0]
        n_samples = len(hidden_states[first_key])
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 hidden state vectors, got {n_samples}."
            )

        # Auto-select: prefer deeper layers (more semantic), break ties by dim.
        # Skip layer 0 (embeddings) unless it's the only option.
        best_layer = None
        best_dim = -1

        candidates = layer_indices if len(layer_indices) == 1 else [
            i for i in layer_indices if i > 0
        ] or layer_indices

        for idx in candidates:
            vectors = np.array(hidden_states[idx])
            dim = intrinsic_dim(vectors)
            if dim > best_dim:
                best_dim = dim
                best_layer = idx

        detector.layer_used = best_layer
        ref_vectors = np.array(hidden_states[best_layer])
        detector._reference_dim = float(best_dim)

        # Reject NaN/Inf vectors before any computation
        if not np.all(np.isfinite(ref_vectors)):
            raise ValueError(
                "Reference vectors contain NaN or Inf values. "
                "Check model outputs and input texts."
            )

        # PCA projection to avoid singular covariance (n < d)
        mean = ref_vectors.mean(axis=0)
        centered = ref_vectors - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # Keep components explaining 95% variance
        total_var = np.sum(S ** 2)
        if total_var < 1e-12:
            raise ValueError(
                "All reference vectors are (near-)identical — cannot calibrate. "
                "Provide diverse reference texts."
            )
        var_explained = np.cumsum(S ** 2) / total_var
        n_components = int(np.searchsorted(var_explained, 0.95) + 1)
        detector._pca_mean = mean
        detector._pca_components = Vt[:n_components]  # (k, d)

        # Project into PCA subspace
        projected = centered @ detector._pca_components.T  # (n, k)
        detector._centroid = projected.mean(axis=0)
        cov = np.cov(projected, rowvar=False)
        # Ensure cov is always 2D (np.cov returns scalar for 1D input)
        detector._cov = np.atleast_2d(cov)

        maha_dists = np.array([
            mahalanobis_distance(v, detector._centroid, detector._cov)
            for v in projected
        ])
        # Store max calibration distance for score normalization
        max_cal_dist = float(np.max(maha_dists))
        if np.isnan(max_cal_dist) or np.isinf(max_cal_dist) or max_cal_dist <= 0:
            raise ValueError(
                "Calibration produced invalid distances (NaN/Inf/zero). "
                "Check reference texts for quality and diversity."
            )
        detector._max_cal_dist = max_cal_dist
        # Default threshold: score > 0.5 means OOD
        # score = maha / (max_cal_dist * 2), so threshold = max_cal_dist
        detector._threshold = max_cal_dist

        return detector

    # Keep old name for backwards compat
    _calibrate_from_hidden = calibrate_from_vectors

    def _validate_layer(self, layer: Union[str, int], n_layers: int) -> list[int]:
        """Validate and resolve the layer parameter."""
        from geood.extraction import get_candidate_layers

        if layer == "auto":
            return get_candidate_layers(n_layers)
        if isinstance(layer, bool):
            raise TypeError(
                f"layer must be 'auto' or an int, got bool ({layer}). "
                "Use an integer layer index."
            )
        if not isinstance(layer, int):
            raise TypeError(
                f"layer must be 'auto' or an int, got {type(layer).__name__}."
            )
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"layer={layer} is out of range for model with {n_layers} layers. "
                f"Valid range: 0..{n_layers - 1}."
            )
        return [layer]

    def calibrate(
        self,
        model: Union[str, object],
        ref_texts: list[str],
        tokenizer: object | None = None,
        layer: Union[str, int] = "auto",
        batch_size: int = 4,
        threshold: float = 0.5,
    ) -> Detector:
        """Calibrate the detector from reference texts.

        Runs a forward pass on *ref_texts*, extracts hidden states at
        candidate layers, and selects the layer with highest intrinsic
        dimensionality.  Computes centroid, covariance, and Mahalanobis
        reference statistics for later detection.

        The model is cached internally for subsequent ``detect()`` calls.

        Args:
            model: HuggingFace model name or loaded model object.
            ref_texts: In-distribution reference texts (recommended: 50+).
            tokenizer: Required when *model* is an object.
            layer: ``"auto"`` or an integer layer index.
            batch_size: Batch size for forward passes (default 4).

        Returns:
            self (for chaining).

        Raises:
            ValueError: If *ref_texts* has fewer than 2 items.
        """
        if not ref_texts or len(ref_texts) < 2:
            raise ValueError(
                f"ref_texts must contain at least 2 texts, got {len(ref_texts) if ref_texts else 0}. "
                "Recommended: 50+ for reliable calibration."
            )
        _MAX_REF_TEXTS = 50_000
        if len(ref_texts) > _MAX_REF_TEXTS:
            raise ValueError(
                f"ref_texts has {len(ref_texts)} items (max {_MAX_REF_TEXTS}). "
                "This likely indicates an error or could exhaust memory."
            )
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        from geood.model_loader import resolve_model
        from geood.extraction import extract_hidden_states, get_layer_count

        resolved_model, resolved_tokenizer, should_cleanup = resolve_model(
            model, tokenizer,
        )

        try:
            n_layers = get_layer_count(resolved_model)
            layer_indices = self._validate_layer(layer, n_layers)

            hidden = extract_hidden_states(
                resolved_model, resolved_tokenizer, ref_texts, layer_indices,
                batch_size=batch_size,
            )

            calibrated = self.calibrate_from_vectors(hidden, layer_indices)
        except Exception:
            # On failure, don't leave zombie state — keep previous calibration
            # intact (or uncalibrated if first call)
            if should_cleanup:
                del resolved_model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass  # don't mask the original error
            raise

        # Success — free old cached model before replacing (H2: GPU leak)
        old_model = self._cached_model
        if old_model is not None and old_model is not resolved_model:
            self._cached_model = None
            del old_model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Atomically update all calibration state
        self._centroid = calibrated._centroid
        self._cov = calibrated._cov
        self._pca_mean = calibrated._pca_mean
        self._pca_components = calibrated._pca_components
        self._reference_dim = calibrated._reference_dim
        self.layer_used = calibrated.layer_used
        self._threshold = calibrated._threshold
        self._max_cal_dist = calibrated._max_cal_dist
        self._ood_threshold = threshold
        # Sanitize model_ref: only store HF-style identifiers, not local paths
        self._model_ref = _sanitize_model_ref(
            model if isinstance(model, str) else None
        )
        self._cached_model = resolved_model
        self._cached_tokenizer = resolved_tokenizer

        return self

    def detect(
        self,
        input_text: Union[str, list[str]],
        model: Union[str, object, None] = None,
        tokenizer: object | None = None,
        batch_size: int = 4,
    ) -> Union[DetectionResult, list[DetectionResult]]:
        """Score one or more texts against the calibrated reference.

        If the detector was calibrated with a model, it reuses the cached
        model.  Pass *model* explicitly to override.

        Args:
            input_text: A single string or a list of strings.
            model: Override model (str or object).  If *None*, reuses
                the cached model from calibration.
            tokenizer: Required when *model* is an object.
            batch_size: Batch size for forward passes (default 4).

        Returns:
            A single :class:`DetectionResult` when *input_text* is a
            string, or a list of results when it is a list.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Detector is not calibrated. Call calibrate() or load() first."
            )
        if input_text is None:
            raise ValueError("input_text must be a string or list of strings, got None.")
        if isinstance(input_text, list) and not all(isinstance(t, str) for t in input_text):
            raise ValueError("All items in input_text must be strings.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        from geood.model_loader import resolve_model
        from geood.extraction import extract_hidden_states

        # Reuse cached model when possible
        if model is None and self._cached_model is not None:
            resolved_model = self._cached_model
            resolved_tokenizer = self._cached_tokenizer
            should_cleanup = False
        elif model is None:
            if self._model_ref is None:
                raise ValueError(
                    "No model available. Either pass model= to detect() "
                    "or calibrate with a model name string."
                )
            resolved_model, resolved_tokenizer, should_cleanup = resolve_model(
                self._model_ref, tokenizer,
            )
            # Cache for next call
            self._cached_model = resolved_model
            self._cached_tokenizer = resolved_tokenizer
        else:
            resolved_model, resolved_tokenizer, should_cleanup = resolve_model(
                model, tokenizer,
            )

        texts = [input_text] if isinstance(input_text, str) else input_text
        if not texts:
            return []

        # Validate layer_used against actual model
        from geood.extraction import get_layer_count, _get_transformer_layers
        try:
            n_layers = len(_get_transformer_layers(resolved_model))
        except ValueError:
            n_layers = None
        if n_layers is not None and self.layer_used >= n_layers:
            raise ValueError(
                f"Detector uses layer {self.layer_used}, but model only has "
                f"{n_layers} layers. Use the same model architecture."
            )

        hidden = extract_hidden_states(
            resolved_model, resolved_tokenizer, texts, [self.layer_used],
            batch_size=batch_size,
        )

        vectors = hidden[self.layer_used]

        # Cross-check hidden dim matches calibration (M3)
        expected_dim = self._pca_mean.shape[0]
        actual_dim = vectors[0].shape[0] if vectors else 0
        if actual_dim != expected_dim:
            raise ValueError(
                f"Hidden dim mismatch: model produces {actual_dim}, "
                f"but detector was calibrated with {expected_dim}. "
                "Use the same model for calibration and detection."
            )

        # Compute intrinsic dim for batch (>= 2 samples)
        batch_dim = None
        if len(vectors) >= 2:
            batch_dim = intrinsic_dim(np.array(vectors))

        results = [
            self._detect_from_vector(v, batch_dim) for v in vectors
        ]

        if should_cleanup:
            del resolved_model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results[0] if isinstance(input_text, str) else results

    def _detect_from_vector(
        self, vector: np.ndarray, batch_dim: int | None = None,
    ) -> DetectionResult:
        # Project into the same PCA subspace used during calibration
        projected = (vector - self._pca_mean) @ self._pca_components.T
        maha = mahalanobis_distance(projected, self._centroid, self._cov)

        # NaN fail-closed: treat NaN as OOD (never silently pass)
        if np.isnan(maha):
            return DetectionResult(
                is_ood=True, score=1.0, mahalanobis=float("nan"),
                intrinsic_dim=batch_dim, reference_dim=self._reference_dim,
                layer=self.layer_used,
            )

        # Score normalized so calibration texts ≈ 0-0.5, OOD > 0.5
        # max_cal_dist * 2 maps the max calibration distance to 0.5
        normalizer = max(self._max_cal_dist * 2, 1e-8)
        score = max(0.0, min(maha / normalizer, 1.0))
        is_ood = score > self._ood_threshold

        return DetectionResult(
            is_ood=is_ood,
            score=float(score),
            mahalanobis=float(maha),
            intrinsic_dim=batch_dim,
            reference_dim=self._reference_dim,
            layer=self.layer_used,
        )

    def save(self, path: str | os.PathLike) -> None:
        """Serialize the detector to disk.

        Uses numpy ``.npz`` for arrays and JSON-encoded metadata.
        No pickle — safe to load from untrusted sources.

        The file is always saved with a ``.npz`` extension.  If *path*
        does not end in ``.npz``, the extension is appended.
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Cannot save an uncalibrated detector. Call calibrate() first."
            )
        path = os.fspath(path)
        if not path.endswith(_NPZ_EXT):
            path = path + _NPZ_EXT
        meta_str = json.dumps({
            "reference_dim": self._reference_dim,
            "layer_used": self.layer_used,
            "threshold": self._threshold,
            "max_cal_dist": self._max_cal_dist,
            "ood_threshold": self._ood_threshold,
            "model_ref": self._model_ref,
            "hidden_dim": int(self._pca_mean.shape[0]),
        }, allow_nan=False)
        meta_arr = np.frombuffer(meta_str.encode("utf-8"), dtype=np.uint8)
        # Atomic write: write to temp file then rename to avoid corruption
        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(suffix=_NPZ_EXT, dir=dir_name)
        try:
            os.close(fd)
            np.savez_compressed(
                tmp_path,
                centroid=self._centroid,
                cov=self._cov,
                pca_mean=self._pca_mean,
                pca_components=self._pca_components,
                meta=meta_arr,
            )
            # On Windows, os.rename fails if target exists; use os.replace
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def release_model(self) -> None:
        """Release cached model and tokenizer to free memory."""
        self._cached_model = None
        self._cached_tokenizer = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load(cls, path: str | os.PathLike) -> Detector:
        """Load a detector from a saved ``.npz`` file.

        Safe to use with untrusted files — no pickle deserialization.
        Arrays are validated for reasonable sizes and shape consistency.

        .. warning::

            The saved file may contain a ``model_ref`` string used to
            download a model from HuggingFace Hub on ``detect()``.
            If loading from an untrusted source, pass ``model=`` explicitly
            to ``detect()`` to override.
        """
        path = os.fspath(path)
        if not path.endswith(_NPZ_EXT) and not os.path.exists(path):
            path = path + _NPZ_EXT
        data = np.load(path, allow_pickle=False)

        try:
            return cls._load_from_npz(data)
        finally:
            data.close()

    @classmethod
    def _load_from_npz(cls, data) -> Detector:
        """Internal: validate and load from an open NpzFile."""
        # Validate required keys exist
        required_keys = {"centroid", "cov", "pca_mean", "pca_components", "meta"}
        missing = required_keys - set(data.files)
        if missing:
            raise ValueError(f"Invalid detector file: missing keys {missing}")

        # Validate array sizes and dtypes before materializing (OOM protection)
        _MAX_DIM = 100_000
        _MAX_ELEMENTS = 10_000_000  # ~76 MB for float64
        _ALLOWED_DTYPES = {"float32", "float64", "uint8", "int64", "int32"}
        for key in ("centroid", "cov", "pca_mean", "pca_components"):
            arr = data[key]
            if arr.dtype.name not in _ALLOWED_DTYPES:
                raise ValueError(
                    f"Array '{key}' has dtype {arr.dtype} — only standard "
                    f"numeric types allowed."
                )
            shape = arr.shape
            if any(s > _MAX_DIM for s in shape):
                raise ValueError(
                    f"Array '{key}' has shape {shape} exceeding max "
                    f"dimension {_MAX_DIM} — file may be crafted."
                )
            n_elements = 1
            for s in shape:
                n_elements *= s
            if n_elements > _MAX_ELEMENTS:
                raise ValueError(
                    f"Array '{key}' has {n_elements} elements (max "
                    f"{_MAX_ELEMENTS}) — file may be crafted."
                )

        centroid = data["centroid"]
        cov = data["cov"]
        pca_mean = data["pca_mean"]
        pca_components = data["pca_components"]

        # Shape consistency
        k = centroid.shape[0]
        if cov.shape != (k, k):
            raise ValueError(
                f"Shape mismatch: centroid ({centroid.shape}) vs "
                f"cov ({cov.shape}). Expected cov ({k}, {k})."
            )
        if pca_components.shape[0] != k:
            raise ValueError(
                f"Shape mismatch: pca_components rows ({pca_components.shape[0]}) "
                f"!= centroid dim ({k})."
            )
        if pca_mean.shape[0] != pca_components.shape[1]:
            raise ValueError(
                f"Shape mismatch: pca_mean ({pca_mean.shape[0]}) != "
                f"pca_components cols ({pca_components.shape[1]})."
            )

        # Parse and validate metadata
        try:
            meta = json.loads(data["meta"].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid metadata in detector file: {exc}") from exc

        # Type validation on metadata
        reference_dim = meta.get("reference_dim")
        layer_used = meta.get("layer_used")
        threshold = meta.get("threshold")
        if not isinstance(reference_dim, (int, float)):
            raise ValueError(f"Invalid reference_dim type: {type(reference_dim)}")
        if not np.isfinite(reference_dim) or reference_dim <= 0:
            raise ValueError(
                f"Invalid reference_dim: {reference_dim} — must be finite and positive."
            )
        if not isinstance(layer_used, int):
            raise ValueError(f"Invalid layer_used type: {type(layer_used)}")
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Invalid threshold type: {type(threshold)}")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                f"Invalid threshold: {threshold} — must be finite and positive."
            )
        if layer_used < 0:
            raise ValueError(f"Invalid layer_used: {layer_used} — must be >= 0.")

        detector = cls()
        detector._centroid = centroid
        detector._cov = cov
        detector._pca_mean = pca_mean
        detector._pca_components = pca_components
        detector._reference_dim = float(reference_dim)
        detector.layer_used = layer_used
        detector._threshold = float(threshold)
        detector._max_cal_dist = float(meta.get("max_cal_dist", threshold))
        detector._ood_threshold = float(meta.get("ood_threshold", 0.5))
        model_ref = meta.get("model_ref")
        if model_ref is not None and not isinstance(model_ref, str):
            model_ref = None
        detector._model_ref = _sanitize_model_ref(model_ref)
        return detector


def _sanitize_model_ref(model_ref: str | None) -> str | None:
    """Strip local filesystem paths from model_ref, keep only HF identifiers."""
    if model_ref is None:
        return None
    if len(model_ref) > _MAX_MODEL_REF_LEN:
        return None
    # Reject path traversal components
    if any(part in (".", "..") for part in model_ref.split("/")):
        return None
    # Must match ASCII-only HF model ID pattern
    if _HF_MODEL_RE.match(model_ref):
        return model_ref
    return None
