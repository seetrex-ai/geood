"""Hidden state extraction from transformer models via forward hooks."""

from __future__ import annotations

import threading
import warnings

import torch
import numpy as np

__all__ = ["get_layer_count", "get_candidate_layers", "extract_hidden_states"]

# Guard against concurrent hook registration on the same model
_extraction_lock = threading.RLock()


def get_layer_count(model: object) -> int:
    """Return the number of transformer layers in *model*.

    Supports LLaMA/Mistral (``model.model.layers``) and GPT-style
    (``model.transformer.h``) architectures.

    Raises:
        ValueError: If the layer structure is not recognised.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise ValueError(
        f"Cannot determine layer count for {type(model).__name__}. "
        "Please specify layer= manually."
    )


def get_candidate_layers(n_layers: int) -> list[int]:
    """Return five evenly spaced layer indices for auto-selection."""
    if n_layers <= 5:
        return list(range(n_layers))
    return [
        0,
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]


def _get_transformer_layers(model: object):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def _get_device(model: object) -> torch.device:
    """Get model device, handling empty parameter edge case (L4)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def extract_hidden_states(
    model: object,
    tokenizer: object,
    texts: list[str],
    layer_indices: list[int],
    batch_size: int = 4,
) -> dict[int, list[np.ndarray]]:
    """Extract mean-pooled hidden states at the requested layers.

    Registers temporary forward hooks, runs batched inference, and
    returns one numpy vector per (layer, sample) pair.

    Thread-safe: uses a lock to prevent concurrent hook registration
    on the same model.
    """
    device = _get_device(model)
    transformer_layers = _get_transformer_layers(model)
    all_hidden: dict[int, list[np.ndarray]] = {idx: [] for idx in layer_indices}
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # M6: validate hook output
            if output is None:
                return
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if tensor is not None:
                captured[layer_idx] = tensor.detach()
        return hook_fn

    # M8: thread-safe hook registration; M4: register inside try
    with _extraction_lock:
        try:
            for idx in layer_indices:
                h = transformer_layers[idx].register_forward_hook(make_hook(idx))
                hooks.append(h)

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(device)
                with torch.no_grad():
                    model(**inputs)
                for idx in layer_indices:
                    if idx not in captured:
                        warnings.warn(
                            f"Layer {idx} hook did not capture output. "
                            "The model may use a non-standard architecture.",
                            stacklevel=2,
                        )
                        continue
                    h_tensor = captured[idx]
                    for j in range(len(batch)):
                        mask = inputs["attention_mask"][j].bool()
                        # M5: warn on all-zero attention mask
                        if not mask.any():
                            warnings.warn(
                                f"Sample {i + j} has an empty attention mask. "
                                "This produces NaN vectors (treated as OOD).",
                                stacklevel=2,
                            )
                        pooled = h_tensor[j][mask].float().mean(dim=0).cpu().numpy()
                        all_hidden[idx].append(pooled)
                captured.clear()
        finally:
            for h in hooks:
                h.remove()

    return all_hidden
