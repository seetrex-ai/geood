"""Polymorphic model resolution: HuggingFace name or pre-loaded object."""

from __future__ import annotations

import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model(
    model: str | object, tokenizer: object | None = None,
) -> tuple[object, object, bool]:
    """Resolve *model* to a ``(model, tokenizer, should_cleanup)`` triple.

    When *model* is a string the model is loaded from HuggingFace Hub.
    When it is an object it is used directly (requires *tokenizer*).

    User-provided tokenizers are deep-copied to avoid mutating the
    caller's object (pad_token, padding_side).

    Returns:
        A tuple of ``(model, tokenizer, should_cleanup)`` where
        *should_cleanup* indicates the caller should free GPU memory
        after use.
    """
    if isinstance(model, str):
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype=torch.float16,
        )
        loaded_tokenizer = AutoTokenizer.from_pretrained(model)
        if loaded_tokenizer.pad_token is None:
            loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
        loaded_tokenizer.padding_side = "left"
        loaded_model.eval()
        return loaded_model, loaded_tokenizer, True
    if tokenizer is None:
        raise ValueError(
            "When passing a model object, you must also provide a tokenizer. "
            "Use: geood.calibrate(model, ref_texts, tokenizer=tokenizer)"
        )
    # Deep copy to avoid mutating caller's tokenizer (pad_token, padding_side)
    try:
        tok = copy.deepcopy(tokenizer)
    except Exception:
        # Fallback for tokenizers that can't be deepcopied
        tok = copy.copy(tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model.eval()
    return model, tok, False
