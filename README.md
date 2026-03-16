# GeoOD

[![PyPI](https://img.shields.io/pypi/v/geood)](https://pypi.org/project/geood/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/geood/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/seetrex-ai/geood/actions/workflows/ci.yml/badge.svg)](https://github.com/seetrex-ai/geood/actions)

**Geometric out-of-distribution detection for LLM hidden states.**

Perplexity misses OOD inputs when the model is fluent in the OOD domain. GeoOD detects them using hidden-state geometry (Mahalanobis distance + intrinsic dimensionality).

> **Paper:** [The Geometric Blind Spot of Perplexity: When Low Loss Hides Out-of-Distribution](https://zenodo.org/records/19039654)

## Why GeoOD?

Perplexity-based OOD detection has a blind spot: it cannot distinguish "the model knows this domain" from "this input belongs to the task distribution." Code snippets score *lower* perplexity than math problems on a math-trained model — but their hidden states collapse to 4 dimensions (vs 34 for math). GeoOD catches this.

| Method | Code AUROC (LLaMA-3-8B) | Code AUROC (Mistral-7B) |
|---|---|---|
| Perplexity | 0.352 | 0.150 |
| **GeoOD** | **1.000** | **1.000** |

## Installation

```bash
pip install geood
```

## Quick start

```python
import geood

# 1. Calibrate with in-distribution reference texts (50+ recommended)
detector = geood.calibrate("gpt2", ref_texts, threshold=0.5)

# 2. Detect OOD inputs — score is the primary signal
result = detector.detect("def quicksort(arr): ...")

print(result.score)      # 0.0-1.0 (higher = more OOD)
print(result.is_ood)     # True if score > threshold
print(result.explain())  # "OOD detected: ref_dim=33.8, mahalanobis=12.3"

# 3. Save for deployment (no pickle — safe to share)
detector.save("my_detector")
detector = geood.load("my_detector")
```

Works with any HuggingFace causal LM. Detection is strongest with 7B+ models; GPT-2 works for testing but produces weaker separation.

**Important:** `score` is the primary output. Calibration texts score ~0-0.5; OOD texts score higher. The `is_ood` flag uses a configurable `threshold` (default 0.5) — adjust it based on your precision/recall requirements.

## Minimal example (runs on CPU)

```python
import geood

# Calibrate on English sentences
ref_texts = [
    "The weather today is sunny and warm.",
    "She walked to the store to buy groceries.",
    "The meeting was scheduled for three o'clock.",
    # ... 50+ reference texts recommended for reliable calibration
]

detector = geood.calibrate("gpt2", ref_texts)

# Test: in-distribution (low score)
result = detector.detect("He drove to work in the morning.")
print(result.score)  # ~0.5 (in-distribution range)

# Test: out-of-distribution (higher score)
result = detector.detect("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)")
print(result.score)  # ~0.7+ (OOD range)
print(result.explain())

# Note: GPT-2 produces moderate separation. Use 7B+ models for strong OOD detection.
```

## API

### `geood.calibrate(model, ref_texts, tokenizer=None, layer="auto", threshold=0.5)`

Calibrate a detector from reference in-distribution texts.

- **model** — HuggingFace model name (`str`) or a loaded `PreTrainedModel`
- **ref_texts** — list of in-distribution reference strings (50+ recommended)
- **tokenizer** — required when `model` is an object
- **layer** — `"auto"` selects the best layer automatically, or pass an `int`
- **threshold** — score threshold for `is_ood` (default 0.5). Lower = more sensitive.

Returns a `Detector`. The model is cached internally for subsequent `detect()` calls.

### `detector.detect(input_text)`

Score one or more texts against the calibrated reference.

- Pass a single `str` to get a single `DetectionResult`
- Pass a `list[str]` to get a `list[DetectionResult]` (batch detection also computes `intrinsic_dim`)

### `DetectionResult`

| Field | Type | Description |
|---|---|---|
| `is_ood` | `bool` | Whether the input is out-of-distribution |
| `score` | `float` | OOD score (0 = in-distribution, 1 = OOD) |
| `mahalanobis` | `float` | Raw Mahalanobis distance |
| `intrinsic_dim` | `int \| None` | Intrinsic dimensionality (available in batch mode) |
| `reference_dim` | `float` | Reference corpus dimensionality |
| `layer` | `int` | Transformer layer used |
| `explain()` | `str` | Human-readable explanation |

### `detector.save(path)` / `geood.load(path)`

Serialize and load detectors. Files are saved as `.npz` (numpy compressed). No pickle — safe to load from untrusted sources.

### `Detector.calibrate_from_vectors(hidden_states, layer_indices)`

Calibrate directly from pre-extracted hidden state vectors. Useful for testing or custom extraction pipelines.

## Use cases

### Data curation

Filter OOD samples before they enter your training dataset:

```python
detector = geood.calibrate(model, clean_samples, tokenizer=tokenizer)

for text in new_data:
    result = detector.detect(text, model=model, tokenizer=tokenizer)
    if not result.is_ood:
        dataset.append(text)
```

### Safety filtering

Reject inputs outside your model's intended domain:

```python
detector = geood.calibrate(model, domain_examples, tokenizer=tokenizer, threshold=0.35)

result = detector.detect(user_input, model=model, tokenizer=tokenizer)
if result.is_ood:
    return "This question is outside my area of expertise."
```

### Deployment monitoring

Log and alert on distribution shift in production:

```python
result = detector.detect(request.text, model=model, tokenizer=tokenizer)
metrics.record("ood_score", result.score)
if result.is_ood:
    logger.warning(f"OOD input detected: {result.explain()}")
```

## How it works

1. **Calibrate:** Run a forward pass on reference texts. Extract hidden states at candidate layers. Compute centroid, covariance, and intrinsic dimensionality. Auto-select the layer with highest representational capacity.

2. **Detect:** Run a forward pass on new input. Compute Mahalanobis distance to the calibrated reference. OOD inputs produce geometrically distinct representations — even when the model assigns them low perplexity.

## Supported models

Any HuggingFace causal LM with a standard transformer architecture:

- GPT-2, GPT-Neo, GPT-J
- LLaMA, Llama-2, Llama-3
- Mistral, Mixtral
- Qwen, Gemma, Phi

## Development

```bash
git clone https://github.com/seetrex-ai/geood
cd geood
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@article{tabares2026geood,
  title={The Geometric Blind Spot of Perplexity: When Low Loss Hides Out-of-Distribution},
  author={Tabares Montilla, Jesus},
  year={2026},
  doi={10.5281/zenodo.19039654}
}
```

## License

[MIT](LICENSE)
