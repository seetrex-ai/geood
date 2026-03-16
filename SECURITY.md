# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

- **Email**: security@seetrex.com
- **Expected response**: Within 72 hours
- **Please do NOT** open a public GitHub issue for security vulnerabilities

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Design

- **No pickle**: All serialization uses `numpy.npz` with `allow_pickle=False`
- **No code execution**: No `eval()`, `exec()`, `subprocess`, or `trust_remote_code`
- **Validated loading**: Array sizes, shapes, metadata types, and value ranges are validated on load
- **NaN fail-closed**: NaN scores are treated as OOD (never silently pass)
- **Finite-only calibration**: NaN/Inf vectors are rejected during calibration

## Loading Detectors from Untrusted Sources

Saved detector files (`.npz`) may contain a `model_ref` field — a HuggingFace
model name used to download a model when calling `detect()`. If you load a
detector from an untrusted source, always pass `model=` explicitly to
`detect()` to override any embedded `model_ref`:

```python
detector = geood.load("untrusted_detector.npz")
result = detector.detect("text", model="your-trusted-model", tokenizer=tok)
```
