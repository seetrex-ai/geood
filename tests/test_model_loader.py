import pytest
from unittest.mock import MagicMock, patch
from geood.model_loader import resolve_model


def test_resolve_object_passthrough():
    mock_model = MagicMock()
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_tokenizer = MagicMock()
    model, tokenizer, should_cleanup = resolve_model(mock_model, mock_tokenizer)
    assert model is mock_model
    # Tokenizer is shallow-copied to avoid mutating caller's object
    assert tokenizer is not mock_tokenizer
    assert should_cleanup is False


def test_resolve_object_requires_tokenizer():
    mock_model = MagicMock()
    with pytest.raises(ValueError, match="tokenizer"):
        resolve_model(mock_model, None)


def test_resolve_string_type():
    with patch("geood.model_loader.AutoModelForCausalLM") as mock_auto:
        with patch("geood.model_loader.AutoTokenizer") as mock_tok:
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_tok.from_pretrained.return_value = MagicMock()
            model, tokenizer, should_cleanup = resolve_model("some/model", None)
            mock_auto.from_pretrained.assert_called_once()
            assert should_cleanup is True
