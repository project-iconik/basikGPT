"""Unit tests for GPT2Tokenizer wrapper and special token handling."""

import pytest
from basikgpt.data.tokenizer import GPT2Tokenizer


@pytest.fixture
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()


def test_tokenizer_invariants(tokenizer: GPT2Tokenizer) -> None:
    """Verifies canonical vocabulary size and EOT token ID."""
    assert tokenizer.vocab_size == 50257
    assert tokenizer.eot_token_id == 50256


def test_encode_document_ordinary_text(tokenizer: GPT2Tokenizer) -> None:
    """Verifies that normal text is encoded and terminated with a single EOT token."""
    text = "Hello world! This is basikGPT."
    tokens = tokenizer.encode_document(text)

    assert len(tokens) > 1
    assert tokens[-1] == 50256  # Ends with EOT
    assert 50256 not in tokens[:-1]  # No other EOT inside ordinary text

    # Decoding without the trailing EOT restores original text
    decoded = tokenizer.decode(tokens[:-1])
    assert decoded == text


def test_encode_document_literal_eot_string(tokenizer: GPT2Tokenizer) -> None:
    """Verifies that literal '<|endoftext|>' inside text is NOT parsed as special token ID 50256."""
    text = "Sentence one. <|endoftext|> Sentence two."
    tokens = tokenizer.encode_document(text)

    # The body must not contain 50256; only the final appended token is 50256
    body_tokens = tokens[:-1]
    assert 50256 not in body_tokens, "Literal '<|endoftext|>' was incorrectly encoded as token 50256!"
    assert tokens[-1] == 50256

    # Full decode of body preserves the exact literal substring
    assert tokenizer.decode(body_tokens) == text


def test_encode_document_empty_and_whitespace(tokenizer: GPT2Tokenizer) -> None:
    """Verifies that empty and whitespace-only documents produce no tokens (skipped)."""
    assert tokenizer.encode_document("") == []
    assert tokenizer.encode_document("   ") == []
    assert tokenizer.encode_document("\n\t\r \n") == []


def test_tokenizer_round_trip(tokenizer: GPT2Tokenizer) -> None:
    """Verifies lossless round-trip tokenization and decoding."""
    text = "Deep learning with PyTorch and Python 3.12 is educational & fun! 12345"
    tokens = tokenizer.encode_document(text)
    decoded = tokenizer.decode(tokens[:-1])
    assert decoded == text
