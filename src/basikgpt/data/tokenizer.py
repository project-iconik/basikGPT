"""GPT-2 BPE Tokenizer Wrapper using tiktoken.

Encodes raw documents using ordinary BPE tokenization and appends the canonical
<|endoftext|> (EOT, ID: 50256) token as a strict document boundary separator.
"""

from collections.abc import Sequence
import tiktoken


class GPT2Tokenizer:
    """Canonical GPT-2 BPE Tokenizer wrapper around tiktoken.

    Encodes text into discrete token IDs in the range [0, 50256] using the standard
    GPT-2 Byte-Pair Encoding (BPE) vocabulary.
    """

    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of GPT-2 (50,257)."""
        return self.encoding.n_vocab

    @property
    def eot_token_id(self) -> int:
        """End-Of-Text token ID (50,256) serving as the canonical document boundary separator."""
        return self.encoding.eot_token

    def encode_document(self, text: str) -> list[int]:
        """Encodes a single document string into a sequence of token IDs ending with EOT.

        Special Token Handling:
            Uses `encode_ordinary(text)` to ensure literal '<|endoftext|>' substrings
            inside document text are treated as regular characters rather than special tokens.
            A single EOT token (50,256) is explicitly appended at the end of valid documents.

        Args:
            text: Raw document text string.

        Returns:
            List of integer token IDs ending with 50,256, or empty list if text is empty/whitespace.
        """
        if not text or not text.strip():
            return []

        # Encode body without interpreting special tokens
        tokens = self.encoding.encode_ordinary(text)

        # Append single document boundary separator
        tokens.append(self.eot_token_id)
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decodes a sequence of integer token IDs back into text."""
        return self.encoding.decode(token_ids)
