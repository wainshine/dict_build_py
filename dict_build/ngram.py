"""N-gram generation from preprocessed Chinese sentences.

For each sentence (padded with '$' sentinels):
- Forward: generate all substrings of length 1..max_len -> right entropy
- Backward: strip sentinels, reverse content, re-add sentinels, generate -> left entropy

Format: word<TAB>next_char (suffix context for entropy)
"""

from typing import Iterator

from .config import SENTINEL


def generate_ngrams(sentence: str, max_len: int = 6) -> Iterator[str]:
    """Generate n-grams from a sentinel-padded sentence.

    For each position i in [1, len-2]:
        For each length j in [1, min(max_len, len-i-1)]:
            Yield: sentence[i:i+j]\tsentence[i+j]

    Where sentence[i+j] is the next character following the n-gram.
    """
    n = len(sentence)
    for i in range(1, n - 1):
        max_j = min(max_len, n - i - 1)
        for j in range(1, max_j + 1):
            word = sentence[i:i + j]
            next_char = sentence[i + j]
            yield word + "\t" + next_char


def generate_ngrams_direct(sentence: str, max_len: int = 6) -> Iterator[str]:
    """Generate n-grams as words only (no suffix info)."""
    n = len(sentence)
    for i in range(1, n - 1):
        max_j = min(max_len, n - i - 1)
        for j in range(1, max_j + 1):
            yield sentence[i:i + j]


def generate_reverse_ngrams(sentence: str, max_len: int = 6) -> Iterator[str]:
    """Generate n-grams from reversed sentence content (for left entropy).

    sentence is sentinel-padded: $word$. We strip sentinels, reverse the
    content, re-add sentinels, then generate n-grams.

    The resulting n-gram words will be reversed back later (in entropy.py).
    """
    inner = sentence[1:-1]
    reversed_s = SENTINEL + inner[::-1] + SENTINEL
    yield from generate_ngrams(reversed_s, max_len)
