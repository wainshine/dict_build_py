"""Text preprocessing: clean text, extract Chinese sentences."""

import regex as re
from typing import Iterator

from .config import SENTINEL, STOPWORDS

# Punctuation/symbols/whitespace/control chars and stopwords all become
# sentence boundaries in a single regex pass.
_PUNCT_PATTERN = re.compile(
    r"[\p{P}\p{S}\p{Z}\p{C}\p{M}　" + re.escape("".join(STOPWORDS)) + r"]"
)
_CHINESE_RE = re.compile(r"[一-龥]+")


def preprocess_line(line: str) -> Iterator[str]:
    """Process a single line of text, yield cleaned Chinese sentences."""
    line = _PUNCT_PATTERN.sub(" ", line)
    for match in _CHINESE_RE.finditer(line):
        token = match.group()
        if len(token) >= 2:
            yield SENTINEL + token + SENTINEL


def is_chinese(char: str) -> bool:
    """Check if a single character is in the Chinese Unicode range."""
    return 0x4E00 <= ord(char) <= 0x9FA5


def all_chinese(word: str) -> bool:
    """Check if all characters in the string are Chinese."""
    return all(is_chinese(c) for c in word)
