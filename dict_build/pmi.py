"""PMI (Pointwise Mutual Information) computation and word filtering.

Uses marisa-trie with mmap for memory-efficient O(k) word lookup.
Trie is built from file, saved to disk, then loaded via mmap.
"""

import math

import marisa_trie
import tqdm

from .preprocess import is_chinese
from .entropy import read_entropy_from_file


def build_and_mmap_trie(
    entropy_file: str,
    trie_file: str,
    min_freq: int = 10,
) -> tuple["marisa_trie.RecordTrie", int]:
    """Build a marisa-trie from an entropy file, save to disk, load via mmap.

    Args:
        entropy_file: Tab-separated file of (word, freq, entropy).
        trie_file: Path to save the compiled trie.
        min_freq: Minimum frequency to include.

    Returns:
        (mmap-loaded trie, total_single_char_freq).
    """
    pairs: list[tuple[str, tuple[int, ...]]] = []
    total_single = 0

    for word, freq, _entropy in read_entropy_from_file(entropy_file):
        if not word or freq < min_freq:
            continue
        if not all(is_chinese(c) for c in word):
            continue
        pairs.append((word, (freq,)))
        if len(word) == 1:
            total_single += freq

    trie = marisa_trie.RecordTrie("I", pairs)
    trie.save(trie_file)

    del pairs
    del trie

    trie_mmap = marisa_trie.RecordTrie("I").mmap(trie_file)
    return trie_mmap, total_single


def compute_pmi(
    word: str,
    freq: int,
    trie: marisa_trie.RecordTrie,
    total_single: int,
) -> float:
    """Compute PMI for a single word."""
    n = len(word)
    if n <= 1:
        return 0.0

    max_prod = 0
    for s in range(1, n):
        left = word[:s]
        right = word[s:]
        try:
            left_freq = trie[left][0][0]
            right_freq = trie[right][0][0]
            prod = left_freq * right_freq
            if prod > max_prod:
                max_prod = prod
        except KeyError:
            continue

    if max_prod == 0:
        return 0.0

    pf = freq * total_single / max_prod
    if pf <= 0:
        return 0.0
    return math.log2(pf)


def extract_words(
    merged_data: list[tuple[str, int, float]],
    pos_prob: dict[str, tuple[float, float, float]],
    trie: marisa_trie.RecordTrie,
    total_single: int,
    pmi_threshold: float,
    entropy_threshold: float,
    pos_threshold: float,
) -> list[tuple[str, int, float, float, float]]:
    """Filter and score candidate words.

    Returns list of (word, freq, pmi, entropy, pos_prob) sorted by freq desc.
    """
    results: list[tuple[str, int, float, float, float]] = []

    for word, freq, entropy in tqdm.tqdm(
        merged_data, desc="  Computing PMI", unit="words"
    ):
        if len(word) < 2:
            continue

        pmi = compute_pmi(word, freq, trie, total_single)
        if pmi < pmi_threshold:
            continue

        first_char = word[0]
        last_char = word[-1]
        pp = 0.0
        if first_char in pos_prob and last_char in pos_prob:
            p_first_s = pos_prob[first_char][0]
            p_last_e = pos_prob[last_char][2]
            pp = min(p_first_s, p_last_e)

        if pp < pos_threshold:
            continue

        results.append((word, freq, pmi, entropy, pp))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

