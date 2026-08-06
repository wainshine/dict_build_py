"""Frequency and entropy calculation from sorted n-gram files.

Streaming approach: the n-gram file is byte-sorted (LC_ALL=C), and the
TAB separator (0x09) sorts before any word character, so all lines for
the same word are adjacent. Each word's suffix counts are flushed as
soon as the word changes — memory is O(distinct suffixes per word),
independent of corpus size.
"""

import math
import os
import shutil
import subprocess
from typing import Iterator


# ---- Core entropy computation ----

def compute_entropy_from_sorted(
    lines: Iterator[str],
    min_freq: int = 0,
    min_entropy: float = -1.0,
) -> Iterator[tuple[str, int, float]]:
    """Process sorted n-gram lines, yield (word, freq, entropy).

    Input must be sorted so that all lines for a word are adjacent
    (guaranteed by LC_ALL=C byte sort of "word<TAB>suffix" lines).
    """
    current_word: str | None = None
    suffix_counts: dict[str, int] = {}
    total = 0

    for line in lines:
        tab_pos = line.find("\t")
        if tab_pos < 0:
            continue
        word = line[:tab_pos]
        suffix = line[tab_pos + 1:].rstrip("\n\r")
        if not word:
            continue

        if word != current_word:
            if current_word is not None:
                result = _score_word(current_word, suffix_counts, total,
                                     min_freq, min_entropy)
                if result is not None:
                    yield result
                suffix_counts.clear()
                total = 0
            current_word = word

        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        total += 1

    if current_word is not None:
        result = _score_word(current_word, suffix_counts, total,
                             min_freq, min_entropy)
        if result is not None:
            yield result


def compute_entropy_from_sorted_left(
    lines: Iterator[str],
    min_freq: int = 0,
    min_entropy: float = -1.0,
) -> Iterator[tuple[str, int, float]]:
    """Same as above but reverses words back (for left entropy).

    NOTE: Reversing breaks sort order.
    """
    for word, freq, entropy in compute_entropy_from_sorted(lines, min_freq, min_entropy):
        yield (word[::-1], freq, entropy)


def _score_word(
    word: str,
    suffix_counts: dict[str, int],
    total: int,
    min_freq: int,
    min_entropy: float,
) -> tuple[str, int, float] | None:
    if total < min_freq:
        return None
    entropy = 0.0
    for count in suffix_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    if entropy < min_entropy:
        return None
    return (word, total, entropy)


# ---- File I/O helpers ----

def write_entropy_to_file(
    iterator: Iterator[tuple[str, int, float]],
    filepath: str,
) -> int:
    """Write entropy tuples to a tab-separated file. Returns count."""
    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for word, freq, entropy in iterator:
            f.write(f"{word}\t{freq}\t{entropy:.6f}\n")
            count += 1
    return count


def read_entropy_from_file(filepath: str) -> Iterator[tuple[str, int, float]]:
    """Read entropy tuples from a tab-separated file.

    Lines with unparseable freq/entropy values (e.g. from an interrupted
    previous run) are skipped instead of aborting the whole merge.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            try:
                yield (parts[0], int(parts[1]), float(parts[2]))
            except ValueError:
                continue


def sort_file_inplace(filepath: str) -> None:
    """Sort a file in-place using system sort.

    Without a system sort, falls back to in-memory sorting for small
    files only; large files raise a clear error instead of OOM-ing.
    """
    if shutil.which("sort") is not None:
        subprocess.run(
            ["sort", "-o", filepath, filepath],
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    else:
        if os.path.getsize(filepath) > 256 * 1024 * 1024:
            raise RuntimeError(
                f"Cannot sort {filepath}: no system 'sort' available and "
                f"file exceeds 256 MB in-memory fallback limit. "
                f"Install GNU coreutils (or run on Linux/macOS)."
            )
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines.sort(key=lambda s: s.encode("utf-8"))
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)


# ---- Merge ----

def merge_entropy_files_sorted(
    right_file: str,
    left_file: str,
    min_entropy: float = -1.0,
) -> list[tuple[str, int, float]]:
    """Merge sorted right and left entropy files, taking min(entropy).

    Both files must be sorted by word. The merge compares Python str
    directly: this is correct because all entropy files are UTF-8 and
    UTF-8 byte order (used by LC_ALL=C sort) equals code-point order.
    Do not switch to locale-aware sorting without revisiting this.
    """
    r_iter = read_entropy_from_file(right_file)
    l_iter = read_entropy_from_file(left_file)
    result: list[tuple[str, int, float]] = []

    try:
        r = next(r_iter)
    except StopIteration:
        return result
    try:
        l = next(l_iter)
    except StopIteration:
        return result

    while True:
        rw, rf, re = r
        lw, lf, le = l

        if rw == lw:
            merged = min(re, le)
            if merged >= min_entropy:
                result.append((rw, rf, merged))
            try:
                r = next(r_iter)
            except StopIteration:
                break
            try:
                l = next(l_iter)
            except StopIteration:
                break
        elif rw < lw:
            try:
                r = next(r_iter)
            except StopIteration:
                break
        else:
            try:
                l = next(l_iter)
            except StopIteration:
                break
    return result
