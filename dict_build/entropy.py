"""Frequency and entropy calculation from sorted n-gram files.

Memory-efficient approach: split sorted n-gram file by first character,
process each part independently, then merge results.
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

    Groups by first character for memory efficiency.
    """
    group: dict[str, dict[str, int]] = {}
    current_first_char = ""

    for line in lines:
        tab_pos = line.find("\t")
        if tab_pos < 0:
            continue
        word = line[:tab_pos]
        suffix = line[tab_pos + 1:].rstrip("\n\r")
        if not word:
            continue

        first_char = word[0]
        if first_char != current_first_char:
            if group:
                yield from _flush_group(group, min_freq, min_entropy)
                group.clear()
            current_first_char = first_char

        word_map = group.get(word)
        if word_map is None:
            group[word] = {suffix: 1}
        else:
            word_map[suffix] = word_map.get(suffix, 0) + 1

    if group:
        yield from _flush_group(group, min_freq, min_entropy)


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


def _flush_group(
    group: dict[str, dict[str, int]],
    min_freq: int,
    min_entropy: float,
) -> Iterator[tuple[str, int, float]]:
    for word, suffix_counts in group.items():
        total = sum(suffix_counts.values())
        if total < min_freq:
            continue
        entropy = 0.0
        for count in suffix_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        if entropy >= min_entropy:
            yield (word, total, entropy)


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
    """Read entropy tuples from a tab-separated file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            parts = line.split("\t")
            if len(parts) == 3:
                yield (parts[0], int(parts[1]), float(parts[2]))


def sort_file_inplace(filepath: str) -> None:
    """Sort a file in-place using system sort."""
    if shutil.which("sort") is not None:
        subprocess.run(
            ["sort", "-o", filepath, filepath],
            check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
    else:
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

    Both files must be sorted by word.
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
