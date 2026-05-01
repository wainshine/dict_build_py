"""Pipeline orchestrator for the full word extraction process."""

import os
import tempfile
from datetime import date
from multiprocessing import Pool
from typing import Iterator

import tqdm

from .config import (
    DEFAULT_MAX_LEN, DEFAULT_MEM_MB, WORKERS, MIN_FREQ,
    CHUNK_LINES, WRITE_BATCH, OUTPUT_FILE_SUFFIX,
    ENTROPY_THRESHOLD, PMI_THRESHOLD, POS_PROB_THRESHOLD,
)
from .preprocess import preprocess_line
from .ngram import generate_ngrams, generate_reverse_ngrams
from .entropy import (
    compute_entropy_from_sorted,
    compute_entropy_from_sorted_left,
    write_entropy_to_file,
    sort_file_inplace,
    merge_entropy_files_sorted,
)
from .pmi import build_and_mmap_trie, extract_words
from .pos_tag import tag_word
from .pos_prob import load_pos_prob

TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".sql", ".md", ".html", ".htm"}

SINGLE_LINE_CHAR_CHUNK = 200_000
BYTE_BUF = 8 * 1024 * 1024


def _collect_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        result = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in TEXT_EXTENSIONS:
                    result.append(os.path.join(root, f))
        return sorted(result)
    raise FileNotFoundError(f"Path not found: {path}")


def _make_output_path(input_path: str) -> str:
    today = date.today().strftime("%Y%m%d")
    if os.path.isfile(input_path):
        stem = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(os.path.dirname(input_path) or ".",
                            f"{stem}_{today}_{OUTPUT_FILE_SUFFIX}")
    else:
        name = os.path.basename(input_path.rstrip("/"))
        return os.path.join(input_path, f"{name}_{today}_{OUTPUT_FILE_SUFFIX}")


def _is_single_line_file(filepath: str) -> bool:
    """Check if a file is a huge single-line file."""
    size = os.path.getsize(filepath)
    if size < 100 * 1024 * 1024:
        return False
    with open(filepath, "rb") as f:
        count = 0
        while True:
            data = f.read(4 * 1024 * 1024)
            if not data:
                break
            count += data.count(b"\n")
            if count > 10:
                return False
    return count <= 10


def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    mem_mb: int = DEFAULT_MEM_MB,
    workers: int = WORKERS,
    min_freq: int = MIN_FREQ,
    pos_prob_path: str | None = None,
) -> str:
    txt_files = _collect_files(input_path)
    print(f"Found {len(txt_files)} text file(s) to process")

    if output_path is None:
        output_path = _make_output_path(input_path)

    temp_dir = tempfile.mkdtemp(prefix="dict_build_")
    ngram_fw_path = os.path.join(temp_dir, "ngram_forward.txt")
    ngram_bw_path = os.path.join(temp_dir, "ngram_backward.txt")

    try:
        print(f"Stage 1-2: Preprocessing + N-gram generation (workers={workers})...")

        huge_files = [f for f in txt_files if _is_single_line_file(f)]
        normal_files = [f for f in txt_files if f not in huge_files]

        if normal_files:
            _generate_ngrams_parallel(
                normal_files, ngram_fw_path, ngram_bw_path, max_len, workers
            )
        for hf in huge_files:
            print(f"  Single-line mode: {os.path.basename(hf)} "
                  f"({os.path.getsize(hf)/1e9:.1f} GB)")
            _generate_ngrams_single_line(
                hf, ngram_fw_path, ngram_bw_path, max_len
            )

        fw_size = os.path.getsize(ngram_fw_path)
        bw_size = os.path.getsize(ngram_bw_path)
        print(f"  Forward n-grams: {fw_size / 1e9:.2f} GB")
        print(f"  Backward n-grams: {bw_size / 1e9:.2f} GB")

        ngram_fw_sorted = os.path.join(temp_dir, "ngram_forward_sorted.txt")
        ngram_bw_sorted = os.path.join(temp_dir, "ngram_backward_sorted.txt")

        print("Stage 3: Sorting n-grams (parallel system sort)...")
        import subprocess as _sp
        _p1 = _sp.Popen([
            "sort", "-S", f"{mem_mb}M", f"--parallel={workers}",
            "-o", ngram_fw_sorted, ngram_fw_path,
        ])
        _p2 = _sp.Popen([
            "sort", "-S", f"{mem_mb}M", f"--parallel={workers}",
            "-o", ngram_bw_sorted, ngram_bw_path,
        ])
        _p1.wait()
        _p2.wait()
        if _p1.returncode != 0 or _p2.returncode != 0:
            raise RuntimeError("Sort failed")
        for _p in (ngram_fw_path, ngram_bw_path):
            try: os.remove(_p)
            except OSError: pass
        print("  Sorting complete")

        right_entropy_file = os.path.join(temp_dir, "right_entropy.txt")
        left_entropy_unsorted = os.path.join(temp_dir, "left_entropy_unsorted.txt")
        left_entropy_file = os.path.join(temp_dir, "left_entropy.txt")

        print(f"Stage 4a: Computing right entropy (min_freq={min_freq})...")
        count = _write_entropy_from_ngram(ngram_fw_sorted, right_entropy_file, min_freq, True)
        print(f"  Right: {count} unique words")

        print(f"Stage 4b: Computing left entropy (min_freq={min_freq})...")
        count = _write_entropy_from_ngram(ngram_bw_sorted, left_entropy_unsorted, min_freq, False)
        print(f"  Left: {count} unique words")

        print("Stage 4c: Sorting left entropy file...")
        sort_file_inplace(left_entropy_unsorted)
        os.rename(left_entropy_unsorted, left_entropy_file)

        print("Stage 5: Merging left and right entropy...")
        merged = merge_entropy_files_sorted(
            right_entropy_file, left_entropy_file,
            min_entropy=ENTROPY_THRESHOLD,
        )
        print(f"  Merged: {len(merged)} candidates (entropy >= {ENTROPY_THRESHOLD})")

        if not merged:
            print("  No candidates pass entropy threshold. Writing empty output.")
            with open(output_path, "w") as f: f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
            return output_path

        trie_file = os.path.join(temp_dir, "freq.trie")
        print("Stage 6a: Building frequency trie (stream from file)...")
        trie, total_single = build_and_mmap_trie(right_entropy_file, trie_file, min_freq=min_freq)
        print(f"  Trie ready. Total single-char freq: {total_single}")

        print("Stage 6b: Computing PMI and filtering...")
        pos_prob = load_pos_prob(pos_prob_path)
        results = extract_words(merged, pos_prob, trie, total_single)
        print(f"  Final words: {len(results)}")

        # Attach POS tags
        for i, (word, freq, pmi, entropy, pp) in enumerate(results):
            pos = tag_word(word)
            results[i] = (word, freq, pmi, entropy, pp, pos)

        _write_output(results, output_path)
        print(f"Output: {output_path}")
        return output_path

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---- N-gram generation: normal files (parallel) ----

def _generate_ngrams_parallel(
    txt_files: list[str],
    ngram_fw_path: str,
    ngram_bw_path: str,
    max_len: int,
    workers: int,
) -> None:
    with open(ngram_fw_path, "a", encoding="utf-8") as fw_out, \
         open(ngram_bw_path, "a", encoding="utf-8") as bw_out:

        fw_batch: list[str] = []
        bw_batch: list[str] = []

        def flush():
            nonlocal fw_batch, bw_batch
            if fw_batch:
                fw_out.writelines(fw_batch)
                fw_batch.clear()
            if bw_batch:
                bw_out.writelines(bw_batch)
                bw_batch.clear()

        total_chunks = 0
        for f in txt_files:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                lc = sum(1 for _ in fh)
            total_chunks += (lc + CHUNK_LINES - 1) // CHUNK_LINES
        if total_chunks == 0:
            total_chunks = len(txt_files)

        with Pool(processes=workers) as pool:
            pbar = tqdm.tqdm(total=total_chunks, desc="  N-grams", unit="chunk")
            for txt_file in txt_files:
                for chunk in _read_chunks_by_lines(txt_file, CHUNK_LINES):
                    fut = pool.apply_async(_process_chunk, (chunk, max_len))
                    try:
                        fw, bw = fut.get()
                    except Exception:
                        pool.terminate()
                        raise
                    fw_batch.extend(fw)
                    bw_batch.extend(bw)
                    if len(fw_batch) >= WRITE_BATCH:
                        flush()
                    pbar.update(1)
            flush()
            pbar.close()
            pool.close()
            pool.join()


def _read_chunks_by_lines(path: str, n: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= n:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _process_chunk(lines: list[str], max_len: int) -> tuple[list[str], list[str]]:
    fw: list[str] = []
    bw: list[str] = []
    for line in lines:
        for sent in preprocess_line(line):
            for ng in generate_ngrams(sent, max_len):
                fw.append(ng + "\n")
            for ng in generate_reverse_ngrams(sent, max_len):
                bw.append(ng + "\n")
    return fw, bw


# ---- N-gram generation: huge single-line files (sequential, no serialization) ----

def _generate_ngrams_single_line(
    filepath: str,
    ngram_fw_path: str,
    ngram_bw_path: str,
    max_len: int,
) -> None:
    """Process a huge single-line file by:
    1. Reading in 8MB binary blocks
    2. Slicing decoded text into ~200K char segments
    3. Preprocessing + n-gram generation inline (no multiprocessing)
    4. Writing n-grams directly to output files
    """
    fw_out = open(ngram_fw_path, "a", encoding="utf-8")
    bw_out = open(ngram_bw_path, "a", encoding="utf-8")

    total_chars = os.path.getsize(filepath) // 3
    total_chunks = (total_chars + SINGLE_LINE_CHAR_CHUNK - 1) // SINGLE_LINE_CHAR_CHUNK
    pbar = tqdm.tqdm(total=total_chunks, desc="  S-line", unit="chunk")

    carryover = b""
    with open(filepath, "rb") as fin:
        while True:
            data = fin.read(BYTE_BUF)
            if not data:
                if carryover:
                    _process_text_segment(
                        carryover.decode("utf-8", errors="replace"),
                        max_len, fw_out, bw_out
                    )
                break

            segment = carryover + data
            try:
                text = segment.decode("utf-8")
                carryover = b""
            except UnicodeDecodeError:
                for cut in range(-1, -5, -1):
                    try:
                        text = segment[:cut].decode("utf-8")
                        carryover = segment[cut:]
                        break
                    except UnicodeDecodeError:
                        continue

            for i in range(0, len(text), SINGLE_LINE_CHAR_CHUNK):
                sub = text[i:i + SINGLE_LINE_CHAR_CHUNK]
                _process_text_segment(sub, max_len, fw_out, bw_out)
                pbar.update(1)

    pbar.close()
    fw_out.close()
    bw_out.close()


def _process_text_segment(
    text: str,
    max_len: int,
    fw_out,
    bw_out,
) -> None:
    """Preprocess a text segment and write n-grams directly to files."""
    fw_lines: list[str] = []
    bw_lines: list[str] = []

    for sent in preprocess_line(text):
        for ng in generate_ngrams(sent, max_len):
            fw_lines.append(ng + "\n")
        for ng in generate_reverse_ngrams(sent, max_len):
            bw_lines.append(ng + "\n")

    if fw_lines:
        fw_out.writelines(fw_lines)
    if bw_lines:
        bw_out.writelines(bw_lines)


# ---- Entropy helpers ----

def _write_entropy_from_ngram(ng: str, out: str, mf: int, d: bool) -> int:
    with open(ng, "r", encoding="utf-8", buffering=16 * 1024 * 1024) as fin:
        gen = compute_entropy_from_sorted(fin, min_freq=mf) if d \
             else compute_entropy_from_sorted_left(fin, min_freq=mf)
        return write_entropy_to_file(gen, out)


# ---- Output ----

def _write_output(res: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
        for row in res:
            w, fr, p, e, pp, pos = row
            f.write(f"{w}\t{fr}\t{p:.6f}\t{e:.6f}\t{pp:.6f}\t{pos}\n")
