"""Pipeline orchestrator for the full word extraction process.

Handles:
- Normal multi-line files -> multiprocessing
- Huge single-line files -> sequential direct write
- Directories of text files -> recursive scan
"""

import os
import shutil
import time
import threading
import subprocess
import tempfile
from datetime import date
from multiprocessing import Pool

import tqdm

from .config import (
    DEFAULT_MAX_LEN, DEFAULT_MEM_MB, WORKERS, MIN_FREQ,
    CHUNK_LINES, OUTPUT_FILE_SUFFIX,
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

        print("Stage 3: Sorting n-grams (LC_ALL=C for speed)...")
        sort_env = {**os.environ, "LC_ALL": "C"}
        p1 = subprocess.Popen([
            "sort", "-S", f"{mem_mb}M", f"--parallel={workers}",
            "-o", ngram_fw_sorted, ngram_fw_path,
        ], stderr=subprocess.PIPE, text=True, env=sort_env)
        p2 = subprocess.Popen([
            "sort", "-S", f"{mem_mb}M", f"--parallel={workers}",
            "-o", ngram_bw_sorted, ngram_bw_path,
        ], stderr=subprocess.PIPE, text=True, env=sort_env)

        fw_in_size = os.path.getsize(ngram_fw_path)
        bw_in_size = os.path.getsize(ngram_bw_path)
        total_in = fw_in_size + bw_in_size

        def _monitor():
            while p1.poll() is None or p2.poll() is None:
                fw_out = os.path.getsize(ngram_fw_sorted) if os.path.exists(ngram_fw_sorted) else 0
                bw_out = os.path.getsize(ngram_bw_sorted) if os.path.exists(ngram_bw_sorted) else 0
                pct = (fw_out + bw_out) / total_in * 100 if total_in > 0 else 0
                elapsed = time.time() - _monitor.t0
                print(f"\r  Sorting... {pct:.0f}% ({elapsed:.0f}s)", end="", flush=True)
                time.sleep(3)
            elapsed = time.time() - _monitor.t0
            print(f"\r  Sorting... 100% ({elapsed:.0f}s)", flush=True)
        _monitor.t0 = time.time()
        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

        _, stderr1 = p1.communicate()
        _, stderr2 = p2.communicate()
        t.join(timeout=1)
        if p1.returncode != 0:
            raise RuntimeError(f"Forward sort failed: {stderr1}")
        if p2.returncode != 0:
            raise RuntimeError(f"Backward sort failed: {stderr2}")
        for fp in (ngram_fw_path, ngram_bw_path):
            try: os.remove(fp)
            except OSError: pass
        print()
        print("  Sorting complete")

        right_entropy_file = os.path.join(temp_dir, "right_entropy.txt")
        left_entropy_unsorted = os.path.join(temp_dir, "left_entropy_unsorted.txt")
        left_entropy_file = os.path.join(temp_dir, "left_entropy.txt")

        print(f"Stage 4a: Computing right entropy (min_freq={min_freq})...")
        count = _write_entropy_from_ngram(
            ngram_fw_sorted, right_entropy_file, min_freq, direct=True
        )
        print(f"  Right: {count} unique words")

        print(f"Stage 4b: Computing left entropy (min_freq={min_freq})...")
        count = _write_entropy_from_ngram(
            ngram_bw_sorted, left_entropy_unsorted, min_freq, direct=False
        )
        print(f"  Left: {count} unique words")

        print("Stage 4c: Sorting left entropy file...")
        sort_file_inplace(left_entropy_unsorted)
        os.rename(left_entropy_unsorted, left_entropy_file)

        from dict_build.config import (
            ENTROPY_THRESHOLD, PMI_THRESHOLD, POS_PROB_THRESHOLD,
        )

        print("Stage 5: Merging left and right entropy...")
        merged = merge_entropy_files_sorted(
            right_entropy_file, left_entropy_file,
            min_entropy=ENTROPY_THRESHOLD,
        )
        print(f"  Merged: {len(merged)} candidates (entropy >= {ENTROPY_THRESHOLD})")

        if not merged:
            print("  No candidates pass entropy threshold.")
            with open(output_path, "w") as f:
                f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
            return output_path

        trie_file = os.path.join(temp_dir, "freq.trie")
        print("Stage 6a: Building frequency trie (stream from file)...")
        trie, total_single = build_and_mmap_trie(
            right_entropy_file, trie_file, min_freq=min_freq
        )
        print(f"  Trie ready. Total single-char freq: {total_single}")

        print("Stage 6b: Computing PMI and filtering...")
        pos_prob = load_pos_prob(pos_prob_path)
        results_raw = extract_words(
            merged, pos_prob, trie, total_single,
            pmi_threshold=PMI_THRESHOLD,
            entropy_threshold=ENTROPY_THRESHOLD,
            pos_threshold=POS_PROB_THRESHOLD,
        )
        print(f"  Candidates after PMI/filter: {len(results_raw)}")

        # Attach POS tags via list comprehension
        results = [
            (w, f, p, e, pp, tag_word(w))
            for w, f, p, e, pp in results_raw
        ]

        _write_output(results, output_path)
        print(f"Output: {output_path}")
        return output_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---- N-gram generation: normal files (parallel) ----

def _generate_ngrams_parallel(
    txt_files: list[str],
    ngram_fw_path: str,
    ngram_bw_path: str,
    max_len: int,
    workers: int,
) -> None:
    """Generate n-grams with multiprocessing, writing directly to temp files.

    Each worker writes its chunk output to a unique temp file pair.
    Results are never pickled back; the main process only concatenates.
    """
    ngram_tmp_dir = tempfile.mkdtemp(prefix="dict_build_ngrams_")

    total_chunks = 0
    for fp in txt_files:
        with open(fp, "r", encoding="utf-8", errors="replace") as fh:
            lc = sum(1 for _ in fh)
        total_chunks += (lc + CHUNK_LINES - 1) // CHUNK_LINES if lc > 0 else 1

    max_pending = workers * 4
    fw_tmp_paths: list[str] = []
    bw_tmp_paths: list[str] = []

    try:
        with Pool(processes=workers) as pool:
            pbar = tqdm.tqdm(total=total_chunks, desc="  N-grams", unit="chunk")
            futures: list = []
            chunk_id = 0

            for txt_file in txt_files:
                for chunk in _read_chunks_by_lines(txt_file, CHUNK_LINES):
                    fw_tmp = os.path.join(ngram_tmp_dir, f"fw_{chunk_id:08d}.txt")
                    bw_tmp = os.path.join(ngram_tmp_dir, f"bw_{chunk_id:08d}.txt")
                    fw_tmp_paths.append(fw_tmp)
                    bw_tmp_paths.append(bw_tmp)

                    if len(futures) >= max_pending:
                        futures.pop(0).get()
                        pbar.update(1)

                    futures.append(pool.apply_async(
                        _process_chunk_direct,
                        (chunk, max_len, fw_tmp, bw_tmp),
                    ))
                    chunk_id += 1

            for fut in futures:
                try:
                    fut.get()
                except Exception:
                    pool.terminate()
                    raise
                pbar.update(1)
            pbar.close()

        for out_path, tmp_paths in [
            (ngram_fw_path, fw_tmp_paths),
            (ngram_bw_path, bw_tmp_paths),
        ]:
            with open(out_path, "wb") as out:
                for tp in tmp_paths:
                    with open(tp, "rb") as f:
                        shutil.copyfileobj(f, out)
    finally:
        shutil.rmtree(ngram_tmp_dir, ignore_errors=True)


def _process_chunk_direct(
    lines: list[str],
    max_len: int,
    fw_path: str,
    bw_path: str,
) -> None:
    """Process a chunk of lines, write n-grams directly to temp files.

    No n-gram lists are accumulated in memory; each fragment is written
    immediately. Worker memory = chunk text + file buffer (~ few MB).
    """
    with open(fw_path, "w", encoding="utf-8") as fw, \
         open(bw_path, "w", encoding="utf-8") as bw:
        for line in lines:
            for sent in preprocess_line(line):
                for ng in generate_ngrams(sent, max_len):
                    fw.write(ng)
                    fw.write("\n")
                for ng in generate_reverse_ngrams(sent, max_len):
                    bw.write(ng)
                    bw.write("\n")


def _read_chunks_by_lines(path: str, n: int):
    chunk: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= n:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


# ---- N-gram generation: huge single-line files (sequential) ----

def _generate_ngrams_single_line(
    filepath: str,
    ngram_fw_path: str,
    ngram_bw_path: str,
    max_len: int,
) -> None:
    fw_out = open(ngram_fw_path, "a", encoding="utf-8")
    bw_out = open(ngram_bw_path, "a", encoding="utf-8")
    try:
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
                    else:
                        text = segment.decode("utf-8", errors="replace")
                        carryover = b""

                for i in range(0, len(text), SINGLE_LINE_CHAR_CHUNK):
                    sub = text[i:i + SINGLE_LINE_CHAR_CHUNK]
                    _process_text_segment(sub, max_len, fw_out, bw_out)
                    pbar.update(1)

        pbar.close()
    finally:
        fw_out.close()
        bw_out.close()


def _process_text_segment(text: str, max_len: int, fw_out, bw_out) -> None:
    """Preprocess a text segment and write n-grams directly."""
    for sent in preprocess_line(text):
        for ng in generate_ngrams(sent, max_len):
            fw_out.write(ng)
            fw_out.write("\n")
        for ng in generate_reverse_ngrams(sent, max_len):
            bw_out.write(ng)
            bw_out.write("\n")


# ---- Entropy helpers ----

def _write_entropy_from_ngram(
    ngram_file: str,
    output_file: str,
    min_freq: int,
    direct: bool,
) -> int:
    with open(ngram_file, "r", encoding="utf-8", buffering=16 * 1024 * 1024) as fin:
        gen = compute_entropy_from_sorted(fin, min_freq=min_freq) if direct \
             else compute_entropy_from_sorted_left(fin, min_freq=min_freq)
        pbar = tqdm.tqdm(gen, desc="    Entropy", unit="words", unit_scale=True)
        return write_entropy_to_file(pbar, output_file)


# ---- Output ----

def _write_output(results: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
        for w, fr, p, e, pp, pos in results:
            f.write(f"{w}\t{fr}\t{p:.6f}\t{e:.6f}\t{pp:.6f}\t{pos}\n")
