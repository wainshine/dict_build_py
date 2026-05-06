"""Pipeline orchestrator for the full word extraction process.

Handles:
- Normal multi-line files -> multiprocessing
- Huge single-line files -> sequential direct write
- Directories of text files -> recursive scan
"""

import io
import os
import shutil
import time
import threading
import subprocess
import tempfile
import zlib
from datetime import date
from multiprocessing import Pool

import tqdm

from .config import (
    DEFAULT_MAX_LEN, DEFAULT_MEM_MB, WORKERS, MIN_FREQ,
    CHUNK_LINES, CHUNKS_PER_BATCH, OUTPUT_FILE_SUFFIX,
    BUCKET_SORT_MIN_BYTES, BUCKET_TARGET_BYTES, MIN_SORT_MEM_MB,
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


def _detect_file_encoding(filepath: str) -> str:
    """Detect the encoding of a text file.

    Tries UTF-8 first (if clean, it's the real encoding). For ambiguous
    files, compares the CJK character *ratio* (not absolute count) across
    GB18030/GBK/BIG5. Falls back to charset_normalizer if inconclusive.
    """
    with open(filepath, "rb") as f:
        sample = f.read(64 * 1024)

    # Try UTF-8 first.
    # If clean: definitely UTF-8.
    # If dirty with U+FFFD but still has CJK chars: it's probably damaged
    # UTF-8 (or double-encoded), not pure GBK. Using GBK on these produces
    # artifact characters. Only fall to GBK when UTF-8 has essentially
    # zero CJK.
    try:
        text = sample.decode("utf-8")
        if "\ufffd" not in text:
            return "utf-8"
    except UnicodeDecodeError:
        # UTF-8 strict decode failed. Check if errors=replace still
        # recovers meaningful CJK — double-encoded GBK files masquerading
        # as valid GBK will produce garbage when decoded as GBK, so
        # preferring UTF-8+replace is the safer choice.
        text = sample.decode("utf-8", errors="replace")
        ch = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FA5)
        if ch > len(sample) * 0.005:
            return "utf-8"

    # UTF-8 gave no meaningful CJK — try GBK/GB18030/BIG5
    candidates = ["gb18030", "gbk", "big5"]
    best_enc = "utf-8"
    best_ratio = 0.0
    total = len(sample)

    for enc in candidates:
        try:
            text = sample.decode(enc, errors="replace")
            ch = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FA5)
            ratio = ch / total if total > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_enc = enc
        except (UnicodeDecodeError, LookupError):
            continue

    # If heuristic found a clear winner, use it
    if best_ratio > 0.01:
        return best_enc

    # Still inconclusive — try charset_normalizer
    try:
        import charset_normalizer
        result = charset_normalizer.from_path(
            filepath,
            preemptive_behaviour=False,
        )
        best = result.best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass

    return best_enc


# ---- Bucket-sort helpers ----

def _hash_line(line_bytes: bytes) -> int:
    """Stable hash of n-gram word (pre-tab) for bucket assignment."""
    tab = line_bytes.find(b"\t")
    key = line_bytes[:tab] if tab > 0 else line_bytes
    return zlib.crc32(key) & 0x7FFFFFFF


def _calc_bucket_params(
    total_size: int,
    workers: int,
    mem_mb: int,
) -> tuple[int, int, int]:
    """Return (num_buckets, max_concurrent_sorts, sort_mem_mb_per)."""
    max_concurrent = min(workers, max(1, mem_mb // MIN_SORT_MEM_MB))
    sort_mem_per = max(MIN_SORT_MEM_MB, mem_mb // max_concurrent)
    num = max(1, int(total_size / BUCKET_TARGET_BYTES) + 1)
    return num, max_concurrent, sort_mem_per


def _sort_bucket(path: str, sort_mem_mb: int) -> str:
    """Sort a single bucket file, return path to sorted result."""
    out = path + ".sorted"
    env = {**os.environ, "LC_ALL": "C"}
    result = subprocess.run(
        ["sort", "-S", f"{sort_mem_mb}M", "--parallel=1",
         "-o", out, path],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Bucket sort failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    os.remove(path)
    return out


def _distribute_batches(
    paths: list[str],
    bucket_dir: str,
    num_buckets: int,
    buf_limit: int,
    batch_size: int,
) -> None:
    """Process a list of temp files in batches into bucket files."""
    for batch_start in range(0, len(paths), batch_size):
        batch = paths[batch_start:batch_start + batch_size]
        fw_bh = [open(os.path.join(bucket_dir, f"fw_b{i:04d}.txt"), "ab")
                 for i in range(num_buckets)]
        bw_bh = [open(os.path.join(bucket_dir, f"bw_b{i:04d}.txt"), "ab")
                 for i in range(num_buckets)]
        fw_bufs = [io.BytesIO() for _ in range(num_buckets)]
        bw_bufs = [io.BytesIO() for _ in range(num_buckets)]

        for tp in batch:
            carry = b""
            is_fw = os.path.basename(tp).startswith("fw_")
            buckets = fw_bh if is_fw else bw_bh
            bufs = fw_bufs if is_fw else bw_bufs

            with open(tp, "rb") as src:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    chunk = carry + chunk
                    carry = b""
                    lines = chunk.split(b"\n")
                    for i in range(len(lines) - 1):
                        line = lines[i] + b"\n"
                        if not line.strip(b"\r\n"):
                            continue
                        h = _hash_line(line) % num_buckets
                        bufs[h].write(line)
                        if bufs[h].tell() >= buf_limit:
                            buckets[h].write(bufs[h].getvalue())
                            bufs[h].seek(0)
                            bufs[h].truncate()
                    carry = lines[-1]
                if carry and carry.strip(b"\r\n"):
                    h = _hash_line(carry + b"\n") % num_buckets
                    bufs[h].write(carry + b"\n")
            os.remove(tp)

        for side_bufs, side_bh in [(fw_bufs, fw_bh), (bw_bufs, bw_bh)]:
            for i in range(num_buckets):
                if side_bufs[i].tell() > 0:
                    side_bh[i].write(side_bufs[i].getvalue())
                side_bufs[i].close()
            for bh in side_bh:
                bh.close()


def _distribute_and_sort_ngrams(
    temp_dir: str,
    fw_tmp_paths: list[str],
    bw_tmp_paths: list[str],
    num_buckets: int,
    max_concurrent: int,
    sort_mem_mb: int,
    mem_mb: int,
) -> tuple[list[str], list[str]]:
    """Partition n-gram temp files into hash buckets, sort each bucket.

    Processes temp files in batches (avoiding open-file thrash on 153+
    file handles). fw and bw streams run in parallel via two threads.
    Buffer size per bucket adapts to available memory.
    """
    bucket_dir = tempfile.mkdtemp(prefix="dict_build_buckets_",
                                   dir=temp_dir)

    # Adaptive per-bucket buffer: 1/4 of mem, split across buckets,
    # floor 128MB, cap 256MB.
    mem_bytes = mem_mb * 1024 * 1024
    buf_limit = max(128 * 1024**2,
                    min(256 * 1024**2,
                        (mem_bytes // 4) // max(num_buckets, 1)))

    BATCH_SIZE = 15

    # Two processes (no GIL) process fw and bw simultaneously
    from multiprocessing import Process
    fw_proc = Process(target=_distribute_batches,
                      args=(fw_tmp_paths, bucket_dir, num_buckets,
                            buf_limit, BATCH_SIZE))
    bw_proc = Process(target=_distribute_batches,
                      args=(bw_tmp_paths, bucket_dir, num_buckets,
                            buf_limit, BATCH_SIZE))
    fw_proc.start()
    bw_proc.start()
    fw_proc.join()
    bw_proc.join()

    if fw_proc.exitcode != 0:
        raise RuntimeError(f"Forward distribution failed (exit {fw_proc.exitcode})")
    if bw_proc.exitcode != 0:
        raise RuntimeError(f"Backward distribution failed (exit {bw_proc.exitcode})")

    # Collect bucket file paths
    fw_bucket_paths = [os.path.join(bucket_dir, f"fw_b{i:04d}.txt")
                       for i in range(num_buckets)]
    bw_bucket_paths = [os.path.join(bucket_dir, f"bw_b{i:04d}.txt")
                       for i in range(num_buckets)]

    # Sort each bucket (parallel)
    fw_sorted: list[str] = []
    bw_sorted: list[str] = []
    all_buckets = fw_bucket_paths + bw_bucket_paths
    is_fw = [True] * len(fw_bucket_paths) + [False] * len(bw_bucket_paths)

    with Pool(processes=max_concurrent) as pool:
        futs = [pool.apply_async(_sort_bucket, (bp, sort_mem_mb))
                for bp in all_buckets]
        for fut, fw_flag in zip(futs, is_fw):
            sp = fut.get()
            if fw_flag:
                fw_sorted.append(sp)
            else:
                bw_sorted.append(sp)

    return fw_sorted, bw_sorted


# ---- File collection ----

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
    total_size = sum(os.path.getsize(fp) for fp in txt_files)
    print(f"Found {len(txt_files)} text file(s) to process "
          f"({total_size / 1e9:.1f} GB)")
    if total_size > 2 * 1024 * 1024 * 1024:
        print(f"  ⚠ Input > 2 GB, processing may take a long time."
              f" Consider --min-freq higher or sample first.")

    if output_path is None:
        output_path = _make_output_path(input_path)

    temp_dir = tempfile.mkdtemp(prefix="dict_build_")
    ngram_fw_path = os.path.join(temp_dir, "ngram_forward.txt")
    ngram_bw_path = os.path.join(temp_dir, "ngram_backward.txt")

    try:
        print(f"Stage 1-2: Preprocessing + N-gram generation (workers={workers})...")
        huge_files = [f for f in txt_files if _is_single_line_file(f)]
        normal_files = [f for f in txt_files if f not in huge_files]

        fw_tmp_paths: list[str] = []
        bw_tmp_paths: list[str] = []
        ngram_tmp_dir: str | None = None

        if normal_files:
            fw_tmp_paths, bw_tmp_paths, ngram_tmp_dir = _generate_ngrams_parallel(
                normal_files, ngram_fw_path, ngram_bw_path, max_len, workers,
            )
        for hf in huge_files:
            print(f"  Single-line mode: {os.path.basename(hf)} "
                  f"({os.path.getsize(hf)/1e9:.1f} GB)")
            _generate_ngrams_single_line(
                hf, ngram_fw_path, ngram_bw_path, max_len,
            )

        total_ngram = sum(os.path.getsize(tp)
                          for tp in fw_tmp_paths + bw_tmp_paths)
        if os.path.exists(ngram_fw_path):
            total_ngram += os.path.getsize(ngram_fw_path)
            total_ngram += os.path.getsize(ngram_bw_path)

        fw_display = total_ngram / 2 / 1e9 if total_ngram > 0 else 0
        print(f"  Forward n-grams: {fw_display:.2f} GB")
        print(f"  Backward n-grams: {fw_display:.2f} GB")

        ngram_fw_sorted = os.path.join(temp_dir, "ngram_forward_sorted.txt")
        ngram_bw_sorted = os.path.join(temp_dir, "ngram_backward_sorted.txt")

        right_entropy_file = os.path.join(temp_dir, "right_entropy.txt")
        left_entropy_unsorted = os.path.join(temp_dir, "left_entropy_unsorted.txt")
        left_entropy_file = os.path.join(temp_dir, "left_entropy.txt")

        use_buckets = (total_ngram >= BUCKET_SORT_MIN_BYTES
                       and not huge_files
                       and fw_tmp_paths)

        if use_buckets:
            num_buckets, max_conc, sort_mem = _calc_bucket_params(
                total_ngram, workers, mem_mb,
            )
            print(f"Stage 3: Bucket-sorting n-grams"
                  f" ({num_buckets} buckets, {max_conc} concurrent)...")

            fw_sorted, bw_sorted = _distribute_and_sort_ngrams(
                temp_dir, fw_tmp_paths, bw_tmp_paths,
                num_buckets, max_conc, sort_mem, mem_mb,
            )
            if ngram_tmp_dir:
                shutil.rmtree(ngram_tmp_dir, ignore_errors=True)

            # Entropy on sorted buckets, then concat
            print(f"Stage 4a: Computing right entropy (min_freq={min_freq})...")
            r_count = _compute_entropy_from_sorted_list(
                fw_sorted, right_entropy_file, min_freq, direct=True,
                workers=workers,
            )
            print(f"  Right: {r_count} unique words")
            # Buckets produce per-group-sorted entropy; re-sort for merge
            sort_file_inplace(right_entropy_file)

            print(f"Stage 4b: Computing left entropy (min_freq={min_freq})...")
            l_count = _compute_entropy_from_sorted_list(
                bw_sorted, left_entropy_unsorted, min_freq, direct=False,
                workers=workers,
            )
            print(f"  Left: {l_count} unique words")
        else:
            # Old path: concat temp files → single sort → entropy
            if fw_tmp_paths:
                _concat_temp_files(fw_tmp_paths, ngram_fw_path)
                _concat_temp_files(bw_tmp_paths, ngram_bw_path)
            if ngram_tmp_dir:
                shutil.rmtree(ngram_tmp_dir, ignore_errors=True)

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

            fw_in = os.path.getsize(ngram_fw_path)
            bw_in = os.path.getsize(ngram_bw_path)
            total_in = fw_in + bw_in

            def _monitor():
                while p1.poll() is None or p2.poll() is None:
                    fw_o = os.path.getsize(ngram_fw_sorted) if os.path.exists(ngram_fw_sorted) else 0
                    bw_o = os.path.getsize(ngram_bw_sorted) if os.path.exists(ngram_bw_sorted) else 0
                    pct = (fw_o + bw_o) / total_in * 100 if total_in > 0 else 0
                    e = time.time() - _monitor.t0
                    print(f"\r  Sorting... {pct:.0f}% ({e:.0f}s)", end="", flush=True)
                    time.sleep(3)
                e = time.time() - _monitor.t0
                print(f"\r  Sorting... 100% ({e:.0f}s)", flush=True)
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

            PARALLEL_ENTROPY_THRESHOLD = 1024 * 1024 * 1024

            print(f"Stage 4a: Computing right entropy (min_freq={min_freq})...")
            if os.path.getsize(ngram_fw_sorted) > PARALLEL_ENTROPY_THRESHOLD:
                count = _write_entropy_from_ngram_parallel(
                    ngram_fw_sorted, right_entropy_file, min_freq,
                    direct=True, workers=workers,
                )
            else:
                count = _write_entropy_from_ngram(
                    ngram_fw_sorted, right_entropy_file, min_freq, direct=True,
                )
            print(f"  Right: {count} unique words")

            print(f"Stage 4b: Computing left entropy (min_freq={min_freq})...")
            if os.path.getsize(ngram_bw_sorted) > PARALLEL_ENTROPY_THRESHOLD:
                count = _write_entropy_from_ngram_parallel(
                    ngram_bw_sorted, left_entropy_unsorted, min_freq,
                    direct=False, workers=workers,
                )
            else:
                count = _write_entropy_from_ngram(
                    ngram_bw_sorted, left_entropy_unsorted, min_freq, direct=False,
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
) -> tuple[list[str], list[str], str]:
    """Generate n-grams with multiprocessing, writing directly to temp files.

    Returns (fw_tmp_paths, bw_tmp_paths, tmp_dir). Caller must process
    the temp files and then delete tmp_dir.
    """
    ngram_tmp_dir = tempfile.mkdtemp(prefix="dict_build_ngrams_")

    # Estimate total chunks from file sizes (avoid reading every file twice)
    total_chunks = 0
    file_encodings: dict[str, str] = {}
    for fp in txt_files:
        enc = _detect_file_encoding(fp)
        file_encodings[fp] = enc
        est_lines = max(1, os.path.getsize(fp) // 100)
        total_chunks += (est_lines + CHUNK_LINES - 1) // CHUNK_LINES

    max_pending = workers * 4
    fw_tmp_paths: list[str] = []
    bw_tmp_paths: list[str] = []

    with Pool(processes=workers) as pool:
        pbar = tqdm.tqdm(total=total_chunks, desc="  N-grams", unit="chunk")
        futures: list = []
        chunks_per_task: list[int] = []
        batch_id = 0
        batch_lines: list[str] = []
        batch_chunks: int = 0

        for txt_file in txt_files:
            enc = file_encodings[txt_file]
            for chunk in _read_chunks_by_lines(txt_file, CHUNK_LINES, encoding=enc):
                batch_lines.extend(chunk)
                batch_chunks += 1

                if batch_chunks >= CHUNKS_PER_BATCH:
                    fw_tmp = os.path.join(ngram_tmp_dir, f"fw_{batch_id:06d}.txt")
                    bw_tmp = os.path.join(ngram_tmp_dir, f"bw_{batch_id:06d}.txt")
                    fw_tmp_paths.append(fw_tmp)
                    bw_tmp_paths.append(bw_tmp)

                    if len(futures) >= max_pending:
                        futures.pop(0).get()
                        pbar.update(chunks_per_task.pop(0))

                    futures.append(pool.apply_async(
                        _process_chunk_direct,
                        (batch_lines, max_len, fw_tmp, bw_tmp),
                    ))
                    chunks_per_task.append(batch_chunks)
                    batch_id += 1
                    batch_lines = []
                    batch_chunks = 0

        if batch_lines:
            fw_tmp = os.path.join(ngram_tmp_dir, f"fw_{batch_id:06d}.txt")
            bw_tmp = os.path.join(ngram_tmp_dir, f"bw_{batch_id:06d}.txt")
            fw_tmp_paths.append(fw_tmp)
            bw_tmp_paths.append(bw_tmp)
            futures.append(pool.apply_async(
                _process_chunk_direct,
                (batch_lines, max_len, fw_tmp, bw_tmp),
            ))
            chunks_per_task.append(batch_chunks)

        for fut, n_chunks in zip(futures, chunks_per_task):
            try:
                fut.get()
            except Exception:
                pool.terminate()
                raise
            pbar.update(n_chunks)
        pbar.close()

    return fw_tmp_paths, bw_tmp_paths, ngram_tmp_dir


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


def _read_chunks_by_lines(path: str, n: int, encoding: str = "utf-8"):
    chunk: list[str] = []
    with open(path, "r", encoding=encoding, errors="replace") as f:
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
    encoding = _detect_file_encoding(filepath)
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
                            carryover.decode(encoding, errors="replace"),
                            max_len, fw_out, bw_out
                        )
                    break

                segment = carryover + data
                try:
                    text = segment.decode(encoding)
                    carryover = b""
                except UnicodeDecodeError:
                    for cut in range(-1, -5, -1):
                        try:
                            text = segment[:cut].decode(encoding)
                            carryover = segment[cut:]
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = segment.decode(encoding, errors="replace")
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

def _split_sorted_ngram_by_chars(
    ngram_file: str,
    output_dir: str,
    chars_per_file: int = 10,
) -> list[str]:
    """Split sorted n-gram file at first-character boundaries.

    The n-gram file is sorted lexically (by word). Splitting at a character
    boundary is safe — all lines for a given first character stay together.
    """
    split_paths: list[str] = []
    current_fh = None
    chars_since_split = 0
    prev_char = ""
    chunk_idx = 0

    with open(ngram_file, "r", encoding="utf-8",
              buffering=16 * 1024 * 1024) as fin:
        for line in fin:
            if not line:
                continue
            first = line[0]
            if first != prev_char:
                prev_char = first
                chars_since_split += 1
                if current_fh is None or chars_since_split > chars_per_file:
                    if current_fh:
                        current_fh.close()
                    path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.txt")
                    split_paths.append(path)
                    current_fh = open(path, "w", encoding="utf-8")
                    chunk_idx += 1
                    chars_since_split = 1
            if current_fh:
                current_fh.write(line)
    if current_fh:
        current_fh.close()
    return split_paths


def _process_entropy_split(
    split_path: str,
    min_freq: int,
    direct: bool,
) -> tuple[int, str]:
    """Compute entropy for one split file. Returns (count, out_path)."""
    out_path = split_path + ".out"
    with open(split_path, "r", encoding="utf-8",
              buffering=16 * 1024 * 1024) as fin:
        gen = compute_entropy_from_sorted(fin, min_freq=min_freq) if direct \
             else compute_entropy_from_sorted_left(fin, min_freq=min_freq)
        count = write_entropy_to_file(gen, out_path)
    return count, out_path


def _write_entropy_from_ngram_parallel(
    ngram_file: str,
    output_file: str,
    min_freq: int,
    direct: bool,
    workers: int,
    chars_per_file: int = 10,
) -> int:
    """Split n-gram file by first-char groups, compute entropy in parallel."""
    split_dir = tempfile.mkdtemp(prefix="dict_build_entropy_")
    try:
        print("    Splitting by first-char groups...", end="", flush=True)
        split_paths = _split_sorted_ngram_by_chars(
            ngram_file, split_dir, chars_per_file,
        )
        print(f" {len(split_paths)} chunks")

        total = 0
        with Pool(processes=workers) as pool:
            desc = "    Entropy" if direct else "    Entropy(L)"
            pbar = tqdm.tqdm(total=len(split_paths), desc=desc, unit="chunk")

            futures = [
                pool.apply_async(_process_entropy_split, (sp, min_freq, direct))
                for sp in split_paths
            ]

            with open(output_file, "w", encoding="utf-8") as out:
                for fut in futures:
                    try:
                        count, tmp_path = fut.get()
                    except Exception:
                        pool.terminate()
                        raise
                    total += count
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)
                    os.remove(tmp_path)
                    pbar.update(1)
            pbar.close()
        return total
    finally:
        shutil.rmtree(split_dir, ignore_errors=True)


def _write_entropy_from_ngram(
    ngram_file: str,
    output_file: str,
    min_freq: int,
    direct: bool,
) -> int:
    """Legacy single-threaded entropy computation (kept for small files)."""
    with open(ngram_file, "r", encoding="utf-8", buffering=16 * 1024 * 1024) as fin:
        gen = compute_entropy_from_sorted(fin, min_freq=min_freq) if direct \
             else compute_entropy_from_sorted_left(fin, min_freq=min_freq)
        pbar = tqdm.tqdm(gen, desc="    Entropy", unit="words", unit_scale=True)
        return write_entropy_to_file(pbar, output_file)


def _concat_temp_files(paths: list[str], output_path: str) -> None:
    """Concatenate multiple files into one output."""
    with open(output_path, "wb") as out:
        for tp in paths:
            with open(tp, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(tp)


def _compute_entropy_from_sorted_list(
    sorted_paths: list[str],
    output_file: str,
    min_freq: int,
    direct: bool,
    workers: int,
) -> int:
    """Compute entropy on each sorted bucket, concat results, return total."""
    total = 0
    with Pool(processes=min(workers, len(sorted_paths))) as pool:
        desc = "    Entropy" if direct else "    Entropy(L)"
        pbar = tqdm.tqdm(total=len(sorted_paths), desc=desc, unit="file")
        futs = [
            pool.apply_async(_process_entropy_split, (sp, min_freq, direct))
            for sp in sorted_paths
        ]
        with open(output_file, "w", encoding="utf-8") as out:
            for fut in futs:
                try:
                    count, tmp_path = fut.get()
                except Exception:
                    pool.terminate()
                    raise
                total += count
                with open(tmp_path, "r", encoding="utf-8") as f:
                    shutil.copyfileobj(f, out)
                os.remove(tmp_path)
                pbar.update(1)
        pbar.close()
    return total


# ---- Output ----

def _write_output(results: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
        for w, fr, p, e, pp, pos in results:
            f.write(f"{w}\t{fr}\t{p:.6f}\t{e:.6f}\t{pp:.6f}\t{pos}\n")
