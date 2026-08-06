"""Pipeline orchestrator for the full word extraction process.

Handles:
- Normal multi-line files -> multiprocessing
- Huge single-line files -> sequential direct write
- Directories of text files -> recursive scan
"""

import io
import heapq
import itertools
import json
import logging
import os
import shutil
import time
import subprocess
import tempfile
import zlib
from datetime import date
from multiprocessing import Pool

import tqdm

from .config import (
    DEFAULT_MAX_LEN, DEFAULT_MEM_MB, WORKERS, MIN_FREQ,
    PMI_THRESHOLD, ENTROPY_THRESHOLD, POS_PROB_THRESHOLD,
    CHUNK_LINES, CHUNKS_PER_BATCH, OUTPUT_FILE_SUFFIX, SENTINEL,
    BUCKET_SORT_MIN_BYTES, BUCKET_TARGET_BYTES, MIN_SORT_MEM_MB,
    MAX_BUCKETS, INTERMEDIATE_SIZE_FACTOR,
    PENDING_TASK_FACTOR, NGRAM_WRITE_BATCH, AVG_LINE_BYTES,
    BUCKET_DISTRIBUTE_BATCH, BUCKET_BUF_MIN_BYTES, BUCKET_BUF_MAX_BYTES,
    PARALLEL_ENTROPY_MIN_BYTES, ENTROPY_SPLIT_MIN_BYTES,
    SINGLE_LINE_MIN_BYTES, SINGLE_LINE_MAX_NEWLINES,
    SINGLE_LINE_SAMPLE_BYTES,
    ENCODING_SAMPLE_BYTES, CJK_RATIO_UTF8_MIN, CJK_RATIO_WINNER_MIN,
)
from .preprocess import preprocess_line, iter_chinese_tokens
from .ngram import generate_ngrams, generate_reverse_ngrams
from .entropy import (
    compute_entropy_from_sorted,
    compute_entropy_from_sorted_left,
    write_entropy_to_file,
    sort_file_inplace,
    iter_merge_entropy_files_sorted,
    read_entropy_from_file,
)
from .pmi import build_and_mmap_trie, extract_words
from .pos_prob import load_pos_prob

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".sql", ".md", ".html", ".htm"}
SINGLE_LINE_CHAR_CHUNK = 200_000
BYTE_BUF = 8 * 1024 * 1024

# Probed lazily: (sort_executable, supports_gnu_flags)
_SORT_CAPS: tuple[str, bool] | None = None


def _sort_command(mem_mb: int, workers: int,
                  tmp_dir: str | None = None) -> list[str]:
    """Build a system sort command, probing capabilities once per process.

    GNU sort gets -S/--parallel; BSD sort gets plain flags (it accepts
    but ignores -S on some versions, and silently ignores invalid
    memsize units, so flags are only passed when the probe succeeds).
    -T pins sort's own temp files to tmp_dir so --temp-dir/--work-dir
    actually govern the sort stage (POSIX: supported by GNU and BSD).
    Raises RuntimeError when no sort executable is available.
    """
    global _SORT_CAPS
    if _SORT_CAPS is None:
        exe = shutil.which("sort")
        if exe is None:
            raise RuntimeError(
                "System 'sort' command not found. Install GNU coreutils "
                "(or run on Linux/macOS) to process large files."
            )
        gnu = False
        try:
            r = subprocess.run(
                [exe, "-S", "1M", "--parallel=2"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=10,
            )
            gnu = r.returncode == 0
        except Exception:
            gnu = False
        _SORT_CAPS = (exe, gnu)

    exe, gnu = _SORT_CAPS
    cmd = [exe]
    if gnu:
        cmd += ["-S", f"{max(1, mem_mb)}M", f"--parallel={max(1, workers)}"]
    if tmp_dir is not None:
        cmd += ["-T", tmp_dir]
    return cmd


def _read_samples(filepath: str, sample_size: int = ENCODING_SAMPLE_BYTES) -> list[bytes]:
    """Read head/middle/tail samples of a file for encoding detection."""
    size = os.path.getsize(filepath)
    if size <= sample_size:
        with open(filepath, "rb") as f:
            return [f.read()]
    offsets = [0, max(0, size // 2 - sample_size // 2), size - sample_size]
    samples = []
    with open(filepath, "rb") as f:
        for off in offsets:
            f.seek(off)
            samples.append(f.read(sample_size))
    return samples


def _decode_utf8_tolerant(sample: bytes) -> str:
    """Strict UTF-8 decode, tolerating a character split at the sample edge.

    Raises UnicodeDecodeError if the failure is not at the trailing edge.
    """
    for trim in range(0, 4):
        try:
            return sample[:len(sample) - trim].decode("utf-8") if trim else sample.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start < len(sample) - trim - 4:
                raise
    raise UnicodeDecodeError("utf-8", sample, 0, 1, "undecodable")


def _detect_file_encoding(filepath: str) -> str:
    """Detect the encoding of a text file.

    Samples head/middle/tail so an ASCII header does not mask a GBK body.
    Tries UTF-8 first (if clean, it's the real encoding). For ambiguous
    files, compares the CJK character *ratio* (not absolute count) across
    GB18030/GBK/BIG5. Falls back to charset_normalizer if inconclusive.
    """
    samples = _read_samples(filepath)
    total = sum(len(s) for s in samples)

    # Try UTF-8 first.
    # If clean: definitely UTF-8.
    # If dirty with U+FFFD but still has CJK chars: it's probably damaged
    # UTF-8 (or double-encoded), not pure GBK. Using GBK on these produces
    # artifact characters. Only fall to GBK when UTF-8 has essentially
    # zero CJK.
    try:
        text = "".join(_decode_utf8_tolerant(s) for s in samples)
        if "�" not in text:
            return "utf-8"
    except UnicodeDecodeError:
        # UTF-8 strict decode failed. Check if errors=replace still
        # recovers meaningful CJK — double-encoded GBK files masquerading
        # as valid GBK will produce garbage when decoded as GBK, so
        # preferring UTF-8+replace is the safer choice.
        text = "".join(s.decode("utf-8", errors="replace") for s in samples)
        ch = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FA5)
        if ch > total * CJK_RATIO_UTF8_MIN:
            return "utf-8"

    # UTF-8 gave no meaningful CJK — try GBK/GB18030/BIG5
    candidates = ["gb18030", "gbk", "big5"]
    best_enc = "utf-8"
    best_ratio = 0.0

    for enc in candidates:
        try:
            text = "".join(s.decode(enc, errors="replace") for s in samples)
            ch = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FA5)
            ratio = ch / total if total > 0 else 0
            if ratio > best_ratio:
                best_ratio = ratio
                best_enc = enc
        except (UnicodeDecodeError, LookupError):
            continue

    # If heuristic found a clear winner, use it
    if best_ratio > CJK_RATIO_WINNER_MIN:
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
    num = min(num, MAX_BUCKETS)
    return num, max_concurrent, sort_mem_per


def _sort_bucket(paths: list[str], sort_mem_mb: int) -> str:
    """Sort one bucket (one or more shard files), return sorted path."""
    out = paths[0] + ".sorted"
    env = {**os.environ, "LC_ALL": "C"}
    result = subprocess.run(
        [*_sort_command(sort_mem_mb, 1, os.path.dirname(paths[0])),
         "-o", out, *paths],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Bucket sort failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass
    return out


def _distribute_batches(
    paths: list[str],
    bucket_dir: str,
    num_buckets: int,
    buf_limit: int,
    batch_size: int,
    shard: str = "",
) -> None:
    """Process a list of temp files in batches into bucket files.

    Bucket files are named {fw|bw}{shard}_b{NNNN}.txt so multiple
    distribution processes can write disjoint shard sets concurrently.
    """
    crc32 = zlib.crc32
    for batch_start in range(0, len(paths), batch_size):
        batch = paths[batch_start:batch_start + batch_size]
        fw_bh = [open(os.path.join(bucket_dir, f"fw{shard}_b{i:04d}.txt"), "ab")
                 for i in range(num_buckets)]
        bw_bh = [open(os.path.join(bucket_dir, f"bw{shard}_b{i:04d}.txt"), "ab")
                 for i in range(num_buckets)]
        fw_bufs = [io.BytesIO() for _ in range(num_buckets)]
        bw_bufs = [io.BytesIO() for _ in range(num_buckets)]

        for tp in batch:
            carry = b""
            is_fw = os.path.basename(tp).startswith("fw_")
            buckets = fw_bh if is_fw else bw_bh
            bufs = fw_bufs if is_fw else bw_bufs
            writes = [b.write for b in bufs]
            tells = [b.tell for b in bufs]

            with open(tp, "rb") as src:
                while True:
                    chunk = src.read(BYTE_BUF)
                    if not chunk:
                        break
                    chunk = carry + chunk
                    lines = chunk.split(b"\n")
                    carry = lines.pop()
                    for line in lines:
                        if not line or line == b"\r":
                            continue
                        tab = line.find(b"\t")
                        key = line[:tab] if tab > 0 else line
                        h = (crc32(key) & 0x7FFFFFFF) % num_buckets
                        writes[h](line)
                        writes[h](b"\n")
                        if tells[h]() >= buf_limit:
                            buckets[h].write(bufs[h].getvalue())
                            bufs[h].seek(0)
                            bufs[h].truncate()
                if carry and carry != b"\r":
                    tab = carry.find(b"\t")
                    key = carry[:tab] if tab > 0 else carry
                    h = (crc32(key) & 0x7FFFFFFF) % num_buckets
                    writes[h](carry)
                    writes[h](b"\n")
            os.remove(tp)

        for side_bufs, side_bh in [(fw_bufs, fw_bh), (bw_bufs, bw_bh)]:
            for i in range(num_buckets):
                if side_bufs[i].tell() > 0:
                    side_bh[i].write(side_bufs[i].getvalue())
                side_bufs[i].close()
            for bh in side_bh:
                bh.close()


def _shard_list(paths: list[str], n: int) -> list[list[str]]:
    """Split paths into n round-robin shards (temp files are ~equal size)."""
    n = max(1, min(n, len(paths)))
    return [paths[i::n] for i in range(n)]


def _distribute_and_sort_ngrams(
    temp_dir: str,
    fw_tmp_paths: list[str],
    bw_tmp_paths: list[str],
    num_buckets: int,
    max_concurrent: int,
    sort_mem_mb: int,
    mem_mb: int,
    workers: int = 2,
) -> tuple[list[str], list[str]]:
    """Partition n-gram temp files into hash buckets, sort each bucket.

    Distribution runs in up to `workers` processes (fw/bw sharded),
    each writing disjoint shard files; a bucket's shards are sorted
    together (sort natively merges multiple inputs). Buffer memory
    across all distribution processes stays within mem_mb / 2.
    """
    bucket_dir = tempfile.mkdtemp(prefix="dict_build_buckets_",
                                   dir=temp_dir)

    num_dist = max(2, workers)
    fw_shards = _shard_list(fw_tmp_paths,
                            max(1, min(len(fw_tmp_paths), num_dist // 2)))
    bw_shards = _shard_list(bw_tmp_paths,
                            max(1, min(len(bw_tmp_paths),
                                       num_dist - len(fw_shards))))
    num_procs = len(fw_shards) + len(bw_shards)

    # Adaptive per-bucket buffer: half of mem across all distribution
    # processes, split across buckets, floor/cap from config.
    mem_bytes = mem_mb * 1024 * 1024
    buf_limit = max(BUCKET_BUF_MIN_BYTES,
                    min(BUCKET_BUF_MAX_BYTES,
                        (mem_bytes // 2) // num_procs // max(num_buckets, 1)))

    from multiprocessing import Process
    procs: list[tuple[str, Process]] = []
    for idx, shard_paths in enumerate(fw_shards):
        procs.append(("Forward", Process(
            target=_distribute_batches,
            args=(shard_paths, bucket_dir, num_buckets, buf_limit,
                  BUCKET_DISTRIBUTE_BATCH, f"_s{idx}"))))
    for idx, shard_paths in enumerate(bw_shards):
        procs.append(("Backward", Process(
            target=_distribute_batches,
            args=(shard_paths, bucket_dir, num_buckets, buf_limit,
                  BUCKET_DISTRIBUTE_BATCH, f"_s{idx}"))))
    for _, p in procs:
        p.start()
    for _, p in procs:
        p.join()
    for side, p in procs:
        if p.exitcode != 0:
            raise RuntimeError(
                f"{side} distribution failed (exit {p.exitcode})")

    # Bucket i's inputs = its shard files from every distribution process
    fw_bucket_inputs = [
        [os.path.join(bucket_dir, f"fw_s{s}_b{i:04d}.txt")
         for s in range(len(fw_shards))]
        for i in range(num_buckets)
    ]
    bw_bucket_inputs = [
        [os.path.join(bucket_dir, f"bw_s{s}_b{i:04d}.txt")
         for s in range(len(bw_shards))]
        for i in range(num_buckets)
    ]

    # Sort each bucket (parallel)
    fw_sorted: list[str] = []
    bw_sorted: list[str] = []
    all_buckets = fw_bucket_inputs + bw_bucket_inputs
    is_fw = [True] * len(fw_bucket_inputs) + [False] * len(bw_bucket_inputs)

    with Pool(processes=max_concurrent) as pool:
        futs = [pool.apply_async(_sort_bucket, (bps, sort_mem_mb))
                for bps in all_buckets]
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
    """Heuristic: >100MB and <=10 newlines in head/mid/tail samples.

    Samples 4MB at the head, middle and tail instead of scanning the
    whole file, so a 5GB single-line file is not read twice.
    """
    size = os.path.getsize(filepath)
    if size < SINGLE_LINE_MIN_BYTES:
        return False
    chunk = SINGLE_LINE_SAMPLE_BYTES
    offsets = [0, max(0, size // 2 - chunk // 2), max(0, size - chunk)]
    count = 0
    with open(filepath, "rb") as f:
        for off in offsets:
            f.seek(off)
            count += f.read(chunk).count(b"\n")
            if count > SINGLE_LINE_MAX_NEWLINES:
                return False
    return True


# ---- Checkpoint helpers (resume via --work-dir) ----

_CHECKPOINT_STAGES = ("ngrams", "ngrams_finalized", "sorted", "entropy", "merged")
_META_FILE = "run_meta.json"
_NGRAM_MANIFEST = "ngram_manifest.txt"
_BUCKETS_MANIFEST = "buckets_manifest.txt"
_MERGED_FILE = "merged.tsv"


def _ckpt_path(run_dir: str, stage: str) -> str:
    return os.path.join(run_dir, f".{stage}.done")


def _ckpt_done(run_dir: str, stage: str) -> bool:
    return os.path.exists(_ckpt_path(run_dir, stage))


def _ckpt_mark(run_dir: str, stage: str) -> None:
    open(_ckpt_path(run_dir, stage), "w").close()


def _ckpt_unmark(run_dir: str, stage: str) -> None:
    try:
        os.remove(_ckpt_path(run_dir, stage))
    except OSError:
        pass


def _ckpt_clear(run_dir: str) -> None:
    """Remove all checkpoint markers, manifests and known intermediates."""
    names = [f".{s}.done" for s in _CHECKPOINT_STAGES]
    names += [_META_FILE, _MERGED_FILE,
              _NGRAM_MANIFEST + ".fw", _NGRAM_MANIFEST + ".bw",
              _BUCKETS_MANIFEST + ".fw", _BUCKETS_MANIFEST + ".bw",
              "ngram_forward.txt", "ngram_backward.txt",
              "ngram_forward_sorted.txt", "ngram_backward_sorted.txt",
              "right_entropy.txt", "left_entropy_unsorted.txt",
              "left_entropy.txt", "freq.trie"]
    for name in names:
        try:
            os.remove(os.path.join(run_dir, name))
        except OSError:
            pass
    for entry in os.scandir(run_dir):
        if entry.is_dir(follow_symlinks=False) and entry.name.startswith(
                ("dict_build_ngrams_", "dict_build_buckets_")):
            shutil.rmtree(entry.path, ignore_errors=True)


def _ckpt_read_meta(run_dir: str) -> dict | None:
    try:
        with open(os.path.join(run_dir, _META_FILE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _ckpt_write_meta(run_dir: str, meta: dict) -> None:
    with open(os.path.join(run_dir, _META_FILE), "w") as f:
        json.dump(meta, f)


def _write_list_file(path: str, items: list[str]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")


def _read_list_file(path: str) -> list[str] | None:
    """Read a path list; None if the manifest or any listed file is missing."""
    try:
        with open(path) as f:
            items = [line.strip() for line in f if line.strip()]
    except OSError:
        return None
    if not items or not all(os.path.exists(p) for p in items):
        return None
    return items


def _read_ngram_manifest(run_dir: str) -> tuple[list[str], list[str]] | None:
    fw = _read_list_file(os.path.join(run_dir, _NGRAM_MANIFEST + ".fw"))
    bw = _read_list_file(os.path.join(run_dir, _NGRAM_MANIFEST + ".bw"))
    if fw is None or bw is None:
        return None
    return fw, bw


def _write_ngram_manifest(run_dir: str, fw: list[str], bw: list[str]) -> None:
    _write_list_file(os.path.join(run_dir, _NGRAM_MANIFEST + ".fw"), fw)
    _write_list_file(os.path.join(run_dir, _NGRAM_MANIFEST + ".bw"), bw)


def _read_buckets_manifest(run_dir: str) -> tuple[list[str], list[str]] | None:
    fw = _read_list_file(os.path.join(run_dir, _BUCKETS_MANIFEST + ".fw"))
    bw = _read_list_file(os.path.join(run_dir, _BUCKETS_MANIFEST + ".bw"))
    if fw is None or bw is None:
        return None
    return fw, bw


def _write_buckets_manifest(run_dir: str, fw: list[str], bw: list[str]) -> None:
    _write_list_file(os.path.join(run_dir, _BUCKETS_MANIFEST + ".fw"), fw)
    _write_list_file(os.path.join(run_dir, _BUCKETS_MANIFEST + ".bw"), bw)


def run_pipeline(
    input_path: str,
    output_path: str | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    mem_mb: int = DEFAULT_MEM_MB,
    workers: int = WORKERS,
    min_freq: int = MIN_FREQ,
    pos_prob_path: str | None = None,
    temp_dir: str | None = None,
    work_dir: str | None = None,
    force: bool = False,
    pmi_threshold: float = PMI_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    pos_threshold: float = POS_PROB_THRESHOLD,
) -> str:
    """Run the full word-extraction pipeline.

    Args:
        temp_dir: Parent directory for a temporary working directory
            (removed when the run finishes, successfully or not).
        work_dir: Persistent working directory enabling checkpointed
            resume. Stage markers (.done files) let an interrupted run
            restart from the last completed stage. Intermediate files
            are kept on failure and cleaned on success.
        force: Ignore existing checkpoints in work_dir and start over.
        pmi_threshold: Minimum PMI for a candidate word.
        entropy_threshold: Minimum min(left, right) entropy.
        pos_threshold: Minimum position formation probability.

    Note:
        On platforms using the "spawn" multiprocessing start method
        (macOS/Windows), callers must guard entry with
        ``if __name__ == "__main__":``.
    """
    txt_files = _collect_files(input_path)
    total_size = sum(os.path.getsize(fp) for fp in txt_files)
    logger.info("Found %d text file(s) to process (%.1f GB)",
                len(txt_files), total_size / 1e9)
    if total_size > 2 * 1024 * 1024 * 1024:
        logger.warning("Input > 2 GB, processing may take a long time. "
                       "Consider --min-freq higher or sample first.")

    if output_path is None:
        output_path = _make_output_path(input_path)

    checkpointing = work_dir is not None
    if checkpointing:
        if temp_dir is not None:
            logger.warning("--temp-dir is ignored when --work-dir is set; "
                           "intermediate files go to the work directory.")
        os.makedirs(work_dir, exist_ok=True)
        run_dir = work_dir
        if force:
            _ckpt_clear(run_dir)
    else:
        temp_parent = temp_dir if temp_dir is not None else tempfile.gettempdir()
        run_dir = tempfile.mkdtemp(prefix="dict_build_", dir=temp_parent)

    estimated_intermediate = total_size * INTERMEDIATE_SIZE_FACTOR
    free_bytes = shutil.disk_usage(run_dir).free
    if free_bytes < estimated_intermediate:
        raise RuntimeError(
            f"Insufficient disk space in {run_dir}: "
            f"{free_bytes / 1e9:.1f} GB free, "
            f"~{estimated_intermediate / 1e9:.1f} GB needed for intermediate "
            f"data. Use --temp-dir/--work-dir to point at a larger disk, "
            f"or raise --min-freq / sample the input first."
        )

    ngram_fw_path = os.path.join(run_dir, "ngram_forward.txt")
    ngram_bw_path = os.path.join(run_dir, "ngram_backward.txt")
    ngram_fw_sorted = os.path.join(run_dir, "ngram_forward_sorted.txt")
    ngram_bw_sorted = os.path.join(run_dir, "ngram_backward_sorted.txt")
    right_entropy_file = os.path.join(run_dir, "right_entropy.txt")
    left_entropy_unsorted = os.path.join(run_dir, "left_entropy_unsorted.txt")
    left_entropy_file = os.path.join(run_dir, "left_entropy.txt")
    merged_file = os.path.join(run_dir, _MERGED_FILE)
    trie_file = os.path.join(run_dir, "freq.trie")

    signature = {
        "input": os.path.abspath(input_path),
        "files": sorted([os.path.abspath(f), os.path.getsize(f)]
                        for f in txt_files),
        "max_len": max_len,
        "min_freq": min_freq,
        "entropy_threshold": entropy_threshold,
    }
    if checkpointing:
        old_meta = _ckpt_read_meta(run_dir)
        if old_meta is not None and any(
                old_meta.get(k) != v for k, v in signature.items()):
            logger.warning("Input or parameters changed since the last run; "
                           "discarding checkpoints and starting over.")
            _ckpt_clear(run_dir)

    # Stage indices: 0 ngrams, 1 route, 2 sort (non-bucket only),
    # 3 entropy, 4 merge, 5 extract
    stage = 0
    use_buckets: bool | None = None
    fw_tmp_paths: list[str] = []
    bw_tmp_paths: list[str] = []
    fw_sorted: list[str] = []
    bw_sorted: list[str] = []

    if checkpointing:
        meta = _ckpt_read_meta(run_dir) or {}
        mb = meta.get("bucket_mode")
        if _ckpt_done(run_dir, "merged") and os.path.exists(merged_file):
            stage = 5
        elif (_ckpt_done(run_dir, "entropy")
              and os.path.exists(right_entropy_file)
              and os.path.exists(left_entropy_file)):
            stage = 4
        elif _ckpt_done(run_dir, "sorted") and mb is not None and (
                (mb and _read_buckets_manifest(run_dir) is not None)
                or (not mb and os.path.exists(ngram_fw_sorted)
                    and os.path.exists(ngram_bw_sorted))):
            stage = 3
            use_buckets = bool(mb)
        elif (_ckpt_done(run_dir, "ngrams_finalized")
              and os.path.exists(ngram_fw_path)
              and os.path.exists(ngram_bw_path)):
            stage = 2
            use_buckets = False
        elif (_ckpt_done(run_dir, "ngrams")
              # parallel mode leaves a manifest; single-line-only runs
              # leave data directly in the ngram files
              and (_read_ngram_manifest(run_dir) is not None
                   or os.path.exists(ngram_fw_path))):
            stage = 1
        if stage:
            logger.info("Resuming from checkpoint (stage %d)", stage)

    success = False
    try:
        if stage == 0:
            logger.info("Stage 1-2: Preprocessing + N-gram generation "
                        "(workers=%d)...", workers)
            if checkpointing:
                _ckpt_clear(run_dir)
            huge_files = [f for f in txt_files if _is_single_line_file(f)]
            normal_files = [f for f in txt_files if f not in huge_files]

            if normal_files:
                fw_tmp_paths, bw_tmp_paths, _ = _generate_ngrams_parallel(
                    normal_files, ngram_fw_path, ngram_bw_path,
                    max_len, workers,
                )
            for hf in huge_files:
                logger.info("  Single-line mode: %s (%.1f GB)",
                            os.path.basename(hf),
                            os.path.getsize(hf) / 1e9)
                _generate_ngrams_single_line(
                    hf, ngram_fw_path, ngram_bw_path, max_len,
                )
            if checkpointing:
                _write_ngram_manifest(run_dir, fw_tmp_paths, bw_tmp_paths)
                _ckpt_mark(run_dir, "ngrams")
            stage = 1

        if stage == 1:
            if checkpointing and not fw_tmp_paths and not bw_tmp_paths:
                restored = _read_ngram_manifest(run_dir)
                if restored is not None:
                    fw_tmp_paths, bw_tmp_paths = restored

            fw_size = sum(os.path.getsize(tp) for tp in fw_tmp_paths)
            bw_size = sum(os.path.getsize(tp) for tp in bw_tmp_paths)
            if os.path.exists(ngram_fw_path):
                fw_size += os.path.getsize(ngram_fw_path)
                bw_size += os.path.getsize(ngram_bw_path)
            total_ngram = fw_size + bw_size
            logger.info("  Forward n-grams: %.2f GB", fw_size / 1e9)
            logger.info("  Backward n-grams: %.2f GB", bw_size / 1e9)

            if total_ngram == 0:
                logger.info("  No valid Chinese text found in input.")
                _write_output([], output_path)
                success = True
                return output_path

            huge_files = [f for f in txt_files if _is_single_line_file(f)]
            use_buckets = bool(total_ngram >= BUCKET_SORT_MIN_BYTES
                               and not huge_files
                               and fw_tmp_paths)
            if checkpointing:
                _ckpt_write_meta(run_dir,
                                 {**signature, "bucket_mode": use_buckets})

            if use_buckets:
                num_buckets, max_conc, sort_mem = _calc_bucket_params(
                    total_ngram, workers, mem_mb,
                )
                logger.info("Stage 3: Bucket-sorting n-grams "
                            "(%d buckets, %d concurrent)...",
                            num_buckets, max_conc)
                # Distribution consumes the temp files: past this point
                # the ngrams checkpoint is no longer recoverable.
                _ckpt_unmark(run_dir, "ngrams")
                fw_sorted, bw_sorted = _distribute_and_sort_ngrams(
                    run_dir, fw_tmp_paths, bw_tmp_paths,
                    num_buckets, max_conc, sort_mem, mem_mb, workers,
                )
                if checkpointing:
                    _write_buckets_manifest(run_dir, fw_sorted, bw_sorted)
                    _ckpt_mark(run_dir, "sorted")
                stage = 3
            else:
                # Old path: concat temp files → single sort → entropy
                _ckpt_unmark(run_dir, "ngrams")
                if fw_tmp_paths:
                    _concat_temp_files(fw_tmp_paths, ngram_fw_path)
                    _concat_temp_files(bw_tmp_paths, ngram_bw_path)
                if checkpointing:
                    _ckpt_mark(run_dir, "ngrams_finalized")
                stage = 2

        if stage == 2:
            logger.info("Stage 3: Sorting n-grams (LC_ALL=C for speed)...")
            sort_env = {**os.environ, "LC_ALL": "C"}
            # fw and bw sorts run concurrently: split the memory/thread
            # budget so peak usage stays within --mem
            sort_cmd = _sort_command(max(1, mem_mb // 2),
                                     max(1, workers // 2),
                                     run_dir)
            t0 = time.time()
            p1 = subprocess.Popen([
                *sort_cmd, "-o", ngram_fw_sorted, ngram_fw_path,
            ], stderr=subprocess.PIPE, text=True, env=sort_env)
            p2 = subprocess.Popen([
                *sort_cmd, "-o", ngram_bw_sorted, ngram_bw_path,
            ], stderr=subprocess.PIPE, text=True, env=sort_env)
            _, stderr1 = p1.communicate()
            _, stderr2 = p2.communicate()
            if p1.returncode != 0:
                raise RuntimeError(f"Forward sort failed: {stderr1}")
            if p2.returncode != 0:
                raise RuntimeError(f"Backward sort failed: {stderr2}")
            for fp in (ngram_fw_path, ngram_bw_path):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            logger.info("  Sorting complete (%.0fs)", time.time() - t0)
            if checkpointing:
                _ckpt_mark(run_dir, "sorted")
            stage = 3

        if stage == 3:
            if use_buckets is None:
                use_buckets = False
            if use_buckets and not fw_sorted and checkpointing:
                restored = _read_buckets_manifest(run_dir)
                if restored is not None:
                    fw_sorted, bw_sorted = restored

            logger.info("Stage 4a: Computing right entropy (min_freq=%d)...",
                        min_freq)
            if use_buckets:
                r_count = _compute_entropy_from_sorted_list(
                    fw_sorted, right_entropy_file, min_freq, direct=True,
                    workers=workers,
                )
                # k-way merged per-bucket output is already globally sorted
            elif os.path.getsize(ngram_fw_sorted) > PARALLEL_ENTROPY_MIN_BYTES:
                r_count = _write_entropy_from_ngram_parallel(
                    ngram_fw_sorted, right_entropy_file, min_freq,
                    direct=True, workers=workers,
                )
            else:
                r_count = _write_entropy_from_ngram(
                    ngram_fw_sorted, right_entropy_file, min_freq, direct=True,
                )
            logger.info("  Right: %d unique words", r_count)

            logger.info("Stage 4b: Computing left entropy (min_freq=%d)...",
                        min_freq)
            if use_buckets:
                l_count = _compute_entropy_from_sorted_list(
                    bw_sorted, left_entropy_unsorted, min_freq, direct=False,
                    workers=workers,
                )
            elif os.path.getsize(ngram_bw_sorted) > PARALLEL_ENTROPY_MIN_BYTES:
                l_count = _write_entropy_from_ngram_parallel(
                    ngram_bw_sorted, left_entropy_unsorted, min_freq,
                    direct=False, workers=workers,
                )
            else:
                l_count = _write_entropy_from_ngram(
                    ngram_bw_sorted, left_entropy_unsorted, min_freq,
                    direct=False,
                )
            logger.info("  Left: %d unique words", l_count)

            logger.info("Stage 4c: Sorting left entropy file...")
            sort_file_inplace(left_entropy_unsorted)
            os.replace(left_entropy_unsorted, left_entropy_file)
            if checkpointing:
                _ckpt_mark(run_dir, "entropy")
            stage = 4

        if stage == 4:
            logger.info("Stage 5: Merging left and right entropy...")
            merged_count = 0
            with open(merged_file, "w", encoding="utf-8") as mf:
                for word, freq, ent in iter_merge_entropy_files_sorted(
                        right_entropy_file, left_entropy_file,
                        min_entropy=entropy_threshold):
                    mf.write(f"{word}\t{freq}\t{ent:.6f}\n")
                    merged_count += 1
            logger.info("  Merged: %d candidates (entropy >= %s)",
                        merged_count, entropy_threshold)
            if checkpointing:
                _ckpt_mark(run_dir, "merged")
            stage = 5

        if stage == 5:
            merged_iter = read_entropy_from_file(merged_file)
            first = next(merged_iter, None)
            if first is None:
                logger.info("  No candidates pass entropy threshold.")
                _write_output([], output_path)
                success = True
                return output_path
            merged_iter = itertools.chain([first], merged_iter)

            logger.info("Stage 6a: Building frequency trie "
                        "(stream from file)...")
            trie, total_single = build_and_mmap_trie(
                right_entropy_file, trie_file, min_freq=min_freq
            )
            logger.info("  Trie ready. Total single-char freq: %d",
                        total_single)

            logger.info("Stage 6b: Computing PMI and filtering...")
            pos_prob = load_pos_prob(pos_prob_path)
            results_raw = extract_words(
                merged_iter, pos_prob, trie, total_single,
                pmi_threshold=pmi_threshold,
                pos_threshold=pos_threshold,
            )
            logger.info("  Candidates after PMI/filter: %d", len(results_raw))
            del trie  # release the mmap before work_dir cleanup (Windows)

            from .pos_tag import tag_word

            results = [
                (w, f, p, e, pp, tag_word(w))
                for w, f, p, e, pp in results_raw
            ]

            _write_output(results, output_path)
            logger.info("Output: %s", output_path)
            success = True
            return output_path

        raise RuntimeError(f"Invalid pipeline stage: {stage}")

    finally:
        if not checkpointing:
            shutil.rmtree(run_dir, ignore_errors=True)
        elif success:
            _ckpt_clear(run_dir)


# ---- N-gram generation: normal files (parallel) ----

def _generate_ngrams_parallel(
    txt_files: list[str],
    ngram_fw_path: str,
    ngram_bw_path: str,
    max_len: int,
    workers: int,
) -> tuple[list[str], list[str], str]:
    """Generate n-grams with multiprocessing, writing directly to temp files.

    Returns (fw_tmp_paths, bw_tmp_paths, tmp_dir). tmp_dir lives inside
    the caller's temp_dir and is removed with it.
    """
    ngram_tmp_dir = tempfile.mkdtemp(
        prefix="dict_build_ngrams_", dir=os.path.dirname(ngram_fw_path),
    )

    # Estimate total chunks from file sizes (avoid reading every file twice)
    total_chunks = 0
    file_encodings: dict[str, str] = {}
    for fp in txt_files:
        enc = _detect_file_encoding(fp)
        file_encodings[fp] = enc
        est_lines = max(1, os.path.getsize(fp) // AVG_LINE_BYTES)
        total_chunks += (est_lines + CHUNK_LINES - 1) // CHUNK_LINES

    max_pending = workers * PENDING_TASK_FACTOR
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

    No n-gram lists are accumulated in memory; writes are batched to
    keep interpreter overhead low. Worker memory = chunk text + write
    buffers (~ few MB).
    """
    with open(fw_path, "w", encoding="utf-8") as fw, \
         open(bw_path, "w", encoding="utf-8") as bw:
        _write_ngrams_batched(
            (sent for line in lines for sent in preprocess_line(line)),
            max_len, fw, bw,
        )


_NGRAM_WRITE_BATCH = NGRAM_WRITE_BATCH


def _write_ngrams_batched(sentences, max_len: int, fw, bw) -> None:
    """Generate n-grams for sentences and write with batched flushes."""
    fw_buf: list[str] = []
    bw_buf: list[str] = []
    fw_append = fw_buf.append
    bw_append = bw_buf.append
    count = 0
    for sent in sentences:
        for ng in generate_ngrams(sent, max_len):
            fw_append(ng)
            fw_append("\n")
        for ng in generate_reverse_ngrams(sent, max_len):
            bw_append(ng)
            bw_append("\n")
        count += 1
        if count >= 256:
            count = 0
            if len(fw_buf) >= _NGRAM_WRITE_BATCH:
                fw.write("".join(fw_buf))
                fw_buf.clear()
            if len(bw_buf) >= _NGRAM_WRITE_BATCH:
                bw.write("".join(bw_buf))
                bw_buf.clear()
    if fw_buf:
        fw.write("".join(fw_buf))
    if bw_buf:
        bw.write("".join(bw_buf))


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
    bytes_per_char = 2 if encoding in ("gb18030", "gbk", "big5") else 3
    fw_out = open(ngram_fw_path, "a", encoding="utf-8")
    bw_out = open(ngram_bw_path, "a", encoding="utf-8")
    try:
        total_chars = os.path.getsize(filepath) // bytes_per_char
        total_chunks = (total_chars + SINGLE_LINE_CHAR_CHUNK - 1) // SINGLE_LINE_CHAR_CHUNK
        pbar = tqdm.tqdm(total=total_chunks, desc="  S-line", unit="chunk")

        carryover = b""
        pending = ""
        with open(filepath, "rb") as fin:
            while True:
                data = fin.read(BYTE_BUF)
                if not data:
                    if carryover:
                        pending = _process_text_with_carry(
                            carryover.decode(encoding, errors="replace"),
                            max_len, fw_out, bw_out, pending,
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
                    pending = _process_text_with_carry(
                        sub, max_len, fw_out, bw_out, pending,
                    )
                    pbar.update(1)

        # trailing token held back at EOF
        if len(pending) >= 2:
            _write_ngrams_batched(iter([SENTINEL + pending + SENTINEL]),
                                  max_len, fw_out, bw_out)
        pbar.close()
    finally:
        fw_out.close()
        bw_out.close()


def _process_text_with_carry(
    text: str,
    max_len: int,
    fw_out,
    bw_out,
    pending: str,
) -> str:
    """Process one text chunk; return the trailing fragment to carry over.

    A Chinese token reaching the chunk end is held back and prepended to
    the next chunk, so words spanning chunk boundaries are not lost.
    (Holding back an actually-complete token is harmless: it is re-emitted
    identically when the next chunk starts with punctuation, or flushed
    at EOF.)
    """
    text = pending + text
    tokens = list(iter_chinese_tokens(text))
    pending = ""
    if tokens:
        tok, _s, e = tokens[-1]
        if e == len(text):
            pending = tok
            tokens.pop()
    sents = (SENTINEL + t + SENTINEL for t, _, _ in tokens if len(t) >= 2)
    _write_ngrams_batched(sents, max_len, fw_out, bw_out)
    return pending


# ---- Entropy helpers ----

def _split_sorted_ngram_by_chars(
    ngram_file: str,
    output_dir: str,
    target_chars: int,
) -> list[str]:
    """Split sorted n-gram file at first-character boundaries.

    Splitting at a character boundary is safe — all lines for a given
    first character stay together. A new chunk starts once the current
    one reaches target_chars, so hot first characters no longer produce
    oversized chunks (byte-balanced instead of count-balanced).
    """
    split_paths: list[str] = []
    current_fh = None
    chars_in_current = 0
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
                if current_fh is None or chars_in_current >= target_chars:
                    if current_fh:
                        current_fh.close()
                    path = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.txt")
                    split_paths.append(path)
                    current_fh = open(path, "w", encoding="utf-8")
                    chunk_idx += 1
                    chars_in_current = 0
            if current_fh:
                current_fh.write(line)
                chars_in_current += len(line)
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
) -> int:
    """Split n-gram file at char boundaries (~3 chunks per worker, byte
    balanced), compute entropy in parallel."""
    split_dir = tempfile.mkdtemp(prefix="dict_build_entropy_",
                                 dir=os.path.dirname(output_file))
    try:
        size = os.path.getsize(ngram_file)
        # chars ≈ bytes / 3 for CJK-heavy UTF-8 content
        target_chars = max(ENTROPY_SPLIT_MIN_BYTES // 3,
                           size // 3 // max(1, workers * 3))
        split_paths = _split_sorted_ngram_by_chars(
            ngram_file, split_dir, target_chars,
        )
        logger.info("    Split into %d first-char chunks", len(split_paths))

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
    """Concatenate multiple files into one output.

    Appends when output_path already exists (single-line mode may have
    written n-grams there before the parallel temp files are merged).
    """
    mode = "ab" if os.path.exists(output_path) else "wb"
    with open(output_path, mode) as out:
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
    """Compute entropy on each sorted bucket, combine, return total.

    direct=True (right entropy): each bucket's output is sorted by word,
    so results are combined with a k-way merge — no full re-sort needed.
    direct=False (left entropy): words are reversed after computation,
    breaking order, so results are concatenated and sorted later.
    """
    total = 0
    tmp_paths: list[str] = []
    with Pool(processes=min(workers, len(sorted_paths))) as pool:
        desc = "    Entropy" if direct else "    Entropy(L)"
        pbar = tqdm.tqdm(total=len(sorted_paths), desc=desc, unit="file")
        futs = [
            pool.apply_async(_process_entropy_split, (sp, min_freq, direct))
            for sp in sorted_paths
        ]
        for fut in futs:
            try:
                count, tmp_path = fut.get()
            except Exception:
                pool.terminate()
                raise
            total += count
            tmp_paths.append(tmp_path)
            pbar.update(1)
        pbar.close()

    try:
        if direct:
            _merge_sorted_entropy_files(tmp_paths, output_file)
        else:
            with open(output_file, "w", encoding="utf-8") as out:
                for tp in tmp_paths:
                    with open(tp, "r", encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)
    finally:
        for tp in tmp_paths:
            try:
                os.remove(tp)
            except OSError:
                pass
    return total


def _merge_sorted_entropy_files(paths: list[str], output_file: str) -> None:
    """K-way merge of per-bucket entropy files (each sorted by word).

    Words are unique across buckets (hash partition), and str comparison
    matches the LC_ALL=C byte order because all files are UTF-8.
    """
    iterators = [read_entropy_from_file(p) for p in paths]
    with open(output_file, "w", encoding="utf-8") as out:
        for word, freq, entropy in heapq.merge(*iterators,
                                               key=lambda t: t[0]):
            out.write(f"{word}\t{freq}\t{entropy:.6f}\n")


# ---- Output ----

def _write_output(results: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("word\tfreq\tpmi\tentropy\tpos_prob\tpos\n")
        for w, fr, p, e, pp, pos in results:
            f.write(f"{w}\t{fr}\t{p:.6f}\t{e:.6f}\t{pp:.6f}\t{pos}\n")
