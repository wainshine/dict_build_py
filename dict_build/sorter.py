"""External merge sort using system sort(1) command.

Uses Apple sort (BSD) or GNU sort - both support -S and --parallel options.
Falls back to pure Python external merge sort if unavailable.
"""

import os
import subprocess
import tempfile
import shutil
from typing import Iterator


def sort_file(
    input_path: str,
    output_path: str,
    max_memory_mb: int = 4096,
    delete_input: bool = False,
    parallel: int | None = None,
) -> None:
    """Sort a text file using system sort command.

    Args:
        input_path: Path to unsorted file (one record per line).
        output_path: Path for sorted output.
        max_memory_mb: Maximum memory for sort buffer.
        delete_input: Whether to delete the input file after sorting.
        parallel: Number of parallel sort threads.
    """
    if shutil.which("sort") is not None:
        if parallel is None:
            parallel = os.cpu_count() or 4
        mem_str = f"{max_memory_mb}M"
        cmd = [
            "sort",
            "-S", mem_str,
            f"--parallel={parallel}",
            "-o", output_path,
            input_path,
        ]
        subprocess.run(cmd, check=True)
    else:
        _sort_with_python(input_path, output_path, max_memory_mb)

    if delete_input:
        try:
            os.remove(input_path)
        except OSError:
            pass


def _sort_with_python(
    input_path: str, output_path: str, max_memory_mb: int
) -> None:
    """Pure Python external merge sort (fallback)."""
    import heapq

    chunk_size_bytes = max_memory_mb * 1024 * 1024
    temp_dir = os.path.dirname(output_path) or "."
    temp_files: list[str] = []

    try:
        chunk: list[str] = []
        chunk_bytes = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line_bytes = len(line.encode("utf-8"))
                if chunk_bytes + line_bytes > chunk_size_bytes and chunk:
                    _dump_sorted_chunk(chunk, temp_dir, temp_files)
                    chunk = []
                    chunk_bytes = 0
                chunk.append(line)
                chunk_bytes += line_bytes
        if chunk:
            _dump_sorted_chunk(chunk, temp_dir, temp_files)

        if not temp_files:
            open(output_path, "w").close()
        elif len(temp_files) == 1:
            os.rename(temp_files[0], output_path)
            temp_files.clear()
        else:
            file_iters = []
            for tf in temp_files:
                file_iters.append(_file_iter(tf))
            with open(output_path, "w", encoding="utf-8") as out:
                for line in heapq.merge(*file_iters):
                    out.write(line)
    finally:
        for tf in temp_files:
            try:
                os.remove(tf)
            except OSError:
                pass


def _dump_sorted_chunk(
    chunk: list[str], temp_dir: str, temp_files: list[str]
) -> None:
    chunk.sort()
    fd, path = tempfile.mkstemp(suffix=".sort", dir=temp_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.writelines(chunk)
    except Exception:
        os.close(fd)
        raise
    temp_files.append(path)


def _file_iter(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        yield from f
