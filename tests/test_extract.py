"""Tests for dict_build."""
import io
import os
import shutil
import tempfile

import pytest

requires_sort = pytest.mark.skipif(
    shutil.which("sort") is None, reason="system sort not available",
)

from dict_build.preprocess import preprocess_line, is_chinese, all_chinese
from dict_build.ngram import generate_ngrams, generate_reverse_ngrams
from dict_build.entropy import (
    compute_entropy_from_sorted,
    compute_entropy_from_sorted_left,
    write_entropy_to_file,
    read_entropy_from_file,
    sort_file_inplace,
    merge_entropy_files_sorted,
)
from dict_build.pmi import compute_pmi, build_and_mmap_trie, extract_words
from dict_build.pos_prob import load_pos_prob
from dict_build.pos_tag import tag_word
from dict_build.pipeline import (
    _detect_file_encoding,
    _hash_line,
    _calc_bucket_params,
    _concat_temp_files,
    _compute_entropy_from_sorted_list,
    _distribute_batches,
    _sort_bucket,
    _read_chunks_by_lines,
    _process_chunk_direct,
    run_pipeline,
)


# ============================================================
# helpers
# ============================================================

def _make_text_file(tmp: str, name: str, content: str, encoding: str = "utf-8") -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding=encoding) as f:
        f.write(content)
    return path


def _make_entropy_file(tmp: str, name: str, rows) -> str:
    path = os.path.join(tmp, name)
    write_entropy_to_file(iter(rows), path)
    return path


# ============================================================
# preprocess
# ============================================================

def test_preprocess_line():
    sentences = list(preprocess_line("Hello 你好世界！这是一个测试。"))
    assert len(sentences) >= 1
    for s in sentences:
        assert s.startswith("$")
        assert s.endswith("$")
        assert all(is_chinese(c) or c == "$" for c in s)


def test_preprocess_line_empty():
    assert list(preprocess_line("")) == []


def test_preprocess_line_no_chinese():
    assert list(preprocess_line("Hello world 123!")) == []


def test_preprocess_line_too_short():
    assert list(preprocess_line("哈")) == []


def test_is_chinese():
    assert is_chinese("世")
    assert not is_chinese("a")


def test_all_chinese():
    assert all_chinese("你好世界")
    assert not all_chinese("hello世界")


# ============================================================
# ngram
# ============================================================

def test_generate_ngrams():
    sent = "$你好世界$"
    ngrams = list(generate_ngrams(sent, max_len=3))
    assert len(ngrams) > 0
    words = {ng.split("\t")[0] for ng in ngrams}
    assert "你" in words
    assert "你好" in words


def test_generate_ngrams_format():
    sent = "$你好世界$"
    for ng in generate_ngrams(sent, max_len=4):
        assert "\t" in ng
        word, next_char = ng.split("\t")
        assert len(next_char) == 1


def test_generate_ngrams_max_len_respected():
    sent = "$你好世界$"
    ngrams = list(generate_ngrams(sent, max_len=2))
    for ng in ngrams:
        word = ng.split("\t")[0]
        assert len(word) <= 2


def test_generate_reverse_ngrams():
    sent = "$你好世界$"
    ngrams = list(generate_reverse_ngrams(sent, max_len=3))
    assert len(ngrams) > 0


def test_generate_reverse_ngrams_no_sentinel():
    sent = "$你好世界$"
    for ng in generate_reverse_ngrams(sent, max_len=4):
        word = ng.split("\t")[0]
        assert "$" not in word


# ============================================================
# entropy
# ============================================================

def test_entropy_computation():
    results = list(compute_entropy_from_sorted(iter([
        "你\t好\n", "你\t好\n", "你\t们\n",
        "你好\t世\n", "你好\t界\n",
    ])))
    words = {r[0] for r in results}
    assert "你" in words
    assert "你好" in words
    for _, freq, entropy in results:
        assert freq > 0
        assert entropy >= 0.0


def test_entropy_freq_equals_suffix_count():
    lines = ["吃\t饭\n", "吃\t饭\n", "吃\t面\n"]
    results = list(compute_entropy_from_sorted(iter(lines)))
    assert results[0][0] == "吃"
    assert results[0][1] == 3


def test_entropy_min_freq_filter():
    results = list(compute_entropy_from_sorted(
        iter(["你\t好\n", "你好\t世\n", "你好\t界\n"]), min_freq=3))
    assert {r[0] for r in results} == set()


def test_entropy_zero_for_single_suffix():
    lines = ["词\tA\n", "词\tA\n", "词\tA\n"]
    results = list(compute_entropy_from_sorted(iter(lines)))
    assert results[0][2] == 0.0


def test_compute_entropy_from_sorted_left():
    lines = ["界\t世\n", "世\t好\n"]
    results = list(compute_entropy_from_sorted_left(iter(lines)))
    words = {r[0] for r in results}
    assert "界" in words or "世" in words


def test_write_read_entropy_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ent.txt")
        data = [("天下", 100, 2.5), ("天地", 80, 3.1), ("之", 999, 1.0)]
        write_entropy_to_file(iter(data), path)
        assert list(read_entropy_from_file(path)) == data


def test_read_entropy_skips_malformed():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ent.txt")
        with open(path, "w") as f:
            f.write("word_only\n")
            f.write("too\tmany\tcols\textra\n")
            f.write("good\t42\t1.5\n")
        results = list(read_entropy_from_file(path))
        assert len(results) == 1
        assert results[0] == ("good", 42, 1.5)


def test_sort_file_inplace():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unsorted.txt")
        with open(path, "w") as f:
            f.write("c\t1\t0.1\n")
            f.write("a\t3\t0.3\n")
            f.write("b\t2\t0.2\n")
        sort_file_inplace(path)
        results = list(read_entropy_from_file(path))
        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == "c"


def test_merge_entropy_files_sorted_basic():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt",
                                [("天下", 100, 3.0), ("天地", 80, 2.5)])
        l = _make_entropy_file(tmp, "l.txt",
                                [("天下", 100, 4.0), ("天地", 80, 2.0)])
        merged = merge_entropy_files_sorted(r, l)
        assert merged == [("天下", 100, 3.0), ("天地", 80, 2.0)]


def test_merge_entropy_files_sorted_min_entropy_filter():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [("天", 100, 3.0), ("地", 60, 1.0)])
        l = _make_entropy_file(tmp, "l.txt", [("天", 100, 4.0), ("地", 60, 0.8)])
        merged = merge_entropy_files_sorted(r, l, min_entropy=2.0)
        assert merged == [("天", 100, 3.0)]


def test_merge_entropy_empty():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [])
        l = _make_entropy_file(tmp, "l.txt", [])
        assert merge_entropy_files_sorted(r, l) == []


def test_merge_entropy_non_overlapping():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [("天下", 100, 3.0)])
        l = _make_entropy_file(tmp, "l.txt", [("天地", 80, 4.0)])
        assert merge_entropy_files_sorted(r, l) == []


# ============================================================
# encoding detection
# ============================================================

def test_detect_encoding_utf8():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_text_file(tmp, "u8.txt", "天下太平万物安宁\n天地玄黄宇宙洪荒\n" * 50)
        assert _detect_file_encoding(path) == "utf-8"


def test_detect_encoding_gbk():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_text_file(tmp, "gbk.txt", "天下太平万物安宁\n天地玄黄宇宙洪荒\n" * 50,
                                encoding="gbk")
        enc = _detect_file_encoding(path)
        assert enc in ("gb18030", "gbk")


def test_detect_encoding_utf8_short():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_text_file(tmp, "short.txt", "天下太平")
        assert _detect_file_encoding(path) == "utf-8"


def test_detect_encoding_utf8_no_chinese():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_text_file(tmp, "ascii.txt", "hello world\nfoo bar\n" * 20)
        enc = _detect_file_encoding(path)
        assert enc == "utf-8"


# ============================================================
# hash line
# ============================================================

def test_hash_line_same_word_same_hash():
    h1 = _hash_line(b"\xe5\xa4\xa9\xe4\xb8\x8b\t\xe5\xa4\xaa\n")
    h2 = _hash_line(b"\xe5\xa4\xa9\xe4\xb8\x8b\t\xe4\xb8\x8b\n")
    assert h1 == h2  # "天下" hashes to same bucket


def test_hash_line_different_word_different_hash():
    h1 = _hash_line(b"\xe5\xa4\xa9\xe4\xb8\x8b\t\n")  # 天下
    h2 = _hash_line(b"\xe5\xa4\xa9\xe5\x9c\xb0\t\n")  # 天地
    assert h1 != h2


def test_hash_line_modulo_distribution():
    """Check that hash mod B is approximately uniform."""
    words = [f"word{i}".encode() for i in range(1000)]
    buckets = [0] * 8
    for w in words:
        line = w + b"\tX\n"
        buckets[_hash_line(line) % 8] += 1
    # No bucket should be empty with 1000 random-like keys
    assert all(b > 0 for b in buckets)


# ============================================================
# bucket params
# ============================================================

def test_calc_bucket_params_small():
    n, c, m = _calc_bucket_params(500 * 1024 * 1024, 16, 4096)  # 500MB
    assert n == 1  # single bucket


def test_calc_bucket_params_large():
    n, c, m = _calc_bucket_params(10 * 1024 * 1024 * 1024, 16, 4096)  # 10GB
    assert n >= 3  # at least 3 buckets with 4GB target
    assert c <= 16


def test_calc_bucket_params_memory_bound():
    n, c, m = _calc_bucket_params(10 * 1024 * 1024 * 1024, 16, 1024)  # 1GB mem
    assert c <= 2  # limited by mem


# ============================================================
# concat temp files
# ============================================================

def test_concat_temp_files():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f"part_{i}.txt")
            with open(p, "w") as f:
                f.write(f"line{i}a\nline{i}b\n")
            paths.append(p)

        out = os.path.join(tmp, "merged.txt")
        _concat_temp_files(paths, out)

        with open(out) as f:
            lines = f.readlines()
        assert len(lines) == 6
        assert lines[0] == "line0a\n"
        assert not os.path.exists(paths[0])  # source deleted


# ============================================================
# compute entropy from sorted list
# ============================================================

def test_compute_entropy_from_sorted_list():
    with tempfile.TemporaryDirectory() as tmp:
        # Create two sorted n-gram files (same as we'd get from buckets)
        ngrams_0 = "天下\t太\n天下\t下\n"
        ngrams_1 = "天地\t玄\n天地\t黄\n"

        p0 = os.path.join(tmp, "b0.txt")
        p1 = os.path.join(tmp, "b1.txt")
        with open(p0, "w") as f: f.write(ngrams_0)
        with open(p1, "w") as f: f.write(ngrams_1)

        out = os.path.join(tmp, "entropy.txt")
        count = _compute_entropy_from_sorted_list(
            [p0, p1], out, min_freq=1, direct=True, workers=2,
        )
        assert count == 2  # 天下 + 天地
        results = list(read_entropy_from_file(out))
        words = {r[0] for r in results}
        assert "天下" in words
        assert "天地" in words


# ============================================================
# sort bucket
# ============================================================

@requires_sort
def test_sort_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "unsorted.txt")
        with open(path, "w") as f:
            f.write("c\t1\nb\t2\na\t3\n")
        sorted_path = _sort_bucket(path, sort_mem_mb=256)
        with open(sorted_path) as f:
            lines = f.readlines()
        assert lines[0].startswith("a")
        assert lines[-1].startswith("c")
        os.remove(sorted_path)


# ============================================================
# trie / pmi
# ============================================================

def test_build_and_mmap_trie():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n世界\t8\t2.5\n你\t5\t0.0\n好\t3\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        assert total == 8
        assert trie["你好"][0][0] == 10


def test_build_and_mmap_trie_min_freq():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n世界\t2\t2.5\n你\t5\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, _ = build_and_mmap_trie(ef, tf, min_freq=5)
        try:
            trie["世界"]
            assert False
        except KeyError:
            pass


def test_compute_pmi():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n你\t5\t0.0\n好\t3\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        assert compute_pmi("你好", 10, trie, total) > 0.0


def test_compute_pmi_single_char():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你\t5\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        assert compute_pmi("你", 5, trie, total) == 0.0


def test_extract_words_basic():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n你\t5\t0.0\n好\t3\t0.0\n世界\t4\t2.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        pp = {"你": (0.5, 0.3, 0.2), "好": (0.1, 0.4, 0.5),
              "世": (0.4, 0.3, 0.3), "界": (0.1, 0.3, 0.6)}
        merged = [("你好", 10, 3.0), ("世界", 4, 2.0)]
        results = extract_words(merged, pp, trie, total,
                                pmi_threshold=0, pos_threshold=0)
        assert len(results) == 2


def test_extract_words_skips_short():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你\t5\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        results = extract_words([("你", 5, 1.0)], {}, trie, total,
                                pmi_threshold=0, pos_threshold=0)
        assert len(results) == 0


def test_extract_words_empty_pos_prob_skips_filter():
    """Missing pos_prop file (empty dict) must not filter out everything."""
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n你\t5\t0.0\n好\t3\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        results = extract_words([("你好", 10, 3.0)], {}, trie, total,
                                pmi_threshold=0, pos_threshold=0.1)
        assert len(results) == 1
        assert results[0][4] == 0.0


def test_extract_words_pos_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n你\t5\t0.0\n好\t3\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        pp = {"你": (0.1, 0.4, 0.5), "好": (0.5, 0.4, 0.1)}
        merged = [("你好", 10, 3.0)]
        # pp = min(0.1, 0.1) = 0.1, threshold 0.2 → filtered
        assert len(extract_words(merged, pp, trie, total,
                                 pmi_threshold=0, pos_threshold=0.2)) == 0
        assert len(extract_words(merged, pp, trie, total,
                                 pmi_threshold=0, pos_threshold=0.05)) == 1


def test_extract_words_blocklist():
    """Words containing encoding artifacts are filtered."""
    with tempfile.TemporaryDirectory() as tmp:
        ef = os.path.join(tmp, "entropy.txt")
        with open(ef, "w") as f:
            f.write("你好\t10\t3.0\n你\t5\t0.0\n好\t3\t0.0\n")
        tf = os.path.join(tmp, "freq.trie")
        trie, total = build_and_mmap_trie(ef, tf, min_freq=2)
        pp = {"你": (0.5, 0.5, 0.5), "好": (0.5, 0.5, 0.5)}
        merged = [("你好", 10, 3.0), ("锟斤拷", 10, 3.0), ("烫烫烫工程", 10, 3.0)]
        results = extract_words(merged, pp, trie, total,
                                pmi_threshold=0, pos_threshold=0)
        words = {r[0] for r in results}
        assert "你好" in words
        assert "锟斤拷" not in words
        assert "烫烫烫工程" not in words


# ============================================================
# pos_tag
# ============================================================

def test_tag_word_known():
    assert len(tag_word("中国")) > 0


def test_tag_word_unknown_returns_x():
    assert tag_word("囧囧囧囧囧囧唔") == "x"


def test_tag_word_empty():
    assert tag_word("") == "x"


# ============================================================
# pos_prob
# ============================================================

def test_load_pos_prob():
    pos_prob = load_pos_prob()
    assert len(pos_prob) > 0
    char = next(iter(pos_prob))
    ps, pm, pe = pos_prob[char]
    assert 0.0 <= ps <= 1.0


def test_load_pos_prob_custom_path():
    with tempfile.TemporaryDirectory() as tmp:
        custom = os.path.join(tmp, "pos.txt")
        with open(custom, "w") as f:
            f.write("中\t0.1\t0.3\t0.6\n国\t0.2\t0.4\t0.4\n")
        result = load_pos_prob(custom)
        assert len(result) == 2
        assert result["中"] == (0.1, 0.3, 0.6)


# ============================================================
# encoding detection – edge cases
# ============================================================

def test_detect_encoding_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.txt")
        with open(path, "w") as f:
            f.write("")
        assert _detect_file_encoding(path) == "utf-8"


def test_detect_encoding_binary():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bin.txt")
        with open(path, "wb") as f:
            f.write(bytes(range(256)) * 4)
        enc = _detect_file_encoding(path)
        # Binary should not crash — returns some encoding
        assert enc in ("utf-8", "gb18030", "gbk", "big5")


# ============================================================
# read_chunks_by_lines
# ============================================================

def test_read_chunks_by_lines_basic():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lines.txt")
        with open(path, "w") as f:
            f.write("\n".join(str(i) for i in range(25)))

        chunks = list(_read_chunks_by_lines(path, 10))
        assert len(chunks) == 3  # 25 lines / 10 = 3 chunks
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5


def test_read_chunks_by_lines_single():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "one.txt")
        with open(path, "w") as f:
            f.write("hello\n")

        chunks = list(_read_chunks_by_lines(path, 100))
        assert len(chunks) == 1
        assert chunks[0] == ["hello\n"]


# ============================================================
# _process_chunk_direct
# ============================================================

def test_process_chunk_direct():
    with tempfile.TemporaryDirectory() as tmp:
        fw_path = os.path.join(tmp, "fw.txt")
        bw_path = os.path.join(tmp, "bw.txt")
        lines = ["你好世界。测试内容。"]

        _process_chunk_direct(lines, max_len=4, fw_path=fw_path, bw_path=bw_path)

        assert os.path.getsize(fw_path) > 0
        assert os.path.getsize(bw_path) > 0
        with open(fw_path) as f:
            content = f.read()
        assert "\t" in content  # n-gram format: word<TAB>next_char


# ============================================================
# _distribute_batches
# ============================================================

def test_distribute_batches():
    with tempfile.TemporaryDirectory() as tmp:
        # Create temp n-gram files
        fw_path = os.path.join(tmp, "fw_000.txt")
        bw_path = os.path.join(tmp, "bw_000.txt")
        with open(fw_path, "w", encoding="utf-8") as f:
            f.write("天下\t太\n天下\t下\n")
        with open(bw_path, "w", encoding="utf-8") as f:
            f.write("天下\t太\n天下\t下\n")

        bucket_dir = os.path.join(tmp, "buckets")
        os.makedirs(bucket_dir)
        _distribute_batches(
            [fw_path, bw_path], bucket_dir,
            num_buckets=2, buf_limit=1024, batch_size=10,
        )

        # Both lines of "天下" should be in the same bucket
        fw_b0 = os.path.join(bucket_dir, "fw_b0000.txt")
        bw_b0 = os.path.join(bucket_dir, "bw_b0000.txt")
        assert os.path.exists(fw_b0)
        assert os.path.exists(bw_b0)


# ============================================================
# pipeline integration
# ============================================================

@requires_sort
def test_pipeline_utf8():
    """Full pipeline with UTF-8 corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "utf8.txt",
            ("天下太平万物安宁\n天地玄黄宇宙洪荒\n"
             "日月盈昃辰宿列张\n寒来暑往秋收冬藏\n"
             "闰余成岁律吕调阳\n云腾致雨露结为霜\n"
             "金生丽水玉出昆冈\n剑号巨阙珠称夜光\n") * 30)

        out = run_pipeline(corpus, max_len=4, mem_mb=256, workers=2,
                           min_freq=5, entropy_threshold=0.0,
                           pmi_threshold=0.0, pos_threshold=0.0)

        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert len(lines) >= 2
        assert lines[0].strip().split("\t") == ["word", "freq", "pmi", "entropy", "pos_prob", "pos"]
        for line in lines[1:]:
            assert len(line.strip().split("\t")) == 6


@requires_sort
def test_pipeline_gbk():
    """Full pipeline with GBK-encoded corpus (encoding detection)."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "gbk.txt")
        content = "天下太平万物安宁\n天地玄黄宇宙洪荒\n" * 30
        with open(corpus, "w", encoding="gbk") as f:
            f.write(content)

        out = run_pipeline(corpus, max_len=4, mem_mb=256, workers=2,
                           min_freq=5, entropy_threshold=0.0,
                           pmi_threshold=0.0, pos_threshold=0.0)

        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert len(lines) >= 2
        assert "天下" in "".join(lines)


@requires_sort
def test_pipeline_mixed_single_line_and_normal(monkeypatch):
    """Mixed input: single-line huge file + normal file must both contribute."""
    import dict_build.pipeline as pl

    with tempfile.TemporaryDirectory() as tmp:
        normal = _make_text_file(tmp, "normal.txt",
                                 "刀枪剑戟斧钺钩叉\n" * 60)
        huge = _make_text_file(tmp, "huge.txt",
                               "苹果香蕉橘子葡萄" * 100)  # no newline

        real_check = pl._is_single_line_file
        monkeypatch.setattr(pl, "_is_single_line_file",
                            lambda p: p == huge or real_check(p))

        out = run_pipeline(tmp, max_len=2, mem_mb=256, workers=2,
                           min_freq=5, entropy_threshold=0.0,
                           pmi_threshold=0.0, pos_threshold=0.0)

        with open(out) as f:
            content = f.read()
        assert "苹果" in content  # from single-line mode
        assert "刀枪" in content  # from parallel mode


def test_pipeline_empty_directory():
    """Empty input directory produces a header-only output, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        out = run_pipeline(tmp, max_len=4, mem_mb=256, workers=2, min_freq=5)
        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert lines == ["word\tfreq\tpmi\tentropy\tpos_prob\tpos\n"]


def test_pipeline_no_chinese():
    """Corpus with no Chinese text produces a header-only output, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "ascii.txt", "hello world\n" * 100)
        out = run_pipeline(corpus, max_len=4, mem_mb=256, workers=2, min_freq=5)
        assert os.path.exists(out)
        with open(out) as f:
            lines = f.readlines()
        assert lines == ["word\tfreq\tpmi\tentropy\tpos_prob\tpos\n"]


@requires_sort
def test_pipeline_temp_dir_used():
    """--temp-dir places intermediate files under the given directory."""
    import dict_build.pipeline as pl

    with tempfile.TemporaryDirectory() as tmp:
        tdir = os.path.join(tmp, "work")
        os.makedirs(tdir)
        corpus = _make_text_file(tmp, "c.txt", "天下太平\n" * 50)

        created: list[str] = []
        real_mkdtemp = tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            p = real_mkdtemp(*args, **kwargs)
            created.append(p)
            return p

        orig = pl.tempfile.mkdtemp
        pl.tempfile.mkdtemp = spy_mkdtemp
        try:
            run_pipeline(corpus, max_len=2, mem_mb=256, workers=2,
                         min_freq=5, temp_dir=tdir)
        finally:
            pl.tempfile.mkdtemp = orig

        assert created
        assert all(p.startswith(tdir) for p in created)


def test_pipeline_disk_space_preflight(monkeypatch):
    """Insufficient free space aborts before any processing."""
    import dict_build.pipeline as pl
    import collections

    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "big.txt", "天下太平\n" * 100)
        usage = collections.namedtuple("usage", "total used free")
        monkeypatch.setattr(pl.shutil, "disk_usage",
                            lambda p: usage(10**12, 10**12, 1024))
        try:
            run_pipeline(corpus, max_len=2, mem_mb=256, workers=2, min_freq=5)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "Insufficient disk space" in str(e)


def test_calc_bucket_params_capped():
    """num_buckets never exceeds MAX_BUCKETS (fd-exhaustion guard)."""
    from dict_build.config import MAX_BUCKETS
    num, _, _ = _calc_bucket_params(10**13, workers=8, mem_mb=4096)
    assert num <= MAX_BUCKETS


@requires_sort
def test_pipeline_resume_after_failure(monkeypatch):
    """A run interrupted after entropy completes resumes from the merge stage."""
    import dict_build.pipeline as pl

    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "c.txt",
            ("天下太平万物安宁\n天地玄黄宇宙洪荒\n"
             "日月盈昃辰宿列张\n寒来暑往秋收冬藏\n") * 60)
        work = os.path.join(tmp, "work")
        out1 = os.path.join(tmp, "out1.data")
        out2 = os.path.join(tmp, "out2.data")

        kw = dict(max_len=4, mem_mb=256, workers=2, min_freq=5,
                  entropy_threshold=0.0, pmi_threshold=0.0, pos_threshold=0.0)
        # First run: clean, for reference output
        run_pipeline(corpus, output_path=out1, **kw)

        # Second run: crash at the merge stage
        real_merge = pl.merge_entropy_files_sorted
        monkeypatch.setattr(pl, "merge_entropy_files_sorted",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("simulated crash")))
        try:
            run_pipeline(corpus, output_path=out2, work_dir=work, **kw)
            raise AssertionError("expected crash")
        except RuntimeError:
            pass
        assert os.path.exists(os.path.join(work, ".entropy.done"))
        monkeypatch.setattr(pl, "merge_entropy_files_sorted", real_merge)

        # Third run: resumes from checkpoint, same output as clean run
        run_pipeline(corpus, output_path=out2, work_dir=work, **kw)

        with open(out1) as f1, open(out2) as f2:
            assert f1.read() == f2.read()
        # Successful resume cleans the work dir intermediates
        assert not os.path.exists(os.path.join(work, ".entropy.done"))


@requires_sort
def test_pipeline_force_ignores_checkpoints():
    """--force restarts even when complete checkpoints exist."""
    import dict_build.pipeline as pl

    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "c.txt", "天下太平万物安宁\n" * 80)
        work = os.path.join(tmp, "work")
        os.makedirs(work)
        # Fake a fully-checkpointed run with garbage artifacts
        for stage in ("ngrams", "ngrams_finalized", "sorted", "entropy",
                      "merged"):
            open(os.path.join(work, f".{stage}.done"), "w").close()
        with open(os.path.join(work, "merged.tsv"), "w") as f:
            f.write("垃圾词\t1\t0.0\n")

        out = run_pipeline(corpus, max_len=2, mem_mb=256, workers=2,
                           min_freq=5, work_dir=work, force=True,
                           entropy_threshold=0.0, pmi_threshold=0.0,
                           pos_threshold=0.0)

        with open(out) as f:
            content = f.read()
        assert "垃圾词" not in content
        assert "天下" in content


def test_generate_ngrams_single_line_direct():
    """Single-line mode writes n-grams for a file with no newlines."""
    from dict_build.pipeline import _generate_ngrams_single_line

    with tempfile.TemporaryDirectory() as tmp:
        src = _make_text_file(tmp, "sl.txt", "天地玄黄宇宙洪荒" * 50)
        fw = os.path.join(tmp, "fw.txt")
        bw = os.path.join(tmp, "bw.txt")
        _generate_ngrams_single_line(src, fw, bw, max_len=2)

        with open(fw, encoding="utf-8") as f:
            fw_content = f.read()
        assert "天地\t" in fw_content
        assert os.path.getsize(bw) > 0


@requires_sort
def test_cli_end_to_end():
    """CLI entry point parses args and produces output."""
    from click.testing import CliRunner
    from dict_build.__main__ import main

    with tempfile.TemporaryDirectory() as tmp:
        corpus = _make_text_file(tmp, "cli.txt", "天下太平万物安宁\n" * 80)
        out = os.path.join(tmp, "out.data")
        result = CliRunner().invoke(main, [
            corpus, "-o", out, "--max-len", "2", "--min-freq", "5",
            "--workers", "2", "--entropy-threshold", "0",
            "--pmi-threshold", "0", "--pos-threshold", "0",
        ])
        assert result.exit_code == 0, result.output
        with open(out) as f:
            content = f.read()
        assert "天下" in content


def test_cli_help():
    from click.testing import CliRunner
    from dict_build.__main__ import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for opt in ("--work-dir", "--force", "--temp-dir", "--verbose", "--quiet"):
        assert opt in result.output
