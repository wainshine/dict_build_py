"""Smoke tests for dict_build."""

import os
import tempfile

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
from dict_build.pipeline import run_pipeline


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
    # 2-char minimum per sentence
    result = list(preprocess_line("哈"))
    assert len(result) == 0


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
    """Each n-gram is word<TAB>next_char."""
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
    """Reverse n-grams should not contain the sentinel in the word."""
    sent = "$你好世界$"
    for ng in generate_reverse_ngrams(sent, max_len=4):
        word = ng.split("\t")[0]
        assert "$" not in word


# ============================================================
# entropy
# ============================================================

def test_entropy_computation():
    sorted_lines = [
        "你\t好\n",
        "你\t好\n",
        "你\t们\n",
        "你好\t世\n",
        "你好\t界\n",
    ]
    results = list(compute_entropy_from_sorted(iter(sorted_lines)))
    words = {r[0] for r in results}
    assert "你" in words
    assert "你好" in words
    for word, freq, entropy in results:
        assert freq > 0
        assert entropy >= 0.0


def test_entropy_freq_equals_suffix_count():
    """Frequency = sum of all suffix occurrences."""
    lines = [
        "吃\t饭\n",
        "吃\t饭\n",
        "吃\t面\n",
    ]
    results = list(compute_entropy_from_sorted(iter(lines)))
    (word, freq, entropy) = results[0]
    assert word == "吃"
    assert freq == 3


def test_entropy_min_freq_filter():
    lines = [
        "你\t好\n",
        "你好\t世\n",
        "你好\t界\n",
    ]
    results = list(compute_entropy_from_sorted(iter(lines), min_freq=3))
    # "你" has 1 occurrence -> filtered; "你好" has 2 -> filtered
    words = {r[0] for r in results}
    assert "你" not in words
    assert "你好" not in words


def test_entropy_zero_for_single_suffix():
    """Entropy is 0 when there is only one distinct suffix."""
    lines = [
        "词\tA\n",
        "词\tA\n",
        "词\tA\n",
    ]
    results = list(compute_entropy_from_sorted(iter(lines)))
    assert len(results) == 1
    assert results[0][2] == 0.0


def test_compute_entropy_from_sorted_left():
    """Left entropy reverses the word back."""
    # Input: reversed word "$界世好你$" → ngrams of reversed sentence
    # Forward ngrams on reversed sentence produce word_tab_next
    # Left entropy reverses the word back
    lines = [
        "界\t世\n",
        "世\t好\n",
    ]
    # These are from reversed sentence. compute_entropy_from_sorted_left
    # should reverse word components back.
    results = list(compute_entropy_from_sorted_left(iter(lines)))
    # Words are reversed: "界"→"界"? Actually: "界" reversed is "界",
    # "世" reversed is "世". The function reverses the word.
    words = {r[0] for r in results}
    # For single chars, reverse is same char
    assert "界" in words or "世" in words


def test_write_read_entropy_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "ent.txt")
        data = [("天下", 100, 2.5), ("天地", 80, 3.1), ("之", 999, 1.0)]
        write_entropy_to_file(iter(data), filepath)
        read_back = list(read_entropy_from_file(filepath))
        assert read_back == data


def test_read_entropy_skips_malformed():
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "ent.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("word_only\n")
            f.write("too\tmany\tcols\textra\n")
            f.write("good\t42\t1.5\n")
        results = list(read_entropy_from_file(filepath))
        assert len(results) == 1
        assert results[0] == ("good", 42, 1.5)


def test_sort_file_inplace():
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "unsorted.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("c\t1\t0.1\n")
            f.write("a\t3\t0.3\n")
            f.write("b\t2\t0.2\n")
        sort_file_inplace(filepath)
        results = list(read_entropy_from_file(filepath))
        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == "c"


# ----- merge -----

def _make_entropy_file(tmp: str, name: str, rows: list[tuple[str, int, float]]):
    path = os.path.join(tmp, name)
    write_entropy_to_file(iter(rows), path)
    return path


def test_merge_entropy_files_sorted_basic():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [
            ("天下", 100, 3.0),
            ("天地", 80, 2.5),
            ("自然", 60, 1.8),
        ])
        l = _make_entropy_file(tmp, "l.txt", [
            ("天下", 100, 4.0),
            ("天地", 80, 2.0),
            ("自然", 60, 2.2),
        ])
        merged = merge_entropy_files_sorted(r, l)
        assert len(merged) == 3
        # merged entropy = min(right, left)
        assert merged[0] == ("天下", 100, 3.0)
        assert merged[1] == ("天地", 80, 2.0)
        assert merged[2] == ("自然", 60, 1.8)


def test_merge_entropy_files_sorted_min_entropy_filter():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [("天下", 100, 3.0), ("自然", 60, 1.0)])
        l = _make_entropy_file(tmp, "l.txt", [("天下", 100, 4.0), ("自然", 60, 0.8)])
        merged = merge_entropy_files_sorted(r, l, min_entropy=2.0)
        assert len(merged) == 1
        assert merged[0][0] == "天下"


def test_merge_entropy_empty_right():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [])
        l = _make_entropy_file(tmp, "l.txt", [("天下", 100, 4.0)])
        merged = merge_entropy_files_sorted(r, l)
        assert merged == []


def test_merge_entropy_empty_left():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [("天下", 100, 3.0)])
        l = _make_entropy_file(tmp, "l.txt", [])
        merged = merge_entropy_files_sorted(r, l)
        assert merged == []


def test_merge_entropy_non_overlapping():
    with tempfile.TemporaryDirectory() as tmp:
        r = _make_entropy_file(tmp, "r.txt", [("天下", 100, 3.0)])
        l = _make_entropy_file(tmp, "l.txt", [("天地", 80, 4.0)])
        merged = merge_entropy_files_sorted(r, l)
        assert merged == []


# ============================================================
# pmi / trie
# ============================================================

def test_build_and_mmap_trie():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("世界\t8\t2.5\n")
            f.write("你\t5\t0.0\n")
            f.write("好\t3\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)
        assert total_single == 8
        assert trie["你好"][0][0] == 10
        assert trie["世界"][0][0] == 8
        assert trie["你"][0][0] == 5


def test_build_and_mmap_trie_min_freq():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("世界\t2\t2.5\n")
            f.write("你\t5\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, _ = build_and_mmap_trie(entropy_file, trie_file, min_freq=5)
        # "世界" freq=2 < 5, should be excluded
        try:
            trie["世界"]
            assert False, "world should be excluded by min_freq"
        except KeyError:
            pass


def test_compute_pmi():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("你\t5\t0.0\n")
            f.write("好\t3\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)
        pmi = compute_pmi("你好", 10, trie, total_single)
        assert pmi > 0.0


def test_compute_pmi_single_char_returns_zero():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你\t5\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)
        assert compute_pmi("你", 5, trie, total_single) == 0.0


def test_extract_words_basic():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("你\t5\t0.0\n")
            f.write("好\t3\t0.0\n")
            f.write("世界\t4\t2.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)

        pos_prob = {
            "你": (0.5, 0.3, 0.2),
            "好": (0.1, 0.4, 0.5),
            "世": (0.4, 0.3, 0.3),
            "界": (0.1, 0.3, 0.6),
        }
        merged = [("你好", 10, 3.0), ("世界", 4, 2.0)]
        results = extract_words(
            merged, pos_prob, trie, total_single,
            pmi_threshold=0.0, entropy_threshold=0.0, pos_threshold=0.0,
        )
        assert len(results) == 2


def test_extract_words_skips_short():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你\t5\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)
        merged = [("你", 5, 1.0)]
        results = extract_words(
            merged, {}, trie, total_single,
            pmi_threshold=0.0, entropy_threshold=0.0, pos_threshold=0.0,
        )
        assert len(results) == 0


def test_extract_words_pos_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("你\t5\t0.0\n")
            f.write("好\t3\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)

        # First char "你" has P_S=0.1, last char "好" has P_E=0.1
        # min(0.1, 0.1) = 0.1
        pos_prob = {"你": (0.1, 0.4, 0.5), "好": (0.5, 0.4, 0.1)}
        merged = [("你好", 10, 3.0)]
        # pos_threshold=0.2 > 0.1 => filtered out
        results_high = extract_words(
            merged, pos_prob, trie, total_single,
            pmi_threshold=0.0, entropy_threshold=0.0, pos_threshold=0.2,
        )
        assert len(results_high) == 0
        # pos_threshold=0.05 < 0.1 => passes
        results_low = extract_words(
            merged, pos_prob, trie, total_single,
            pmi_threshold=0.0, entropy_threshold=0.0, pos_threshold=0.05,
        )
        assert len(results_low) == 1


def test_extract_words_missing_from_pos_prob():
    """Words whose first/last char are not in pos_prob get pp=0 and fail."""
    with tempfile.TemporaryDirectory() as tmp:
        entropy_file = os.path.join(tmp, "entropy.txt")
        with open(entropy_file, "w", encoding="utf-8") as f:
            f.write("你好\t10\t3.0\n")
            f.write("你\t5\t0.0\n")
            f.write("好\t3\t0.0\n")

        trie_file = os.path.join(tmp, "freq.trie")
        trie, total_single = build_and_mmap_trie(entropy_file, trie_file, min_freq=2)
        # pos_prob is empty → pp=0 for all words → all filtered (pos_threshold=0.1)
        merged = [("你好", 10, 3.0)]
        results = extract_words(
            merged, {}, trie, total_single,
            pmi_threshold=0.0, entropy_threshold=0.0, pos_threshold=0.1,
        )
        assert len(results) == 0


# ============================================================
# pos_tag
# ============================================================

def test_tag_word_known():
    """Common Chinese words should return meaningful POS tags."""
    assert len(tag_word("中国")) > 0


def test_tag_word_unknown_returns_x():
    """Gibberish should return 'x'."""
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
        custom_path = os.path.join(tmp, "pos.txt")
        with open(custom_path, "w", encoding="utf-8") as f:
            f.write("中\t0.1\t0.3\t0.6\n")
            f.write("国\t0.2\t0.4\t0.4\n")
        result = load_pos_prob(custom_path)
        assert len(result) == 2
        assert result["中"] == (0.1, 0.3, 0.6)


# ============================================================
# pipeline integration (smoke)
# ============================================================

def test_pipeline_smoke():
    """Full pipeline on a minimal corpus."""
    with tempfile.TemporaryDirectory() as tmp:
        corpus = os.path.join(tmp, "test.txt")
        with open(corpus, "w", encoding="utf-8") as f:
            for _ in range(20):
                f.write("天下太平万物安宁。\n")
                f.write("天地玄黄宇宙洪荒。\n")
                f.write("日月盈昃辰宿列张。\n")
                f.write("寒来暑往秋收冬藏。\n")
                f.write("闰余成岁律吕调阳。\n")
                f.write("云腾致雨露结为霜。\n")
                f.write("金生丽水玉出昆冈。\n")
                f.write("剑号巨阙珠称夜光。\n")

            # Lower thresholds so the small corpus can produce results
            import dict_build.config as cfg
            cfg.ENTROPY_THRESHOLD = 0.0
            cfg.PMI_THRESHOLD = 0.0
            cfg.POS_PROB_THRESHOLD = 0.0

        out = run_pipeline(
            input_path=corpus,
            max_len=4,
            mem_mb=256,
            workers=2,
            min_freq=5,
        )

        # Restore defaults for other tests
        cfg.ENTROPY_THRESHOLD = 2.0
        cfg.PMI_THRESHOLD = 1.0
        cfg.POS_PROB_THRESHOLD = 0.1

        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2, f"Expected >=2 lines, got {len(lines)}"
        header = lines[0].strip().split("\t")
        assert header == ["word", "freq", "pmi", "entropy", "pos_prob", "pos"]
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == 6, f"bad line: {line!r}"
