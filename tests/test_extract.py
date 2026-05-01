"""Smoke tests for dict_build."""

import os
import tempfile

from dict_build.preprocess import preprocess_line, is_chinese, all_chinese
from dict_build.ngram import generate_ngrams, generate_reverse_ngrams
from dict_build.entropy import compute_entropy_from_sorted
from dict_build.pmi import compute_pmi, build_and_mmap_trie
from dict_build.pos_prob import load_pos_prob


def test_preprocess_line():
    sentences = list(preprocess_line("Hello 你好世界！这是一个测试。"))
    assert len(sentences) >= 1
    for s in sentences:
        assert s.startswith("$")
        assert s.endswith("$")
        assert all(is_chinese(c) or c == "$" for c in s)


def test_is_chinese():
    assert is_chinese("世")
    assert not is_chinese("a")


def test_all_chinese():
    assert all_chinese("你好世界")
    assert not all_chinese("hello世界")


def test_generate_ngrams():
    sent = "$你好世界$"
    ngrams = list(generate_ngrams(sent, max_len=3))
    assert len(ngrams) > 0
    words = {ng.split("\t")[0] for ng in ngrams}
    assert "你" in words
    assert "你好" in words


def test_generate_reverse_ngrams():
    sent = "$你好世界$"
    ngrams = list(generate_reverse_ngrams(sent, max_len=3))
    assert len(ngrams) > 0


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


def test_load_pos_prob():
    pos_prob = load_pos_prob()
    assert len(pos_prob) > 0
    char = next(iter(pos_prob))
    ps, pm, pe = pos_prob[char]
    assert 0.0 <= ps <= 1.0


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
