"""Tests for word-level ROUGE-L F1."""

from prkit.evaluation.similarities.rouge_l import rouge_l_f1


class TestRougeLF1:
    def test_identical(self):
        assert rouge_l_f1("hello world", "hello world") == 1.0

    def test_empty(self):
        assert rouge_l_f1("", "a") == 0.0
        assert rouge_l_f1("a", "") == 0.0

    def test_partial_overlap(self):
        s = rouge_l_f1("the cat sat", "the dog sat")
        assert 0.0 < s < 1.0

    def test_case_insensitive(self):
        assert rouge_l_f1("Hello WORLD", "hello world") == 1.0
