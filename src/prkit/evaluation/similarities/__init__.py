"""Text similarity helpers for evaluation (e.g. cross-typed matching)."""

from .rouge_l import rouge_l_f1

__all__ = ["rouge_l_f1"]
