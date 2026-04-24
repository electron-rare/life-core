"""Evaluations module (P4 T1.9).

Four comparators + harness + router used to score LLM deliverables
against a human gold standard.
"""
from .harness import run_evaluation

__all__ = ["run_evaluation"]
