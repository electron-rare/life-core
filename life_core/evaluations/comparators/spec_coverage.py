"""LLM-as-judge comparator for spec deliverables (P4 T1.9a).

The comparator asks Claude Sonnet 4.6 to rate how well the LLM-authored
spec covers the requirements of the human gold-standard spec. The
response is parsed as a single float in ``[0, 1]``; any parsing
failure is treated as a 0 score. Bounds are enforced in :func:`compare`.
"""
from __future__ import annotations

from litellm import completion

JUDGE_PROMPT = """You are a requirements auditor. Rate how well the LLM spec covers the
requirements of the human spec. Answer with a single float in [0,1] (no prose).

=== HUMAN SPEC ===
{human}

=== LLM SPEC ===
{llm}

Score:"""


def _llm_judge_coverage(human: str, llm: str) -> float:
    """Invoke the LLM judge and return a raw ``[0, 1]`` score."""

    resp = completion(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(human=human, llm=llm),
            }
        ],
    )
    try:
        return float(resp["choices"][0]["message"]["content"].strip().split()[0])
    except (ValueError, IndexError):
        return 0.0


def compare(human: str, llm: str) -> dict:
    """Compare two spec strings, returning a bounded score + details dict."""

    score = _llm_judge_coverage(human, llm)
    return {
        "score": max(0.0, min(1.0, score)),
        "details": {"method": "llm_as_judge"},
    }
