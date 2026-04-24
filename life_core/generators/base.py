"""Abstract base class for LLM-backed generators with auto-reprompt."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationContext:
    """Input context for a single ``generate()`` call."""

    deliverable_slug: str
    deliverable_type: str
    prompt_template: str
    llm_model: str
    prompt_vars: dict[str, Any] = field(default_factory=dict)
    max_reprompts: int = 2
    n_candidates: int = 1


@dataclass
class GenerationOutcome:
    """Result of ``generate()``: success flag, bytes, errors, attempt count."""

    ok: bool
    data: bytes
    errors: list[str] = field(default_factory=list)
    attempts: int = 1
    score: float = 1.0


class BaseGenerator(ABC):
    """Shared auto-reprompt loop: call_llm → validate → optionally re-prompt.

    Subclasses implement ``call_llm`` (the model-specific HTTP call) and
    ``validate`` (deliverable-type-specific checks). ``generate`` loops up
    to ``max_reprompts + 1`` attempts, accumulating per-attempt feedback in
    ``prompt_vars['attempt_history']`` (structured) and a backward-compat
    ``prompt_vars['human_feedback']`` (string of the last failure).

    P3: ``validate`` may return either ``(ok, errors)`` or
    ``(ok, errors, score)``. The 2-tuple is auto-promoted to score
    ``1.0`` if ``ok`` else ``0.0``. When ``ctx.n_candidates > 1`` each
    attempt fans out K parallel ``call_llm`` + ``validate`` evaluations
    and the highest-scoring outcome (preferring ``ok``) is selected.
    """

    @abstractmethod
    def call_llm(self, ctx: GenerationContext) -> bytes:
        """Render the prompt and call the model; return raw bytes."""

    @abstractmethod
    def validate(
        self, data: bytes, ctx: GenerationContext
    ) -> tuple[bool, list[str]] | tuple[bool, list[str], float]:
        """Validate ``data`` for this deliverable type.

        Return ``(ok, issues)`` (legacy) or ``(ok, issues, score)`` where
        ``score`` is in ``[0.0, 1.0]``.
        """

    def _validate_with_score(
        self, data: bytes, ctx: GenerationContext
    ) -> tuple[bool, list[str], float]:
        """Call ``validate`` and normalise the result to a 3-tuple."""
        result = self.validate(data, ctx)
        if len(result) == 2:
            ok, errors = result  # type: ignore[misc]
            score = 1.0 if ok else 0.0
            return ok, list(errors), score
        ok, errors, score = result  # type: ignore[misc]
        return ok, list(errors), float(score)

    def _run_one_candidate(
        self, ctx: GenerationContext
    ) -> tuple[bytes, bool, list[str], float]:
        """Single candidate: call_llm + validate with normalised score."""
        data = self.call_llm(ctx)
        ok, errors, score = self._validate_with_score(data, ctx)
        return data, ok, errors, score

    def _record_attempt(
        self,
        ctx: GenerationContext,
        attempt_number: int,
        data: bytes,
        errors: list[str],
    ) -> None:
        """Append a structured entry to ``attempt_history`` and refresh
        the backward-compat ``human_feedback`` string."""
        history = ctx.prompt_vars.setdefault("attempt_history", [])
        entry = {
            "attempt": attempt_number,
            "data_hash": hashlib.sha256(data).hexdigest()[:12],
            "errors": list(errors),
        }
        history.append(entry)
        ctx.prompt_vars["human_feedback"] = (
            f"Attempt {entry['attempt']} (hash {entry['data_hash']}) failed: "
            f"{'; '.join(entry['errors'])}. Fix only these issues."
        )

    def generate(self, ctx: GenerationContext) -> GenerationOutcome:
        """Call-validate-reprompt loop. Never raises for validation errors."""
        ctx.prompt_vars.setdefault("attempt_history", [])
        errors: list[str] = []
        data: bytes = b""
        score: float = 0.0
        n_candidates = max(1, int(ctx.n_candidates or 1))
        for attempt in range(1, ctx.max_reprompts + 2):
            if n_candidates == 1:
                data, ok, errors, score = self._run_one_candidate(ctx)
            else:
                with ThreadPoolExecutor(max_workers=n_candidates) as pool:
                    results = list(
                        pool.map(
                            lambda _i: self._run_one_candidate(ctx),
                            range(n_candidates),
                        )
                    )
                # Prefer the highest-scoring passing candidate; else the
                # highest-scoring overall (whose errors feed the reprompt).
                passing = [r for r in results if r[1]]
                pool_to_pick = passing if passing else results
                data, ok, errors, score = max(
                    pool_to_pick, key=lambda r: r[3]
                )
            if ok:
                return GenerationOutcome(
                    ok=True,
                    data=data,
                    errors=[],
                    attempts=attempt,
                    score=score,
                )
            self._record_attempt(ctx, attempt, data, errors)
        return GenerationOutcome(
            ok=False,
            data=data,
            errors=errors,
            attempts=ctx.max_reprompts + 1,
            score=score,
        )
