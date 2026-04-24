"""Abstract base class for LLM-backed generators with auto-reprompt."""
from __future__ import annotations

from abc import ABC, abstractmethod
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


@dataclass
class GenerationOutcome:
    """Result of ``generate()``: success flag, bytes, errors, attempt count."""

    ok: bool
    data: bytes
    errors: list[str] = field(default_factory=list)
    attempts: int = 1


class BaseGenerator(ABC):
    """Shared auto-reprompt loop: call_llm → validate → optionally re-prompt.

    Subclasses implement ``call_llm`` (the model-specific HTTP call) and
    ``validate`` (deliverable-type-specific checks). ``generate`` loops up
    to ``max_reprompts + 1`` attempts, injecting validation feedback into
    ``prompt_vars['human_feedback']`` between attempts.
    """

    @abstractmethod
    def call_llm(self, ctx: GenerationContext) -> bytes:
        """Render the prompt and call the model; return raw bytes."""

    @abstractmethod
    def validate(
        self, data: bytes, ctx: GenerationContext
    ) -> tuple[bool, list[str]]:
        """Validate ``data`` for this deliverable type; return (ok, issues)."""

    def generate(self, ctx: GenerationContext) -> GenerationOutcome:
        """Call-validate-reprompt loop. Never raises for validation errors."""
        errors: list[str] = []
        data: bytes = b""
        for attempt in range(1, ctx.max_reprompts + 2):
            data = self.call_llm(ctx)
            ok, issues = self.validate(data, ctx)
            if ok:
                return GenerationOutcome(
                    ok=True, data=data, errors=[], attempts=attempt
                )
            errors = issues
            ctx.prompt_vars["human_feedback"] = (
                f"Previous attempt failed validation: {issues}. "
                "Fix only these issues."
            )
        return GenerationOutcome(
            ok=False,
            data=data,
            errors=errors,
            attempts=ctx.max_reprompts + 1,
        )
