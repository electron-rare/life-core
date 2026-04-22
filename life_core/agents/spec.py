"""Spec agent: produces an RFC2119 specification from an intake record."""

from __future__ import annotations

from .base import AgentBase, AgentResult

PROMPT_TEMPLATE = """You are a spec agent. Produce an RFC2119 specification for:

Title: {title}
Goal: {goal}
Constraints: {constraints}
Compliance profile: {profile}

Write a markdown document with a # heading, a ## Requirements section,
and numbered requirements using MUST / SHOULD / MAY exclusively.
Do not include any text outside the markdown.
"""


async def call_llm(prompt: str) -> str:
    """Stub for real LLM call, replaced at runtime by the mascarade client."""
    raise NotImplementedError("LLM not wired in unit test context")


class SpecAgent(AgentBase):
    role = "spec"

    async def run(self, payload):
        intake = payload["intake"]
        profile = payload.get("compliance_profile", "prototype")
        prompt = PROMPT_TEMPLATE.format(
            title=intake.get("title", ""),
            goal=intake.get("normalized_payload", {}).get("goal", ""),
            constraints=", ".join(
                intake.get("normalized_payload", {}).get("constraints", [])
            ),
            profile=profile,
        )
        try:
            text = await call_llm(prompt)
        except Exception as e:
            return AgentResult(ok=False, output="", reasons=[str(e)])
        if "MUST" not in text or "## Requirements" not in text:
            return AgentResult(
                ok=False, output=text, reasons=["missing RFC2119 markers"]
            )
        return AgentResult(ok=True, output=text, reasons=[])
