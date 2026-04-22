"""QA agent: reviews artefacts against a compliance profile and returns a verdict."""

from __future__ import annotations

import json

from .base import AgentBase, AgentResult

PROMPT = """You are a QA agent reviewing artefacts for deliverable {did}, gate {gate}.
Compliance profile: {profile}
Artefacts summary (JSON): {artefacts}

Decide whether the gate passes. Respond with a JSON object of the form:
{{"verdict": "pass|fail", "reasons": ["..."], "category": "compliance|build|test|null"}}
Only emit the JSON, no extra text.
"""


async def call_llm(prompt: str) -> str:
    """Stub for real LLM call, replaced at runtime by the mascarade client."""
    raise NotImplementedError("LLM not wired in unit test context")


class QaAgent(AgentBase):
    role = "qa"

    async def run(self, payload):
        prompt = PROMPT.format(
            did=payload["deliverable_id"],
            gate=payload["gate"],
            profile=payload.get("compliance_profile", "prototype"),
            artefacts=json.dumps(payload.get("artefacts", {})),
        )
        try:
            text = await call_llm(prompt)
            parsed = json.loads(text)
        except Exception as e:
            return AgentResult(ok=False, output="fail", reasons=[f"llm error: {e}"])
        verdict = parsed.get("verdict", "fail")
        reasons = list(parsed.get("reasons", []))
        category = parsed.get("category")
        if category:
            reasons.append(f"category:{category}")
        return AgentResult(
            ok=(verdict == "pass"), output=verdict, reasons=reasons
        )
