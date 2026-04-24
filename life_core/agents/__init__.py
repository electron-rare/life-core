"""Inner HITL agents module (P4 Sprint 1, T1.8).

Existing legacy LLM agents (``base``, ``llm``, ``qa``, ``spec``) remain
importable via their submodule paths. New orchestration contracts live
in ``life_core.agents.contract`` and are re-exported here for engine
callers.
"""
from .contract import AgentEnvelope, AgentRequest, AgentResult, ArtifactRef

__all__ = ["AgentRequest", "AgentResult", "AgentEnvelope", "ArtifactRef"]
