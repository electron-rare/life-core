"""Handler for POST /audit/analyze — delegates to AuditAnalyzer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# makelife package must be importable; add to sys.path if needed
try:
    from makelife.audit_analyzer import AnalysisError, AuditAnalyzer
except ImportError as exc:
    raise ImportError(
        "makelife package not found. Install it: cd makelife && uv pip install -e ."
    ) from exc


class AuditAnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the audit file to analyse")
    cross_paths: list[str] = Field(
        default_factory=list,
        description="Optional additional file paths for cross-file analysis",
    )
    model: str = Field(
        default="claude-3-5-haiku-latest",
        description="LLM model to use for analysis",
    )


class AuditAnalyzeResponse(BaseModel):
    issues: list[dict[str, Any]]
    summary: str
    mode: str  # "single" or "cross"


def handle_audit_analyze(request: AuditAnalyzeRequest) -> AuditAnalyzeResponse:
    """Perform LLM analysis on one or more audit files.

    Raises:
        FileNotFoundError: if file_path or any cross_paths file does not exist.
        AnalysisError: if the LLM call fails.
    """
    analyzer = AuditAnalyzer(model=request.model)

    if request.cross_paths:
        all_paths = [request.file_path] + request.cross_paths
        result = analyzer.analyze_cross(all_paths)
        mode = "cross"
    else:
        result = analyzer.analyze_single(request.file_path)
        mode = "single"

    return AuditAnalyzeResponse(
        issues=result.get("issues", []),
        summary=result.get("summary", ""),
        mode=mode,
    )
