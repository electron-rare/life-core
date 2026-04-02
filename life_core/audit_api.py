"""Audit governance API."""
import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

from .audit_analyze_handler import (
    AuditAnalyzeRequest,
    AuditAnalyzeResponse,
    handle_audit_analyze,
)
from makelife.audit_analyzer import AnalysisError

logger = logging.getLogger("life_core.audit_api")
audit_router = APIRouter(prefix="/audit", tags=["Audit"])


def _load_report():
    for p in [Path("/data/audit_report.json"), Path("audit_report.json")]:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


@audit_router.get("/status")
async def audit_status():
    report = _load_report()
    if not report:
        return {"status": "no_report", "message": "Run validator first."}
    return {
        "last_run": report.get("timestamp", "unknown"),
        "total_audits": report.get("total_files", 0),
        **report.get("summary", {}),
    }


@audit_router.get("/report")
async def audit_report():
    return _load_report() or {"status": "no_report", "results": []}


@audit_router.post("/analyze", response_model=AuditAnalyzeResponse, tags=["audit"])
async def analyze_audit(request: AuditAnalyzeRequest) -> AuditAnalyzeResponse:
    """Run LLM analysis on an audit file via life-core /chat.

    Accepts a single file for single-file analysis, or file_path + cross_paths
    for cross-file contradiction/overlap detection.
    """
    try:
        return handle_audit_analyze(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except AnalysisError as exc:
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {exc}")
