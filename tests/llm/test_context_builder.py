from pathlib import Path
from unittest.mock import MagicMock

from life_core.llm.context_builder import build_context


def test_build_context_includes_brief(tmp_path):
    ctx = build_context(
        deliverable_slug="s",
        deliverable_type="spec",
        upstream_artifacts=[],
        brief="Test brief.",
        constraints=["cap BOM 15 eur"],
    )
    assert ctx["brief"] == "Test brief."
    assert "cap BOM 15 eur" in ctx["constraints"]


def test_build_context_reads_upstream_artifact_bytes(tmp_path):
    artifact_file = tmp_path / "prd.md"
    artifact_file.write_text("# Spec")
    ref = MagicMock(
        deliverable_slug="x-spec",
        type="spec",
        version=1,
        storage_path=artifact_file,
    )
    ctx = build_context(
        deliverable_slug="s",
        deliverable_type="hardware",
        upstream_artifacts=[ref],
        brief="...",
        constraints=[],
    )
    assert ctx["upstream"][0]["content"] == "# Spec"
    assert ctx["upstream"][0]["type"] == "spec"
