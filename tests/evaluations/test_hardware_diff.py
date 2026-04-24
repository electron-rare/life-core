"""Tests for ``life_core.evaluations.comparators.hardware_diff`` (T1.9b)."""
from __future__ import annotations

from unittest.mock import patch

from life_core.evaluations.comparators import hardware_diff
from life_core.evaluations.comparators.hardware_diff import (
    _value_close,
    compare,
)
from life_core.tools.kicad_cli import DRCResult


def _mock_bom(items: list[dict]):
    """Return a patch target for :func:`_extract_bom` yielding ``items``."""

    return patch.object(hardware_diff, "_extract_bom", return_value=items)


def test_extract_bom_missing_file_returns_empty(tmp_path):
    """Non-existent schematic paths must return an empty BOM (no raise)."""
    assert hardware_diff._extract_bom(str(tmp_path / "nope.kicad_sch")) == []


def test_extract_bom_parses_kiutils_schematic(tmp_path, monkeypatch):
    """Exercise the kiutils parsing path via a patched Schematic class."""
    sch_file = tmp_path / "good.kicad_sch"
    sch_file.write_text("(kicad_sch)")  # content irrelevant; parser is stubbed

    class _P:
        def __init__(self, key: str, value: str):
            self.key = key
            self.value = value

    class _Sym:
        def __init__(self, ref: str, val: str):
            self.properties = [_P("Reference", ref), _P("Value", val)]

    class _StubSchematic:
        def from_file(self, _path: str):
            stub = _StubSchematic()
            stub.schematicSymbols = [_Sym("R1", "10k"), _Sym("C1", "100nF")]
            return stub

    import kiutils.schematic as real_kiutils_sch  # already installed as dep

    monkeypatch.setattr(real_kiutils_sch, "Schematic", _StubSchematic)

    bom = hardware_diff._extract_bom(str(sch_file))

    assert bom == [
        {"ref": "R1", "value": "10k", "type": "R"},
        {"ref": "C1", "value": "100nF", "type": "C"},
    ]


def test_extract_bom_handles_missing_props(tmp_path, monkeypatch):
    """Symbols missing Reference/Value fall back to '?'."""
    sch_file = tmp_path / "partial.kicad_sch"
    sch_file.write_text("(kicad_sch)")

    class _EmptySym:
        properties: list = []

    class _StubSchematic:
        def from_file(self, _path: str):
            stub = _StubSchematic()
            stub.schematicSymbols = [_EmptySym()]
            return stub

    import kiutils.schematic as real_kiutils_sch

    monkeypatch.setattr(real_kiutils_sch, "Schematic", _StubSchematic)

    bom = hardware_diff._extract_bom(str(sch_file))

    assert bom == [{"ref": "?", "value": "?", "type": "?"}]


def test_value_close_within_tolerance():
    assert _value_close("10kOhm", "11kOhm") is True  # 10 % diff, < 20 %
    assert _value_close("100uF", "119uF") is True  # 19 % diff
    assert _value_close("100uF", "121uF") is False  # 21 % diff


def test_value_close_non_numeric_strict():
    assert _value_close("NE555", "NE555") is True
    assert _value_close("NE555", "NE556") is False


def test_value_close_zero_pair():
    assert _value_close("0", "0") is True
    assert _value_close("0", "1") is False


def test_compare_identical_boms_full_score(tmp_path):
    """Matching types + values + no DRC errors → score 1.0."""
    bom = [
        {"ref": "R1", "value": "10k", "type": "R"},
        {"ref": "C1", "value": "100nF", "type": "C"},
    ]
    human_sch = tmp_path / "human.kicad_sch"
    llm_sch = tmp_path / "llm.kicad_sch"
    human_sch.write_text("")
    llm_sch.write_text("")

    calls: list[str] = []

    def fake_extract(path: str) -> list[dict]:
        calls.append(path)
        return bom

    with patch.object(hardware_diff, "_extract_bom", side_effect=fake_extract):
        with patch.object(
            hardware_diff,
            "run_drc",
            return_value=DRCResult(passed=True, errors=[]),
        ):
            result = compare(str(human_sch), str(llm_sch))

    assert result["score"] == 1.0
    assert result["details"]["types_match"] is True
    assert result["details"]["bom_match"] == 2
    assert result["details"]["bom_total"] == 2
    assert result["details"]["drc_errors"] == 0
    assert len(calls) == 2


def test_compare_drc_penalty(tmp_path):
    """Each DRC error subtracts 0.1 from the LLM score."""
    bom = [{"ref": "R1", "value": "10k", "type": "R"}]
    human_sch = tmp_path / "h.kicad_sch"
    llm_sch = tmp_path / "l.kicad_sch"
    human_sch.write_text("")
    llm_sch.write_text("")

    with patch.object(hardware_diff, "_extract_bom", return_value=bom):
        with patch.object(
            hardware_diff,
            "run_drc",
            return_value=DRCResult(
                passed=False, errors=[{"id": 1}, {"id": 2}]
            ),
        ):
            result = compare(str(human_sch), str(llm_sch))

    # Full type+BOM match = 1.0 ; minus 2 × 0.1 = 0.8.
    assert result["score"] == 0.8
    assert result["details"]["drc_errors"] == 2


def test_compare_partial_value_match(tmp_path):
    """Half the components match → types_match (0.5) + 0.5 × 0.5 = 0.75."""
    human_bom = [
        {"ref": "R1", "value": "10k", "type": "R"},
        {"ref": "R2", "value": "22k", "type": "R"},
    ]
    llm_bom = [
        {"ref": "R1", "value": "10k", "type": "R"},
        {"ref": "R2", "value": "100k", "type": "R"},  # out of tolerance
    ]
    human_sch = tmp_path / "h.kicad_sch"
    llm_sch = tmp_path / "l.kicad_sch"
    human_sch.write_text("")
    llm_sch.write_text("")

    call_count = {"n": 0}

    def fake_extract(path: str) -> list[dict]:
        call_count["n"] += 1
        return human_bom if call_count["n"] == 1 else llm_bom

    with patch.object(hardware_diff, "_extract_bom", side_effect=fake_extract):
        with patch.object(
            hardware_diff,
            "run_drc",
            return_value=DRCResult(passed=True, errors=[]),
        ):
            result = compare(str(human_sch), str(llm_sch))

    assert result["score"] == 0.75
    assert result["details"]["bom_match"] == 1
    assert result["details"]["types_match"] is True


def test_compare_missing_files_returns_zero_bom(tmp_path):
    """Missing paths skip DRC and return a low score deterministically."""
    with patch.object(hardware_diff, "_extract_bom", return_value=[]):
        result = compare(
            str(tmp_path / "nope1.sch"), str(tmp_path / "nope2.sch")
        )

    assert result["details"]["drc_errors"] is None
    # No BOM, no types, no DRC → bom_score = 0 / 1 = 0, same_types True
    # (both empty sets equal) → 0.5 + 0 - 0 = 0.5.
    assert result["score"] == 0.5


def test_compare_different_types_no_half_bonus(tmp_path):
    """Different type sets drop the 0.5 bonus."""
    human_bom = [{"ref": "R1", "value": "10k", "type": "R"}]
    llm_bom = [{"ref": "C1", "value": "10k", "type": "C"}]
    human_sch = tmp_path / "h.kicad_sch"
    llm_sch = tmp_path / "l.kicad_sch"
    human_sch.write_text("")
    llm_sch.write_text("")

    call_count = {"n": 0}

    def fake_extract(path: str) -> list[dict]:
        call_count["n"] += 1
        return human_bom if call_count["n"] == 1 else llm_bom

    with patch.object(hardware_diff, "_extract_bom", side_effect=fake_extract):
        with patch.object(
            hardware_diff,
            "run_drc",
            return_value=DRCResult(passed=True, errors=[]),
        ):
            result = compare(str(human_sch), str(llm_sch))

    assert result["details"]["types_match"] is False
    # 0 (types) + 0.5 × 0 (no value matches) - 0 = 0.0.
    assert result["score"] == 0.0
