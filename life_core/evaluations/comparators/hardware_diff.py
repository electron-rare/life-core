"""Hardware diff comparator (P4 T1.9b).

Scores an LLM KiCad schematic against a human gold standard by comparing
the BOM (component types + values within ±20 %) and penalising DRC
errors on the LLM schematic. All values are bounded to ``[0, 1]``.
"""
from __future__ import annotations

import re
from pathlib import Path

from life_core.tools.kicad_cli import run_drc


def _extract_bom(sch_path: str) -> list[dict]:
    """Parse a KiCad schematic and return a minimal BOM list.

    Returns an empty list when the file is missing so callers exercise
    the math path without raising. ``type`` is the reference prefix
    (``R``, ``C``, …) which is enough for type-set equality.
    """

    if not Path(sch_path).exists():
        return []
    from kiutils.schematic import Schematic

    sch = Schematic().from_file(sch_path)
    items: list[dict] = []
    for symbol in getattr(sch, "schematicSymbols", []):
        props = getattr(symbol, "properties", []) or []
        ref = next(
            (p.value for p in props if getattr(p, "key", None) == "Reference"),
            "?",
        )
        val = next(
            (p.value for p in props if getattr(p, "key", None) == "Value"),
            "?",
        )
        items.append(
            {"ref": ref, "value": val, "type": ref[0] if ref else "?"}
        )
    return items


def _value_close(v1: str, v2: str, tol: float = 0.20) -> bool:
    """Numeric-aware string compare with ±``tol`` fractional tolerance.

    Falls back to strict equality when either side has no leading
    numeric prefix (e.g. ``"NE555"``).
    """

    m1 = re.match(r"([\d.]+)", v1)
    m2 = re.match(r"([\d.]+)", v2)
    if not m1 or not m2:
        return v1 == v2
    x1, x2 = float(m1.group(1)), float(m2.group(1))
    if x1 == 0:
        return x2 == 0
    return abs(x1 - x2) / x1 <= tol


def compare(human_sch: str, llm_sch: str) -> dict:
    """Score the LLM schematic against the human one."""

    human_bom = _extract_bom(human_sch)
    llm_bom = _extract_bom(llm_sch)
    by_type_h = {b["type"] for b in human_bom}
    by_type_l = {b["type"] for b in llm_bom}
    same_types = by_type_h == by_type_l
    value_matches = sum(
        1
        for h in human_bom
        if any(
            l["type"] == h["type"] and _value_close(l["value"], h["value"])
            for l in llm_bom
        )
    )
    bom_score = value_matches / max(len(human_bom), 1)

    drc_llm = run_drc(llm_sch) if Path(llm_sch).exists() else None
    drc_penalty = 0.1 * (len(drc_llm.errors) if drc_llm else 0)

    score = max(
        0.0,
        min(
            1.0,
            (0.5 if same_types else 0.0) + 0.5 * bom_score - drc_penalty,
        ),
    )
    return {
        "score": score,
        "details": {
            "bom_match": value_matches,
            "bom_total": len(human_bom),
            "types_match": same_types,
            "drc_errors": len(drc_llm.errors) if drc_llm else None,
        },
    }
