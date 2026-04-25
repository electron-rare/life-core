"""Hardware diff comparator (P4 T1.9b).

Scores an LLM KiCad schematic against a human gold standard along
three axes:

1. **BOM similarity** — type-set equality plus per-component value
   match within ±20 % (existing structural diff).
2. **ERC cleanliness** — penalises ERC violations on the LLM
   schematic (``kicad-cli sch erc``). Replaces the previous misuse
   of ``run_drc`` which silently no-op'd on ``.kicad_sch`` files.
3. **Required-components coverage** *(new)* — when a spec frontmatter
   declares ``required_components`` (or a heuristic class lookup
   matches the spec body), measure what fraction the LLM schematic
   actually instantiates. Surfaced as ``required_coverage`` in the
   details and folded into the final score with weights 0.6 (BOM
   structural) / 0.4 (required components) when a spec is provided.

All values are bounded to ``[0, 1]``. The ``compare`` signature stays
backward-compatible: callers that do not pass ``spec_path`` get the
legacy structural-only behaviour.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import yaml

from life_core.tools.kicad_cli import run_erc

# Heuristic fallback: when a spec.md has no frontmatter, scan its body
# for known component classes. Patterns are matched case-insensitively
# against both the spec text and each LLM BOM symbol's lib_id / value.
_HEURISTIC_CLASSES: dict[str, str] = {
    "INA226": "current sense",
    "INA237": "current sense",
    "ESP32": "MCU",
    "STM32": "MCU",
    "RP2040": "MCU",
    "ATMEGA": "MCU",
    "MOSFET": "switch",
    "TCA9535": "GPIO expander",
    "TCA9548": "I2C mux",
    "LDO": "regulator",
    "TPS5": "buck regulator",
    "NE555": "timer",
}


def _extract_bom(sch_path: str) -> list[dict]:
    """Parse a KiCad schematic and return a minimal BOM list.

    Returns an empty list when the file is missing so callers exercise
    the math path without raising. ``type`` is the reference prefix
    (``R``, ``C``, …) which is enough for type-set equality. ``lib_id``
    is preserved when present so the required-components matcher can
    use upstream library identifiers.
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
        lib_id = getattr(symbol, "libId", "") or getattr(symbol, "libraryNickname", "")
        items.append(
            {
                "ref": ref,
                "value": val,
                "type": ref[0] if ref else "?",
                "lib_id": lib_id,
            }
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


def _parse_spec_frontmatter(spec_path: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)`` for a Markdown spec file.

    Supports the ``---\\n<yaml>\\n---\\n<body>`` convention. When no
    frontmatter is present, returns ``({}, full_text)``.
    """

    p = Path(spec_path)
    if not p.exists():
        return {}, ""
    text = p.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return fm, parts[2]


def _required_components_from_spec(
    spec_path: str,
) -> list[dict[str, str]]:
    """Resolve the required-components list for a spec file.

    Order of precedence:

    1. ``required_components`` array in the YAML frontmatter.
    2. Heuristic lookup over the spec body using
       :data:`_HEURISTIC_CLASSES`.

    Each entry has the shape ``{"lib_id_match": "*PAT*", "role": "..."}``.
    """

    fm, body = _parse_spec_frontmatter(spec_path)
    declared = fm.get("required_components") if isinstance(fm, dict) else None
    if isinstance(declared, list) and declared:
        out: list[dict[str, str]] = []
        for item in declared:
            if isinstance(item, dict) and "lib_id_match" in item:
                out.append(
                    {
                        "lib_id_match": str(item["lib_id_match"]),
                        "role": str(item.get("role", "")),
                    }
                )
        if out:
            return out

    # Heuristic fallback.
    body_upper = body.upper()
    out = []
    for pattern, role in _HEURISTIC_CLASSES.items():
        if pattern in body_upper:
            out.append({"lib_id_match": f"*{pattern}*", "role": role})
    return out


def _bom_matches_required(
    bom: list[dict], requirement: dict[str, str]
) -> bool:
    """Return True when at least one BOM entry matches a requirement.

    Match runs against ``lib_id`` then ``value`` (case-insensitive
    glob on ``lib_id_match``).
    """

    pat = requirement["lib_id_match"].upper()
    for entry in bom:
        for field in ("lib_id", "value", "ref"):
            v = str(entry.get(field, "")).upper()
            if v and fnmatch.fnmatchcase(v, pat):
                return True
    return False


def _required_components_coverage(
    llm_bom: list[dict], requirements: list[dict[str, str]]
) -> tuple[float, int, int]:
    """Compute coverage = matched / required.

    Returns ``(coverage, matched, total)``. When ``requirements`` is
    empty, returns ``(1.0, 0, 0)`` so the score is not penalised.
    """

    if not requirements:
        return 1.0, 0, 0
    matched = sum(1 for r in requirements if _bom_matches_required(llm_bom, r))
    return matched / len(requirements), matched, len(requirements)


def compare(
    human_sch: str, llm_sch: str, spec_path: str | None = None
) -> dict:
    """Score the LLM schematic against the human one.

    When ``spec_path`` is provided, the score blends the structural
    BOM diff (weight 0.6) with the required-components coverage
    (weight 0.4) before applying the ERC penalty. When ``spec_path``
    is ``None``, the legacy structural-only formula is used.
    """

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
    structural = (0.5 if same_types else 0.0) + 0.5 * bom_score

    erc_llm = run_erc(llm_sch) if Path(llm_sch).exists() else None
    erc_penalty = 0.1 * (len(erc_llm.errors) if erc_llm else 0)

    requirements: list[dict[str, str]] = []
    coverage = 1.0
    matched = 0
    total = 0
    if spec_path is not None:
        requirements = _required_components_from_spec(spec_path)
        coverage, matched, total = _required_components_coverage(
            llm_bom, requirements
        )

    if total > 0:
        base = 0.6 * structural + 0.4 * coverage
    else:
        base = structural

    score = max(0.0, min(1.0, base - erc_penalty))

    details: dict[str, Any] = {
        "bom_match": value_matches,
        "bom_total": len(human_bom),
        "types_match": same_types,
        "erc_errors": len(erc_llm.errors) if erc_llm else None,
    }
    if total > 0:
        details["required_coverage"] = coverage
        details["required_matched"] = matched
        details["required_total"] = total
    return {"score": score, "details": details}
