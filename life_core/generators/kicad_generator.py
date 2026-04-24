"""KiCad 8 generator (ADR-006).

LLM emits a JSON description of a schematic (components, wires, labels);
this module translates it into a ``.kicad_sch`` via ``kiutils`` and validates
with ``kicad-cli`` DRC.

ADR-006 rejected SKiDL (no KiCad 8 support) and raw ``sexpdata`` (missing
UUIDs, paper, title_block, lib_symbols). ``kiutils`` 1.4.x is the chosen
backend because it round-trips KiCad 8 conformant files.

kiutils 1.4.8 API notes:
  - ``Schematic.create_new()`` yields a valid empty schematic (version,
    generator, paper, sheet_instances already populated).
  - Components go into ``sch.schematicSymbols`` (list of ``SchematicSymbol``).
  - Wires do not have a dedicated ``Wire`` class at this version; we use
    ``PolyLine`` in ``sch.graphicalItems``.
  - Labels go into ``sch.labels`` (list of ``LocalLabel``).
  - ``Position`` uses keyword args ``X``, ``Y``, ``angle``.
  - ``Property`` uses ``key`` / ``value`` keyword args.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from litellm import completion

from life_core.tools.kicad_cli import run_drc

from .base import BaseGenerator, GenerationContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(),
)


def _render_kiutils_from_json(payload: dict[str, Any]) -> str:
    """Translate the LLM JSON payload into a KiCad 8 ``.kicad_sch`` file.

    The produced file must be parseable by ``kicad-cli`` (contains paper,
    title_block, lib_symbols stubs, and per-symbol UUIDs). Returns the
    absolute path to the generated schematic.
    """
    from kiutils.items.common import Position, Property
    from kiutils.items.schitems import (
        LocalLabel,
        PolyLine,
        SchematicSymbol,
    )
    from kiutils.schematic import Schematic

    sch = Schematic.create_new()

    # Components.
    for comp in payload.get("components", []):
        at = comp.get("at", [0, 0, 0])
        sym = SchematicSymbol()
        sym.libId = comp["lib_id"]
        sym.position = Position(
            X=float(at[0]),
            Y=float(at[1]),
            angle=float(at[2]) if len(at) > 2 else 0.0,
        )
        sym.uuid = str(uuid.uuid4())
        sym.properties = [
            Property(key="Reference", value=str(comp.get("reference", "?"))),
            Property(key="Value", value=str(comp.get("value", "?"))),
            Property(key="Footprint", value=str(comp.get("footprint", ""))),
        ]
        sch.schematicSymbols.append(sym)

    # Wires — kiutils 1.4.8 has no dedicated Wire class; use PolyLine in
    # graphicalItems, which serialises to a compatible s-expression.
    for w in payload.get("wires", []):
        wire = PolyLine()
        wire.points = [
            Position(X=float(p[0]), Y=float(p[1]))
            for p in w.get("pts", [])
        ]
        wire.uuid = str(uuid.uuid4())
        sch.graphicalItems.append(wire)

    # Labels.
    for lab in payload.get("labels", []):
        at = lab.get("at", [0, 0])
        label = LocalLabel()
        label.text = str(lab.get("text", ""))
        label.position = Position(X=float(at[0]), Y=float(at[1]))
        label.uuid = str(uuid.uuid4())
        sch.labels.append(label)

    out_path = Path(tempfile.mkdtemp(prefix="kicad_")) / "output.kicad_sch"
    sch.to_file(str(out_path))
    return str(out_path)


class KicadGenerator(BaseGenerator):
    """LLM-backed KiCad 8 schematic generator with DRC validation loop."""

    def call_llm(self, ctx: GenerationContext) -> bytes:
        prompt = _env.get_template(ctx.prompt_template).render(
            **ctx.prompt_vars,
            deliverable_slug=ctx.deliverable_slug,
            deliverable_type=ctx.deliverable_type,
        )
        response = completion(
            model=ctx.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"].encode()

    def validate(
        self, data: bytes, ctx: GenerationContext
    ) -> tuple[bool, list[str], float]:
        """Return ``(ok, errors, score)``.

        Score is ``1.0 - 0.1 * len(drc_errors)`` clamped to ``[0.0, 1.0]``
        for renderable JSON; ``0.0`` for malformed JSON, missing keys, or
        kiutils render failure.
        """
        text = data.decode(errors="replace").strip()
        # Tolerate triple-backticks wrappers from some models.
        if text.startswith("```"):
            _, _, rest = text.partition("```")
            rest = rest.lstrip("json").lstrip("\n")
            text = rest.rsplit("```", 1)[0]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, [f"Output is not valid JSON: {exc}"], 0.0
        required = {"components", "wires", "labels"}
        missing = required - set(payload)
        if missing:
            return False, [f"JSON missing keys: {sorted(missing)}"], 0.0
        try:
            sch_path = _render_kiutils_from_json(payload)
        except Exception as exc:  # noqa: BLE001 — surface any render error
            return False, [f"kiutils render failed: {exc}"], 0.0
        drc = run_drc(sch_path)
        score = max(0.0, min(1.0, 1.0 - 0.1 * len(drc.errors)))
        if drc.errors:
            return (
                False,
                [
                    f"DRC: {err.get('description', '(no description)')}"
                    for err in drc.errors
                ],
                score,
            )
        return True, [], 1.0
