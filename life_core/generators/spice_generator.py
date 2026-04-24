"""ngspice SPICE generator (T1.7b).

LLM emits raw ngspice netlist text. Validation requires a ``.end`` card and
delegates convergence to ``life_core.tools.ngspice.simulate``. Triple-backtick
wrappers are tolerated.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from litellm import completion

from life_core.tools.ngspice import simulate

from .base import BaseGenerator, GenerationContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(),
)


class SpiceGenerator(BaseGenerator):
    """LLM-backed ngspice netlist generator with convergence validation."""

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
    ) -> tuple[bool, list[str]]:
        text = data.decode(errors="replace").strip()
        # Tolerate triple-backticks wrappers from some models.
        if text.startswith("```"):
            _, _, rest = text.partition("```")
            rest = rest.lstrip("text").lstrip("\n")
            text = rest.rsplit("```", 1)[0].strip()
        if not text.lower().endswith(".end"):
            return False, ["Netlist must end with .end"]
        path = Path(tempfile.mkdtemp(prefix="spice_")) / "tb.cir"
        path.write_text(text)
        result = simulate(str(path))
        if not result.converged:
            return False, [
                f"ngspice did not converge: {result.errors}"
            ]
        return True, []
