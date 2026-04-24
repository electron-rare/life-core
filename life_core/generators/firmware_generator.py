"""PlatformIO firmware generator (T1.7a).

LLM emits a JSON object ``{"platformio_ini": ..., "src_main_cpp": ...}``.
Validation writes a temp project and runs ``pio run -e native`` via
``life_core.tools.platformio.build_native``. Build failure surfaces the
last stderr line as a ``pio run failed: ...`` error so the reprompt loop
can feed it back to the model.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from litellm import completion

from life_core.tools.platformio import build_native

from .base import BaseGenerator, GenerationContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(),
)
_REQUIRED_KEYS = {"platformio_ini", "src_main_cpp"}


class FirmwareGenerator(BaseGenerator):
    """LLM-backed PlatformIO generator with native-build validation."""

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
            rest = rest.lstrip("json").lstrip("\n")
            text = rest.rsplit("```", 1)[0]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return False, [f"Output is not valid JSON: {exc}"]
        missing = _REQUIRED_KEYS - set(payload)
        if missing:
            return False, [f"JSON missing keys: {sorted(missing)}"]

        workdir = Path(tempfile.mkdtemp(prefix="pio_"))
        (workdir / "platformio.ini").write_text(payload["platformio_ini"])
        (workdir / "src").mkdir()
        (workdir / "src" / "main.cpp").write_text(payload["src_main_cpp"])
        build = build_native(str(workdir))
        if not build.ok:
            last = (
                build.stderr.strip().splitlines()[-1]
                if build.stderr
                else "unknown"
            )
            return False, [f"pio run failed: {last}"]
        return True, []
