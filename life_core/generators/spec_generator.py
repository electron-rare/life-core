"""Spec generator: LLM → Markdown + YAML frontmatter with key validation."""
from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from litellm import completion

from .base import BaseGenerator, GenerationContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(),
)

_REQUIRED_FRONTMATTER_KEYS = {
    "description",
    "inputs",
    "outputs",
    "constraints",
    "acceptance_criteria",
    "compliance",
}


class SpecGenerator(BaseGenerator):
    """Produce a spec deliverable as Markdown with YAML frontmatter."""

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
        text = data.decode(errors="replace")
        if not text.startswith("---"):
            return False, [
                "Output must start with '---' YAML frontmatter."
            ]
        try:
            _, frontmatter, _ = text.split("---", 2)
            meta = yaml.safe_load(frontmatter)
        except (ValueError, yaml.YAMLError) as exc:
            return False, [f"Malformed YAML frontmatter: {exc}"]
        missing = _REQUIRED_FRONTMATTER_KEYS - set(meta or {})
        if missing:
            return False, [
                f"Frontmatter missing keys: {sorted(missing)}"
            ]
        return True, []
