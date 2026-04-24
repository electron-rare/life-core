from unittest.mock import patch

from life_core.generators.base import GenerationContext
from life_core.generators.spec_generator import SpecGenerator


def test_spec_generator_calls_llm_and_validates_yaml_frontmatter():
    ctx = GenerationContext(
        deliverable_slug="s-spec",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="mascarade-spec",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
    )
    valid_md = (
        b"---\n"
        b"description: x\n"
        b"inputs: []\n"
        b"outputs: []\n"
        b"constraints: []\n"
        b"acceptance_criteria: []\n"
        b"compliance: prototype\n"
        b"---\n"
        b"# Body\n"
    )
    with patch(
        "life_core.generators.spec_generator.completion",
        return_value={
            "choices": [{"message": {"content": valid_md.decode()}}],
            "usage": {},
        },
    ):
        gen = SpecGenerator()
        outcome = gen.generate(ctx)
    assert outcome.ok is True


def test_spec_generator_rejects_missing_frontmatter():
    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="mascarade-spec",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
        max_reprompts=0,
    )
    with patch(
        "life_core.generators.spec_generator.completion",
        return_value={
            "choices": [{"message": {"content": "no frontmatter here"}}],
            "usage": {},
        },
    ):
        gen = SpecGenerator()
        outcome = gen.generate(ctx)
    assert outcome.ok is False
    assert any("frontmatter" in e for e in outcome.errors)
