from life_core.generators.base import (
    BaseGenerator,
    GenerationContext,
    GenerationOutcome,
)


class DummyGen(BaseGenerator):
    def call_llm(self, ctx: GenerationContext) -> bytes:
        return b"dummy content"

    def validate(
        self, data: bytes, ctx: GenerationContext
    ) -> tuple[bool, list[str]]:
        return True, []


def test_base_generator_round_trip(tmp_path):
    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="mascarade-spec",
        prompt_vars={"brief": "x", "constraints": [], "upstream": []},
    )
    gen = DummyGen()
    outcome = gen.generate(ctx)
    assert isinstance(outcome, GenerationOutcome)
    assert outcome.ok is True
    assert outcome.data == b"dummy content"
