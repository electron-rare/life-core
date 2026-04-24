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


def test_attempt_history_accumulates_on_retries():
    """Each failed attempt is appended to attempt_history."""
    seen_histories = []

    class TwoFailsThenSucceeds(BaseGenerator):
        def __init__(self):
            self.calls = 0

        def call_llm(self, ctx):
            seen_histories.append(
                list(ctx.prompt_vars.get("attempt_history", []))
            )
            self.calls += 1
            return f"call_{self.calls}".encode()

        def validate(self, data, ctx):
            if self.calls < 3:
                return False, [f"err{self.calls}"]
            return True, []

    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="m",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
        max_reprompts=2,
    )
    gen = TwoFailsThenSucceeds()
    outcome = gen.generate(ctx)
    assert outcome.ok is True
    # First call: empty history. Second call: 1 entry. Third call: 2 entries.
    assert seen_histories[0] == []
    assert len(seen_histories[1]) == 1 and seen_histories[1][0]["attempt"] == 1
    assert len(seen_histories[2]) == 2
    assert seen_histories[2][1]["errors"] == ["err2"]


def test_attempt_history_backward_compat_human_feedback_set():
    """human_feedback still set so existing prompts keep working."""
    seen_feedback = []

    class FailsOnce(BaseGenerator):
        def __init__(self):
            self.calls = 0

        def call_llm(self, ctx):
            seen_feedback.append(ctx.prompt_vars.get("human_feedback"))
            self.calls += 1
            return b"ok"

        def validate(self, data, ctx):
            if self.calls < 2:
                return False, ["e"]
            return True, []

    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="m",
        prompt_vars={},
        max_reprompts=1,
    )
    FailsOnce().generate(ctx)
    assert seen_feedback[0] is None
    assert seen_feedback[1] is not None and "e" in seen_feedback[1]


def test_n_candidates_picks_highest_score():
    scores = [0.4, 0.7, 0.6]
    call_idx = [0]

    class ScoredGen(BaseGenerator):
        def call_llm(self, ctx):
            return f"data_{call_idx[0]}".encode()

        def validate(self, data, ctx):
            i = call_idx[0]
            call_idx[0] += 1
            s = scores[i % len(scores)]
            return (
                s == max(scores),
                [] if s == max(scores) else ["lower"],
                s,
            )

    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="m",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
        n_candidates=3,
        max_reprompts=0,
    )
    outcome = ScoredGen().generate(ctx)
    assert outcome.score == 0.7
    assert outcome.ok is True


def test_n_candidates_legacy_validate_two_tuple_still_works():
    """Validators that return 2-tuple are auto-promoted to (ok, errors, 1.0|0.0)."""

    class Legacy(BaseGenerator):
        def call_llm(self, ctx):
            return b"x"

        def validate(self, data, ctx):
            return (True, [])  # 2-tuple

    ctx = GenerationContext(
        deliverable_slug="s",
        deliverable_type="spec",
        prompt_template="spec.j2",
        llm_model="m",
        prompt_vars={},
        max_reprompts=0,
    )
    outcome = Legacy().generate(ctx)
    assert outcome.ok is True
    assert outcome.score == 1.0
