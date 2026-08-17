"""
Conversation compaction: what happens to a turn that falls out of the
window.

Run against a real ConversationMemory - the whole behaviour is about which
rows are read and which id the high-water mark lands on, so a fake memory
would be testing the fake.
"""
import pytest

from memory.compaction import SUMMARY_PREFIX, SUMMARY_ROLE, ConversationCompactor
from memory.conversation_memory import ConversationMemory


class FakeLLMService:
    """Returns canned summaries and records how it was called."""

    def __init__(self, summaries: list[str] | None = None) -> None:
        self._summaries = list(summaries or ["A summary of the earlier turns."])
        self.prompts: list[str] = []
        self.temperatures: list[float | None] = []

    def generate_response(
        self, user_message, history=None, system_prompt=None, temperature=None
    ) -> str:
        self.prompts.append(user_message)
        self.temperatures.append(temperature)
        if len(self._summaries) > 1:
            return self._summaries.pop(0)
        return self._summaries[0]


class FailingLLMService:
    def generate_response(self, *args, **kwargs):
        raise RuntimeError("LLM request failed: upstream timeout")


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(db_path=str(tmp_path / "conversations.sqlite3"))


def _fill(memory, conversation_id: str, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        memory.add_message(conversation_id, role="user", content=f"message {i}")


def _compactor(memory, llm, history_limit=4, trigger=2) -> ConversationCompactor:
    return ConversationCompactor(
        memory=memory,
        llm_service=llm,
        history_limit=history_limit,
        trigger=trigger,
    )


# ---- When it runs ----------------------------------------------------------


def test_a_conversation_inside_the_window_is_not_compacted(memory):
    llm = FakeLLMService()
    _fill(memory, "conv-1", 4)

    _compactor(memory, llm).compact("conv-1")

    assert llm.prompts == []
    assert memory.get_summary("conv-1") is None


def test_nothing_happens_until_the_trigger_is_reached(memory):
    llm = FakeLLMService()
    # 5 messages, window of 4 -> exactly 1 has fallen out, trigger is 2.
    _fill(memory, "conv-1", 5)

    _compactor(memory, llm).compact("conv-1")

    assert llm.prompts == []


def test_turns_that_fall_out_of_the_window_are_summarised(memory):
    llm = FakeLLMService(["The user introduced themselves as Ndriqim."])
    _fill(memory, "conv-1", 6)

    _compactor(memory, llm).compact("conv-1")

    summary = memory.get_summary("conv-1")
    assert summary is not None
    assert summary.summary == "The user introduced themselves as Ndriqim."
    # Only the two that dropped out - the window's four are still being sent
    # verbatim, and summarising them too would put them in the prompt twice.
    assert "message 0" in llm.prompts[0]
    assert "message 1" in llm.prompts[0]
    assert "message 2" not in llm.prompts[0]


def test_the_summary_is_generated_deterministically(memory):
    llm = FakeLLMService()
    _fill(memory, "conv-1", 6)

    _compactor(memory, llm).compact("conv-1")

    # This is a transformation of existing text, and each pass reads the
    # last one's output - sampling noise compounds.
    assert llm.temperatures == [0]


# ---- Not doing the same work twice -----------------------------------------


def test_a_second_pass_only_covers_what_the_first_one_did_not(memory):
    llm = FakeLLMService(["first summary", "second summary"])
    compactor = _compactor(memory, llm)

    _fill(memory, "conv-1", 6)
    compactor.compact("conv-1")
    _fill(memory, "conv-1", 2, start=6)
    compactor.compact("conv-1")

    second_prompt = llm.prompts[1]
    # The turns the first pass already covered are not re-read...
    assert "message 0" not in second_prompt
    assert "message 1" not in second_prompt
    # ...but its output is, since the new summary replaces it wholesale.
    assert "first summary" in second_prompt
    assert memory.get_summary("conv-1").summary == "second summary"


def test_the_high_water_mark_advances(memory):
    llm = FakeLLMService()
    compactor = _compactor(memory, llm)
    _fill(memory, "conv-1", 6)

    compactor.compact("conv-1")
    first_mark = memory.get_summary("conv-1").covered_through_id

    _fill(memory, "conv-1", 4, start=6)
    compactor.compact("conv-1")

    assert memory.get_summary("conv-1").covered_through_id > first_mark


def test_a_repeated_pass_with_no_new_turns_does_nothing(memory):
    llm = FakeLLMService()
    compactor = _compactor(memory, llm)
    _fill(memory, "conv-1", 6)

    compactor.compact("conv-1")
    compactor.compact("conv-1")

    assert len(llm.prompts) == 1


def test_no_message_is_both_summarised_and_in_the_window(memory):
    llm = FakeLLMService()
    compactor = _compactor(memory, llm, history_limit=4, trigger=1)
    _fill(memory, "conv-1", 10)

    compactor.compact("conv-1")

    covered = memory.get_summary("conv-1").covered_through_id
    window = memory.get_history("conv-1", limit=4)
    # The boundary belongs to exactly one side. In it twice would waste the
    # budget; in neither is the amnesia this exists to fix.
    assert window[0].message_id == covered + 1


# ---- Failure ---------------------------------------------------------------


def test_a_failed_pass_does_not_raise(memory):
    _fill(memory, "conv-1", 6)

    # The user's answer has already been produced by this point; a
    # summarising call that times out must not turn it into a 502.
    _compactor(memory, FailingLLMService()).compact("conv-1")

    assert memory.get_summary("conv-1") is None


def test_a_failed_pass_is_retried_next_time(memory):
    _fill(memory, "conv-1", 6)
    _compactor(memory, FailingLLMService()).compact("conv-1")

    llm = FakeLLMService(["recovered summary"])
    _compactor(memory, llm).compact("conv-1")

    assert memory.get_summary("conv-1").summary == "recovered summary"
    assert "message 0" in llm.prompts[0]


def test_an_empty_summary_never_replaces_a_good_one(memory):
    compactor_llm = FakeLLMService(["a real summary"])
    compactor = _compactor(memory, compactor_llm)
    _fill(memory, "conv-1", 6)
    compactor.compact("conv-1")

    _fill(memory, "conv-1", 4, start=6)
    _compactor(memory, FakeLLMService(["   "])).compact("conv-1")

    assert memory.get_summary("conv-1").summary == "a real summary"


# ---- What the prompt gets --------------------------------------------------


def test_a_service_sends_the_summary_and_compacts_after_the_turn(memory):
    """
    The wiring, not the compactor: a summary is worth nothing if the
    services don't put it in the prompt.
    """
    from app.services.chat_service import ChatService

    class RecordingLLM(FakeLLMService):
        def __init__(self):
            super().__init__(["a summary of the early turns"])
            self.histories: list[list[dict]] = []

        def generate_response(
            self, user_message, history=None, system_prompt=None, temperature=None
        ):
            self.histories.append(list(history or []))
            return super().generate_response(
                user_message, history, system_prompt, temperature
            )

    llm = RecordingLLM()
    service = ChatService(
        llm_service=llm,
        memory=memory,
        history_limit=4,
        # A separate LLM for compaction, so `llm.histories` records the
        # prompts the *user's* turns were answered with and nothing else.
        compactor=_compactor(
            memory,
            FakeLLMService(["a summary of the early turns"]),
            history_limit=4,
            trigger=2,
        ),
    )

    _fill(memory, "conv-1", 6)
    service.get_answer("first", conversation_id="conv-1")
    # That turn's own history predates any summary; the compaction it
    # triggers is what the *next* turn reads.
    service.get_answer("second", conversation_id="conv-1")

    assert llm.histories[0][0]["role"] != SUMMARY_ROLE
    assert llm.histories[-1][0]["role"] == SUMMARY_ROLE
    assert "a summary of the early turns" in llm.histories[-1][0]["content"]


def test_a_service_without_a_compactor_behaves_exactly_as_before(memory):
    from app.services.chat_service import ChatService

    llm = FakeLLMService(["an answer"])
    service = ChatService(llm_service=llm, memory=memory, history_limit=4)

    _fill(memory, "conv-1", 6)
    service.get_answer("hello", conversation_id="conv-1")

    assert memory.get_summary("conv-1") is None


def test_summary_prefix_is_empty_for_an_uncompacted_conversation(memory):
    assert _compactor(memory, FakeLLMService()).summary_prefix("conv-1") == []


def test_summary_prefix_carries_the_summary_as_a_system_message(memory):
    llm = FakeLLMService(["The user is deploying to Vercel."])
    _fill(memory, "conv-1", 6)
    compactor = _compactor(memory, llm)
    compactor.compact("conv-1")

    prefix = compactor.summary_prefix("conv-1")

    assert len(prefix) == 1
    # Not "user" or "assistant": neither party said this, and labelling it
    # as one of them invites the model to answer it or to treat it as its
    # own previous words.
    assert prefix[0]["role"] == SUMMARY_ROLE
    assert SUMMARY_PREFIX in prefix[0]["content"]
    assert "The user is deploying to Vercel." in prefix[0]["content"]
