"""
Tests for the hand-written research loop (agents/research_agent.py).

Split in two, matching where the risk actually is:

- The parser tests feed it the malformed replies real models produce. This
  is the part with no framework behind it, so it is the part that has to be
  pinned down case by case.

- The loop tests script the LLM's replies and assert on control flow -
  which tool ran, how many times, why the loop stopped. Nothing here calls
  a model or touches a vector store; a "reasoning loop" that can only be
  tested against a live LLM is one you cannot debug.
"""
import pytest

from agents.agent_loop import ActionParseError, _extract_json_object, _parse_action
from agents.research_agent import ResearchAgent
from rag.document_registry import DocumentRecord
from rag.retriever import RetrievedChunk


# ---- Fakes -----------------------------------------------------------------


class FakeLLMService:
    """
    Replays a scripted list of replies, one per call.

    Deliberately raises when the script runs out rather than returning a
    default: a loop that took an extra turn is exactly the bug these tests
    exist to catch, and a lenient fake would let it pass silently.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.system_prompts: list[str | None] = []
        self.temperatures: list[float | None] = []

    def generate_response(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(user_message)
        self.system_prompts.append(system_prompt)
        self.temperatures.append(temperature)
        if not self.replies:
            raise AssertionError(
                f"LLM called {len(self.prompts)} times but only "
                f"{len(self.prompts) - 1} replies were scripted."
            )
        return self.replies.pop(0)


class FakeRetriever:
    """Returns a preset result per search, in order, and records the queries."""

    def __init__(self, results: list[list[RetrievedChunk]]) -> None:
        self.results = list(results)
        self.queries: list[str] = []

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.queries.append(question)
        return self.results.pop(0) if self.results else []


class FakeRegistry:
    def __init__(self, records: list[DocumentRecord] | None = None) -> None:
        self.records = records if records is not None else [_record("handbook.pdf")]

    def list_documents(self) -> list[DocumentRecord]:
        return self.records


def _record(filename: str, status: str = "READY") -> DocumentRecord:
    return DocumentRecord(
        document_id=f"doc-{filename}",
        filename=filename,
        page_count=10,
        chunk_count=40,
        status=status,
        created_at="2026-01-01T00:00:00Z",
        title=filename.removesuffix(".pdf").title(),
    )


def _chunk(chunk_id: str, text: str = "some passage text", **overrides) -> RetrievedChunk:
    fields = {
        "chunk_id": chunk_id,
        "text": text,
        "filename": "handbook.pdf",
        "page": 3,
        "score": 0.81,
    }
    fields.update(overrides)
    return RetrievedChunk(**fields)


def _agent(llm, retriever=None, registry=None, **overrides) -> ResearchAgent:
    return ResearchAgent(
        llm_service=llm,
        retriever=retriever or FakeRetriever([]),
        document_registry=registry or FakeRegistry(),
        **{"max_steps": 5, "search_top_k": 3, **overrides},
    )


def _finish(answer: str) -> str:
    return f'Thought: I have enough.\nAction: {{"tool": "finish", "input": {{"answer": "{answer}"}}}}'


def _search(query: str) -> str:
    return f'Thought: Looking this up.\nAction: {{"tool": "search", "input": {{"query": "{query}"}}}}'


# ---- The brace scanner -----------------------------------------------------


def test_extract_json_object_stops_at_the_matching_brace():
    """Prose after the action must not be swallowed into it."""
    assert (
        _extract_json_object('{"tool": "search"} and then I will summarise.')
        == '{"tool": "search"}'
    )


def test_extract_json_object_handles_nesting():
    text = '{"tool": "search", "input": {"query": "refunds"}} trailing'
    assert _extract_json_object(text) == '{"tool": "search", "input": {"query": "refunds"}}'


def test_extract_json_object_ignores_braces_inside_strings():
    """A finish answer quoting a document can contain a brace."""
    text = '{"tool": "finish", "input": {"answer": "the template is {name}"}}'
    assert _extract_json_object(text) == text


def test_extract_json_object_ignores_escaped_quotes():
    text = r'{"tool": "finish", "input": {"answer": "he said \"no\" clearly"}}'
    assert _extract_json_object(text) == text


def test_extract_json_object_returns_none_when_unbalanced():
    """A reply truncated by a token limit mid-object."""
    assert _extract_json_object('{"tool": "finish", "input": {"answer": "half') is None


# ---- The action parser -----------------------------------------------------


def test_parse_action_reads_a_clean_reply():
    thought, _, tool, tool_input = _parse_action(
        'Thought: I should look this up.\nAction: {"tool": "search", "input": {"query": "refunds"}}'
    )
    assert thought == "I should look this up."
    assert tool == "search"
    assert tool_input == {"query": "refunds"}


def test_parse_action_strips_a_code_fence():
    raw = '```json\nThought: checking.\nAction: {"tool": "search", "input": {"query": "x"}}\n```'
    _, _, tool, tool_input = _parse_action(raw)
    assert (tool, tool_input) == ("search", {"query": "x"})


def test_parse_action_ignores_prose_after_the_action():
    _, _, tool, _ = _parse_action(
        'Action: {"tool": "list_documents", "input": {}}\nThen I will search each one.'
    )
    assert tool == "list_documents"


def test_parse_action_accepts_other_frameworks_key_names():
    """ReAct's original spelling and the OpenAI tool-calling spelling."""
    _, _, tool, tool_input = _parse_action(
        'Action: {"action": "search", "action_input": {"query": "refunds"}}'
    )
    assert (tool, tool_input) == ("search", {"query": "refunds"})


def test_parse_action_accepts_a_bare_string_input():
    _, _, tool, tool_input = _parse_action('Action: {"tool": "search", "input": "refunds"}')
    assert tool == "search"
    assert tool_input == {"__bare__": "refunds"}


def test_parse_action_accepts_a_missing_input_for_no_arg_tools():
    _, _, tool, tool_input = _parse_action('Action: {"tool": "list_documents"}')
    assert (tool, tool_input) == ("list_documents", {})


def test_parse_action_skips_braces_in_the_thought():
    """A thought mentioning braces must not be parsed as the action."""
    _, _, tool, _ = _parse_action(
        'Thought: the {placeholder} section looks relevant.\n'
        'Action: {"tool": "search", "input": {"query": "placeholder"}}'
    )
    assert tool == "search"


def test_parse_action_rejects_a_reply_with_no_action():
    with pytest.raises(ActionParseError, match="no Action"):
        _parse_action("I think the refund window is 14 days.")


def test_parse_action_rejects_malformed_json():
    with pytest.raises(ActionParseError, match="not valid JSON"):
        _parse_action('Action: {"tool": "search", "input": {query: refunds}}')


def test_parse_action_rejects_a_missing_tool_key():
    with pytest.raises(ActionParseError, match='missing the "tool" key'):
        _parse_action('Action: {"input": {"query": "refunds"}}')


# ---- The loop --------------------------------------------------------------


def test_run_rejects_an_empty_question():
    with pytest.raises(ValueError, match="must not be empty"):
        _agent(FakeLLMService([])).run("   ")


def test_multi_hop_run_searches_twice_then_answers():
    """The whole point of the agent: two retrievals for one question."""
    llm = FakeLLMService(
        [
            _search("refund policy"),
            _search("returns policy"),
            _finish("Refunds run 14 days [E1]; returns run 30 [E2]."),
        ]
    )
    retriever = FakeRetriever([[_chunk("c1")], [_chunk("c2")]])
    result = _agent(llm, retriever).run("How do refunds and returns differ?")

    assert retriever.queries == ["refund policy", "returns policy"]
    assert result.answer == "Refunds run 14 days [E1]; returns run 30 [E2]."
    assert result.stopped_because == "finished"
    assert len(result.steps) == 3
    assert [source.chunk_id for source in result.sources] == ["c1", "c2"]


def test_evidence_labels_are_stable_and_passages_are_not_repeated():
    """A chunk seen twice keeps its first label and is not re-sent in full."""
    llm = FakeLLMService([_search("first"), _search("second"), _finish("done")])
    shared = _chunk("shared", text="the passage body")
    retriever = FakeRetriever([[shared], [shared, _chunk("fresh")]])

    result = _agent(llm, retriever).run("q")

    first_observation = result.steps[0].observation
    second_observation = result.steps[1].observation
    assert "[E1]" in first_observation and "the passage body" in first_observation
    # Second sighting: same label, pointer instead of the text.
    assert "[E1] handbook.pdf p.3 (already shown above, not repeated)" in second_observation
    assert second_observation.count("the passage body") == 0
    assert "[E2]" in second_observation
    # Deduped in the citations too - one entry per distinct passage.
    assert [source.chunk_id for source in result.sources] == ["shared", "fresh"]


def test_a_repeated_search_is_not_re_run():
    """Identical query, differently cased and spaced - still a repeat."""
    llm = FakeLLMService([_search("refund policy"), _search("Refund   Policy"), _finish("done")])
    retriever = FakeRetriever([[_chunk("c1")]])

    result = _agent(llm, retriever).run("q")

    assert retriever.queries == ["refund policy"]
    assert "You already searched" in result.steps[1].observation


def test_an_unknown_tool_becomes_an_observation_and_the_loop_continues():
    llm = FakeLLMService(
        [
            'Action: {"tool": "read_email", "input": {}}',
            _search("refunds"),
            _finish("answered"),
        ]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert 'no tool called "read_email"' in result.steps[0].observation
    assert result.stopped_because == "finished"
    assert result.answer == "answered"


def test_a_failing_tool_becomes_an_observation_rather_than_crashing():
    class ExplodingRetriever(FakeRetriever):
        def retrieve(self, question, top_k=5):
            raise KeyError("vector store went away")

    llm = FakeLLMService([_search("refunds"), _finish("answered anyway")])
    result = _agent(llm, ExplodingRetriever([])).run("q")

    assert "The search tool failed" in result.steps[0].observation
    assert result.answer == "answered anyway"


def test_the_loop_recovers_from_one_unparseable_reply():
    llm = FakeLLMService(
        ["I'll just answer: refunds take 14 days.", _search("refunds"), _finish("done")]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert "no Action" in result.steps[0].observation
    # The broken reply itself is replayed, so the model can see what it did.
    assert "I'll just answer" in result.steps[0].action_json
    assert result.stopped_because == "finished"


def test_two_consecutive_parse_failures_end_the_loop_with_a_synthesis():
    llm = FakeLLMService(
        [
            _search("refunds"),
            "no action here",
            "still no action",
            "Refunds take 14 days according to [E1].",
        ]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert result.stopped_because == "parse_failures"
    assert result.answer == "Refunds take 14 days according to [E1]."
    # The evidence gathered before the failures is still cited.
    assert [source.chunk_id for source in result.sources] == ["c1"]


def test_parse_failures_are_counted_consecutively_not_cumulatively():
    """One bad reply early must not doom a run that then behaves."""
    llm = FakeLLMService(
        ["bad", _search("a"), "bad again", _search("b"), _finish("recovered")]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")], [_chunk("c2")]])).run("q")

    assert result.stopped_because == "finished"
    assert result.answer == "recovered"


def test_hitting_the_step_limit_still_produces_an_answer():
    """The budget running out must not throw the retrieved evidence away."""
    llm = FakeLLMService(
        [_search("a"), _search("b"), "Partial answer from [E1] and [E2]."]
    )
    retriever = FakeRetriever([[_chunk("c1")], [_chunk("c2")]])
    result = _agent(llm, retriever, max_steps=2).run("q")

    assert result.stopped_because == "step_limit"
    assert result.answer == "Partial answer from [E1] and [E2]."
    assert len(result.steps) == 2
    assert len(result.sources) == 2


def test_the_synthesis_call_does_not_use_the_loops_system_prompt():
    """Sent with the loop's rules attached, the model writes another Action."""
    llm = FakeLLMService([_search("a"), "the written-up answer"])
    _agent(llm, FakeRetriever([[_chunk("c1")]]), max_steps=1).run("q")

    loop_system_prompt, synthesis_system_prompt = llm.system_prompts
    assert "Action:" in loop_system_prompt
    assert "Action" not in synthesis_system_prompt


def test_finishing_without_searching_is_nudged_once_then_accepted():
    llm = FakeLLMService([_finish("from my own knowledge"), _finish("still nothing to search")])
    result = _agent(llm, FakeRetriever([])).run("q")

    assert "have not searched" in result.steps[0].observation
    assert result.answer == "still nothing to search"
    assert result.stopped_because == "finished"


def test_finishing_after_a_search_is_not_nudged():
    llm = FakeLLMService([_search("refunds"), _finish("grounded answer")])
    result = _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert len(result.steps) == 2
    assert result.answer == "grounded answer"


def test_finish_with_an_empty_answer_is_rejected_and_retried():
    llm = FakeLLMService(
        [
            _search("refunds"),
            'Action: {"tool": "finish", "input": {"answer": ""}}',
            _finish("a real answer"),
        ]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert "called finish with no answer" in result.steps[1].observation
    assert result.answer == "a real answer"


def test_list_documents_reports_only_ready_documents():
    registry = FakeRegistry(
        [_record("ready.pdf"), _record("broken.pdf", status="FAILED")]
    )
    llm = FakeLLMService(
        ['Action: {"tool": "list_documents", "input": {}}', _search("x"), _finish("done")]
    )
    result = _agent(llm, FakeRetriever([[_chunk("c1")]]), registry).run("q")

    observation = result.steps[0].observation
    assert "ready.pdf" in observation
    assert "broken.pdf" not in observation


def test_an_empty_search_result_says_so_rather_than_returning_nothing():
    llm = FakeLLMService([_search("nothing here"), _finish("not covered")])
    result = _agent(llm, FakeRetriever([[]])).run("q")

    assert "No passages matched" in result.steps[0].observation
    assert result.sources == []


def test_history_reaches_the_prompt_as_context_not_as_chat_turns():
    llm = FakeLLMService([_finish("done"), _finish("done")])
    _agent(llm, FakeRetriever([])).run(
        "and the other one?", history=[{"role": "user", "content": "what is the refund window?"}]
    )

    assert "what is the refund window?" in llm.prompts[0]


def test_each_run_starts_from_a_clean_scratchpad():
    """The agent is a reusable singleton - one question must not leak into the next."""
    agent = _agent(
        FakeLLMService([_search("a"), _finish("one"), _search("a"), _finish("two")]),
        FakeRetriever([[_chunk("c1")], [_chunk("c2")]]),
    )

    first = agent.run("q1")
    second = agent.run("q2")

    assert first.stopped_because == second.stopped_because == "finished"
    # The repeated query was re-run rather than reported as a repeat.
    assert [source.chunk_id for source in second.sources] == ["c2"]
    assert len(second.steps) == 2


def test_the_research_loop_runs_deterministically():
    """Shared with the vision agent via AgentLoop - pinned on both sides."""
    llm = FakeLLMService([_search("refunds"), _finish("done")])
    _agent(llm, FakeRetriever([[_chunk("c1")]])).run("q")

    assert llm.temperatures == [0.0, 0.0]


def test_the_synthesis_call_is_deterministic_too():
    llm = FakeLLMService([_search("a"), "the written-up answer"])
    _agent(llm, FakeRetriever([[_chunk("c1")]]), max_steps=1).run("q")

    assert llm.temperatures == [0.0, 0.0]
