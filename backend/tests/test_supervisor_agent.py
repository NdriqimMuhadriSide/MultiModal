"""
Tests for the supervisor and its delegation (agents/supervisor_agent.py).

The risk in a delegating agent is not the loop - that is
tests/test_research_agent.py's subject and is shared code. It is everything
that only goes wrong once one agent calls another:

- a specialist handed a fragment it cannot resolve
- two specialists numbering citations from separate ledgers
- sub-steps vanishing into a single opaque observation
- a tree spending its specialists' budgets on top of its own

Nothing here calls a model or touches a vector store: the LLM replies are
scripted, so what is asserted is control flow.
"""
import pytest

from agents.agent_loop import StepEvent
from agents.supervisor_agent import BoundImage, SupervisorAgent, SupervisorResultEvent
from rag.retriever import RetrievedChunk
from tests.test_research_agent import FakeLLMService, FakeRegistry, FakeRetriever


def chunk(chunk_id: str, text: str, filename: str = "policy.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        filename=filename,
        page=1,
        score=0.9,
        section="S",
    )


def action(tool: str, **kwargs) -> str:
    import json

    return f"Thought: t\nAction: {json.dumps({'tool': tool, 'input': kwargs})}"


class FakeVisionService:
    """Returns a fixed description, recording what it was asked."""

    def __init__(self, description: str = "A receipt for 84.50.") -> None:
        self.description = description
        self.questions: list[str] = []

    def analyze_image(self, image_bytes, mime_type, question, temperature=None) -> str:
        self.questions.append(question)
        return self.description


def build(
    replies: list[str],
    retrieval: list[list[RetrievedChunk]] | None = None,
    vision: FakeVisionService | None = None,
    critic_enabled: bool = False,
    **kwargs,
) -> tuple[SupervisorAgent, FakeLLMService, FakeRetriever]:
    """
    A supervisor with the critic OFF unless a test asks for it.

    Off by default here on purpose. The critic makes an extra LLM call on
    every delegating run, so leaving it on would make each test below script
    a verdict it does not care about - and, worse, a test that forgot to
    would still pass, because the critic approves when its call fails. That
    is right in production and useless in a test. Its own behaviour is
    covered in tests/test_critic.py and in the gate tests at the bottom of
    this file, both of which turn it on explicitly.
    """
    llm = FakeLLMService(replies)
    retriever = FakeRetriever(retrieval or [])
    agent = SupervisorAgent(
        llm_service=llm,
        vision_service=vision or FakeVisionService(),
        retriever=retriever,
        document_registry=FakeRegistry(),
        critic_enabled=critic_enabled,
        **kwargs,
    )
    return agent, llm, retriever


def approve() -> str:
    return '{"verdict": "approve"}'


def revise(problem: str) -> str:
    import json

    return json.dumps({"verdict": "revise", "problem": problem})


# ---------------------------------------------------------------------------
# Answering without delegating
# ---------------------------------------------------------------------------


def test_a_question_it_can_answer_itself_costs_one_step_and_no_specialist():
    """
    The common case, and the one a supervisor most easily gets wrong: given
    specialists, models reach for them. Delegating "what is the capital of
    France?" would spend several seconds to be told the documents do not
    cover it.
    """
    agent, llm, retriever = build([action("finish", answer="Paris.")])

    result = agent.run("what is the capital of France?")

    assert result.answer == "Paris."
    assert result.tool_used == "answer_directly"
    assert len(result.steps) == 1
    assert retriever.queries == []
    assert len(llm.prompts) == 1


def test_finishing_with_no_answer_is_rejected_and_costs_a_step():
    agent, _, _ = build(
        [action("finish"), action("finish", answer="Second time.")]
    )

    result = agent.run("hello")

    assert result.answer == "Second time."
    assert len(result.steps) == 2
    assert "no answer" in result.steps[0].observation


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def test_a_document_question_is_delegated_and_the_specialists_answer_comes_back():
    agent, _, retriever = build(
        [
            action("research_documents", question="What is the refund window?"),
            action("search", query="refund window"),
            action("finish", answer="Refunds within 30 days [E1]."),
            action("finish", answer="You have 30 days to claim a refund [E1]."),
        ],
        retrieval=[[chunk("c1", "Refunds are accepted within 30 days.")]],
    )

    result = agent.run("how long do I have to get a refund?")

    assert result.answer == "You have 30 days to claim a refund [E1]."
    assert result.tool_used == "research_documents"
    assert retriever.queries == ["refund window"]


def test_a_delegations_sub_steps_are_nested_under_the_step_that_caused_them():
    """
    Without this the trace reports "asked the document specialist" and
    silently drops everything behind the answer - which is exactly the
    hidden work the trace exists to expose.
    """
    agent, _, _ = build(
        [
            action("research_documents", question="What is the refund window?"),
            action("search", query="refund window"),
            action("finish", answer="30 days [E1]."),
            action("finish", answer="30 days [E1]."),
        ],
        retrieval=[[chunk("c1", "Refunds within 30 days.")]],
    )

    result = agent.run("refund window?")

    delegation = result.steps[0]
    assert delegation.tool == "research_documents"
    assert [child.tool for child in delegation.children] == ["search", "finish"]
    # The supervisor's own `finish` delegated to nothing.
    assert result.steps[1].children == []


def test_a_specialists_steps_stream_before_the_delegation_that_produced_them():
    """
    A specialist that thinks for several steps behind one blocking call is
    seconds of silence. Its steps arrive as they land, tagged one level
    deeper so the client can indent them.
    """
    agent, _, _ = build(
        [
            action("research_documents", question="What is the refund window?"),
            action("search", query="refund window"),
            action("finish", answer="30 days [E1]."),
            action("finish", answer="30 days [E1]."),
        ],
        retrieval=[[chunk("c1", "Refunds within 30 days.")]],
    )

    events = [e for e in agent.stream("refund window?") if isinstance(e, StepEvent)]

    assert [(e.depth, e.step.tool) for e in events] == [
        (1, "search"),
        (1, "finish"),
        (0, "research_documents"),
        (0, "finish"),
    ]


def test_a_delegation_with_no_question_is_rejected_rather_than_sent_empty():
    agent, _, retriever = build(
        [
            action("research_documents"),
            action("finish", answer="Answered without the documents."),
        ]
    )

    result = agent.run("something")

    assert retriever.queries == []
    assert "needs a question" in result.steps[0].observation


def test_a_failing_specialist_costs_a_step_rather_than_the_run():
    """
    A tool raising is information the supervisor can act on - it can answer
    from its own knowledge and say so. Only the LLM failing ends a run.
    """

    class ExplodingRetriever:
        queries: list[str] = []

        def retrieve(self, query, top_k=5):
            raise RuntimeError("the vector store is down")

    llm = FakeLLMService(
        [
            action("research_documents", question="What is the refund window?"),
            action("search", query="refund window"),
            action("finish", answer="I could not reach the documents."),
            action("finish", answer="I could not check your documents just now."),
        ]
    )
    agent = SupervisorAgent(
        llm_service=llm,
        vision_service=FakeVisionService(),
        retriever=ExplodingRetriever(),
        document_registry=FakeRegistry(),
        critic_enabled=False,
    )

    result = agent.run("refund window?")

    assert result.answer == "I could not check your documents just now."
    assert result.stopped_because == "finished"


# ---------------------------------------------------------------------------
# The image specialist, reached from a text endpoint
# ---------------------------------------------------------------------------


def test_read_image_reports_there_is_no_image_rather_than_failing():
    """
    /agent/ask carries no upload. A conversation that never went through
    /vision/ask has no picture, and that is an ordinary fact the supervisor
    can act on - not a reason to fail the turn.
    """
    agent, _, _ = build(
        [
            action("read_image", question="What is the total?"),
            action("finish", answer="You would need to attach an image."),
        ]
    )

    result = agent.run("what does this receipt say?")

    assert "no image in this conversation" in result.steps[0].observation
    assert result.tool_used == "answer_directly"


def test_an_image_carried_by_the_conversation_reaches_the_vision_specialist():
    vision = FakeVisionService("A receipt. The total is 84.50.")
    agent, _, _ = build(
        [
            action("read_image", question="What is the total?"),
            action("inspect_image", question="What is the total?"),
            action("finish", answer="The total is 84.50."),
            action("finish", answer="The receipt total is 84.50."),
        ],
        vision=vision,
    )

    result = agent.run(
        "what is the total on this receipt?",
        image=BoundImage(data=b"fake-png-bytes", mime_type="image/png"),
    )

    assert result.answer == "The receipt total is 84.50."
    assert result.tool_used == "read_image"
    assert vision.questions == ["What is the total?"]


# ---------------------------------------------------------------------------
# The two things only a tree gets wrong
# ---------------------------------------------------------------------------


def test_two_specialists_number_their_citations_from_one_shared_ledger():
    """
    The bug this is here to prevent: each specialist numbering from its own
    ledger hands out [E1] for a different passage, and the merged citation
    list then points at the wrong text with nothing raised anywhere.
    """
    vision = FakeVisionService("A receipt for 84.50.")
    agent, _, _ = build(
        [
            # The image specialist searches the corpus first.
            action("read_image", question="What is the total?"),
            action("search_knowledge_base", query="expense limit"),
            action("inspect_image", question="What is the total?"),
            action("finish", answer="Total 84.50, limit is [E1]."),
            # Then the document specialist searches something else.
            action("research_documents", question="What is the approval process?"),
            action("search", query="approval process"),
            action("finish", answer="Approval is [E2]."),
            action("finish", answer="Total 84.50 [E1], approval [E2]."),
        ],
        retrieval=[
            [chunk("limits-c1", "The expense limit is 50 per head.")],
            [chunk("approval-c1", "Claims over 100 need director approval.")],
        ],
        vision=vision,
    )

    result = agent.run(
        "does this receipt comply, and what approval does it need?",
        image=BoundImage(data=b"png", mime_type="image/png"),
    )

    # Two distinct passages, each labelled once, in the order first seen -
    # so index i is the passage labelled [E(i+1)].
    assert [source.chunk_id for source in result.sources] == [
        "limits-c1",
        "approval-c1",
    ]
    assert result.tool_used == "multiple_specialists"


def test_the_tree_budget_bounds_the_whole_run_not_each_agent_separately():
    """
    Without a shared pool a 3-step supervisor free to call a 3-step
    specialist has a worst case of 9 LLM calls plus synthesis. The pool is
    what makes the tree's cost additive rather than multiplicative.
    """
    # A specialist that never finishes, so it would spend its whole ceiling
    # every time it is called.
    replies = [action("research_documents", question="A self-contained question?")]
    replies += [action("search", query=f"phrasing {n}") for n in range(10)]
    replies += ["Final write-up."] * 4

    agent, llm, _ = build(
        replies,
        retrieval=[[chunk(f"c{n}", f"passage {n}")] for n in range(10)],
        max_steps=3,
        tree_budget=4,
        research_max_steps=3,
    )

    result = agent.run("something that needs the documents")

    # 4 steps from the pool, plus the synthesis calls the two exhausted
    # agents each make. Never the 9+ an unshared budget would allow.
    assert len(llm.prompts) <= 7
    assert result.stopped_because == "step_limit"


def test_a_second_run_starts_from_a_clean_budget_and_ledger():
    """
    The budget and ledger are shared by reference with every specialist, so
    they are reset in place rather than rebound - rebinding would leave the
    specialists writing into the previous run's objects.
    """
    agent, llm, _ = build(
        [
            action("research_documents", question="What is the refund window?"),
            action("search", query="refund window"),
            action("finish", answer="30 days [E1]."),
            action("finish", answer="30 days [E1]."),
        ],
        retrieval=[[chunk("c1", "Refunds within 30 days.")]],
        tree_budget=10,
    )
    first = agent.run("refund window?")
    assert len(first.sources) == 1

    llm.replies = [action("finish", answer="Hello.")]
    second = agent.run("hello")

    assert second.sources == []
    assert second.tool_used == "answer_directly"
    assert second.answer == "Hello."


def test_an_empty_message_is_rejected_before_any_llm_call():
    agent, llm, _ = build([])

    with pytest.raises(ValueError):
        agent.run("   ")

    assert llm.prompts == []


# ---------------------------------------------------------------------------
# The critic, as a gate on finish
# ---------------------------------------------------------------------------


def _delegating(*extra: str) -> list[str]:
    """A run that delegates once, then finishes - plus whatever follows."""
    return [
        action("research_documents", question="What is the refund window?"),
        action("search", query="refund window"),
        action("finish", answer="30 days [E1]."),
        action("finish", answer="You have 30 days [E1]."),
        *extra,
    ]


def _one_passage():
    return [[chunk("c1", "Refunds are accepted within 30 days.")]]


def test_an_approved_draft_is_delivered_unchanged_and_marked_reviewed():
    agent, llm, _ = build(
        _delegating(approve()), retrieval=_one_passage(), critic_enabled=True
    )

    result = agent.run("refund window?")

    assert result.answer == "You have 30 days [E1]."
    assert result.reviewed is True
    # The critic's call is the last one, and is not a step in the trace: it
    # is a gate on finishing, not something the supervisor chose to do.
    assert len(llm.prompts) == 5
    assert len(result.steps) == 2


def test_a_rejected_draft_is_sent_back_with_the_objection_as_the_observation():
    agent, _, _ = build(
        _delegating(
            revise("the 14-day figure is not in the evidence; the passage says 30"),
            action("finish", answer="You have 30 days [E1]."),
            approve(),
        ),
        retrieval=_one_passage(),
        critic_enabled=True,
    )

    result = agent.run("refund window?")

    assert result.answer == "You have 30 days [E1]."
    rejected_step = result.steps[1]
    assert rejected_step.tool == "finish"
    assert "the 14-day figure is not in the evidence" in rejected_step.observation
    # The rejection cost a step, so the run took one more than it otherwise
    # would - the price of the gate, paid only when it fires.
    assert len(result.steps) == 3


def test_the_critic_may_only_send_a_draft_back_once():
    """
    A reviewer that can reject indefinitely is a run that never finishes. The
    second draft was written with the objection in hand; a third rejection
    has nothing new to say.
    """
    agent, llm, _ = build(
        _delegating(
            revise("first objection"),
            action("finish", answer="Second attempt [E1]."),
        ),
        retrieval=_one_passage(),
        critic_enabled=True,
    )

    result = agent.run("refund window?")

    assert result.answer == "Second attempt [E1]."
    # Five loop replies and exactly one review. The second finish is not
    # reviewed at all, which is why no second verdict is scripted - and the
    # fake would raise if one were asked for.
    assert len(llm.prompts) == 6
    # `reviewed` says a review happened during the run, which it did. It does
    # not claim this rewrite was itself re-checked - see SupervisorResult.
    assert result.reviewed is True


def test_a_direct_answer_is_never_reviewed_and_costs_no_extra_call():
    """
    Nothing was gathered, so there is nothing to review the draft against.
    Skipping it keeps the common case at the latency it had before the critic
    existed.
    """
    agent, llm, _ = build([action("finish", answer="Paris.")], critic_enabled=True)

    result = agent.run("what is the capital of France?")

    assert result.answer == "Paris."
    assert result.reviewed is False
    assert len(llm.prompts) == 1


def test_an_unreadable_verdict_lets_the_answer_through_unreviewed():
    """
    Fail open. A reviewer that blocks answers when it is broken turns a
    degradation into an outage - the supervisor's work is already done.
    """
    agent, _, _ = build(
        _delegating("I think it looks fine, honestly."),
        retrieval=_one_passage(),
        critic_enabled=True,
    )

    result = agent.run("refund window?")

    assert result.answer == "You have 30 days [E1]."
    # Delivered, but not claimed to have been checked.
    assert result.reviewed is False


def test_a_rejection_with_no_stated_problem_is_treated_as_unreviewed():
    """
    "Fix it" without saying what is not actionable: the supervisor would
    resubmit the same draft and burn the revision on nothing.
    """
    agent, _, _ = build(
        _delegating('{"verdict": "revise"}'),
        retrieval=_one_passage(),
        critic_enabled=True,
    )

    result = agent.run("refund window?")

    assert result.answer == "You have 30 days [E1]."
    assert result.reviewed is False


def test_the_critic_can_be_switched_off_entirely():
    agent, llm, _ = build(
        _delegating(), retrieval=_one_passage(), critic_enabled=False
    )

    result = agent.run("refund window?")

    assert result.answer == "You have 30 days [E1]."
    assert result.reviewed is False
    assert len(llm.prompts) == 4
