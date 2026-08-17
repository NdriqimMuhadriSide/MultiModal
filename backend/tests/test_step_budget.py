"""
Tests for StepBudget and how AgentLoop spends it (agents/agent_loop.py).

Two bounds apply to every loop and they are independent: the agent's own
`max_steps`, and the pool shared with everything else in its tree. Most of
what can go wrong here is a run that quietly costs more than it was allowed
to, which no assertion about answers would catch.
"""
import json

from agents.agent_loop import AgentLoop, StepBudget, Tool
from tests.test_research_agent import FakeLLMService


def action(tool: str, **kwargs) -> str:
    return f"Thought: t\nAction: {json.dumps({'tool': tool, 'input': kwargs})}"


def loop(llm, max_steps: int = 6, budget: StepBudget | None = None) -> AgentLoop:
    return AgentLoop(
        llm_service=llm,
        tools=[
            Tool(
                name="noop",
                description="does nothing",
                input_spec="{}",
                run=lambda _: "nothing happened",
            ),
            Tool(
                name="finish",
                description="ends the run",
                input_spec='{"answer": "..."}',
                run=lambda payload: payload.get("answer", ""),
                terminal=True,
            ),
        ],
        system_prompt="s",
        synthesis_system_prompt="s",
        max_steps=max_steps,
        budget=budget,
    )


# ---- The primitive ---------------------------------------------------------


def test_take_spends_one_step_and_refuses_when_empty():
    budget = StepBudget(2)

    assert budget.take() is True
    assert budget.take() is True
    assert budget.take() is False
    assert budget.remaining == 0


def test_a_budget_is_shared_by_reference_not_copied():
    """
    The whole mechanism. A copied budget would give each agent its own
    private allowance and restore exactly the multiplication it exists to
    prevent.
    """
    budget = StepBudget(5)
    first = loop(FakeLLMService([action("finish", answer="a")]), budget=budget)
    second = loop(FakeLLMService([action("finish", answer="b")]), budget=budget)

    first.run("q")
    second.run("q")

    assert budget.remaining == 3


# ---- How the loop spends it ------------------------------------------------


def test_a_root_loop_gets_its_own_pool_sized_to_max_steps():
    """A loop with no budget passed is the whole of its own tree."""
    llm = FakeLLMService([action("noop")] * 3 + ["Final write-up."])

    result = loop(llm, max_steps=3).run("q")

    assert len(result.steps) == 3
    assert result.stopped_because == "step_limit"


def test_a_shared_pool_can_stop_a_loop_before_its_own_ceiling_does():
    llm = FakeLLMService([action("noop")] * 2 + ["Final write-up."])

    result = loop(llm, max_steps=6, budget=StepBudget(2)).run("q")

    assert len(result.steps) == 2
    assert result.stopped_because == "step_limit"


def test_a_loops_own_ceiling_can_stop_it_before_the_shared_pool_does():
    """
    Both bounds, not just the pool. Without the local ceiling one specialist
    could spend everything the rest of the tree still needs.
    """
    llm = FakeLLMService([action("noop")] * 2 + ["Final write-up."])
    budget = StepBudget(10)

    result = loop(llm, max_steps=2, budget=budget).run("q")

    assert len(result.steps) == 2
    assert result.stopped_because == "step_limit"
    assert budget.remaining == 8


def test_the_prompt_tells_the_agent_the_smaller_of_the_two_bounds():
    """
    A sub-agent told it has six steps when the tree has two left plans work
    it will not get to finish - which is why the figure in the prompt is
    whichever bound bites first, not the agent's own.
    """
    llm = FakeLLMService([action("noop"), action("finish", answer="done")])

    loop(llm, max_steps=6, budget=StepBudget(2)).run("q")

    # Two steps left including this one, then the last-step warning.
    assert "You have 2 steps left" in llm.prompts[0]
    assert "This is your LAST step" in llm.prompts[1]


def test_an_exhausted_pool_still_produces_an_answer_rather_than_an_apology():
    """
    The evidence is already paid for. Reporting the limit instead of
    answering would throw it away and tell the user about an implementation
    detail they cannot act on.
    """
    llm = FakeLLMService([action("noop"), "The answer from what was gathered."])

    result = loop(llm, max_steps=6, budget=StepBudget(1)).run("q")

    assert result.answer == "The answer from what was gathered."
    assert result.stopped_because == "step_limit"
