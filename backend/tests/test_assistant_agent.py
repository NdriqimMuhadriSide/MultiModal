import pytest

from agents.assistant_agent import AssistantAgent
from rag.rag_service import RAGAnswer


class FakeLLMService:
    """Deterministic router + direct-answer stand-in - no real Groq call."""

    def __init__(self, routing_response: str, direct_answer: str = "a direct answer") -> None:
        self.routing_response = routing_response
        self.direct_answer = direct_answer
        self.prompts_seen: list[str] = []
        self.history_seen: list[list[dict[str, str]] | None] = []

    def generate_response(
        self, user_message: str, history: list[dict[str, str]] | None = None
    ) -> str:
        self.prompts_seen.append(user_message)
        self.history_seen.append(history)
        # The router prompt always contains this exact instruction line -
        # use it to tell a routing call apart from a direct-answer call.
        if "routing component" in user_message:
            return self.routing_response
        return self.direct_answer


class FakeRAGService:
    def __init__(self) -> None:
        self.asked_with: str | None = None

    def ask(self, question: str, top_k: int = 5) -> RAGAnswer:
        self.asked_with = question
        return RAGAnswer(answer="answer from the knowledge base", sources=[])


def test_agent_routes_to_knowledge_base_tool():
    llm = FakeLLMService(routing_response="KNOWLEDGE_BASE")
    rag = FakeRAGService()
    agent = AssistantAgent(llm_service=llm, rag_service=rag)

    result = agent.run("What is our refund policy?")

    assert result.tool_used == "search_knowledge_base"
    assert result.answer == "answer from the knowledge base"
    assert rag.asked_with == "What is our refund policy?"


def test_agent_routes_to_direct_answer_tool():
    llm = FakeLLMService(routing_response="DIRECT_ANSWER", direct_answer="Paris is the capital of France.")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    result = agent.run("What is the capital of France?")

    assert result.tool_used == "answer_directly"
    assert result.answer == "Paris is the capital of France."


def test_agent_falls_back_to_direct_answer_on_unclear_routing():
    """If the router's output doesn't cleanly match a known tool keyword,
    the agent should not crash - it should default to answering directly."""
    llm = FakeLLMService(routing_response="not sure, maybe something else?")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    result = agent.run("some ambiguous message")

    assert result.tool_used == "answer_directly"


def test_agent_rejects_empty_message():
    agent = AssistantAgent(
        llm_service=FakeLLMService(routing_response="DIRECT_ANSWER"),
        rag_service=FakeRAGService(),
    )

    with pytest.raises(ValueError):
        agent.run("")


def test_agent_routes_to_external_api_tool_and_extracts_city(monkeypatch):
    import agents.assistant_agent as agent_module

    def fake_fetch_weather(city: str) -> str:
        return f"The current weather in {city} is sunny."

    monkeypatch.setattr(agent_module, "_fetch_weather", fake_fetch_weather)

    llm = FakeLLMService(routing_response="EXTERNAL_API")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    result = agent.run("What's the weather in Paris?")

    assert result.tool_used == "call_external_api"
    assert "Paris" in result.answer
    assert "sunny" in result.answer


def test_agent_passes_history_to_direct_answer_call():
    """The direct-answer tool receives history as real conversation turns,
    so follow-ups ("what about its population?") have their referent."""
    llm = FakeLLMService(routing_response="DIRECT_ANSWER")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    agent.run("What is its population?", history=history)

    # Two calls: routing first, then the direct answer. Only the second
    # gets history as actual turns.
    assert llm.history_seen[0] is None or llm.history_seen[0] == []
    assert llm.history_seen[1] == history


def test_router_prompt_includes_history_as_context():
    """History reaches routing as inert prompt text, not as chat turns -
    otherwise a follow-up like 'what is its population?' routes on those
    four words alone and lands on the wrong tool."""
    llm = FakeLLMService(routing_response="DIRECT_ANSWER")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    agent.run(
        "What is its population?",
        history=[{"role": "assistant", "content": "The capital of France is Paris."}],
    )

    router_prompt = llm.prompts_seen[0]
    assert "routing component" in router_prompt
    assert "The capital of France is Paris." in router_prompt


def test_agent_runs_without_history():
    """History is optional - omitting it must not break the graph."""
    llm = FakeLLMService(routing_response="DIRECT_ANSWER", direct_answer="ok")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    result = agent.run("hello")

    assert result.answer == "ok"


def test_agent_external_api_tool_handles_missing_city():
    llm = FakeLLMService(routing_response="EXTERNAL_API")
    agent = AssistantAgent(llm_service=llm, rag_service=FakeRAGService())

    result = agent.run("What's the weather like today?")

    assert result.tool_used == "call_external_api"
    assert "city" in result.answer.lower()
