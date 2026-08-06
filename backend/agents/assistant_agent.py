"""
Assistant agent — LangGraph-based reasoning loop with three tools.

This is what actually distinguishes an "agent" from the chat/RAG/vision
services built so far: instead of always running the same fixed pipeline
(e.g. RAGService.ask always embeds -> searches -> prompts), this module
lets the LLM itself decide, per incoming message, which of three paths to
take:

    Tool 1: search_knowledge_base - reuses rag/rag_service.py's retrieval +
            generation pipeline (ingested PDFs in Chroma).
    Tool 2: call_external_api     - calls a real external API (Open-Meteo,
            free, no key required) for information the system doesn't store.
    Tool 3: answer_directly       - answers straight from the LLM's own
            knowledge, no tool needed.

Graph shape:

              +-----------+
   message -> |  route    |  (LLM decides which tool applies)
              +-----------+
                 |    |    |
         KB tool |    |    | DIRECT_ANSWER
                 v    |    v
        +-----------+ |  +----------------+
        | search_kb | |  | answer_directly|
        +-----------+ |  +----------------+
                 EXTERNAL_API
                       v
              +------------------+
              | call_external_api|
              +------------------+
                       |
                       v
                     END (all paths converge here)

Each tool node is a plain Python function operating on a shared state
dict - this is the "reasoning loop" in its simplest form: one routing
decision, one tool execution, done. A more advanced agent would loop back
to the router after a tool runs (e.g. to decide if a second tool call is
needed), but a single-hop routing graph is the clearest way to demonstrate
the tool-calling architecture without extra moving parts.
"""
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, TypedDict

import requests
from langgraph.graph import END, StateGraph

from ai.llm_service import LLMService
from prompts.assistant_prompts import format_router_prompt
from rag.rag_service import RAGService

ToolChoice = Literal["KNOWLEDGE_BASE", "EXTERNAL_API", "DIRECT_ANSWER"]

# What the router's choice is reported as once a tool has actually run.
# Distinct from ToolChoice: that names the decision, this names the node
# that produced the answer, and it's the value the UI labels a bubble with.
ToolUsed = Literal["search_knowledge_base", "call_external_api", "answer_directly"]

# Open-Meteo: free, no API key, no signup - used for the EXTERNAL_API tool
# so this is a real external call, not a mocked/fake one.
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Matches a city name mentioned after "in"/"for" in a weather-style question,
# e.g. "What's the weather in Paris?" -> "Paris". Deliberately simple regex
# extraction (not another LLM call) to keep the external-API tool fast and
# cheap; a production agent would likely use structured tool-calling
# (see the "Tool calling architecture" explanation) instead of regex.
_CITY_PATTERN = re.compile(r"(?:in|for)\s+([A-Za-z\s]+?)[\?\.]?$", re.IGNORECASE)


class AgentState(TypedDict, total=False):
    """Shared state threaded through every node in the graph."""

    message: str
    history: list[dict[str, str]]
    tool_choice: ToolChoice
    answer: str
    tool_used: str


@dataclass
class AgentResult:
    answer: str
    tool_used: str


def _extract_city(message: str) -> str | None:
    match = _CITY_PATTERN.search(message.strip())
    return match.group(1).strip() if match else None


def _fetch_weather(city: str) -> str:
    """Call Open-Meteo's free geocoding + forecast APIs for `city`."""
    geo_response = requests.get(
        GEOCODING_API_URL, params={"name": city, "count": 1}, timeout=10
    )
    geo_response.raise_for_status()
    geo_results = geo_response.json().get("results")
    if not geo_results:
        return f"I couldn't find a location named '{city}'."

    location = geo_results[0]
    weather_response = requests.get(
        WEATHER_API_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current_weather": "true",
        },
        timeout=10,
    )
    weather_response.raise_for_status()
    current = weather_response.json()["current_weather"]

    return (
        f"The current weather in {location['name']}, {location.get('country', '')} "
        f"is {current['temperature']}°C with wind speed {current['windspeed']} km/h."
    )


class AssistantAgent:
    """
    Wraps a compiled LangGraph graph that routes a message to exactly one
    of three tools, then returns the resulting answer.
    """

    def __init__(self, llm_service: LLMService, rag_service: RAGService) -> None:
        self._llm_service = llm_service
        self._rag_service = rag_service
        self._graph = self._build_graph()

    # ---- Nodes -----------------------------------------------------------

    def _route_node(self, state: AgentState) -> AgentState:
        """
        The reasoning step: ask the LLM which tool applies to this message.
        Falls back to DIRECT_ANSWER if the model's response doesn't cleanly
        match one of the three expected keywords, so a routing hiccup never
        crashes the request - it just answers directly instead.

        Recent turns are folded into the prompt text (see
        format_router_prompt) rather than sent as real conversation history:
        the router is a single-shot classification asking for one word back,
        and passing prior turns as actual chat messages makes the model
        answer the follow-up conversationally instead of routing it. As
        inert context it can still resolve what "it" refers to - without
        that, "what is its population?" routes on those four words alone and
        lands on the wrong tool.
        """
        router_prompt = format_router_prompt(
            state["message"], history=state.get("history")
        )
        raw_choice = self._llm_service.generate_response(router_prompt).strip().upper()

        if "KNOWLEDGE_BASE" in raw_choice:
            tool_choice: ToolChoice = "KNOWLEDGE_BASE"
        elif "EXTERNAL_API" in raw_choice:
            tool_choice = "EXTERNAL_API"
        else:
            tool_choice = "DIRECT_ANSWER"

        return {"tool_choice": tool_choice}

    def _search_knowledge_base_node(self, state: AgentState) -> AgentState:
        """Tool 1: delegate to the existing RAG pipeline (embed -> Chroma -> LLM)."""
        result = self._rag_service.ask(state["message"])
        return {"answer": result.answer, "tool_used": "search_knowledge_base"}

    def _call_external_api_node(self, state: AgentState) -> AgentState:
        """Tool 2: call a real external API (Open-Meteo) for live data."""
        city = _extract_city(state["message"])
        if not city:
            answer = (
                "I wasn't able to tell which city you're asking about. "
                "Try asking like: \"What's the weather in Paris?\""
            )
        else:
            try:
                answer = _fetch_weather(city)
            except requests.RequestException as exc:
                answer = f"I couldn't reach the weather service: {exc}"

        return {"answer": answer, "tool_used": "call_external_api"}

    def _answer_directly_node(self, state: AgentState) -> AgentState:
        """Tool 3: no retrieval, no external call - answer from the LLM directly."""
        answer = self._llm_service.generate_response(
            state["message"], history=state.get("history")
        )
        return {"answer": answer, "tool_used": "answer_directly"}

    # ---- Graph wiring ------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("route", self._route_node)
        graph.add_node("search_knowledge_base", self._search_knowledge_base_node)
        graph.add_node("call_external_api", self._call_external_api_node)
        graph.add_node("answer_directly", self._answer_directly_node)

        graph.set_entry_point("route")

        # Conditional edge: the router's output ("tool_choice") decides
        # which single node runs next - this is the tool-dispatch step.
        graph.add_conditional_edges(
            "route",
            lambda state: state["tool_choice"],
            {
                "KNOWLEDGE_BASE": "search_knowledge_base",
                "EXTERNAL_API": "call_external_api",
                "DIRECT_ANSWER": "answer_directly",
            },
        )

        # All three tool nodes are terminal - this agent makes exactly one
        # routing decision and one tool call per request (no looping back
        # to "route" for a second decision).
        graph.add_edge("search_knowledge_base", END)
        graph.add_edge("call_external_api", END)
        graph.add_edge("answer_directly", END)

        return graph.compile()

    # ---- Public entrypoint --------------------------------------------------

    def run(
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> AgentResult:
        """
        `history` is a list of {"role": ..., "content": ...} dicts, oldest
        first - the same shape ai/llm_service.py takes and that the memory
        layer produces. It reaches the answering step only (see
        _route_node's docstring for why routing stays history-free), and is
        ignored entirely by the knowledge-base and external-API tools, whose
        answers come from retrieved chunks / live API data rather than from
        the conversation so far.

        Raises:
            ValueError: if `message` is empty.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty.")

        final_state = self._graph.invoke(
            {"message": message, "history": history or []}
        )
        return AgentResult(
            answer=final_state["answer"],
            tool_used=final_state["tool_used"],
        )

    def stream(
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> tuple[ToolUsed, Iterator[str]]:
        """
        Route as `run` does, then yield the chosen tool's answer in pieces.

        Returns `(tool_used, chunks)`. Routing is resolved eagerly because
        the caller wants to tell the user *which* tool is answering before
        the answer itself arrives - and because routing is a single-shot
        classification returning one word, there is nothing to stream about
        it anyway.

        This deliberately does not go through the compiled graph.
        `graph.invoke` runs to completion and hands back a finished state,
        which is the one thing a streaming caller cannot use; LangGraph's
        own streaming APIs stream *state updates between nodes*, not tokens
        within a node, so they would deliver the same single answer blob one
        step later. Routing still calls `_route_node`, so the decision logic
        has exactly one implementation - only the dispatch is restated here.

        Not every branch actually streams, and that is honest rather than a
        gap: the external-API tool builds its sentence from a weather JSON
        response with no LLM involved, so there is nothing to arrive
        incrementally. It yields once.

        Raises:
            ValueError: if `message` is empty.
            RuntimeError: if an LLM call fails - possibly mid-stream.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty.")

        state: AgentState = {"message": message, "history": history or []}
        tool_choice = self._route_node(state)["tool_choice"]

        if tool_choice == "KNOWLEDGE_BASE":
            _sources, chunks = self._rag_service.stream_ask(message)
            return "search_knowledge_base", chunks

        if tool_choice == "EXTERNAL_API":
            answer = self._call_external_api_node(state)["answer"]

            def single() -> Iterator[str]:
                yield answer

            return "call_external_api", single()

        return "answer_directly", self._llm_service.stream_response(
            message, history=history or []
        )
