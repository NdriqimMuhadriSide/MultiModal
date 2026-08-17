"""
Supervisor agent — answers directly, or delegates to a specialist.

WHAT THIS REPLACES

agents/assistant_agent.py, which routed a message to one of three tools and
stopped. Its own docstring called that "the reasoning loop in its simplest
form: one routing decision, one tool execution, done", and named the missing
piece: looping back after a tool runs, so a second call can be decided on the
strength of what the first one returned.

That missing piece is the whole difference between routing and supervising,
and it is what makes this the first thing in the codebase that can answer:

    "Does this receipt comply with our expense policy?"

which needs the image read AND the corpus searched, in that order, with the
second question shaped by what the first returned. No single specialist can
do it, and a router that picks exactly one of them never could.

THE DELEGATION INTERFACE ALREADY EXISTED

`Tool.run: Callable[[dict], str]` (agents/agent_loop.py) is all a delegating
tool needs: a sub-agent is a tool whose implementation happens to be another
loop. Nothing about AgentLoop knows that agents can contain agents, and it
did not need changing to find out.

Three things about a *tree* did need care, and all three are silent
corruption rather than crashes:

    the budget   a 6-step supervisor calling a 6-step specialist is 36 LLM
                 calls. StepBudget is one pool, shared by reference, and
                 every loop stops at whichever bites first - the pool or its
                 own ceiling.

    the labels   two specialists each numbering from their own ledger both
                 hand out [E1] for different passages, and the merged
                 citation list then points at the wrong text with nothing
                 raised anywhere. One EvidenceLedger, owned here, passed to
                 every specialist.

    the trace    a delegation that returned only its answer would collapse
                 four steps into one string, and the trace is the only way a
                 reader can check an answer assembled out of sight. Sub-steps
                 arrive as children, and stream live rather than after.

THE REVIEWER

`finish` is a gate, not just an exit. Before an answer reaches the user, a
critic (agents/critic.py) checks it against what the specialists actually
reported, and may send it back once. It is a gate rather than a fourth tool
because a reviewer the supervisor could choose to call would be skipped on
exactly the answers that most need one.

It costs nothing on the common path: with no findings to check against there
is nothing to review, so a directly-answered question is as fast as it was
before the critic existed.

WHAT SPECIALISTS ARE NOT GIVEN

Conversation history. A specialist gets one self-contained question and
nothing else, which is why prompts/supervisor_prompts.py spends a paragraph
on resolving pronouns before delegating. Handing the corpus specialist "what
about the other one?" produces a confident answer about nothing, and it is
the single most common way a multi-agent system goes wrong.
"""
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import requests

from agents.agent_loop import (
    ActionRejected,
    AgentLoop,
    AgentStep,
    ResultEvent,
    StepBudget,
    StepEvent,
    Tool,
    ToolResult,
)
from agents.critic import Critic, Verdict
from agents.knowledge_base_tool import EvidenceLedger
from agents.research_agent import ResearchAgent, ResearchResultEvent
from agents.vision_agent import VisionAgent, VisionResultEvent
from ai.llm_service import LLMService
from ai.vision_service import VisionService
from prompts.supervisor_prompts import (
    SUPERVISOR_SYNTHESIS_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from rag.rag_service import RAGSource

# Open-Meteo: free, no API key, no signup - kept from assistant_agent.py so
# this remains a real external call rather than a mocked one.
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# What the UI labels the answer with. Distinct from the trace, which carries
# what actually happened - this is one word for a badge.
ToolUsed = Literal[
    "answer_directly",
    "research_documents",
    "read_image",
    "call_external_api",
    "multiple_specialists",
]

# The tool names that are worth a badge of their own. `finish` is absent
# because it is not a source of evidence - a run that only ever called it
# answered from the model's own knowledge, which is what "answer_directly"
# says.
_BADGE_TOOLS: frozenset[str] = frozenset(
    {"read_image", "research_documents", "call_external_api"}
)


@dataclass
class BoundImage:
    """The picture this turn is about, resolved by the caller."""

    data: bytes
    mime_type: str


@dataclass
class SupervisorResult:
    answer: str
    # The supervisor's own steps. A delegation's step carries the
    # specialist's steps in its `children`, so this is the whole tree.
    steps: list[AgentStep] = field(default_factory=list)
    # Every distinct passage any specialist retrieved, in label order - so
    # index i is [E(i+1)] wherever that label appears in the answer. Read off
    # the shared ledger rather than assembled from the specialists, which is
    # what keeps the numbering consistent across all of them.
    sources: list[RAGSource] = field(default_factory=list)
    stopped_because: str = "finished"
    tool_used: ToolUsed = "answer_directly"
    # Whether a critic produced a usable verdict during this run.
    #
    # Precisely that, and not "this exact text was approved". On a run where
    # the draft was sent back, this answer is the rewrite - written to
    # address the objection, but not itself re-reviewed, because the critic
    # gets one push-back and spending a second call to produce a verdict it
    # is not allowed to act on would buy nothing.
    #
    # False for a direct answer, where there was no evidence to review
    # against, and for a review that could not be completed - see
    # agents/critic.py on why both let the answer through. What it tells a
    # reader is whether the gate ran, which is the fact they cannot get from
    # anywhere else.
    reviewed: bool = False


@dataclass
class SupervisorResultEvent:
    """The run ended. Always last, exactly once."""

    result: SupervisorResult


SupervisorEvent = StepEvent | SupervisorResultEvent


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


class SupervisorAgent:
    """
    A ReAct loop whose tools are other agents.

    Stateless between calls in the same sense the specialists are: `stream`
    resets the shared budget and the shared ledger at the top of every run.
    Conversation continuity is the caller's job (see
    app/services/agent_service.py).
    """

    def __init__(
        self,
        llm_service: LLMService,
        vision_service: VisionService,
        retriever,
        document_registry,
        max_steps: int = 6,
        tree_budget: int = 14,
        research_max_steps: int = 6,
        vision_max_steps: int = 5,
        search_top_k: int = 4,
        temperature: float | None = 0.0,
        critic_enabled: bool = True,
    ) -> None:
        # Both created once and shared by reference with every specialist, so
        # a withdrawal or a label assignment anywhere in the tree is visible
        # everywhere in it. Reset in place per run - never rebound, which
        # would leave the specialists holding the previous run's objects.
        self._budget = StepBudget(tree_budget)
        self._tree_budget = tree_budget
        self._ledger = EvidenceLedger()

        self._research = ResearchAgent(
            llm_service=llm_service,
            retriever=retriever,
            document_registry=document_registry,
            max_steps=research_max_steps,
            search_top_k=search_top_k,
            temperature=temperature,
            budget=self._budget,
            ledger=self._ledger,
        )
        self._vision = VisionAgent(
            llm_service=llm_service,
            vision_service=vision_service,
            retriever=retriever,
            max_steps=vision_max_steps,
            search_top_k=search_top_k,
            temperature=temperature,
            budget=self._budget,
            ledger=self._ledger,
        )

        # Not given the shared budget. A review is not a step the supervisor
        # chose to take, and charging the tree for it would mean a run that
        # delegated twice had fewer steps left to answer with than one that
        # delegated once badly. It is a fixed, bounded cost per run: at most
        # two calls, and usually one.
        self._critic = Critic(llm_service, temperature=temperature) if critic_enabled else None

        # Per-run state. Reset at the top of `stream`.
        self._image: BoundImage | None = None
        self._delegated_to: list[str] = []
        self._message = ""
        # What each specialist reported, in the order it came back. This is
        # what the critic reviews the draft against - the findings, not the
        # scratchpad, since how the supervisor decided to go looking for them
        # is not evidence about anything.
        self._findings: list[tuple[str, str]] = []
        self._verdict: Verdict | None = None
        # One push-back per run, the same policy the specialists use for
        # their own nudges. A critic that can reject indefinitely is a run
        # that can never finish, and the second draft is the one written
        # with the objection in hand - a third rejection has nothing new to
        # tell the supervisor.
        self._revision_requested = False

        self._loop = AgentLoop(
            llm_service=llm_service,
            tools=[
                Tool(
                    name="research_documents",
                    description=(
                        "Ask the document specialist a question about the "
                        "user's ingested documents. It can search the corpus "
                        "several times, following up on what it finds, and "
                        "returns an answer citing [E1]-style evidence "
                        "labels. Use it for anything that turns on what the "
                        "documents actually say."
                    ),
                    input_spec=(
                        '{"question": "a complete, self-contained question - '
                        'the specialist cannot see this conversation"}'
                    ),
                    run=self._run_research,
                    stream=self._stream_research,
                ),
                Tool(
                    name="read_image",
                    description=(
                        "Ask the image specialist a question about the "
                        "picture in this conversation. It decides how to read "
                        "it - vision model for layout and meaning, character "
                        "recognition for exact numbers and codes - and can "
                        "check what it sees against the documents."
                    ),
                    input_spec=(
                        '{"question": "a complete, self-contained question '
                        'about the image"}'
                    ),
                    run=self._run_vision,
                    stream=self._stream_vision,
                ),
                Tool(
                    name="call_external_api",
                    description=(
                        "Get the current weather for a city from a live "
                        "weather service. This is the only live data you can "
                        "reach - it cannot answer anything else."
                    ),
                    # The model names the city, where assistant_agent.py
                    # scraped it out of the message with a regex that only
                    # worked when the sentence ended in "in <City>". Asking
                    # for it as an argument is both more reliable and one
                    # fewer thing to maintain.
                    input_spec='{"city": "the city name, e.g. Paris"}',
                    run=self._run_weather,
                ),
                Tool(
                    name="finish",
                    description=(
                        "Give your final answer to the user and end the turn. "
                        "Use this as soon as you can answer - including "
                        "straight away, when the question needs no specialist."
                    ),
                    input_spec='{"answer": "your full answer to the user"}',
                    run=self._run_finish,
                    terminal=True,
                ),
            ],
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            synthesis_system_prompt=SUPERVISOR_SYNTHESIS_SYSTEM_PROMPT,
            max_steps=max_steps,
            temperature=temperature,
            budget=self._budget,
            run_name="supervisor",
        )

    # ---- Tools ---------------------------------------------------------------

    @staticmethod
    def _sub_question(tool_input: dict, tool_name: str) -> str:
        """
        Pull the question out of a delegation, or refuse the call.

        A rejection rather than a failure: nothing went wrong, the supervisor
        just did not say what to ask. The message becomes the next
        observation, so it is written for the model.
        """
        question = tool_input.get("question") or tool_input.get("__bare__") or ""
        if not isinstance(question, str) or not question.strip():
            raise ActionRejected(
                f'The {tool_name} tool needs a question. Call it as {{"tool": '
                f'"{tool_name}", "input": {{"question": "..."}}}} with a '
                "question that makes sense on its own."
            )
        return question.strip()

    def _stream_research(self, tool_input: dict) -> Iterator[StepEvent | ToolResult]:
        """Delegate to the document specialist, forwarding its steps live."""
        question = self._sub_question(tool_input, "research_documents")
        self._delegated_to.append("research_documents")

        for event in self._research.stream(question):
            if isinstance(event, ResearchResultEvent):
                self._findings.append(
                    (f"the document specialist, asked {question!r}", event.result.answer)
                )
                yield ToolResult(observation=event.result.answer)
            else:
                yield event

    def _run_research(self, tool_input: dict) -> str:
        """Non-streaming form. A drain of `_stream_research`, never a second copy."""
        return _drain(self._stream_research(tool_input))

    def _stream_vision(self, tool_input: dict) -> Iterator[StepEvent | ToolResult]:
        """Delegate to the image specialist, forwarding its steps live."""
        question = self._sub_question(tool_input, "read_image")

        if self._image is None:
            # An observation rather than an error. "There is no picture here"
            # is an ordinary fact the supervisor can act on - by answering
            # from the documents, or by telling the user to attach one - and
            # failing the run would take that choice away.
            yield ToolResult(
                observation=(
                    "There is no image in this conversation. Nothing has been "
                    "uploaded, so there is nothing to look at. Answer the "
                    "user without it, and say that you would need an image "
                    "attached if the question genuinely requires one."
                )
            )
            return

        self._delegated_to.append("read_image")
        for event in self._vision.stream(
            question, image_bytes=self._image.data, mime_type=self._image.mime_type
        ):
            if isinstance(event, VisionResultEvent):
                self._findings.append(
                    (f"the image specialist, asked {question!r}", event.result.answer)
                )
                yield ToolResult(observation=event.result.answer)
            else:
                yield event

    def _run_vision(self, tool_input: dict) -> str:
        """Non-streaming form. A drain of `_stream_vision`."""
        return _drain(self._stream_vision(tool_input))

    def _run_weather(self, tool_input: dict) -> str:
        """Tool: current weather for a named city."""
        city = tool_input.get("city") or tool_input.get("__bare__") or ""
        if not isinstance(city, str) or not city.strip():
            raise ActionRejected(
                'The call_external_api tool needs a city. Call it as {"tool": '
                '"call_external_api", "input": {"city": "Paris"}}.'
            )

        self._delegated_to.append("call_external_api")
        try:
            reading = _fetch_weather(city.strip())
        except requests.RequestException as exc:
            # Returned rather than raised so the supervisor can tell the user
            # the service was unreachable instead of the run dying on it.
            return f"The weather service could not be reached: {exc}"

        self._findings.append((f"a live weather lookup for {city.strip()!r}", reading))
        return reading

    def _run_finish(self, tool_input: dict) -> str:
        """
        Terminal tool: review the draft, then accept it or send it back once.

        No "you must delegate first" policy, unlike the specialists'. For
        them, answering without looking means describing an image or a
        document they never opened. For a supervisor, answering directly is
        the correct and most common move, and a nudge here would tax exactly
        the behaviour prompts/supervisor_prompts.py works to encourage.

        The critic is a gate here rather than a tool the supervisor may
        choose, because an optional reviewer is skipped hardest on the
        answers that most need one - the confident ones. Its objection is
        raised as ActionRejected, which the loop already turns into the next
        observation at a cost of one step, so the supervisor rewrites with
        the objection in front of it.
        """
        answer = tool_input.get("answer") or tool_input.get("__bare__") or ""
        if not isinstance(answer, str) or not answer.strip():
            raise ActionRejected(
                "You called finish with no answer. Call it again as "
                '{"tool": "finish", "input": {"answer": "..."}} with the '
                "answer text."
            )

        if self._critic is None:
            return answer

        # Reviewed at most once. A second draft was written with the
        # objection in hand; a further rejection would have nothing new to
        # say and could keep a run from ever finishing.
        if self._revision_requested:
            return answer

        verdict = self._critic.review(self._message, self._findings, answer)
        # Kept even when it approves, because "was checked and passed" is a
        # fact worth reporting and is not the same as "was never checked".
        self._verdict = verdict

        if verdict.approved:
            return answer

        self._revision_requested = True
        raise ActionRejected(
            f"A reviewer checked that draft against your evidence and asked "
            f"for one change: {verdict.problem}\n\nRewrite the answer to fix "
            "that and call finish again. Do not run more tools - use what you "
            "already have, and say plainly where the evidence runs out."
        )

    # ---- Entrypoint ----------------------------------------------------------

    def run(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        image: BoundImage | None = None,
    ) -> SupervisorResult:
        """
        Answer `message`, delegating where it helps, and return only the result.

        A drain of `stream`, for the same reason every other agent's `run` is.

        Raises:
            ValueError: if `message` is empty.
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        result: SupervisorResult | None = None
        for event in self.stream(message, history=history, image=image):
            if isinstance(event, SupervisorResultEvent):
                result = event.result

        assert result is not None, "stream() ended without a SupervisorResultEvent"
        return result

    def stream(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        image: BoundImage | None = None,
    ) -> Iterator[SupervisorEvent]:
        """
        Answer `message`, emitting each turn as it completes.

        Yields a `StepEvent` per turn - the supervisor's own at depth 0, and
        every specialist's at depth 1 as they happen - then exactly one
        `SupervisorResultEvent`, on every exit path.

        `image` is the picture this turn is about, resolved by the caller
        from conversation memory. Passed in rather than looked up, so this
        agent stays as ignorant of the memory layer as VisionAgent is; None
        means the conversation has never carried one, and `read_image` says
        so if the model tries.

        Raises:
            ValueError: if `message` is empty - eagerly, before the first
                yield, so a caller that has already sent HTTP headers is not
                the one to discover it.
            RuntimeError: if an LLM call fails - possibly mid-stream.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty.")

        # In place on both, for the reason given in __init__: the specialists
        # hold these same objects, and rebinding would leave them writing to
        # the previous run's.
        self._budget.remaining = self._tree_budget
        self._ledger.reset()
        self._image = image
        self._message = message
        self._delegated_to = []
        self._findings = []
        self._verdict = None
        self._revision_requested = False

        def events() -> Iterator[SupervisorEvent]:
            for event in self._loop.stream(message, history=history):
                if isinstance(event, StepEvent):
                    yield event
                    continue

                yield SupervisorResultEvent(self._to_result(event))

        return events()

    def _to_result(self, event: ResultEvent) -> SupervisorResult:
        return SupervisorResult(
            answer=event.result.answer,
            steps=event.result.steps,
            sources=self._ledger.sources(),
            stopped_because=event.result.stopped_because,
            tool_used=self._badge(),
            # False when no critic ran (a direct answer, or the feature off)
            # and when one ran but could not produce a usable verdict.
            reviewed=self._verdict is not None and self._verdict.reviewed,
        )

    def _badge(self) -> ToolUsed:
        """
        One word for the UI, from what the run actually delegated to.

        Reported honestly rather than flatteringly: a turn that read the
        image and then checked a policy says `multiple_specialists`, because
        naming either one alone would tell the reader the answer rests on
        less than it does. The trace carries the detail; this is the label on
        the bubble.
        """
        distinct = set(self._delegated_to) & _BADGE_TOOLS
        if not distinct:
            return "answer_directly"
        if len(distinct) > 1:
            return "multiple_specialists"
        return next(iter(distinct))  # type: ignore[return-value]


def _drain(events: Iterator[StepEvent | ToolResult]) -> str:
    """
    Run a delegating tool's stream to completion and return its observation.

    What `run` is for every tool that also has a `stream`: one dispatch, one
    set of semantics, and a caller that does not want progress simply
    discards it.
    """
    observation = ""
    for event in events:
        if isinstance(event, ToolResult):
            observation = event.observation
    return observation
