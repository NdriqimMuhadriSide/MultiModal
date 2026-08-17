"""
Research agent — multi-hop document research over the ingested corpus.

WHAT THIS IS FOR

A single retrieve-and-answer pass is the right shape for "what does the
handbook say about parental leave?" and the wrong shape for anything that
needs a second look at the documents:

    "How does our refund policy differ from the returns policy?"
    "Which of these reports mentions the Q3 shortfall, and what does it say?"
    "What does the handbook say about the process the contract requires?"

Each of those needs more than one retrieval, and - crucially - the second
retrieval depends on what the first one returned. RAGService.ask cannot do
it at any retrieval quality, because it embeds once, searches once, and
answers once. That is the gap this agent fills.

WHERE THE MACHINERY LIVES

The reasoning loop itself is in agents/agent_loop.py, and the search tool
in agents/knowledge_base_tool.py - both shared with agents/vision_agent.py.
What is left here is what makes this agent *this* agent:

    - which tools it has (corpus listing, search, finish)
    - who it is, and how it is told to research (prompts/research_prompts.py)
    - one policy: it may not finish without having looked at the documents,
      unless it insists

That split is the point. Two agents with completely different jobs share a
parser, a step budget, and a termination story, because none of those are
about the job.
"""
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from agents.agent_loop import (
    ActionRejected,
    AgentLoop,
    AgentStep,
    ResultEvent,
    StepBudget,
    StepEvent,
    Tool,
)
from agents.knowledge_base_tool import EvidenceLedger, KnowledgeBaseSearch
from ai.llm_service import LLMService
from prompts.research_prompts import (
    RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from rag.rag_service import RAGSource

# The trace type this agent returns is the loop's own step record. Aliased
# rather than wrapped: callers that predate the extraction (the service
# layer, the endpoint schema) import ResearchStep, and there is nothing
# research-specific to add to it.
ResearchStep = AgentStep


@dataclass
class ResearchResult:
    answer: str
    # The full trace. This is a first-class part of the result, not debug
    # output: an answer assembled over four hidden retrievals is not
    # something a user can sanity-check, and the trace is what makes it
    # checkable. It is also the only view that shows *why* the agent
    # searched what it searched.
    steps: list[ResearchStep] = field(default_factory=list)
    # Every distinct passage retrieved across every hop, in the order the
    # agent first saw them - so [E1] in the answer is sources[0] here.
    sources: list[RAGSource] = field(default_factory=list)
    stopped_because: Literal["finished", "step_limit", "parse_failures"] = "finished"


@dataclass
class ResearchResultEvent:
    """The run ended. Always last, exactly once."""

    result: ResearchResult


# What `stream` yields: the loop's own StepEvent as each turn lands, then one
# of these. The step type is reused rather than re-wrapped, exactly as
# agents/vision_agent.py does it - a step is a step, and a consumer rendering
# one should not need a second shape per agent.
ResearchEvent = StepEvent | ResearchResultEvent


class ResearchAgent:
    """
    A ReAct-style loop over the document corpus, written without a framework.

    Stateless between calls: every `run` resets the search tool's ledger and
    the finish policy's memory. Conversation continuity is the caller's job
    (see app/services/research_service.py), exactly as it is for
    every other agent here.
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever,
        document_registry,
        max_steps: int = 6,
        search_top_k: int = 4,
        temperature: float | None = 0.0,
        budget: StepBudget | None = None,
        ledger: EvidenceLedger | None = None,
        run_name: str = "research",
    ) -> None:
        self._registry = document_registry
        self._search = KnowledgeBaseSearch(
            retriever, top_k=search_top_k, ledger=ledger
        )
        # The model gets exactly one nudge if it tries to answer without
        # having looked at anything. One, not a rule: some questions really
        # are answerable from the conversation alone, and an agent that
        # cannot ever decline to search would burn a step on every one of
        # them. A second `finish` is honoured whatever it says.
        self._nudged_to_search = False

        self._loop = AgentLoop(
            llm_service=llm_service,
            tools=[
                Tool(
                    name="list_documents",
                    description=(
                        "List every document in the knowledge base with its "
                        "filename, title and length. Use this first when the "
                        "question is about a specific document, or when it asks "
                        "you to compare documents - you need their real names "
                        "before you can search them."
                    ),
                    input_spec="no arguments - use {}",
                    run=self._run_list_documents,
                ),
                self._search.as_tool(),
                Tool(
                    name="finish",
                    description=(
                        "Give your final answer and end the research. Use this "
                        "as soon as the evidence supports an answer - including "
                        "when the evidence shows the documents do not cover the "
                        "question."
                    ),
                    input_spec='{"answer": "your full answer, citing [E1] style labels"}',
                    run=self._run_finish,
                    terminal=True,
                ),
            ],
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            synthesis_system_prompt=RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
            max_steps=max_steps,
            temperature=temperature,
            budget=budget,
            run_name=run_name,
        )

    # ---- Tools ---------------------------------------------------------------

    def _run_list_documents(self, tool_input: dict) -> str:
        """Tool: what is in the corpus at all."""
        records = self._registry.list_documents()
        # PROCESSING and FAILED documents are excluded rather than listed
        # with their status: a document the agent cannot search is one it
        # would otherwise plan searches against and find nothing in.
        ready = [record for record in records if record.status == "READY"]

        if not ready:
            return (
                "The knowledge base has no documents ready to search. Any "
                "question about document contents cannot be answered - say so."
            )

        lines = [
            f"- {record.filename}"
            f" | title: {record.title or 'unknown'}"
            f" | {record.page_count} pages, {record.chunk_count} searchable passages"
            for record in ready
        ]
        return f"{len(ready)} document(s) in the knowledge base:\n" + "\n".join(lines)

    def _run_finish(self, tool_input: dict) -> str:
        """
        Terminal tool: accept the final answer, or push back once.

        Both rejections are policy rather than error - nothing went wrong,
        the agent just is not allowed to stop like that - which is why they
        raise ActionRejected and cost a step rather than failing the run.
        """
        answer = tool_input.get("answer") or tool_input.get("__bare__") or ""
        if not isinstance(answer, str) or not answer.strip():
            raise ActionRejected(
                "You called finish with no answer. Call it again as "
                '{"tool": "finish", "input": {"answer": "..."}} with the '
                "answer text."
            )

        if self._search.attempts == 0 and not self._nudged_to_search:
            self._nudged_to_search = True
            raise ActionRejected(
                "You have not searched the documents yet, so that answer is "
                "from your own knowledge rather than from the knowledge "
                "base. Run a search first. If the question genuinely cannot "
                "be helped by the documents, call finish again and it will "
                "be accepted."
            )

        return answer

    # ---- Entrypoint ----------------------------------------------------------

    def run(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> ResearchResult:
        """
        Research `question` over up to `max_steps` tool calls and return the
        answer, the trace, and every passage seen.

        A drain of `stream`, for the same reason AgentLoop.run and
        VisionAgent.run are: one dispatch, one set of stopping conditions,
        and a caller that does not want progress simply discards it.

        Raises:
            ValueError: if `question` is empty.
            RuntimeError: if an LLM call fails (propagated from llm_service).
        """
        result: ResearchResult | None = None
        for event in self.stream(question, history=history):
            if isinstance(event, ResearchResultEvent):
                result = event.result

        assert result is not None, "stream() ended without a ResearchResultEvent"
        return result

    def stream(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> Iterator["ResearchEvent"]:
        """
        Research `question`, emitting each turn as it completes.

        Added when the supervisor began delegating here: a specialist that
        thinks for four steps behind a single blocking call is fifteen
        seconds in which the caller can be told nothing, and the whole reason
        the trace is first-class is that hidden work should not stay hidden.

        Yields a `StepEvent` per completed turn and exactly one
        `ResearchResultEvent` last, on every exit path.

        Raises:
            ValueError: if `question` is empty - eagerly, before the first
                yield.
            RuntimeError: if an LLM call fails - possibly mid-stream.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty.")

        self._search.reset()
        self._nudged_to_search = False

        def events() -> Iterator["ResearchEvent"]:
            for event in self._loop.stream(question, history=history):
                if isinstance(event, StepEvent):
                    yield event
                    continue

                yield ResearchResultEvent(self._to_result(event))

        return events()

    def _to_result(self, event: ResultEvent) -> ResearchResult:
        return ResearchResult(
            answer=event.result.answer,
            steps=event.result.steps,
            # The ledger's whole list, which under delegation is the *tree's*
            # passages rather than only this run's. That is deliberate: the
            # list has to stay index-aligned with the [E#] labels (see
            # EvidenceLedger.sources), and filtering it to what this agent
            # personally retrieved would renumber it and break every citation
            # in the answer. A delegated run's extra entries are unused, and
            # the supervisor - which owns the ledger - is the one whose list
            # reaches the API.
            sources=self._search.sources(),
            stopped_because=event.result.stopped_because,
        )
