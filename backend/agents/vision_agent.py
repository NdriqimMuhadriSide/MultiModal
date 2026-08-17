"""
Vision & OCR agent — reads an uploaded image by deciding *how* to read it.

WHAT THIS IS FOR

This project is called MultiModal and, until this module, every agent in it
was text-only. `VisionService` sends an image to a vision model,
`rag/ocr.py` runs character recognition during PDF ingestion, and no agent
touched either.

More than a wiring gap: there is a real decision here that a fixed pipeline
cannot make. OCR and a vision-language model are good at opposite things.

    Tesseract           exact on printed characters, free, local, and
                        completely blind to meaning. Nothing on diagrams,
                        charts or handwriting.

    Vision model        understands layout, charts, handwriting, and what a
                        document *is* - and misreads long digit strings,
                        which is exactly what an invoice total is.

So "what is the total on this receipt?" needs both: the vision model to
find where the total is, character recognition to read the digits
correctly. Which combination a given image needs depends on what is in it,
and the only way to know is to look. That is an agent's decision, not a
config flag.

The existing pipeline makes this call with a fixed rule -
`ocr_min_text_chars: 32` - and that rule applies to PDF pages during
ingestion only. An uploaded image goes straight to the vision model with no
character-recognition path at all.

THE FOURTH TOOL

`search_knowledge_base` is what makes this more than an image reader. It is
what turns

    "what does this receipt say?"          (an image question)

into

    "does this receipt comply with our expense policy?"

which needs the image *and* the corpus, and is the first thing in this
codebase that genuinely spans two modalities in one answer.

WHERE THE MACHINERY LIVES

The loop, parser and termination logic are in agents/agent_loop.py, and the
search tool in agents/knowledge_base_tool.py - both shared with
agents/research_agent.py. What is here is the tool set, the per-run image,
and one policy: do not answer without having looked at the image.

TWO INTEGRATIONS DELIBERATELY NOT USED

An MCP server. Every tool below is a function in this repo called in this
process; exposing them over MCP would be JSON-RPC to localhost standing in
for a function call. MCP earns its cost at an ownership or deployment
boundary, and there isn't one here. If something outside this codebase ever
needs to read images this way, that is when to revisit it.

An agent as a tool. agents/research_agent.py could be wrapped as
`ask_research_agent`, and it is tempting because the two already share a
loop. It is declined: a multi-hop loop nested inside this one's five-step
budget multiplies the worst case into the tens of LLM calls, against a
question shape - "check this image against a policy" - that a single
retrieval already serves. `search_knowledge_base` covers the need at a
fraction of the cost. Revisit only if image questions start needing several
dependent corpus lookups, which would be evidence rather than a guess.
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
from agents.value_check import unverified_values
from ai.llm_service import LLMService
from ai.vision_service import SUPPORTED_IMAGE_MIME_TYPES, VisionService
from prompts.vision_prompts import (
    VISION_AGENT_SYNTHESIS_SYSTEM_PROMPT,
    VISION_AGENT_SYSTEM_PROMPT,
)
from rag import ocr
from rag.rag_service import RAGSource

# Ceilings on what a single tool may put into the scratchpad.
#
# Both are re-sent on every later step, so an uncapped observation is paid
# for repeatedly. The OCR limit is the looser of the two because its output
# is a layout-preserving character grid - the whitespace that makes columns
# line up is most of the character count, and cutting it too tight destroys
# the alignment that made the grid worth producing.
_OCR_CHAR_LIMIT = 2500
_INSPECTION_CHAR_LIMIT = 1500


@dataclass
class VisionResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    # Knowledge-base passages only. Nothing the image itself produced gets a
    # citation: a RAGSource is a chunk id, a filename and a page, and an OCR
    # grid has none of those. Inventing a source record for the uploaded
    # image would make the citation list lie about where things came from.
    sources: list[RAGSource] = field(default_factory=list)
    stopped_because: Literal["finished", "step_limit", "parse_failures"] = "finished"
    # Number-shaped tokens in the answer that character recognition did not
    # confirm (agents/value_check.py). Empty is the good case.
    #
    # Reported rather than enforced, and not every entry is a fault: a
    # correctly derived "£42.25 a head" and a limit quoted from a retrieved
    # policy both land here legitimately. What it gives a reader is the one
    # thing they could not otherwise tell - which figures came off the image
    # and which did not.
    unverified_values: list[str] = field(default_factory=list)


@dataclass
class VisionResultEvent:
    """The run ended. Always last, exactly once."""

    result: VisionResult


# What `stream` yields: the loop's own StepEvent as each turn lands, then
# one of these. The step type is reused rather than re-wrapped - a step is
# a step, and a consumer that renders one should not need two shapes.
VisionEvent = StepEvent | VisionResultEvent


class VisionAgent:
    """
    A ReAct loop whose tools read one attached image, plus the document corpus.

    The image is bound per run rather than chosen by the model: there is
    exactly one, and making the agent name it would add a way to get it
    wrong without adding anything it could get right.
    """

    def __init__(
        self,
        llm_service: LLMService,
        vision_service: VisionService,
        retriever,
        max_steps: int = 5,
        search_top_k: int = 4,
        temperature: float | None = 0.0,
        budget: StepBudget | None = None,
        ledger: EvidenceLedger | None = None,
        run_name: str = "vision",
    ) -> None:
        self._vision_service = vision_service
        # Applies to the vision call too, not only the loop's reasoning
        # turns. The agent reads that reply for facts about a document, so a
        # description that varies between identical runs makes the whole
        # answer unreproducible.
        self._temperature = temperature
        self._search = KnowledgeBaseSearch(
            retriever,
            name="search_knowledge_base",
            description=(
                "Search the ingested documents - NOT the image - for "
                "passages matching a query, and return them as labelled "
                "evidence. Use it when the question turns on a policy, rule, "
                "limit or specification that the image should be checked "
                "against."
            ),
            top_k=search_top_k,
            ledger=ledger,
        )

        # Per-run state. Reset at the top of `run`.
        self._image_bytes: bytes = b""
        self._mime_type: str = ""
        self._ocr_text: str | None = None
        self._asked_of_image: set[str] = set()
        self._looked_at_image = False
        self._nudged_to_look = False

        self._loop = AgentLoop(
            llm_service=llm_service,
            tools=[
                Tool(
                    name="inspect_image",
                    description=(
                        "Look at the image and answer a question about it. "
                        "Understands layout, diagrams, charts, handwriting "
                        "and document type. Use it first to learn what you "
                        "are looking at and where things are. Do NOT trust "
                        "it for exact numbers, dates or codes."
                    ),
                    input_spec='{"question": "what to look for in the image"}',
                    run=self._run_inspect_image,
                ),
                Tool(
                    name="read_text",
                    description=(
                        "Run character recognition over the image and return "
                        "its text as a layout-preserving character grid. "
                        "Exact on printed text, useless on handwriting, "
                        "photos and diagrams. Use it before quoting any "
                        "number, date, code or exact spelling."
                    ),
                    input_spec="no arguments - use {}",
                    run=self._run_read_text,
                ),
                self._search.as_tool(),
                Tool(
                    name="finish",
                    description=(
                        "Give your final answer and end the analysis. Use "
                        "this as soon as your tools support an answer - "
                        "including when they show the image does not contain "
                        "what was asked about."
                    ),
                    input_spec='{"answer": "your full answer"}',
                    run=self._run_finish,
                    terminal=True,
                ),
            ],
            system_prompt=VISION_AGENT_SYSTEM_PROMPT,
            synthesis_system_prompt=VISION_AGENT_SYNTHESIS_SYSTEM_PROMPT,
            max_steps=max_steps,
            temperature=temperature,
            budget=budget,
            run_name=run_name,
        )

    # ---- Tools ---------------------------------------------------------------

    def _run_inspect_image(self, tool_input: dict) -> str:
        """Tool: ask the vision model about the attached image."""
        question = tool_input.get("question") or tool_input.get("__bare__") or ""
        if not isinstance(question, str) or not question.strip():
            return (
                'The inspect_image tool needs a question. Call it as {"tool": '
                '"inspect_image", "input": {"question": "..."}}.'
            )

        key = " ".join(question.lower().split())
        if key in self._asked_of_image:
            # Unlike a repeated search, this is not free to re-run - it is a
            # provider round trip - and the answer would be substantially
            # the same. Refusing it is cheaper than paying for a duplicate.
            return (
                f'You already asked the image "{question}". Its answer is in '
                "your transcript above. Ask something different, use "
                "read_text for exact characters, or call finish."
            )

        self._looked_at_image = True
        self._asked_of_image.add(key)

        analysis = self._vision_service.analyze_image(
            image_bytes=self._image_bytes,
            mime_type=self._mime_type,
            question=question,
            temperature=self._temperature,
        ).strip()

        if not analysis:
            return "The vision model returned nothing for that question."

        if len(analysis) > _INSPECTION_CHAR_LIMIT:
            analysis = analysis[:_INSPECTION_CHAR_LIMIT].rstrip() + " […]"
        return analysis

    def _run_read_text(self, tool_input: dict) -> str:
        """
        Tool: character recognition over the attached image.

        Cached for the run. Unlike `inspect_image`, this takes no arguments,
        so a second call is necessarily identical - and OCR costs around a
        second of local compute, which is worth not paying twice.
        """
        if not ocr.is_available():
            # An honest, actionable observation rather than an empty result:
            # "no text found" and "no OCR engine installed" call for
            # completely different next moves, and only one of them is the
            # model's problem to solve.
            reason = ocr.unavailable_reason() or "Character recognition is unavailable."
            return (
                f"{reason} You cannot read exact text from this image. Use "
                "inspect_image instead, and say in your answer that the text "
                "could not be verified character by character."
            )

        if self._ocr_text is not None:
            return (
                "You already ran read_text. Its output is in your transcript "
                "above and will not change. Use it, or call finish."
            )
        self._ocr_text = ocr.ocr_image(self._image_bytes)

        if not self._ocr_text.strip():
            return (
                "Character recognition found no printed text in this image. "
                "It is most likely a photograph, a diagram, or handwriting. "
                "Use inspect_image to describe it instead, and do not quote "
                "exact values."
            )

        # Only now does this count as having seen the image, and the
        # placement is deliberate. An unavailable engine and an empty grid
        # both mean the agent holds no image content - if either satisfied
        # the finish policy, the agent could answer a question about a
        # picture it never managed to look at.
        self._looked_at_image = True

        text = self._ocr_text
        truncated = ""
        if len(text) > _OCR_CHAR_LIMIT:
            text = text[:_OCR_CHAR_LIMIT].rstrip()
            truncated = (
                "\n\n[…] Text continues beyond this point and was cut for "
                "length. If what you need is missing, use inspect_image to "
                "locate it on the page."
            )

        return (
            "Recognised text, with the page layout preserved as spacing:\n\n"
            f"{text}{truncated}"
        )

    def _run_finish(self, tool_input: dict) -> str:
        """Terminal tool: accept the answer, or push back once."""
        answer = tool_input.get("answer") or tool_input.get("__bare__") or ""
        if not isinstance(answer, str) or not answer.strip():
            raise ActionRejected(
                "You called finish with no answer. Call it again as "
                '{"tool": "finish", "input": {"answer": "..."}} with the '
                "answer text."
            )

        # One nudge, not a rule - the same policy as the research agent's,
        # for the same reason. An image the model has not opened cannot have
        # been described, but a question that turns out not to be about the
        # image at all should not be un-answerable.
        if not self._looked_at_image and not self._nudged_to_look:
            self._nudged_to_look = True
            raise ActionRejected(
                "You have not looked at the image yet, so that answer "
                "describes an image you have not seen. Use inspect_image or "
                "read_text first. If the question genuinely is not about the "
                "image, call finish again and it will be accepted."
            )

        return answer

    # ---- Entrypoint ----------------------------------------------------------

    def run(
        self,
        question: str,
        image_bytes: bytes,
        mime_type: str,
        history: list[dict[str, str]] | None = None,
    ) -> VisionResult:
        """
        Answer `question` about `image_bytes` and return only the result.

        A drain of `stream`, for the same reason AgentLoop.run is one: two
        implementations of the same dispatch drift, and the streaming path
        is the one that would drift silently.

        Raises:
            ValueError: if the question is empty, the image is empty, or the
                mime type is unsupported.
            RuntimeError: if an LLM or vision call fails.
        """
        result: VisionResult | None = None
        for event in self.stream(question, image_bytes, mime_type, history=history):
            if isinstance(event, VisionResultEvent):
                result = event.result

        assert result is not None, "stream() ended without a VisionResultEvent"
        return result

    def stream(
        self,
        question: str,
        image_bytes: bytes,
        mime_type: str,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[VisionEvent]:
        """
        Answer `question` about `image_bytes`, emitting each turn as it lands.

        Image validation happens here rather than on first tool use, and
        eagerly rather than on first iteration.
        `VisionService.analyze_image` would reject an unsupported type
        itself, but only once the model happened to call it - turning a
        400-shaped problem into a mid-run observation the agent would then
        try to reason around. Doing it before any yield also means the
        endpoint can still answer 400, which it cannot once the stream has
        opened.

        Raises:
            ValueError: if the question is empty, the image is empty, or the
                mime type is unsupported.
            RuntimeError: if an LLM or vision call fails - possibly
                mid-stream.
        """
        if not image_bytes:
            raise ValueError("image must not be empty.")

        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Unsupported image type '{mime_type}'. Supported types: "
                f"{', '.join(sorted(SUPPORTED_IMAGE_MIME_TYPES))}."
            )

        self._image_bytes = image_bytes
        self._mime_type = mime_type
        self._ocr_text = None
        self._asked_of_image = set()
        self._looked_at_image = False
        self._nudged_to_look = False
        self._search.reset()

        # The model is told an image is attached before it plans anything.
        # Without this the first step is often a search, because the prompt
        # otherwise reads as a text question with some unusual tools.
        preamble = (
            f"An image of type {mime_type} is attached to this session. Your "
            "inspect_image and read_text tools operate on it - you do not "
            "need to identify or upload it.\n\n"
        )

        def events() -> Iterator[VisionEvent]:
            for event in self._loop.stream(
                question, history=history, preamble=preamble
            ):
                if isinstance(event, StepEvent):
                    yield event
                    continue

                # The loop finished. Everything this agent adds on top -
                # citations and the numeric check - is computed now rather
                # than per step, because both are properties of the whole
                # run: a label assigned on step 2 is only final once no
                # further search can renumber it, and a figure can only be
                # checked against the recognition the run actually did.
                yield VisionResultEvent(self._to_result(event))

        return events()

    def _to_result(self, event: ResultEvent) -> VisionResult:
        return VisionResult(
            answer=event.result.answer,
            steps=event.result.steps,
            sources=self._search.sources(),
            stopped_because=event.result.stopped_because,
            # Checked against whatever recognition produced this run, which
            # is "" when read_text was never called or found nothing. That
            # is the correct baseline rather than a missing one: an answer
            # full of figures from an image nobody ran OCR over is exactly
            # the case worth flagging.
            unverified_values=unverified_values(
                event.result.answer, self._ocr_text or ""
            ),
        )
