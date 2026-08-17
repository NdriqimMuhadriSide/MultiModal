"""
The knowledge-base search tool, and the evidence ledger behind it.

Shared by every agent that can look things up in the ingested documents -
currently agents/research_agent.py and agents/vision_agent.py. Extracted
when the second one appeared, because a search tool is not really one
function: it is retrieval plus label assignment plus deduplication plus a
rendering format the model has learned to read, and two copies of that
would diverge in ways that only show up as worse answers.

It talks to rag/retriever.py directly rather than to RAGService. RAGService
fuses retrieval and generation and returns an *answer*; an agent needs the
opposite - raw evidence it can accumulate across steps and reason over,
with exactly one generation at the end. Everything the project has tuned
into retrieval (hybrid search, reranking, expansion) still applies, because
all of it lives inside Retriever.retrieve.
"""
from rag.rag_service import RAGSource

# How much of a retrieved passage the model is shown per step.
#
# Not a tuning knob so much as the thing that keeps a loop affordable: at
# four passages a search and several searches a run, untruncated chunks
# would put a dozen full chunks in a prompt that is re-sent on every
# subsequent step, so the cost of step 6 is paid on steps 1 through 5 as
# well. 700 characters is roughly a paragraph - enough to judge relevance
# and quote from, short enough that a dozen of them stay manageable.
#
# The full text is never needed downstream: citations carry chunk_id, so a
# UI can fetch a passage in full if it wants to show one.
DEFAULT_EVIDENCE_CHAR_LIMIT = 700


def _normalise_query(query: str) -> str:
    """Fold a search query to a key for repeat detection."""
    return " ".join(query.lower().split())


class EvidenceLedger:
    """
    Assigns each distinct retrieved passage a stable [E#] label, once.

    Two jobs, both about the same problem - the same chunk coming back from
    more than one search:

    1. Labels. "[E3]" has to mean the same passage on step 5 as it did on
       step 2, or a citation in the final answer points at nothing. Numbering
       per-search would guarantee collisions.

    2. Budget. A passage already shown is re-rendered as a one-line pointer
       instead of 700 more characters of text. On a question that searches
       several related phrasings - which is exactly what these agents do -
       the overlap between steps is large, and paying for it repeatedly on
       every subsequent turn is the difference between a loop that fits in a
       context window and one that does not.
    """

    def __init__(self) -> None:
        self._labels: dict[str, str] = {}
        self._sources: list[RAGSource] = []

    def record(self, chunk) -> tuple[str, bool]:
        """Return `(label, is_new)` for `chunk`, registering it if unseen."""
        existing = self._labels.get(chunk.chunk_id)
        if existing is not None:
            return existing, False

        label = f"E{len(self._labels) + 1}"
        self._labels[chunk.chunk_id] = label
        self._sources.append(
            RAGSource(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page=chunk.page,
                score=chunk.score,
                section=chunk.section,
                dense_score=chunk.dense_score,
                keyword_score=chunk.keyword_score,
                matched_by=chunk.matched_by,
                rerank_score=chunk.rerank_score,
                found_by_query=chunk.found_by_query,
            )
        )
        return label, True

    def sources(self) -> list[RAGSource]:
        """Every distinct passage seen, in first-seen order - so index i is [E(i+1)]."""
        return list(self._sources)

    def reset(self) -> None:
        """
        Forget everything, in place.

        In place, rather than by building a fresh ledger, because a shared
        one is held by reference by every agent in a tree. Rebinding the
        supervisor's attribute would leave each specialist still numbering
        into last run's ledger, which is the exact collision this object
        exists to prevent - and it would be invisible until two runs in the
        same process produced overlapping labels.
        """
        self._labels.clear()
        self._sources.clear()


class KnowledgeBaseSearch:
    """
    A reusable `search` tool plus the per-run state it needs.

    An object rather than a bare function because the tool is not stateless:
    it has to remember which passages it has already shown (for labels and
    for budget) and which queries it has already run (so a repeat costs
    nothing). That state is per-run, which is why `reset` exists and why
    every agent using this must call it at the top of its own run.
    """

    def __init__(
        self,
        retriever,
        name: str = "search",
        description: str | None = None,
        top_k: int = 4,
        evidence_char_limit: int = DEFAULT_EVIDENCE_CHAR_LIMIT,
        ledger: EvidenceLedger | None = None,
    ) -> None:
        self._retriever = retriever
        self._name = name
        self._description = description or (
            "Search the ingested documents for passages matching a query and "
            "return them as labelled evidence. This is your only way to see "
            "what the documents say. Search once per distinct topic - do not "
            "put two topics in one query."
        )
        self._top_k = top_k
        self._evidence_char_limit = evidence_char_limit

        # A ledger passed in is shared with other searchers - in practice,
        # the one a supervisor hands to every specialist it delegates to.
        # Sharing it is what keeps [E#] labels globally unique across a tree.
        #
        # Two agents each numbering from their own ledger both hand out [E1]
        # for different passages, and the supervisor then merges their
        # answers into one citation list where half the labels point at the
        # wrong text. Nothing raises; the answer simply cites the wrong
        # source. This flag is what makes that impossible.
        self._owns_ledger = ledger is None
        self._ledger = ledger if ledger is not None else EvidenceLedger()
        self._seen_queries: set[str] = set()
        # Counts searches *attempted*, where _seen_queries records only the
        # ones that completed. The difference matters to an agent whose
        # finish policy asks "have you looked yet": a search that crashed
        # has still looked, and telling the model it has not would be both
        # false and useless.
        self.attempts = 0

    def reset(self) -> None:
        """
        Clear per-run state. Every agent must call this at the top of `run`.

        A borrowed ledger is deliberately left alone. A specialist resetting
        at the start of its own run would otherwise wipe the labels the
        supervisor assigned before delegating - and the supervisor, which
        owns the ledger, is the one that resets it once per tree.

        The query cache is cleared either way. It exists to stop one agent
        paying twice for the same retrieval within its own run, and two
        agents in a tree asking the same question is a repeat worth
        answering: they each need the passages in their own scratchpad.
        """
        if self._owns_ledger:
            self._ledger.reset()
        self._seen_queries = set()
        self.attempts = 0

    def sources(self) -> list[RAGSource]:
        return self._ledger.sources()

    def as_tool(self):
        """Build the Tool record the loop dispatches on."""
        from agents.agent_loop import Tool

        return Tool(
            name=self._name,
            description=self._description,
            input_spec='{"query": "the words to search for"}',
            run=self.run,
        )

    def run(self, tool_input: dict) -> str:
        """
        Retrieve passages and render them as labelled evidence.

        `top_k` is fixed by configuration and deliberately not exposed to the
        model. Given the choice it asks for large values on hard questions -
        which is precisely backwards, since a wider net on a vague query
        returns more noise, and every extra passage is re-sent in the prompt
        on every later step. The way to see more here is another,
        better-worded search, which is the behaviour a loop exists to
        encourage.
        """
        query = tool_input.get("query") or tool_input.get("__bare__") or ""
        if not isinstance(query, str) or not query.strip():
            return (
                f'The {self._name} tool needs a query. Call it as '
                f'{{"tool": "{self._name}", "input": {{"query": "..."}}}}.'
            )

        self.attempts += 1

        key = _normalise_query(query)
        if key in self._seen_queries:
            # Answered from the tool's own memory rather than by re-running
            # retrieval: the result would be byte-identical, and the useful
            # thing to return is the fact that it is a repeat.
            return (
                f'You already searched for "{query}" earlier in this run. '
                "The results are in your transcript above. Search different "
                "wording, or call finish if you have enough."
            )

        try:
            chunks = self._retriever.retrieve(query, top_k=self._top_k)
        except ValueError as exc:
            # A bad query is the model's mistake to fix, so it becomes an
            # observation. A RuntimeError - the embedding model or the store
            # being down - is not, and is left to propagate.
            return f"That search was rejected: {exc}"

        self._seen_queries.add(key)

        if not chunks:
            return (
                f'No passages matched "{query}". Try different wording, or '
                "conclude that the documents do not cover this."
            )

        blocks = []
        for chunk in chunks:
            label, is_new = self._ledger.record(chunk)
            where = f"{chunk.filename} p.{chunk.page}"
            if chunk.section:
                where += f" — {chunk.section}"

            if not is_new:
                blocks.append(f"[{label}] {where} (already shown above, not repeated)")
                continue

            text = chunk.text.strip()
            if len(text) > self._evidence_char_limit:
                text = text[: self._evidence_char_limit].rstrip() + " […]"
            blocks.append(f"[{label}] {where}\n{text}")

        return f'{len(chunks)} passage(s) for "{query}":\n\n' + "\n\n".join(blocks)
