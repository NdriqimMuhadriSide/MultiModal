"""
Retriever.

Responsibility: the "retrieval" half of RAG, isolated from context/prompt
construction and generation - turn a user's question into a ranked list of
relevant chunks.

Two retrievers run here, and RETRIEVAL_MODE picks which:

    dense    question -> embed -> Chroma cosine search -> threshold
    keyword  question -> tokenize -> BM25 over chunk text
    hybrid   both of the above -> reciprocal rank fusion (default)

...and RERANK_ENABLED adds a final pass over whichever shortlist comes out,
re-scoring each candidate against the question with a cross-encoder
(rag/reranker.py). Retrieval optimises recall over a large corpus; reranking
optimises precision over the ~20 candidates recall produced. They are
different jobs, which is why one model cannot do both.

They fail in opposite directions, which is the entire argument for running
both. Dense search matches meaning and so finds the paragraph about
cancellations when the question says "how do I stop my subscription" - and
misses "ERR-4021" entirely, because a rare literal token barely registers in
a 384-dimension vector. BM25 matches the words and so nails the error code,
the surname, the regulation number - and returns nothing at all when the
question and the document happen to use different words for the same thing.
A question is usually some of each, and which kind it is isn't known in
advance, so the robust move is to ask both and merge (rag/fusion.py).

This is the RAG layer's first stage (see rag/rag_service.py for how it's
composed with rag/context_builder.py and prompts/rag_prompts.py). Kept
separate from rag/rag_service.py so retrieval-quality concerns - top_k, the
similarity threshold, fusion weights, and eventually re-ranking - can be
tuned and tested in one place, independent of how the retrieved chunks get
turned into a prompt or a generated answer.

Layering rule: this module only talks to rag/embedding_service.py,
rag/keyword_index.py, rag/fusion.py and rag/vector_store.py (the storage
layer). It never imports ai/llm_service.py or prompts/ - retrieval and
generation stay separate concerns.
"""
import logging
from dataclasses import dataclass, field

from app.core.config import settings
from rag.embedding_service import EmbeddingService
from rag.fusion import Ranking, reciprocal_rank_fusion
from rag.grading import CORRECT, grade_retrieval
from rag.keyword_index import KeywordIndex
from rag.reranker import Reranker
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Default number of chunks returned per query - matches the Phase 4B spec
# ("top_k = 5").
DEFAULT_TOP_K = 5

# Chunks with a similarity below this are dropped before ever reaching the
# LLM. Without this, a low-relevance chunk could still be included just
# because top_k asked for N results even when the store only has weak
# matches - which would work against "answer only from context, say you
# don't know otherwise."
#
# Applies to the *dense* half only. A chunk that keyword search found has no
# cosine similarity to compare against this number, and judging it by a score
# it was never given would silently delete the half of hybrid retrieval that
# is best at exact terms. Its equivalent floor is BM25's own: a chunk sharing
# no term with the query scores zero and rag/keyword_index.py never returns it.
DEFAULT_MIN_SIMILARITY = 0.2

# The valid values of RETRIEVAL_MODE.
RETRIEVAL_MODES = ("dense", "keyword", "hybrid")


@dataclass
class _QueryHits:
    """
    What one query phrasing found, before anything is merged.

    Kept separate per query rather than folded straight into RetrievedChunks
    because fusion needs each phrasing's own ordering. `payload` carries the
    text and metadata so a chunk only one phrasing found can still be built
    without a second trip to the store.
    """

    dense: dict[str, float] = field(default_factory=dict)
    keyword: dict[str, float] = field(default_factory=dict)
    payload: dict[str, tuple[str, dict]] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """
    A single chunk retrieved for a question, with its source metadata.

    The first five fields are the Phase 4B retriever contract, unchanged:
        { "text": "", "filename": "", "page": "", "score": "" }
    """

    chunk_id: str
    text: str
    filename: str
    page: int
    # What the final ordering was actually done on. In `dense` mode this is
    # the cosine similarity, exactly as before; in `keyword` mode a BM25
    # score; in `hybrid` mode a fused rank score. Deliberately one field
    # rather than three, so callers that just want "how did this rank" have
    # one thing to read - and deliberately *not* called `similarity`, because
    # outside dense mode it isn't one.
    score: float
    # Heading path the chunk came from (see rag/structure.py), or "" for a
    # document with no detected structure. Defaulted so the Phase 4B contract
    # above still constructs unchanged.
    section: str = ""
    # The per-retriever scores behind `score`, each None when that retriever
    # didn't surface this chunk. Kept because a fused score is opaque on its
    # own: when a hybrid result looks wrong, the useful question is which half
    # put it there, and these are the answer.
    dense_score: float | None = None
    keyword_score: float | None = None
    # "dense" | "keyword" | "both". Redundant with the two fields above and
    # worth it anyway - it is the thing you actually want to see in a log line
    # or a debugging UI, and deriving it at every call site invites three
    # slightly different derivations.
    matched_by: str = "dense"
    # Carried through from chunk metadata so collapse_duplicate_passages can
    # run *after* fusion, on these rather than on raw store results - which it
    # has to, or the two halves' duplicates would only be collapsed within
    # each half.
    parent_id: str = ""
    # The phrasing that first surfaced this chunk. The original question is
    # always queried first, so a variant here means expansion is the reason
    # this chunk is in the results at all - which is the one number that says
    # whether MULTI_QUERY_ENABLED is earning its LLM call on this corpus.
    found_by_query: str = ""
    # The cross-encoder's verdict in (0, 1), or None when reranking is off.
    # When it is set it is also what `score` holds, because it is then what
    # the final ordering was done on - kept separately so the fused rank score
    # it replaced is not the only thing lost from the record.
    rerank_score: float | None = None


class Retriever:
    """Runs dense and/or keyword search over the store and returns ranked chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        keyword_index: KeywordIndex | None = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        mode: str | None = None,
        reranker: Reranker | None = None,
        query_expander=None,
        multi_query: bool = False,
        corrective: bool = False,
    ) -> None:
        self._mode = mode if mode is not None else settings.retrieval_mode
        if self._mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval mode '{self._mode}'. "
                f"Valid options: {', '.join(RETRIEVAL_MODES)}."
            )
        if self._mode in ("keyword", "hybrid") and keyword_index is None:
            # Raised rather than quietly dropping to dense-only. A deployment
            # configured for hybrid that silently runs dense would look
            # completely healthy and just answer worse - the same reasoning as
            # rag/chunking/__init__.py's refusal to fall back to `recursive`.
            raise ValueError(
                f"RETRIEVAL_MODE='{self._mode}' needs a keyword index."
            )
        if multi_query and query_expander is None:
            raise ValueError("MULTI_QUERY_ENABLED needs a query expander.")
        if corrective and reranker is None:
            # A hard requirement: without a reranker there is no score that
            # means the same thing from one question to the next, so a
            # threshold on it would be noise (see rag/grading.py).
            raise ValueError("CORRECTIVE_RAG_ENABLED needs a reranker.")
        # A query expander, by contrast, is optional here. With one, a poor
        # grade triggers a rephrase-and-retry; without one, grading still
        # decides whether to answer or refuse, which needs no LLM at all and
        # is - measurably, on this project's own eval set - where all of the
        # benefit came from. Requiring an API key for the retry would have
        # gated the useful half behind the half that did nothing.

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._min_similarity = min_similarity
        self._reranker = reranker
        self._query_expander = query_expander
        self._multi_query = multi_query
        self._corrective = corrective

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """
        Return up to `top_k` chunks most relevant to `question`.

        Raises:
            ValueError: if `question` is empty or top_k is not positive
                (propagated from embedding_service / vector_store).
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        # The original question is always first, and is always present even
        # when expansion produced nothing - see rag/query_expansion.py.
        queries = [question]
        if self._multi_query:
            queries.extend(self._query_expander.expand(question))

        chunks = self._retrieve_for(queries, question, top_k)

        if not self._corrective:
            return chunks[:top_k]

        grade = grade_retrieval(
            chunks,
            accept_score=settings.corrective_accept_score,
            reject_score=settings.corrective_reject_score,
        )

        can_retry = self._query_expander is not None and not self._multi_query
        if grade.verdict != CORRECT and can_retry:
            # Anything short of a confident result gets one more attempt with
            # rephrased queries - including a result bad enough to refuse. A
            # refusal on first impression would give up on a question that was
            # merely worded unlike the document, which is the exact failure
            # expansion exists to fix; refusing should be a considered verdict,
            # not a first reading.
            #
            # Running expansion only here is the point: applied to every
            # question it costs an LLM call on the ones that never needed it,
            # which is what MULTI_QUERY_ENABLED does and what its own
            # measurements showed buys nothing when retrieval was already fine.
            retried_queries = queries + self._query_expander.expand(question)
            retried = self._retrieve_for(retried_queries, question, top_k)
            # Kept only if it actually improved on what we had. A retry that
            # found nothing better should not be allowed to reorder a result
            # set it failed to beat.
            if _best_score(retried) > _best_score(chunks):
                chunks = retried
            grade = grade_retrieval(
                chunks,
                accept_score=settings.corrective_accept_score,
                reject_score=settings.corrective_reject_score,
            )

        if not grade.is_answerable:
            # Returning nothing is how the refusal is expressed: with no
            # context to build, rag/rag_service.py already answers "I don't
            # know" without calling the LLM at all. That path existed for an
            # empty corpus; grading is what makes it fire for a corpus that
            # simply has no answer to this question.
            logger.info(
                "Corrective RAG: refusing to answer, best relevance %.4f is below %.2f.",
                grade.best_score if grade.best_score is not None else -1.0,
                settings.corrective_reject_score,
            )
            return []

        return chunks[:top_k]

    def _retrieve_for(
        self, queries: list[str], question: str, top_k: int
    ) -> list[RetrievedChunk]:
        """
        Run the whole search -> fuse -> collapse -> rerank pipeline for a set
        of query phrasings.

        Split out of `retrieve` so corrective retrieval can run it a second
        time with more phrasings, rather than duplicating the pipeline or
        threading a retry flag through every step of it.
        """
        # Both halves are over-fetched, for two reasons that compound: chunks
        # pointing at the same passage get collapsed below, and fusion needs
        # enough depth in each list for the two to overlap at all. Fusing two
        # 5-long lists that share nothing just concatenates them.
        candidate_count = top_k * settings.retrieval_overfetch

        hits = [self._search_one(query, candidate_count) for query in queries]
        chunks = self._merge(queries, hits)
        rankings = self._rankings(queries, hits)

        if len(rankings) == 1:
            # One retriever, so its own order is already the answer and
            # fusing would only replace real scores with rank scores. This is
            # what keeps `dense` mode byte-for-byte the behaviour that existed
            # before hybrid retrieval - same order, same scores.
            ordered = [chunks[chunk_id] for chunk_id in rankings[0].chunk_ids]
        else:
            fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k)
            ordered = []
            for chunk_id, score in fused:
                chunk = chunks[chunk_id]
                chunk.score = score
                ordered.append(chunk)

        # Collapsed *before* reranking, not after: duplicates of one passage
        # would otherwise each cost a cross-encoder forward pass to establish
        # what they already agree on, and then be thrown away anyway.
        ordered = collapse_duplicate_passages(ordered)

        if self._reranker is not None:
            # Scored against the *original* question, never a variant: the
            # variants exist to widen what retrieval finds, but what the user
            # asked is still the only thing relevance can honestly be measured
            # against.
            ordered = self._rerank(question, ordered)

        return ordered[:top_k]

    def _rerank(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Re-score the shortlist with the cross-encoder and reorder by it.

        This is the point of over-fetching. Retrieval's job was recall - get
        the right chunk into the shortlist at all - and it is scored on
        rank-ish evidence (vector proximity, term overlap) that cannot tell
        "about the same topic" from "answers this question". The cross-encoder
        reads each pair properly and decides which of the candidates actually
        belongs in the top_k that reaches the model.

        Ordering is by rerank score alone rather than blended with the fused
        score: the two are measuring different things, and a blend would need
        a weight nobody can set honestly without an evaluation set. Python's
        sort is stable, so candidates the cross-encoder cannot separate keep
        their retrieval order.
        """
        if not chunks:
            return chunks

        scores = self._reranker.score(question, [chunk.text for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = score
            chunk.score = score

        kept = [chunk for chunk in chunks if chunk.rerank_score >= settings.rerank_min_score]
        return sorted(kept, key=lambda chunk: -chunk.rerank_score)

    def _search_one(self, query: str, candidate_count: int) -> "_QueryHits":
        """
        Run the configured retrievers for a single query phrasing.

        Returns raw per-query scores rather than assembled chunks, because a
        chunk found by three phrasings needs to appear at its own rank in each
        of their rankings - that agreement (or disagreement) is the signal
        fusion reads. Flattening to one score per chunk here would destroy it.
        """
        hits = _QueryHits()

        if self._mode in ("dense", "hybrid"):
            query_embedding = self._embedding_service.embed_text(query)
            for result in self._vector_store.search(query_embedding, top_k=candidate_count):
                if result.similarity < self._min_similarity:
                    continue
                hits.dense[result.chunk_id] = result.similarity
                hits.payload[result.chunk_id] = (result.text, result.metadata)

        if self._mode in ("keyword", "hybrid"):
            assert self._keyword_index is not None  # enforced in __init__
            for match in self._keyword_index.search(query, top_k=candidate_count):
                hits.keyword[match.chunk_id] = match.score
                hits.payload.setdefault(match.chunk_id, (match.text, match.metadata))

        return hits

    def _merge(self, queries: list[str], hits: list["_QueryHits"]) -> dict[str, RetrievedChunk]:
        """
        Collapse every query's hits into one RetrievedChunk per id.

        A chunk found by several phrasings must be one object, or it would
        occupy several of the `top_k` slots with identical text. Where they
        disagree on a score the best one is kept - it is the honest answer to
        "how well did retrieval do on this chunk" - and the per-query values it
        summarises are still in `hits`, which is where fusion reads them.

        `found_by_query` records the first phrasing that surfaced the chunk.
        Since the original question is always queried first, anything labelled
        with a variant is something expansion added and the original question
        alone would have missed.
        """
        chunks: dict[str, RetrievedChunk] = {}

        for query, query_hits in zip(queries, hits):
            for chunk_id, (text, metadata) in query_hits.payload.items():
                dense = query_hits.dense.get(chunk_id)
                keyword = query_hits.keyword.get(chunk_id)

                existing = chunks.get(chunk_id)
                if existing is None:
                    chunks[chunk_id] = _chunk_from(
                        chunk_id,
                        text,
                        metadata,
                        score=dense if dense is not None else (keyword or 0.0),
                        dense_score=dense,
                        keyword_score=keyword,
                        matched_by=_matched_by(dense, keyword),
                        found_by_query=query,
                    )
                    continue

                if dense is not None:
                    existing.dense_score = _best(existing.dense_score, dense)
                if keyword is not None:
                    existing.keyword_score = _best(existing.keyword_score, keyword)
                existing.matched_by = _matched_by(existing.dense_score, existing.keyword_score)

        return chunks

    def _rankings(self, queries: list[str], hits: list["_QueryHits"]) -> list[Ranking]:
        """
        Build one ranked id list per (query phrasing, retriever half).

        With expansion off that is the familiar two lists; with three variants
        it is up to eight. Fusion takes any number, and more lists is exactly
        how multi-query retrieval expresses itself: a chunk that four different
        phrasings all surfaced accumulates four contributions, while a chunk
        one odd rewrite found alone accumulates one.

        Every phrasing carries the same weight, the original included. It is
        tempting to privilege the original, but the variants are the only
        reason a differently-worded passage is in the pool at all, and
        down-weighting them would partly undo the technique. The original's
        real protection is that it is always present, never that it shouts
        louder.

        Zero-weight halves are dropped rather than passed on contributing
        nothing: HYBRID_KEYWORD_WEIGHT=0 is a legitimate way to say "dense
        only for now", and with no expansion it should take the
        single-retriever path above (real scores, no fusion) rather than a
        fused one whose scores all come from the other list.
        """
        rankings: list[Ranking] = []

        for index, (query, query_hits) in enumerate(zip(queries, hits)):
            label = "original" if index == 0 else f"variant{index}"

            if self._mode in ("dense", "hybrid"):
                weight = settings.hybrid_dense_weight if self._mode == "hybrid" else 1.0
                if weight > 0 and query_hits.dense:
                    rankings.append(
                        Ranking(
                            name=f"dense:{label}",
                            chunk_ids=_ranked_ids(query_hits.dense),
                            weight=weight,
                        )
                    )

            if self._mode in ("keyword", "hybrid"):
                weight = settings.hybrid_keyword_weight if self._mode == "hybrid" else 1.0
                if weight > 0 and query_hits.keyword:
                    rankings.append(
                        Ranking(
                            name=f"keyword:{label}",
                            chunk_ids=_ranked_ids(query_hits.keyword),
                            weight=weight,
                        )
                    )

        return rankings


def _chunk_from(
    chunk_id: str,
    text: str,
    metadata: dict,
    score: float,
    matched_by: str,
    dense_score: float | None = None,
    keyword_score: float | None = None,
    found_by_query: str = "",
) -> RetrievedChunk:
    """Build a RetrievedChunk from a store hit, pulling source fields out of metadata."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        filename=metadata.get("filename", "unknown"),
        page=metadata.get("page_number", 0),
        score=score,
        section=metadata.get("section", ""),
        dense_score=dense_score,
        keyword_score=keyword_score,
        matched_by=matched_by,
        parent_id=metadata.get("parent_id", ""),
        found_by_query=found_by_query,
    )


def _ranked_ids(scores: dict[str, float]) -> list[str]:
    """Chunk ids from one retriever's score map, best first."""
    return [chunk_id for chunk_id, _ in sorted(scores.items(), key=lambda item: -item[1])]


def _best_score(chunks: list[RetrievedChunk]) -> float:
    """
    The best cross-encoder score in a result set, for comparing a retry
    against what it was trying to improve on. -1 for an empty set, which is
    below any real score, so anything at all beats having found nothing.
    """
    scores = [chunk.rerank_score for chunk in chunks if chunk.rerank_score is not None]
    return max(scores) if scores else -1.0


def _best(current: float | None, candidate: float) -> float:
    """The better of a score we already had (possibly none) and a new one."""
    return candidate if current is None else max(current, candidate)


def _matched_by(dense: float | None, keyword: float | None) -> str:
    """Which half (or halves) produced a chunk, as a label worth logging."""
    if dense is not None and keyword is not None:
        return "both"
    if keyword is not None:
        return "keyword"
    return "dense"


def _passage_key(result) -> str:
    """
    The passage a result points at, whichever shape the result is.

    A RetrievedChunk carries `parent_id` directly; a raw SearchResult still
    has it in `.metadata`. Checked in that order so the field wins where both
    exist.
    """
    parent_id = getattr(result, "parent_id", None)
    if parent_id is not None:
        return parent_id
    return getattr(result, "metadata", {}).get("parent_id", "")


def collapse_duplicate_passages(results: list) -> list:
    """
    Keep only the best-scoring chunk per passage, preserving rank order.

    Several chunking strategies deliberately point many chunks at one passage:
    parent-document embeds a dozen children of the same parent, sentence-window
    embeds every sentence of a paragraph. Without this, one strong paragraph
    takes every slot in top-k and the model sees the same text five times
    instead of five different pieces of evidence - which is worse than not
    having used the strategy at all.

    Chunks with no parent (the ordinary case) are never collapsed together:
    an empty key means "this chunk stands alone", not "these all match".

    Results are assumed to arrive best-first, which is what the vector store
    and the fusion step both return, so the first chunk seen for a passage is
    its best.

    Takes either shape a passage arrives in: a SearchResult straight from the
    store, which keeps its parent id in `.metadata`
    (app/services/document_service.py's search), or a RetrievedChunk that has
    already been through fusion and carries it as a field.
    """
    kept = []
    seen: set[str] = set()
    for result in results:
        parent_id = _passage_key(result)
        if parent_id:
            if parent_id in seen:
                continue
            seen.add(parent_id)
        kept.append(result)
    return kept
