"""
Pydantic request/response models for the RAG question-answering endpoints.

RAGAskRequest/RAGAskResponse back the original POST /rag/ask endpoint
(snake_case, includes `question` echoed back and full similarity scores -
useful for debugging retrieval quality).

RAGChatRequest/RAGChatResponse back POST /rag/chat, the Phase 4B contract
consumed by the frontend chat UI: a terser {answer, sources} shape with
camelCase source fields (filename, page, chunkId) matching the citation
list the UI renders under each assistant message.
"""
from pydantic import BaseModel, ConfigDict, Field


class RAGAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer using ingested documents.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")


class RAGSourceResponse(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    # Cosine similarity - and *only* that. None when the chunk was found by
    # keyword search alone, because such a chunk has no cosine score: it was
    # never in the dense result set. Reporting a 0.0 there would read as "the
    # embedding says this is irrelevant", which is a claim nothing measured.
    similarity: float | None = None
    # The score the ranking actually used: the similarity above in `dense`
    # mode, a BM25 score in `keyword` mode, a fused rank score in `hybrid`.
    # Comparable within one response, not across responses or modes.
    score: float
    keyword_score: float | None = None
    # The cross-encoder's relevance verdict in (0, 1), or null when
    # RERANK_ENABLED is off. When present it is also what `score` holds, since
    # it is then what the final ordering was done on.
    rerank_score: float | None = None
    # The phrasing that found this chunk. Equal to the asked question
    # unless MULTI_QUERY_ENABLED is on and an LLM-written variant got
    # there first - which is how you tell whether expansion contributed.
    found_by_query: str = ""
    # "dense" | "keyword" | "both" - which half of hybrid retrieval found it.
    matched_by: str = "dense"
    section: str = ""


class RAGAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSourceResponse]


class RAGChatRequest(BaseModel):
    """Request body for POST /rag/chat."""

    question: str = Field(..., min_length=1, description="Question to answer using ingested documents.")


class RAGChatSource(BaseModel):
    """A single citation shown under the assistant's answer in the chat UI."""

    model_config = ConfigDict(populate_by_name=True)

    filename: str
    page: int
    chunk_id: str = Field(..., alias="chunkId")
    # Where in the document the cited chunk sits, so a citation can read
    # "report.pdf · p12 · 2. Methods" instead of just a page number. Empty
    # for documents with no detected heading structure.
    section: str = ""


class RAGChatResponse(BaseModel):
    """Response body for POST /rag/chat."""

    model_config = ConfigDict(populate_by_name=True)

    answer: str
    sources: list[RAGChatSource]
