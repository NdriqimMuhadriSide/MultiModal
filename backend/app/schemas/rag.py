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
    similarity: float


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


class RAGChatResponse(BaseModel):
    """Response body for POST /rag/chat."""

    model_config = ConfigDict(populate_by_name=True)

    answer: str
    sources: list[RAGChatSource]
