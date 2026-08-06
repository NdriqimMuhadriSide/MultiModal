"""
Embedding service.

Responsibility: turn text chunks into vector embeddings. Pure text-in,
vectors-out - no PDF/chunking knowledge, no FastAPI/HTTP knowledge, no
vector database knowledge (that's ChromaDB's job in a later task).

Runs entirely locally via `sentence-transformers`, so there's no API key,
no billing, and no network call for this step - a deliberate choice for
this learning project since both OpenAI (requires billing) and Groq (does
not offer an embeddings endpoint at all) were ruled out. See the module
docstring discussion in the accompanying explanation for the tradeoffs.

Swapping to a hosted embeddings API later (OpenAI's text-embedding-3-small,
Gemini's text-embedding-004, etc.) only means changing this file - nothing
in rag/pdf_loader.py, rag/text_splitter.py, or the future Chroma-writing
service needs to change, since they only depend on this module's public
`embed_texts` / `embed_text` functions, not on how the vectors are produced.
"""
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@dataclass
class EmbeddedChunk:
    """A single text input paired with its resulting embedding vector."""

    text: str
    embedding: list[float]


class EmbeddingService:
    """Wraps a local sentence-transformers model for text -> vector embedding."""

    def __init__(self, model_name: str) -> None:
        # Loading the model reads weights from disk (downloading them once,
        # on first use, to a local cache) - this happens once per process
        # thanks to the lru_cache on get_embedding_service(), not per request.
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        """Length of the vectors this model produces (e.g. 384 for MiniLM-L6)."""
        return self._model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single string.

        Raises:
            ValueError: if `text` is empty.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")

        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[EmbeddedChunk]:
        """
        Embed a batch of strings at once (more efficient than calling
        embed_text in a loop - the model batches the forward pass).

        Raises:
            ValueError: if `texts` is empty or contains a blank string.
        """
        if not texts:
            raise ValueError("texts must not be empty.")
        if any(not text or not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings.")

        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [
            EmbeddedChunk(text=text, embedding=vector.tolist())
            for text, vector in zip(texts, vectors)
        ]


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached EmbeddingService (the model is loaded once per process)."""
    return EmbeddingService(model_name=settings.embedding_model_name)
