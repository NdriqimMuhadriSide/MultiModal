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
in rag/loaders/pdf.py, rag/text_splitter.py, or the future Chroma-writing
service needs to change, since they only depend on this module's public
`embed_texts` / `embed_text` functions, not on how the vectors are produced.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


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

    @property
    def max_tokens(self) -> int:
        """
        The longest input this model actually reads, in tokens (256 for
        MiniLM-L6).

        This is a hard wall, not a guideline, and crossing it is *silent*:
        the model truncates and embeds the prefix, so a chunk gets a vector
        describing only its opening. rag/text_splitter.py sizes chunks against
        this number, which is why it's exposed rather than kept private.
        """
        return self._model.max_seq_length

    def count_tokens(self, text: str) -> int:
        """
        How many tokens this model will see in `text`.

        Uses the model's *own* tokenizer, so the count is what the model will
        actually do rather than an estimate. Characters are a poor proxy: the
        same 800 characters are ~180 tokens of English prose but ~340 tokens
        of Markdown table rows, because every pipe, digit and date fragment
        becomes its own token.

        Special tokens are included ([CLS] and [SEP], two of the 256), so the
        result is directly comparable to `max_tokens` with no off-by-two.
        """
        # verbose=False suppresses transformers' "sequence longer than the
        # maximum" warning: measuring an over-long string is the entire point
        # of this method, so the warning is noise here.
        return len(
            self._model.tokenizer.encode(text, add_special_tokens=True, verbose=False)
        )

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

        self._warn_about_truncation(texts)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [
            EmbeddedChunk(text=text, embedding=vector.tolist())
            for text, vector in zip(texts, vectors)
        ]

    def _warn_about_truncation(self, texts: list[str]) -> None:
        """
        Say something when an input is going to be cut off.

        Truncation is the worst kind of bug: the call succeeds, a vector comes
        back, and it silently describes only the first part of the text. With
        chunks sized in tokens this should never fire - so if it does, it means
        something bypassed the splitter, and that is worth hearing about rather
        than discovering later through bad search results.

        Logged once per batch, with a count, rather than once per chunk: a
        misconfiguration would otherwise produce thousands of identical lines.
        """
        oversized = [text for text in texts if self.count_tokens(text) > self.max_tokens]
        if oversized:
            logger.warning(
                "%d of %d texts exceed the embedding model's %d-token limit and will be "
                "truncated - their trailing content will not affect their embeddings. "
                "Longest was %d tokens.",
                len(oversized),
                len(texts),
                self.max_tokens,
                max(self.count_tokens(text) for text in oversized),
            )


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached EmbeddingService (the model is loaded once per process)."""
    return EmbeddingService(model_name=settings.embedding_model_name)
