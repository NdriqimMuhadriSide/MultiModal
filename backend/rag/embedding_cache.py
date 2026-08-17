"""
Embedding cache.

Responsibility: remember what a piece of text embedded to, so the same text
is never put through the model twice.

The vectors are deterministic - the same string and the same model always
produce the same numbers - which makes this the rare cache with no
staleness question to answer. There is nothing to invalidate, because
nothing can change underneath it. A different model is a different key, so
swapping settings.embedding_model_name doesn't return the old model's
vectors; it just misses.

WHAT IT ACTUALLY SAVES

Re-ingesting a document whose chunks haven't changed, re-running an eval
set, and asking a question someone already asked - the three things this
project does constantly while being developed. Embedding runs locally on
CPU, so the cost is wall-clock rather than billing: a few hundred
milliseconds per query, several seconds per document.

STORAGE

SQLite, one file, same approach as memory/conversation_memory.py. Vectors
are stored as float32 via `array` rather than as JSON, which is not
premature: a 384-dimension vector is ~1.5KB packed and ~7KB as text, and
the model emits float32 anyway - so the packed form is exact, not lossy.

The key is a digest of the model name and the text rather than the text
itself, so a chunk's content never sits in a second database, and the index
stays small whatever the chunk size.

Nothing is ever evicted. For a learning project's corpus that is a file of
a few megabytes; if it ever matters, an eviction pass by `created_at` is
the obvious addition and needs no change to the interface.
"""
import hashlib
import logging
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _key(model_name: str, text: str) -> str:
    """
    The cache key for a (model, text) pair.

    The separator is a NUL byte, which cannot appear in either half, so
    there is no pair of inputs that can collide by concatenation - a model
    called "a" with text "bc" and one called "ab" with text "c" are
    different keys.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache:
    """A local SQLite store of text -> vector, keyed by model and content."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call, so this is safe to use from
        # FastAPI's threadpool without sharing a connection across threads.
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def get_many(self, model_name: str, texts: list[str]) -> dict[str, list[float]]:
        """
        Return the cached vectors for whichever of `texts` have one, keyed
        by the text itself.

        Keyed by text rather than by digest so the caller can look up what
        it asked for without recomputing the hash - and because a batch
        containing the same string twice should cost one lookup and produce
        one entry.

        A read failure returns an empty dict rather than raising: this is a
        cache, and the correct behaviour when it cannot be read is to embed
        the text.
        """
        if not texts:
            return {}

        by_key = {_key(model_name, text): text for text in texts}
        placeholders = ",".join("?" * len(by_key))

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                    tuple(by_key),
                ).fetchall()
        except sqlite3.Error:
            logger.warning("Embedding cache read failed; embedding directly.", exc_info=True)
            return {}

        found: dict[str, list[float]] = {}
        for key, blob in rows:
            vector = array("f")
            vector.frombytes(blob)
            found[by_key[key]] = vector.tolist()
        return found

    def put_many(self, model_name: str, items: dict[str, list[float]]) -> None:
        """
        Store vectors for the given texts.

        A write failure is logged and swallowed, for the same reason as a
        read failure: a full disk should slow this project down, not break
        ingestion.
        """
        if not items:
            return

        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (_key(model_name, text), array("f", vector).tobytes(), created_at)
            for text, vector in items.items()
        ]

        try:
            with self._connect() as conn:
                # INSERT OR REPLACE rather than plain INSERT: two threads
                # embedding the same text concurrently both write, and the
                # second one must not raise on the primary key. They agree
                # on the value, so whichever lands last is correct.
                conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (key, vector, created_at) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
        except sqlite3.Error:
            logger.warning("Embedding cache write failed; continuing.", exc_info=True)

    def count(self) -> int:
        """How many vectors are cached. For diagnostics and tests."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


_cache_instance: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache | None:
    """
    Return a process-wide EmbeddingCache, or None when caching is switched
    off.

    A module global rather than @lru_cache, matching
    memory/conversation_memory.py, so a test can point this at a tmp_path
    without a cached instance outliving it.
    """
    global _cache_instance
    from app.core.config import settings

    if not settings.embedding_cache_enabled:
        return None

    if _cache_instance is None:
        _cache_instance = EmbeddingCache(db_path=settings.embedding_cache_db_path)
    return _cache_instance
