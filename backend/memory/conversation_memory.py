"""
Conversation memory layer.

Architecture:

    User
      |
      v
    Memory Layer  <-- (this module: store the incoming message, load
      |                 recent history for this conversation_id)
      v
      LLM          <-- (ai/llm_service.py: history + new message -> response)
      |
      v
    Memory Layer  <-- (store the LLM's response under the same conversation_id)
      |
      v
    User

This module is the only place that persists conversation turns. It stores,
per conversation_id:
  - the user's messages
  - the AI's responses
  - which conversation they belong to
  - the order they happened in (timestamp + auto-increment id)

Backed by SQLite (stdlib `sqlite3`, one local file) rather than an external
database - consistent with this project's "free, local, no extra infra"
approach, and enough for a single-process learning app. Swapping to
Postgres/Redis later would mean replacing this file's internals while
keeping the same add_message/get_history public interface.

This is short-term / working memory: `get_history` returns only the most
recent N turns of a *specific* conversation (bounded by
settings.conversation_history_limit) to keep prompts small and relevant.
The full conversation is still durably stored - nothing is deleted - so
this is a *view* limit, not a data-retention limit. See the accompanying
explanation for how this differs from long-term memory and vector memory.
"""
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class ConversationMessage:
    conversation_id: str
    role: Role
    content: str
    created_at: str


class ConversationMemory:
    """Wraps a local SQLite database for storing and retrieving conversation turns."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call keeps this safe to use from
        # multiple request threads (FastAPI's threadpool for sync routes)
        # without needing to manage a shared connection's thread-safety.
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id "
                "ON messages (conversation_id, id)"
            )

    @staticmethod
    def new_conversation_id() -> str:
        """Generate a fresh conversation_id for a new conversation."""
        return str(uuid.uuid4())

    def add_message(self, conversation_id: str, role: Role, content: str) -> None:
        """
        Persist a single turn (a user message or an assistant response)
        under `conversation_id`.

        Raises:
            ValueError: if `content` is empty.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, created_at),
            )

    def get_history(self, conversation_id: str, limit: int) -> list[ConversationMessage]:
        """
        Return the most recent `limit` messages for `conversation_id`, in
        chronological order (oldest first) - the shape an LLM expects when
        the history is inserted into its message list.

        Raises:
            ValueError: if `limit` is not positive.
        """
        if limit <= 0:
            raise ValueError("limit must be a positive integer.")

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()

        # Rows come back newest-first (for the LIMIT to keep the *most
        # recent* N) - reverse to chronological order before returning.
        return [
            ConversationMessage(conversation_id=row[0], role=row[1], content=row[2], created_at=row[3])
            for row in reversed(rows)
        ]

    def get_full_history(self, conversation_id: str) -> list[ConversationMessage]:
        """Return every stored message for `conversation_id`, oldest first (no limit)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()

        return [
            ConversationMessage(conversation_id=row[0], role=row[1], content=row[2], created_at=row[3])
            for row in rows
        ]


_memory_instance: ConversationMemory | None = None


def get_conversation_memory() -> ConversationMemory:
    """
    Return a process-wide ConversationMemory instance.

    Not decorated with @lru_cache (same reasoning as rag/vector_store.py's
    get_vector_store) so tests can point this at a temporary DB file via
    monkeypatch/dependency overrides without a stale cached instance
    lingering across test runs.
    """
    global _memory_instance
    if _memory_instance is None:
        from app.core.config import settings

        _memory_instance = ConversationMemory(db_path=settings.conversation_db_path)
    return _memory_instance
