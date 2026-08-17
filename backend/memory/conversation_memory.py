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

A turn also records *what kind of thing it was about*:

    modality        "text" | "image" | "audio" - which surface produced the
                    turn, and therefore what the user was actually looking
                    at when they said it.
    attachment_ref  the memory/attachment_store.py id of the image or audio
                    the turn refers to, or None for plain text.

Both exist for the same reason: a stored turn used to be a bare string, so
"it's red" and "it's blue" from two different photographs were
indistinguishable once written, and a later turn could inherit the wrong
one as context. The ref makes a turn self-describing - a reader can tell
whether the previous answer was about the picture in front of it or a
different one - and it is what lets a follow-up reuse an image instead of
demanding a re-upload.
"""
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Role = Literal["user", "assistant"]
Modality = Literal["text", "image", "audio"]


@dataclass
class ConversationMessage:
    conversation_id: str
    role: Role
    content: str
    created_at: str
    # Defaulted rather than required: every turn written before these
    # columns existed is a text turn with no attachment, and that is also
    # the right answer for the three services that never touch media.
    modality: Modality = "text"
    attachment_ref: str | None = None
    # The autoincrement id, which is also the conversation's ordering. Only
    # compaction needs it - to say "everything up to here is summarised" -
    # so it is defaulted rather than required and every other caller can
    # keep ignoring it.
    message_id: int = 0


@dataclass
class ConversationSummary:
    """
    A conversation's older turns, condensed.

    `covered_through_id` is the high-water mark: every message up to and
    including that id is represented in `summary`, and everything after it
    is not. Compaction reads it to know where to resume, which is what
    keeps a second pass from re-summarising what the first one already did.
    """

    conversation_id: str
    summary: str
    covered_through_id: int
    updated_at: str


# Written once and shared by every read path, so a column added later cannot
# be picked up by one of them and missed by another - the queries differ
# only in their WHERE and ORDER BY, and that is the only place they should
# differ.
_MESSAGE_COLUMNS = (
    "conversation_id, role, content, created_at, modality, attachment_ref, id"
)


def _as_message(row: tuple) -> ConversationMessage:
    """Build a ConversationMessage from a row selected with _MESSAGE_COLUMNS."""
    return ConversationMessage(
        conversation_id=row[0],
        role=row[1],
        content=row[2],
        created_at=row[3],
        # The ALTER backfills existing rows with 'text', so this coalesce is
        # belt-and-braces for a database that reached the current shape some
        # other way (a hand-run migration, a restored dump).
        modality=row[4] or "text",
        attachment_ref=row[5],
        message_id=row[6],
    )


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
                    created_at TEXT NOT NULL,
                    modality TEXT NOT NULL DEFAULT 'text',
                    attachment_ref TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id "
                "ON messages (conversation_id, id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    covered_through_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._add_missing_columns(conn)

    @staticmethod
    def _add_missing_columns(conn: sqlite3.Connection) -> None:
        """
        Bring a database created before `modality`/`attachment_ref` existed
        up to the current shape.

        CREATE TABLE IF NOT EXISTS is a no-op against an existing table, so
        it cannot add a column - a database written by an earlier version
        would keep its four-column table and every read would fail on the
        two names that aren't there. Reading the actual columns back and
        adding only what's absent is what makes this safe to run on every
        startup, which is when it runs.

        Both additions are nullable-or-defaulted, which is the reason this
        can be a plain ALTER TABLE with no table rewrite and no backfill:
        existing rows are text turns with no attachment, and that is exactly
        what they end up saying.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "modality" not in existing:
            conn.execute(
                "ALTER TABLE messages ADD COLUMN modality TEXT NOT NULL DEFAULT 'text'"
            )
        if "attachment_ref" not in existing:
            conn.execute("ALTER TABLE messages ADD COLUMN attachment_ref TEXT")

    @staticmethod
    def new_conversation_id() -> str:
        """Generate a fresh conversation_id for a new conversation."""
        return str(uuid.uuid4())

    def add_message(
        self,
        conversation_id: str,
        role: Role,
        content: str,
        modality: Modality = "text",
        attachment_ref: str | None = None,
    ) -> None:
        """
        Persist a single turn (a user message or an assistant response)
        under `conversation_id`.

        `modality` and `attachment_ref` describe what the turn was about.
        They default to a plain text turn with nothing attached, so the
        three services that only ever handle text call this exactly as they
        did before.

        Both halves of a media turn should carry the same values: the
        question and the answer are about the same picture, and tagging only
        the question would leave the answer - the half a later turn actually
        reads back as context - unattributed.

        Raises:
            ValueError: if `content` is empty.
        """
        if not content or not content.strip():
            raise ValueError("content must not be empty.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, created_at, modality, attachment_ref) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, created_at, modality, attachment_ref),
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
                f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()

        # Rows come back newest-first (for the LIMIT to keep the *most
        # recent* N) - reverse to chronological order before returning.
        return [_as_message(row) for row in reversed(rows)]

    def get_full_history(self, conversation_id: str) -> list[ConversationMessage]:
        """Return every stored message for `conversation_id`, oldest first (no limit)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()

        return [_as_message(row) for row in rows]

    def get_messages_between(
        self, conversation_id: str, after_id: int, before_id: int | None
    ) -> list[ConversationMessage]:
        """
        Return this conversation's messages with `after_id` < id <
        `before_id`, oldest first. `before_id` of None means "to the end".

        This is the slice compaction works on: everything the existing
        summary doesn't cover yet (`after_id` is its high-water mark) and
        that has already dropped out of the history window (`before_id` is
        the oldest message still in it). Both bounds are exclusive, so the
        boundary messages belong to exactly one of the summary and the
        window - never to both, which would put them in the prompt twice,
        and never to neither, which is the amnesia this exists to fix.
        """
        clause = "" if before_id is None else " AND id < ?"
        parameters: tuple = (
            (conversation_id, after_id)
            if before_id is None
            else (conversation_id, after_id, before_id)
        )

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM messages "
                f"WHERE conversation_id = ? AND id > ?{clause} ORDER BY id ASC",
                parameters,
            ).fetchall()

        return [_as_message(row) for row in rows]

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        """Return this conversation's compacted older turns, if it has any."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conversation_id, summary, covered_through_id, updated_at "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

        if row is None:
            return None

        return ConversationSummary(
            conversation_id=row[0],
            summary=row[1],
            covered_through_id=row[2],
            updated_at=row[3],
        )

    def save_summary(
        self, conversation_id: str, summary: str, covered_through_id: int
    ) -> None:
        """
        Store (or replace) this conversation's summary.

        One row per conversation, replaced wholesale rather than appended
        to: each pass re-summarises the previous summary together with the
        turns since, so what is written is always the single current
        account of everything before `covered_through_id`. Keeping the
        older ones would mean deciding which to send.

        Raises:
            ValueError: if `summary` is empty.
        """
        if not summary or not summary.strip():
            raise ValueError("summary must not be empty.")

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversation_summaries "
                "(conversation_id, summary, covered_through_id, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET "
                "summary = excluded.summary, "
                "covered_through_id = excluded.covered_through_id, "
                "updated_at = excluded.updated_at",
                (
                    conversation_id,
                    summary,
                    covered_through_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_last_attachment(
        self, conversation_id: str, modality: Modality
    ) -> str | None:
        """
        Return the most recent `attachment_ref` of the given modality in this
        conversation, or None if the conversation has never carried one.

        This is what makes a follow-up question possible without re-sending
        the file: "and what about the date?" resolves to the image the
        previous turn was about. Scoped by modality so an audio turn cannot
        answer for an image one.

        Deliberately *not* bounded by the history window. The window governs
        what the model is shown; this governs which file the conversation is
        still holding, and a picture does not stop being the subject just
        because ten messages have gone by.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT attachment_ref FROM messages "
                "WHERE conversation_id = ? AND modality = ? AND attachment_ref IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (conversation_id, modality),
            ).fetchone()

        return row[0] if row else None


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
