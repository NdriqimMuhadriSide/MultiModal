"""
Document registry.

Responsibility: track *documents* as first-class records - one row per
uploaded PDF, independent of the chunks that PDF was split into. This is
the piece the earlier /documents/ingest work didn't have: ChromaDB knows
about chunks (via rag/vector_store.py), but nothing durably records "this
document_id maps to this filename, uploaded at this time, with this many
pages/chunks, currently in this status." That's what this module is for.

Backed by SQLite (stdlib `sqlite3`, one local file), same approach as
memory/conversation_memory.py - free, local, no extra infrastructure, and
enough for a single-process learning app. Swapping to Postgres later would
mean replacing this file's internals while keeping the same
create_document/update_status/list_documents public interface.

Status lifecycle, set by the caller (app/services/document_service.py):

    PROCESSING -> READY     (happy path: text extracted, chunks embedded
                              and stored in ChromaDB)
    PROCESSING -> FAILED    (e.g. PDF had no extractable text at all)

A document is inserted as PROCESSING *before* the (potentially slow)
extract -> chunk -> embed -> store pipeline runs, so a crash mid-pipeline
still leaves a visible, honest record instead of the document silently not
existing anywhere.
"""
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Status = Literal["PROCESSING", "READY", "FAILED"]


@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: Status
    created_at: str


class DocumentRegistry:
    """Wraps a local SQLite database for tracking uploaded documents."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # Short-lived connection per call - same reasoning as
        # ConversationMemory._connect: safe across FastAPI's request
        # threadpool without shared-connection thread-safety concerns.
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK (status IN ('PROCESSING', 'READY', 'FAILED')),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_created_at "
                "ON documents (created_at DESC)"
            )

    @staticmethod
    def new_document_id() -> str:
        """Generate a fresh document_id for a newly uploaded PDF."""
        return str(uuid.uuid4())

    def create_document(self, document_id: str, filename: str) -> DocumentRecord:
        """
        Register a new document in PROCESSING status, before the ingestion
        pipeline (extract/chunk/embed/store) runs against it.

        Raises:
            ValueError: if `filename` is empty.
        """
        if not filename or not filename.strip():
            raise ValueError("filename must not be empty.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (document_id, filename, page_count, "
                "chunk_count, status, created_at) VALUES (?, ?, 0, 0, 'PROCESSING', ?)",
                (document_id, filename, created_at),
            )

        return DocumentRecord(
            document_id=document_id,
            filename=filename,
            page_count=0,
            chunk_count=0,
            status="PROCESSING",
            created_at=created_at,
        )

    def mark_ready(self, document_id: str, page_count: int, chunk_count: int) -> None:
        """Mark a document READY once its chunks are embedded and stored."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'READY', page_count = ?, "
                "chunk_count = ? WHERE document_id = ?",
                (page_count, chunk_count, document_id),
            )

    def mark_failed(self, document_id: str, page_count: int = 0) -> None:
        """Mark a document FAILED (e.g. no extractable text -> zero chunks)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'FAILED', page_count = ?, "
                "chunk_count = 0 WHERE document_id = ?",
                (page_count, document_id),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return a single document by id, or None if it doesn't exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document_id, filename, page_count, chunk_count, status, created_at "
                "FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        if row is None:
            return None
        return DocumentRecord(
            document_id=row[0],
            filename=row[1],
            page_count=row[2],
            chunk_count=row[3],
            status=row[4],
            created_at=row[5],
        )

    def list_documents(self) -> list[DocumentRecord]:
        """Return every registered document, most recently uploaded first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT document_id, filename, page_count, chunk_count, status, created_at "
                "FROM documents ORDER BY created_at DESC"
            ).fetchall()

        return [
            DocumentRecord(
                document_id=row[0],
                filename=row[1],
                page_count=row[2],
                chunk_count=row[3],
                status=row[4],
                created_at=row[5],
            )
            for row in rows
        ]


_registry_instance: DocumentRegistry | None = None


def get_document_registry() -> DocumentRegistry:
    """
    Return a process-wide DocumentRegistry instance.

    Not decorated with @lru_cache (same reasoning as
    memory/conversation_memory.get_conversation_memory) so tests can point
    this at a temporary DB file without a stale cached instance lingering
    across test runs.
    """
    global _registry_instance
    if _registry_instance is None:
        from app.core.config import settings

        _registry_instance = DocumentRegistry(db_path=settings.document_registry_db_path)
    return _registry_instance
