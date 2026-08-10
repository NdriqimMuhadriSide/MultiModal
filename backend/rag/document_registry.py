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

FAILED rows also carry a `failure_reason`, because the status alone can't
distinguish a corrupt file from a scan that would ingest fine if OCR were
installed - and only one of those is worth the user retrying.

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
    # SHA-256 of the uploaded bytes, used to recognise a re-upload of the
    # same file. Nullable because rows written before this column existed
    # have no hash, and backfilling one is impossible - the original bytes
    # are not kept anywhere.
    content_hash: str | None = None
    # Why a FAILED document failed, phrased for a user. Always None for
    # PROCESSING and READY rows. Without it "FAILED" is a dead end: a corrupt
    # file and a scan that needs OCR look identical in the list, and only one
    # of them is worth acting on.
    failure_reason: str | None = None
    # What the document says about itself (rag/metadata.py). `title` is always
    # populated - it falls back to the first heading, then the filename - while
    # the rest are None whenever the file didn't carry them.
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    # ISO 8601 date carried by the *document*. Distinct from created_at above,
    # which is when it was uploaded here: a 2019 report ingested today has two
    # very different dates, and only one of them is useful for filtering.
    document_date: str | None = None


# Columns selected by every read below, in DocumentRecord field order.
_COLUMNS = (
    "document_id, filename, page_count, chunk_count, status, created_at, "
    "content_hash, failure_reason, title, author, subject, document_date"
)


def _to_record(row: tuple) -> DocumentRecord:
    return DocumentRecord(
        document_id=row[0],
        filename=row[1],
        page_count=row[2],
        chunk_count=row[3],
        status=row[4],
        created_at=row[5],
        content_hash=row[6],
        failure_reason=row[7],
        title=row[8],
        author=row[9],
        subject=row[10],
        document_date=row[11],
    )


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

            # Added after the table shipped, so existing databases need the
            # column bolted on rather than declared above. SQLite has no
            # "ADD COLUMN IF NOT EXISTS", hence the PRAGMA check.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
            if "content_hash" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            if "failure_reason" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN failure_reason TEXT")
            for column in ("title", "author", "subject", "document_date"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")

            # Not UNIQUE: rows predating the column all have NULL, and a
            # re-upload that fails mid-pipeline should not be blocked from
            # being retried. Uniqueness is enforced by the lookup in
            # app/services/document_service.py, which only matches READY rows.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash "
                "ON documents (content_hash)"
            )

    @staticmethod
    def new_document_id() -> str:
        """Generate a fresh document_id for a newly uploaded PDF."""
        return str(uuid.uuid4())

    def create_document(
        self, document_id: str, filename: str, content_hash: str | None = None
    ) -> DocumentRecord:
        """
        Register a new document in PROCESSING status, before the ingestion
        pipeline (extract/chunk/embed/store) runs against it.

        `content_hash` is optional so existing callers and tests keep
        working; ingestion always supplies it.

        Raises:
            ValueError: if `filename` is empty.
        """
        if not filename or not filename.strip():
            raise ValueError("filename must not be empty.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (document_id, filename, page_count, "
                "chunk_count, status, created_at, content_hash) "
                "VALUES (?, ?, 0, 0, 'PROCESSING', ?, ?)",
                (document_id, filename, created_at, content_hash),
            )

        return DocumentRecord(
            document_id=document_id,
            filename=filename,
            page_count=0,
            chunk_count=0,
            status="PROCESSING",
            created_at=created_at,
            content_hash=content_hash,
        )

    def find_ready_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        """
        Return the oldest READY document with this content hash, or None.

        Restricted to READY on purpose: a PROCESSING row may be a crashed
        ingest, and a FAILED one should stay retryable rather than have its
        hash permanently reject the same file.

        Oldest-first so repeated uploads keep collapsing onto one canonical
        document instead of hopping between duplicates.
        """
        if not content_hash:
            return None

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents "
                "WHERE content_hash = ? AND status = 'READY' "
                "ORDER BY created_at ASC LIMIT 1",
                (content_hash,),
            ).fetchone()

        return None if row is None else _to_record(row)

    def delete_document(self, document_id: str) -> bool:
        """
        Remove a document's registry row. Returns True if a row was
        deleted, False if no such document existed.

        Deleting the row is only half of forgetting a document - its chunks
        live in Chroma and are removed by
        app/services/document_service.py, which owns both sides.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            return cursor.rowcount > 0

    def set_metadata(
        self,
        document_id: str,
        title: str | None,
        author: str | None,
        subject: str | None,
        document_date: str | None,
    ) -> None:
        """
        Record what a document says about itself.

        Separate from create_document because the metadata is only known after
        the file has been read, and the PROCESSING row is deliberately written
        before that - so a crash mid-ingest still leaves a visible record.

        Applied whatever the outcome: a scanned PDF that ends up FAILED still
        has a title and an author worth showing in the list.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET title = ?, author = ?, subject = ?, "
                "document_date = ? WHERE document_id = ?",
                (title, author, subject, document_date, document_id),
            )

    def mark_ready(self, document_id: str, page_count: int, chunk_count: int) -> None:
        """
        Mark a document READY once its chunks are embedded and stored.

        Clears any failure_reason: a retry that succeeds must not leave the
        previous attempt's explanation attached to a working document.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'READY', page_count = ?, "
                "chunk_count = ?, failure_reason = NULL WHERE document_id = ?",
                (page_count, chunk_count, document_id),
            )

    def mark_failed(
        self, document_id: str, page_count: int = 0, reason: str | None = None
    ) -> None:
        """
        Mark a document FAILED (e.g. no extractable text -> zero chunks).

        `reason` is optional so existing callers and tests keep working, but
        ingestion always supplies one.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = 'FAILED', page_count = ?, "
                "chunk_count = 0, failure_reason = ? WHERE document_id = ?",
                (page_count, reason, document_id),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return a single document by id, or None if it doesn't exist."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        return None if row is None else _to_record(row)

    def list_documents(self) -> list[DocumentRecord]:
        """Return every registered document, most recently uploaded first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents ORDER BY created_at DESC"
            ).fetchall()

        return [_to_record(row) for row in rows]


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
