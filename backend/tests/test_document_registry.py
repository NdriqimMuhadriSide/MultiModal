import pytest

from rag.document_registry import DocumentRegistry


@pytest.fixture
def registry(tmp_path) -> DocumentRegistry:
    return DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))


def test_create_document_starts_in_processing_status(registry):
    document_id = DocumentRegistry.new_document_id()
    record = registry.create_document(document_id=document_id, filename="a.pdf")

    assert record.document_id == document_id
    assert record.filename == "a.pdf"
    assert record.status == "PROCESSING"
    assert record.page_count == 0
    assert record.chunk_count == 0


def test_create_document_rejects_empty_filename(registry):
    with pytest.raises(ValueError):
        registry.create_document(document_id="id-1", filename="")


def test_mark_ready_updates_status_and_counts(registry):
    document_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=document_id, filename="a.pdf")

    registry.mark_ready(document_id, page_count=5, chunk_count=12)

    record = registry.get_document(document_id)
    assert record.status == "READY"
    assert record.page_count == 5
    assert record.chunk_count == 12


def test_mark_failed_updates_status_and_zeroes_chunk_count(registry):
    document_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=document_id, filename="scanned.pdf")

    registry.mark_failed(document_id, page_count=3)

    record = registry.get_document(document_id)
    assert record.status == "FAILED"
    assert record.page_count == 3
    assert record.chunk_count == 0


def test_get_document_returns_none_for_unknown_id(registry):
    assert registry.get_document("does-not-exist") is None


def test_list_documents_returns_most_recently_created_first(registry):
    first_id = DocumentRegistry.new_document_id()
    second_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=first_id, filename="first.pdf")
    registry.create_document(document_id=second_id, filename="second.pdf")

    documents = registry.list_documents()

    assert [doc.document_id for doc in documents] == [second_id, first_id]


def test_new_document_id_generates_unique_values():
    ids = {DocumentRegistry.new_document_id() for _ in range(50)}
    assert len(ids) == 50


def test_find_ready_by_content_hash_ignores_processing_and_failed(registry):
    processing = DocumentRegistry.new_document_id()
    registry.create_document(document_id=processing, filename="a.pdf", content_hash="h1")
    # Still PROCESSING (a crashed ingest looks exactly like this) - not a
    # match, or a crash would permanently shadow the real document.
    assert registry.find_ready_by_content_hash("h1") is None

    failed = DocumentRegistry.new_document_id()
    registry.create_document(document_id=failed, filename="b.pdf", content_hash="h2")
    registry.mark_failed(failed)
    # FAILED must stay retryable rather than have its hash reject the file.
    assert registry.find_ready_by_content_hash("h2") is None

    ready = DocumentRegistry.new_document_id()
    registry.create_document(document_id=ready, filename="c.pdf", content_hash="h3")
    registry.mark_ready(ready, page_count=2, chunk_count=5)

    found = registry.find_ready_by_content_hash("h3")
    assert found is not None
    assert found.document_id == ready
    assert found.content_hash == "h3"
    assert found.chunk_count == 5


def test_find_ready_by_content_hash_returns_none_for_unknown_or_empty(registry):
    assert registry.find_ready_by_content_hash("nothing-has-this-hash") is None
    assert registry.find_ready_by_content_hash("") is None


def test_delete_document_removes_the_row_and_reports_whether_it_existed(registry):
    document_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=document_id, filename="a.pdf", content_hash="h")
    registry.mark_ready(document_id, page_count=1, chunk_count=3)

    assert registry.delete_document(document_id) is True
    assert registry.get_document(document_id) is None
    assert registry.list_documents() == []
    # Second delete is False, not an error - the endpoint turns that into a 404.
    assert registry.delete_document(document_id) is False


def test_deleting_a_document_frees_its_content_hash_for_reupload(registry):
    first = DocumentRegistry.new_document_id()
    registry.create_document(document_id=first, filename="a.pdf", content_hash="same")
    registry.mark_ready(first, page_count=1, chunk_count=1)
    registry.delete_document(first)

    assert registry.find_ready_by_content_hash("same") is None


def test_mark_failed_records_the_reason(registry):
    document_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=document_id, filename="scanned.pdf")

    registry.mark_failed(document_id, page_count=2, reason="No text; OCR is not installed.")

    assert registry.get_document(document_id).failure_reason == "No text; OCR is not installed."


def test_marking_ready_clears_a_previous_failure_reason(registry):
    """A retry that works must not leave the failed attempt's excuse attached."""
    document_id = DocumentRegistry.new_document_id()
    registry.create_document(document_id=document_id, filename="scanned.pdf")
    registry.mark_failed(document_id, reason="OCR is not installed.")

    registry.mark_ready(document_id, page_count=2, chunk_count=7)

    record = registry.get_document(document_id)
    assert record.status == "READY"
    assert record.failure_reason is None


def test_content_hash_column_is_added_to_a_pre_existing_database(tmp_path):
    """A registry created before content_hash existed must still open."""
    import sqlite3

    db_path = str(tmp_path / "legacy.sqlite3")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE documents (
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
            "INSERT INTO documents VALUES ('old', 'legacy.pdf', 1, 2, 'READY', '2026-01-01T00:00:00+00:00')"
        )

    registry = DocumentRegistry(db_path=db_path)

    legacy = registry.get_document("old")
    assert legacy is not None
    assert legacy.filename == "legacy.pdf"
    # Rows written before the column existed have no hash and cannot get one:
    # the original bytes were never kept.
    assert legacy.content_hash is None
    # ...and a NULL hash must never match a new upload.
    assert registry.find_ready_by_content_hash("anything") is None
