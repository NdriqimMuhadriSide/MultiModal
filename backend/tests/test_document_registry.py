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
