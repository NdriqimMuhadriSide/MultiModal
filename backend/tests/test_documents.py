import io

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.main import app
from app.services.document_service import DocumentIngestionService, get_document_ingestion_service
from rag.document_registry import DocumentRegistry
from rag.embedding_service import EmbeddedChunk
from rag.vector_store import SearchResult

client = TestClient(app)


class FakeEmbeddingService:
    """Deterministic stand-in so tests don't load the real sentence-transformers model."""

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must not be empty.")
        return [float(len(text))]

    def embed_texts(self, texts: list[str]) -> list[EmbeddedChunk]:
        if not texts:
            raise ValueError("texts must not be empty.")
        return [EmbeddedChunk(text=t, embedding=[float(len(t))]) for t in texts]


class FakeVectorStore:
    """In-memory stand-in so tests don't touch a real Chroma persistence directory."""

    def __init__(self) -> None:
        self.stored_chunks = []

    def store_chunks(self, chunks) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")
        self.stored_chunks.extend(chunks)

    def search(self, query_embedding, top_k: int = 5):
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                similarity=0.9,
            )
            for chunk in self.stored_chunks[:top_k]
        ]

    def count(self) -> int:
        return len(self.stored_chunks)


def _fake_ingestion_service(tmp_path, registry: DocumentRegistry | None = None) -> DocumentIngestionService:
    return DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        document_registry=registry or DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3")),
    )


def _override_with(service: DocumentIngestionService):
    """Build a zero-arg FastAPI dependency override that returns `service`."""

    def _override() -> DocumentIngestionService:
        return service

    return _override


def _pdf_bytes_with_text_pages(page_texts: list[str]) -> bytes:
    """
    Build a real, minimal PDF where the caller controls text extraction
    indirectly isn't practical without a heavier PDF-writing dependency, so
    this test exercises the endpoint with blank pages (valid PDF structure,
    empty extracted text) and relies on test_pdf_loader.py / test_text_splitter.py
    for the actual text-content behavior. This still verifies the full
    request/response wiring: upload -> ingest -> chunk -> metadata -> JSON.
    """
    writer = PdfWriter()
    for _ in page_texts:
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_ingest_document_returns_page_and_chunk_metadata(tmp_path):
    pdf_bytes = _pdf_bytes_with_text_pages(["page one", "page two"])

    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["page_count"] == 2
    # Blank pages produce no text -> no chunks, which is correct behavior.
    assert body["chunk_count"] == 0
    assert body["chunks"] == []


def test_ingest_document_rejects_non_pdf_content_type(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 400


def test_ingest_document_rejects_invalid_pdf_bytes(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 400


def test_ingest_document_chunk_metadata_and_storage(monkeypatch, tmp_path):
    """
    Exercises the metadata attachment + embedding + storage logic directly
    via the service layer (bypassing real PDF parsing) so we can assert on
    filename/page/chunk_id and confirm chunks actually reach the vector store.
    """
    import app.services.document_service as document_service_module
    from rag.pdf_loader import PageContent

    fake_pages = [
        PageContent(page_number=1, text="a" * 1500),
        PageContent(page_number=2, text="short page"),
    ]
    monkeypatch.setattr(document_service_module, "load_pdf", lambda file_bytes: fake_pages)

    fake_store = FakeVectorStore()
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=fake_store,
        document_registry=registry,
    )
    result = service.ingest_pdf(filename="report.pdf", file_bytes=b"irrelevant")

    assert result.filename == "report.pdf"
    assert result.page_count == 2
    assert result.status == "READY"
    assert result.document_id  # a uuid4 string was generated
    # Page 1 (1500 chars) with settings.chunk_size=800/overlap=150 -> 3 chunks
    # ((1500 - 150) / (800 - 150) rounded up = 3)
    page_1_chunks = [c for c in result.chunks if c.page_number == 1]
    page_2_chunks = [c for c in result.chunks if c.page_number == 2]
    assert len(page_1_chunks) == 3
    assert len(page_2_chunks) == 1

    first_chunk = page_1_chunks[0]
    assert first_chunk.chunk_id == f"{result.document_id}::p1::c0"
    assert first_chunk.filename == "report.pdf"
    assert first_chunk.document_id == result.document_id
    assert first_chunk.chunk_index == 0

    # All 4 chunks (3 from page 1, 1 from page 2) should have been embedded
    # and persisted to the vector store with the right metadata.
    assert fake_store.count() == 4
    stored_ids = {chunk.chunk_id for chunk in fake_store.stored_chunks}
    assert stored_ids == {
        f"{result.document_id}::p1::c0",
        f"{result.document_id}::p1::c1",
        f"{result.document_id}::p1::c2",
        f"{result.document_id}::p2::c0",
    }
    assert fake_store.stored_chunks[0].metadata["filename"] == "report.pdf"
    assert fake_store.stored_chunks[0].metadata["document_id"] == result.document_id

    # The registry should reflect the READY document with its final counts.
    record = registry.get_document(result.document_id)
    assert record is not None
    assert record.status == "READY"
    assert record.page_count == 2
    assert record.chunk_count == 4


def test_ingest_document_with_no_extractable_text_marks_registry_failed(tmp_path):
    """A structurally valid PDF with zero extractable text (e.g. a scanned
    page) should ingest successfully (0 chunks) but register as FAILED,
    not silently look identical to a READY document with no content."""
    pdf_bytes = _pdf_bytes_with_text_pages(["blank"])
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        document_registry=registry,
    )

    result = service.ingest_pdf(filename="scanned.pdf", file_bytes=pdf_bytes)

    assert result.status == "FAILED"
    assert result.chunks == []
    record = registry.get_document(result.document_id)
    assert record.status == "FAILED"
    assert record.chunk_count == 0


def test_upload_document_returns_document_id_chunks_created_and_status(monkeypatch, tmp_path):
    # Blank pages (from _pdf_bytes_with_text_pages) extract to empty text -
    # monkeypatch load_pdf so this test exercises a document that actually
    # produces chunks, same approach as
    # test_ingest_document_chunk_metadata_and_storage.
    import app.services.document_service as document_service_module
    from rag.pdf_loader import PageContent

    monkeypatch.setattr(
        document_service_module,
        "load_pdf",
        lambda file_bytes: [PageContent(page_number=1, text="hello world " * 200)],
    )
    pdf_bytes = _pdf_bytes_with_text_pages(["hello world " * 200])

    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"documentId", "chunksCreated", "status"}
    assert isinstance(body["documentId"], str) and body["documentId"]
    assert body["chunksCreated"] > 0
    assert body["status"] == "READY"


def test_upload_document_rejects_non_pdf_content_type(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 400


def test_list_documents_returns_uploaded_documents_most_recent_first(monkeypatch, tmp_path):
    import app.services.document_service as document_service_module
    from rag.pdf_loader import PageContent

    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = _fake_ingestion_service(tmp_path, registry=registry)

    monkeypatch.setattr(
        document_service_module,
        "load_pdf",
        lambda file_bytes: [PageContent(page_number=1, text="first document body text")],
    )
    service.ingest_pdf(filename="a.pdf", file_bytes=b"irrelevant-a")

    monkeypatch.setattr(
        document_service_module,
        "load_pdf",
        lambda file_bytes: [PageContent(page_number=1, text="second document body text")],
    )
    service.ingest_pdf(filename="b.pdf", file_bytes=b"irrelevant-b")

    app.dependency_overrides[get_document_ingestion_service] = _override_with(service)
    try:
        response = client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    body = response.json()
    filenames = [doc["filename"] for doc in body["documents"]]
    # Most recently uploaded first -> "b.pdf" before "a.pdf".
    assert filenames == ["b.pdf", "a.pdf"]
    assert all(doc["status"] == "READY" for doc in body["documents"])
    assert all("documentId" in doc and "chunkCount" in doc for doc in body["documents"])


def test_search_documents_returns_ranked_results(tmp_path):
    service = _fake_ingestion_service(tmp_path)

    from rag.vector_store import StoredChunk

    service._vector_store.store_chunks(
        [
            StoredChunk(
                chunk_id="policy.pdf::p1::c0",
                text="Refunds are issued within 30 days of purchase.",
                embedding=[1.0],
                metadata={"filename": "policy.pdf", "page_number": 1, "chunk_index": 0},
            )
        ]
    )

    app.dependency_overrides[get_document_ingestion_service] = _override_with(service)
    try:
        response = client.post(
            "/api/v1/documents/search",
            json={"query": "What is our refund policy?", "top_k": 3},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is our refund policy?"
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["chunk_id"] == "policy.pdf::p1::c0"
    assert result["filename"] == "policy.pdf"
    assert result["page_number"] == 1
    assert "Refunds are issued" in result["text"]


def test_search_documents_rejects_empty_query(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post("/api/v1/documents/search", json={"query": ""})
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 422
