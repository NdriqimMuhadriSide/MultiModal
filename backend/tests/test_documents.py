import io

import pytest
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

    # Counts characters instead of tokens: deterministic, and it keeps these
    # tests independent of a real model's vocabulary. The limit is set high
    # enough never to clamp - test_chunk_budget_is_clamped_to_the_model covers
    # the clamp itself with a deliberately small one.
    max_tokens = 100_000

    def count_tokens(self, text: str) -> int:
        return len(text)

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

    def search(self, query_embedding, top_k: int = 5, where: dict | None = None):
        matching = [
            chunk
            for chunk in self.stored_chunks
            if all(
                chunk.metadata.get(key) == value
                for key, value in (where or {}).items()
                if value not in (None, "")
            )
        ]
        return [
            SearchResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                similarity=0.9,
            )
            for chunk in matching[:top_k]
        ]

    def get_by_document_id(self, document_id: str):
        from rag.vector_store import ChunkRecord

        records = [
            ChunkRecord(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata)
            for c in self.stored_chunks
            if c.metadata.get("document_id") == document_id
        ]
        records.sort(
            key=lambda r: (r.metadata.get("page_number", 0), r.metadata.get("chunk_index", 0))
        )
        return records

    def delete_by_document_id(self, document_id: str) -> int:
        keep = [c for c in self.stored_chunks if c.metadata.get("document_id") != document_id]
        removed = len(self.stored_chunks) - len(keep)
        self.stored_chunks = keep
        return removed

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


def test_ingest_document_rejects_an_unsupported_extension(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("photo.png", b"\x89PNG", "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


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
    from rag.loaders import PageContent

    # Pinned so the chunk arithmetic below stays readable; FakeEmbeddingService
    # counts characters, so these are effectively the old character sizes.
    monkeypatch.setattr(document_service_module.settings, "chunk_size_tokens", 800)
    monkeypatch.setattr(document_service_module.settings, "chunk_overlap_tokens", 150)

    fake_pages = [
        PageContent(page_number=1, text="a" * 1500),
        PageContent(page_number=2, text="short page"),
    ]
    monkeypatch.setattr(document_service_module, "load_document", lambda filename, file_bytes: fake_pages)

    fake_store = FakeVectorStore()
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=fake_store,
        document_registry=registry,
    )
    result = service.ingest_document(filename="report.pdf", file_bytes=b"irrelevant")

    assert result.filename == "report.pdf"
    assert result.page_count == 2
    assert result.status == "READY"
    assert result.document_id  # a uuid4 string was generated
    # Page 1 (1500 chars) with size 800 / overlap 150 -> 3 chunks
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

    result = service.ingest_document(filename="scanned.pdf", file_bytes=pdf_bytes)

    assert result.status == "FAILED"
    assert result.chunks == []
    record = registry.get_document(result.document_id)
    assert record.status == "FAILED"
    assert record.chunk_count == 0


def _service_over(pages, tmp_path, monkeypatch):
    """An ingestion service whose PDF loader returns exactly `pages`."""
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module, "load_document", lambda filename, file_bytes: pages)
    store = FakeVectorStore()
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        document_registry=DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3")),
    )
    return service, store


def _structured_page():
    from rag.layout import Block
    from rag.loaders import PageContent

    blocks = [
        Block(kind="heading", text="2. Methods", level=1, section_path=("2. Methods",)),
        Block(
            kind="text",
            text="We collected 500 samples over six weeks.",
            section_path=("2. Methods",),
        ),
        Block(
            kind="heading",
            text="3. Results",
            level=1,
            section_path=("3. Results",),
        ),
        Block(kind="text", text="Yields rose sharply.", section_path=("3. Results",)),
    ]
    return PageContent(
        page_number=1,
        text="\n\n".join(block.text for block in blocks),
        blocks=blocks,
    )


def test_chunks_do_not_straddle_a_heading(tmp_path, monkeypatch):
    """
    Sections are chunked one at a time, so a chunk belongs to exactly one of
    them - which is what makes labelling it meaningful.
    """
    service, _ = _service_over([_structured_page()], tmp_path, monkeypatch)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert [chunk.section for chunk in result.chunks] == ["2. Methods", "3. Results"]
    assert "Yields rose sharply." not in result.chunks[0].text
    assert "We collected 500 samples" not in result.chunks[1].text


def test_chunk_text_is_prefixed_with_its_section(tmp_path, monkeypatch):
    # The prefix goes into the embedded text, which is the point: it's what
    # lets a paragraph about samples match a question about methods.
    service, _ = _service_over([_structured_page()], tmp_path, monkeypatch)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert result.chunks[0].text.startswith("[2. Methods]\n\n")


def test_section_headers_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.document_service.settings.chunk_section_headers", False)
    service, _ = _service_over([_structured_page()], tmp_path, monkeypatch)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert result.chunks[0].text.startswith("2. Methods")
    # ...and the section is still recorded, just not written into the text.
    assert result.chunks[0].section == "2. Methods"


def test_section_is_stored_as_chunk_metadata(tmp_path, monkeypatch):
    service, store = _service_over([_structured_page()], tmp_path, monkeypatch)

    service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert [chunk.metadata["section"] for chunk in store.stored_chunks] == [
        "2. Methods",
        "3. Results",
    ]


def test_chunk_ids_stay_unique_across_sections_on_a_page(tmp_path, monkeypatch):
    service, _ = _service_over([_structured_page()], tmp_path, monkeypatch)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    ids = [chunk.chunk_id for chunk in result.chunks]
    assert ids == [f"{result.document_id}::p1::c0", f"{result.document_id}::p1::c1"]


def test_a_page_without_blocks_is_chunked_whole(tmp_path, monkeypatch):
    """Layout analysis switched off still ingests, just with no sections."""
    from rag.loaders import PageContent

    page = PageContent(page_number=1, text="Plain text with no structure at all.")
    service, _ = _service_over([page], tmp_path, monkeypatch)

    result = service.ingest_document(filename="plain.pdf", file_bytes=b"%PDF-")

    assert len(result.chunks) == 1
    assert result.chunks[0].section == ""
    assert result.chunks[0].text == "Plain text with no structure at all."


def _sectioned_page(prose: str):
    from rag.layout import Block
    from rag.loaders import PageContent

    blocks = [
        Block(kind="heading", text="2. Methods", level=1, section_path=("2. Methods",)),
        Block(kind="text", text=prose, section_path=("2. Methods",)),
    ]
    return PageContent(page_number=1, text=prose, blocks=blocks)


@pytest.mark.parametrize(
    "strategy_name", ["recursive", "semantic", "sentence_window", "parent_document"]
)
def test_every_strategy_ingests_end_to_end(strategy_name, monkeypatch, tmp_path):
    """
    The point of the strategy abstraction: ingestion, embedding, storage and
    retrieval are the same code whichever one is selected.
    """
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunking_strategy", strategy_name)
    page = _sectioned_page(
        "The trial began in March. It was halted six weeks later. "
        "Dr. Patel reviewed the samples. Yields rose sharply. " * 4
    )
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, store = _service_with_registry(tmp_path)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert result.status == "READY"
    assert result.chunks
    assert len(store.stored_chunks) == len(result.chunks)
    # Every chunk keeps its section label and stays inside the budget.
    assert all(chunk.section == "2. Methods" for chunk in result.chunks)
    assert all(len(chunk.embed_text) <= service._chunk_budget() for chunk in result.chunks)


def test_what_is_embedded_can_differ_from_what_is_stored(monkeypatch, tmp_path):
    """
    Pins the mechanism the alternative strategies are built on. The vector
    comes from the child; the text handed back is the parent.
    """
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunking_strategy", "parent_document")
    page = _sectioned_page("The trial began in March. " * 30)
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, store = _service_with_registry(tmp_path)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert any(chunk.embed_text != chunk.text for chunk in result.chunks)
    # FakeEmbeddingService embeds len(text), so the stored vector proves which
    # of the two strings was actually sent to the model.
    for chunk, stored in zip(result.chunks, store.stored_chunks):
        assert stored.embedding == [float(len(chunk.embed_text))]
        assert stored.text == chunk.text


def test_chunks_pointing_at_one_passage_share_a_parent_id(monkeypatch, tmp_path):
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunking_strategy", "parent_document")
    page = _sectioned_page("The trial began in March. " * 30)
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, store = _service_with_registry(tmp_path)

    service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    parent_ids = [chunk.metadata["parent_id"] for chunk in store.stored_chunks]
    assert all(parent_ids)
    assert len(set(parent_ids)) < len(parent_ids)


def test_the_default_strategy_leaves_chunks_standing_alone(tmp_path, monkeypatch):
    import app.services.document_service as document_service_module

    page = _sectioned_page("The trial began in March. " * 30)
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, store = _service_with_registry(tmp_path)

    service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    # No parent means retrieval never collapses them together.
    assert all(chunk.metadata["parent_id"] == "" for chunk in store.stored_chunks)


def test_contextual_retrieval_reserves_room_for_its_line(monkeypatch, tmp_path):
    """
    The generated line is part of the embedded text, so a chunk sized without
    it would overflow by exactly the line's length.
    """
    import app.services.document_service as document_service_module

    class StubLLM:
        def generate_response(self, user_message, history=None, system_prompt=""):
            return "This excerpt is from the March trial report."

    monkeypatch.setattr(document_service_module.settings, "contextual_retrieval", True)
    monkeypatch.setattr(document_service_module.settings, "chunk_size_tokens", 200)
    monkeypatch.setattr(document_service_module.settings, "contextual_reserved_tokens", 50)
    page = _sectioned_page("The trial began in March. " * 40)
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        document_registry=registry,
        llm_service=StubLLM(),
    )

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert result.chunks
    # Section breadcrumb first, then the generated line, then the content.
    assert all(chunk.text.startswith("[2. Methods]") for chunk in result.chunks)
    assert all(
        "This excerpt is from the March trial report." in chunk.text
        for chunk in result.chunks
    )
    # Header and context line included, every chunk still fits the budget.
    assert all(len(chunk.embed_text) <= 200 for chunk in result.chunks)


def test_a_strategy_that_needs_an_llm_fails_loudly_without_one(monkeypatch, tmp_path):
    """Better than silently chunking the corpus a different way than configured."""
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunking_strategy", "propositional")
    page = _sectioned_page("The trial began in March.")
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, _ = _service_with_registry(tmp_path)

    with pytest.raises(ValueError, match="LLM service"):
        service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")


def test_chunk_budget_is_clamped_to_the_model(monkeypatch, tmp_path, caplog):
    """
    Asking for chunks larger than the model reads doesn't fail - it silently
    truncates every one of them. Clamping turns that into a log line.
    """
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunk_size_tokens", 4000)
    service, _, _ = _service_with_registry(tmp_path)
    monkeypatch.setattr(service._embedding_service, "max_tokens", 256, raising=False)

    with caplog.at_level("WARNING"):
        budget = service._chunk_budget()

    assert budget == 256
    assert "CHUNK_SIZE_TOKENS" in caplog.text


def test_a_configured_budget_within_the_model_is_used_as_is(monkeypatch, tmp_path):
    import app.services.document_service as document_service_module

    monkeypatch.setattr(document_service_module.settings, "chunk_size_tokens", 128)
    service, _, _ = _service_with_registry(tmp_path)

    assert service._chunk_budget() == 128


def test_the_section_header_is_reserved_out_of_the_budget(monkeypatch, tmp_path):
    """
    The header is part of the embedded text, so a chunk sized without it would
    overflow by exactly the header's length - which is how it used to work.
    """
    import app.services.document_service as document_service_module
    from rag.layout import Block
    from rag.loaders import PageContent

    monkeypatch.setattr(document_service_module.settings, "chunk_size_tokens", 120)
    monkeypatch.setattr(document_service_module.settings, "chunk_overlap_tokens", 20)

    page = PageContent(
        page_number=1,
        text="",
        blocks=[
            Block(kind="heading", text="2. Methods", level=1, section_path=("2. Methods",)),
            Block(kind="text", text="sample " * 200, section_path=("2. Methods",)),
        ],
    )
    monkeypatch.setattr(
        document_service_module, "load_document", lambda filename, file_bytes: [page]
    )
    service, _, _ = _service_with_registry(tmp_path)

    result = service.ingest_document(filename="report.pdf", file_bytes=b"%PDF-")

    assert len(result.chunks) > 1
    assert all(chunk.text.startswith("[2. Methods]") for chunk in result.chunks)
    # Header included, every chunk still fits the budget.
    assert all(len(chunk.text) <= 120 for chunk in result.chunks)


def test_a_long_section_path_falls_back_to_the_innermost_heading(tmp_path):
    """The leaf is the most specific part of the path, so it's what to keep."""
    service, _, _ = _service_with_registry(tmp_path)
    deep = ("Annual Research Report", "3. Methods", "3.2 Field Sampling", "3.2.1 Coastal Sites")

    header = service._section_header(deep, budget=120)

    assert header == "[3.2.1 Coastal Sites]\n\n"


def test_a_short_section_path_keeps_the_whole_breadcrumb(tmp_path):
    service, _, _ = _service_with_registry(tmp_path)

    header = service._section_header(("2. Methods",), budget=256)

    assert header == "[2. Methods]\n\n"


def _markdown_report() -> bytes:
    return (
        b"---\ntitle: Annual Safety Review\nauthor: Compliance Team\ndate: 2026-02-11\n---\n\n"
        b"# Findings\n\nAll sites passed inspection.\n"
    )


def _service_with_registry(tmp_path):
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    store = FakeVectorStore()
    service = DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        document_registry=registry,
    )
    return service, registry, store


def test_ingest_records_what_the_document_says_about_itself(tmp_path):
    service, registry, _ = _service_with_registry(tmp_path)

    result = service.ingest_document(filename="8f2a-v3.md", file_bytes=_markdown_report())

    record = registry.get_document(result.document_id)
    assert record.title == "Annual Safety Review"
    assert record.author == "Compliance Team"
    assert record.document_date == "2026-02-11"
    # The upload time and the document's own date are different facts.
    assert record.created_at != record.document_date


def test_metadata_is_copied_onto_every_chunk(tmp_path):
    """Chroma filters on chunk metadata, so the document row alone isn't enough."""
    service, _, store = _service_with_registry(tmp_path)

    service.ingest_document(filename="report.md", file_bytes=_markdown_report())

    assert store.stored_chunks
    for chunk in store.stored_chunks:
        assert chunk.metadata["author"] == "Compliance Team"
        assert chunk.metadata["document_date"] == "2026-02-11"


def test_search_can_be_filtered_by_author(tmp_path):
    service, _, _ = _service_with_registry(tmp_path)
    service.ingest_document(filename="a.md", file_bytes=_markdown_report())
    service.ingest_document(
        filename="b.md",
        file_bytes=b"---\ntitle: Other\nauthor: Someone Else\n---\n\n# Other\n\nUnrelated text.\n",
    )

    matching = service.search(query="inspection", author="Compliance Team")
    missing = service.search(query="inspection", author="Nobody At All")

    assert matching
    assert all(result.metadata["author"] == "Compliance Team" for result in matching)
    assert missing == []


def test_unfiltered_search_still_returns_everything(tmp_path):
    service, _, _ = _service_with_registry(tmp_path)
    service.ingest_document(filename="a.md", file_bytes=_markdown_report())

    assert service.search(query="inspection")


def test_a_failed_document_still_records_its_metadata(monkeypatch, tmp_path):
    """A scan that ends up FAILED still has a title worth showing in the list."""
    service, registry, _ = _service_with_registry(tmp_path)

    result = service.ingest_document(
        filename="scan.md", file_bytes=b"---\ntitle: Site Photos\nauthor: Surveyor\n---\n"
    )

    assert result.status == "FAILED"
    record = registry.get_document(result.document_id)
    assert record.title == "Site Photos"
    assert record.author == "Surveyor"


def test_document_list_exposes_the_metadata(tmp_path):
    service, _, _ = _service_with_registry(tmp_path)
    service.ingest_document(filename="8f2a-v3.md", file_bytes=_markdown_report())

    app.dependency_overrides[get_document_ingestion_service] = _override_with(service)
    try:
        response = client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    document = response.json()["documents"][0]
    assert document["title"] == "Annual Safety Review"
    assert document["author"] == "Compliance Team"
    assert document["documentDate"] == "2026-02-11"
    # The filename is still reported alongside it, not replaced by it.
    assert document["filename"] == "8f2a-v3.md"


def _failing_service(tmp_path) -> tuple[DocumentIngestionService, DocumentRegistry]:
    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    return (
        DocumentIngestionService(
            embedding_service=FakeEmbeddingService(),
            vector_store=FakeVectorStore(),
            document_registry=registry,
        ),
        registry,
    )


def test_scanned_pdf_failure_says_ocr_is_unavailable(monkeypatch, tmp_path):
    """
    FAILED on its own is a dead end - a corrupt file and a scan that needs OCR
    look identical. The reason is what makes the next step obvious.
    """
    from rag import ocr

    monkeypatch.setattr(ocr, "_probe", lambda: "OCR is not available: the Tesseract binary was not found.")
    monkeypatch.setattr(ocr.settings, "ocr_enabled", True)
    service, registry = _failing_service(tmp_path)

    result = service.ingest_document(
        filename="scanned.pdf", file_bytes=_pdf_bytes_with_text_pages(["blank"])
    )

    assert result.status == "FAILED"
    assert "scanned PDF" in result.failure_reason
    assert "Tesseract" in result.failure_reason
    assert registry.get_document(result.document_id).failure_reason == result.failure_reason


def test_scanned_pdf_failure_says_ocr_found_nothing_when_it_did_run(monkeypatch, tmp_path):
    from rag.loaders import PageContent

    import app.services.document_service as document_service_module

    monkeypatch.setattr(
        document_service_module,
        "load_document",
        lambda filename, file_bytes: [PageContent(page_number=1, text="", from_ocr=True)],
    )
    service, registry = _failing_service(tmp_path)

    result = service.ingest_document(
        filename="scanned.pdf", file_bytes=_pdf_bytes_with_text_pages(["blank"])
    )

    assert result.status == "FAILED"
    assert "OCR ran" in result.failure_reason
    assert "OCR_LANGUAGE" in result.failure_reason


def test_unreadable_pdf_records_the_parse_error_as_the_reason(tmp_path):
    import pytest

    service, registry = _failing_service(tmp_path)

    with pytest.raises(ValueError):
        service.ingest_document(filename="broken.pdf", file_bytes=b"this is not a pdf")

    record = registry.list_documents()[0]
    assert record.status == "FAILED"
    assert "Could not read PDF" in record.failure_reason


def test_document_list_exposes_the_failure_reason(monkeypatch, tmp_path):
    from rag import ocr

    monkeypatch.setattr(ocr, "_probe", lambda: "OCR is not available: the Tesseract binary was not found.")
    monkeypatch.setattr(ocr.settings, "ocr_enabled", True)
    service, _ = _failing_service(tmp_path)
    service.ingest_document(filename="scanned.pdf", file_bytes=_pdf_bytes_with_text_pages(["blank"]))

    app.dependency_overrides[get_document_ingestion_service] = lambda: service
    try:
        response = client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["status"] == "FAILED"
    assert "Tesseract" in document["failureReason"]


def test_upload_document_returns_document_id_chunks_created_and_status(monkeypatch, tmp_path):
    # Blank pages (from _pdf_bytes_with_text_pages) extract to empty text -
    # monkeypatch load_pdf so this test exercises a document that actually
    # produces chunks, same approach as
    # test_ingest_document_chunk_metadata_and_storage.
    import app.services.document_service as document_service_module
    from rag.loaders import PageContent

    monkeypatch.setattr(
        document_service_module,
        "load_document",
        lambda filename, file_bytes: [PageContent(page_number=1, text="hello world " * 200)],
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
    assert set(body.keys()) == {
        "documentId",
        "chunksCreated",
        "status",
        "deduplicated",
        "failureReason",
    }
    assert isinstance(body["documentId"], str) and body["documentId"]
    assert body["chunksCreated"] > 0
    assert body["status"] == "READY"
    # A first upload is never a duplicate.
    assert body["deduplicated"] is False


def test_upload_document_rejects_an_unsupported_extension(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("photo.png", b"\x89PNG", "image/png")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_list_documents_returns_uploaded_documents_most_recent_first(monkeypatch, tmp_path):
    import app.services.document_service as document_service_module
    from rag.loaders import PageContent

    registry = DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3"))
    service = _fake_ingestion_service(tmp_path, registry=registry)

    monkeypatch.setattr(
        document_service_module,
        "load_document",
        lambda filename, file_bytes: [PageContent(page_number=1, text="first document body text")],
    )
    service.ingest_document(filename="a.pdf", file_bytes=b"irrelevant-a")

    monkeypatch.setattr(
        document_service_module,
        "load_document",
        lambda filename, file_bytes: [PageContent(page_number=1, text="second document body text")],
    )
    service.ingest_document(filename="b.pdf", file_bytes=b"irrelevant-b")

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


# --- Re-upload deduplication -------------------------------------------------
#
# Before content hashing, every upload minted a fresh document_id, so chunk
# ids never collided, `upsert` had nothing to overwrite, and the same PDF
# uploaded twice left two full copies competing for the same top-k slots.


def _service_with_fake_pdf(monkeypatch, tmp_path, pages=None):
    """A service whose PDF parsing is stubbed, sharing one store + registry."""
    import app.services.document_service as document_service_module
    from rag.loaders import PageContent

    monkeypatch.setattr(
        document_service_module,
        "load_document",
        lambda filename, file_bytes: pages or [PageContent(page_number=1, text="policy text " * 100)],
    )
    return DocumentIngestionService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        document_registry=DocumentRegistry(db_path=str(tmp_path / "documents.sqlite3")),
    )


def test_reuploading_identical_bytes_reuses_the_existing_document(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)

    first = service.ingest_document(filename="handbook.pdf", file_bytes=b"the same bytes")
    chunks_after_first = service._vector_store.count()

    second = service.ingest_document(filename="handbook-copy.pdf", file_bytes=b"the same bytes")

    assert second.deduplicated is True
    assert second.document_id == first.document_id
    assert first.deduplicated is False
    # The decisive assertion: no second copy was written.
    assert service._vector_store.count() == chunks_after_first
    assert len(service.list_documents()) == 1
    # Chunk content still comes back, so chunksCreated stays honest.
    assert len(second.chunks) == len(first.chunks)
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]
    # The original filename wins - the stored document is what it is.
    assert second.filename == "handbook.pdf"


def test_different_bytes_still_create_a_separate_document(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)

    first = service.ingest_document(filename="a.pdf", file_bytes=b"bytes one")
    second = service.ingest_document(filename="a.pdf", file_bytes=b"bytes two")

    # Same filename is not the same document; content decides.
    assert second.document_id != first.document_id
    assert second.deduplicated is False
    assert len(service.list_documents()) == 2


def test_reupload_after_delete_ingests_again(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)

    first = service.ingest_document(filename="a.pdf", file_bytes=b"same")
    service.delete_document(first.document_id)

    again = service.ingest_document(filename="a.pdf", file_bytes=b"same")

    # Deleting must release the hash, or a document could never be restored.
    assert again.deduplicated is False
    assert again.document_id != first.document_id
    assert service._vector_store.count() == len(again.chunks)


def test_dedupe_falls_through_when_registry_and_store_disagree(monkeypatch, tmp_path):
    """A row whose chunks vanished must re-ingest, not return an empty document."""
    service = _service_with_fake_pdf(monkeypatch, tmp_path)
    first = service.ingest_document(filename="a.pdf", file_bytes=b"same")

    # Registry row survives; its chunks do not.
    service._vector_store.delete_by_document_id(first.document_id)

    again = service.ingest_document(filename="a.pdf", file_bytes=b"same")

    assert again.deduplicated is False
    assert again.chunks
    assert service._vector_store.count() == len(again.chunks)


# --- Deletion ----------------------------------------------------------------


def test_delete_document_removes_chunks_and_registry_row(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)
    keep = service.ingest_document(filename="keep.pdf", file_bytes=b"keep me")
    doomed = service.ingest_document(filename="doomed.pdf", file_bytes=b"delete me")
    total_before = service._vector_store.count()

    result = service.delete_document(doomed.document_id)

    assert result is not None
    assert result.filename == "doomed.pdf"
    assert result.chunks_deleted == len(doomed.chunks)
    assert service._vector_store.count() == total_before - len(doomed.chunks)
    # The other document is untouched.
    assert [d.document_id for d in service.list_documents()] == [keep.document_id]
    assert service._vector_store.get_by_document_id(keep.document_id)


def test_delete_document_returns_none_for_unknown_id(tmp_path):
    service = _fake_ingestion_service(tmp_path)
    assert service.delete_document("does-not-exist") is None


def test_delete_endpoint_returns_chunks_deleted(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)
    uploaded = service.ingest_document(filename="handbook.pdf", file_bytes=b"bytes")

    app.dependency_overrides[get_document_ingestion_service] = _override_with(service)
    try:
        response = client.delete(f"/api/v1/documents/{uploaded.document_id}")
        listing = client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"documentId", "filename", "chunksDeleted"}
    assert body["documentId"] == uploaded.document_id
    assert body["filename"] == "handbook.pdf"
    assert body["chunksDeleted"] == len(uploaded.chunks)
    assert listing.json()["documents"] == []


def test_delete_endpoint_404s_for_unknown_document(tmp_path):
    app.dependency_overrides[get_document_ingestion_service] = _override_with(
        _fake_ingestion_service(tmp_path)
    )
    try:
        response = client.delete("/api/v1/documents/no-such-id")
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    # A stale UI list gets told the truth rather than a silent success.
    assert response.status_code == 404


def test_upload_endpoint_reports_deduplicated_on_second_upload(monkeypatch, tmp_path):
    service = _service_with_fake_pdf(monkeypatch, tmp_path)
    pdf_bytes = _pdf_bytes_with_text_pages(["page one"])

    app.dependency_overrides[get_document_ingestion_service] = _override_with(service)
    try:
        first = client.post(
            "/api/v1/documents/upload",
            files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")},
        )
        second = client.post(
            "/api/v1/documents/upload",
            files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")},
        )
    finally:
        app.dependency_overrides.pop(get_document_ingestion_service, None)

    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True
    assert second.json()["documentId"] == first.json()["documentId"]
    assert second.json()["chunksCreated"] == first.json()["chunksCreated"]
