"""
Document ingestion + retrieval business logic.

Orchestrates the rag/ modules end to end:
  1. rag/loaders/pdf.py         - PDF bytes -> per-page text in reading order
     (via rag/layout.py: multi-column pages are read column by column,
     tables come back as Markdown, running headers/footers are dropped;
     pages with no text layer go through rag/ocr.py; rag/structure.py then
     marks headings and tags every block with the section it sits under)
  2. rag/text_splitter.py      - per-section text -> overlapping chunks, so
     no chunk straddles a heading and each one can be labelled with (and
     prefixed by) the section it came from
     (sized in tokens using the embedding model's own tokenizer, so a
     chunk can never overflow the model's context window - see
     _chunk_budget below)
  3. rag/embedding_service.py  - chunk text -> embedding vectors
  4. rag/vector_store.py       - persist embedded chunks; similarity search
  5. rag/document_registry.py  - track the document itself (id, filename,
     page/chunk counts, status) as a first-class record, independent of
     the chunks stored in Chroma

...attaching the metadata each chunk needs for retrieval: document id,
filename, page number, and a stable chunk id. `ingest_document` covers steps
1-5 (upload -> registered -> stored in Chroma -> READY/FAILED). `search`
covers the query side (question -> embedding -> Chroma search -> ranked
chunks), which a future RAG-answering endpoint will build on to feed
retrieved chunks into an LLM prompt.
"""
import hashlib
import logging
from dataclasses import dataclass, field

from fastapi import HTTPException, status

from ai.llm_service import get_llm_service
from app.core.config import settings
from rag import ocr
from rag.document_registry import DocumentRegistry, DocumentRecord, get_document_registry
from rag.embedding_service import EmbeddingService, get_embedding_service
from rag.layout import Block
from rag.loaders import PageContent, load_document
from rag.metadata import DocumentMetadata, extract_metadata
from rag.retriever import collapse_duplicate_passages
from rag.structure import section_label
from rag.chunking import ChunkBudget, ContextualEnrichment, build_strategy, document_context
from rag.vector_store import SearchResult, StoredChunk, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class IngestedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    # "2. Methods > 2.1 Field Sampling", or "" for text that sits under no
    # heading (a preface, or a document with no detected structure).
    section: str = ""
    # What was embedded, which is not always what is stored - see
    # rag/chunking/base.py. Defaults to `text` via __post_init__.
    embed_text: str = ""
    # Identifies the larger passage this chunk points at, when several chunks
    # point at the same one. Empty when the chunk stands alone.
    parent_id: str = ""

    def __post_init__(self) -> None:
        if not self.embed_text:
            self.embed_text = self.text


@dataclass
class IngestResult:
    document_id: str
    filename: str
    page_count: int
    status: str
    chunks: list[IngestedChunk]
    # True when these bytes were already ingested and the existing document
    # was returned instead of a second copy being created. Defaulted so the
    # ordinary path constructs IngestResult unchanged.
    deduplicated: bool = False
    # Set only when status is FAILED: a user-facing explanation, so the UI can
    # distinguish "this file is corrupt" from "this is a scan and OCR isn't
    # installed" - which are the same status but very different next steps.
    failure_reason: str | None = None
    # What the document says about itself (rag/metadata.py).
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


@dataclass
class DeleteResult:
    document_id: str
    filename: str
    chunks_deleted: int


def _page_sections(page: PageContent) -> list[tuple[tuple[str, ...], list[Block]]]:
    """
    Split a page into runs of consecutive blocks that share a section.

    Runs rather than one group per section: a page can leave a section and
    come back to it (a subsection, then the parent's closing paragraph), and
    merging those would reorder the document.

    The blocks are handed on rather than joined into a string, because a
    strategy needs to know which of them is a table: joining first would leave
    it guessing from Markdown pipes what the loaders already knew for certain.

    Falls back to the whole page as one prose block when there are no blocks -
    which is what layout analysis being switched off looks like from here.
    """
    if not page.blocks:
        return [((), [Block(kind="text", text=page.text)])]

    groups: list[tuple[tuple[str, ...], list[Block]]] = []
    for block in page.blocks:
        if groups and groups[-1][0] == block.section_path:
            groups[-1][1].append(block)
        else:
            groups.append((block.section_path, [block]))

    return groups


# The most of a chunk's token budget a section header is allowed to take. A
# deeply nested path ("Report > 3. Methods > 3.2 Sampling > 3.2.1 Sites") can
# run to dozens of tokens, and a chunk that is mostly its own breadcrumb has
# little room left for the content the breadcrumb is meant to describe.
MAX_HEADER_BUDGET_SHARE = 0.25


def _no_text_reason(pages: list[PageContent]) -> str:
    """
    Explain, in one sentence a user can act on, why a valid PDF produced no
    text.

    There are three genuinely different situations behind the same empty
    result, and the difference is the whole point of the message: OCR wasn't
    available, OCR was available and still found nothing, or the file simply
    has no words in it.
    """
    if any(page.from_ocr for page in pages):
        return (
            "No text could be extracted. OCR ran over the pages but found "
            "nothing readable - the scan may be too low-resolution, skewed, "
            "or in a language other than "
            f"'{settings.ocr_language}' (set OCR_LANGUAGE)."
        )

    blocker = ocr.unavailable_reason()
    if blocker:
        return (
            "No text could be extracted - this looks like a scanned PDF, "
            f"where the pages are images rather than text. {blocker}"
        )

    return "No text could be extracted: the pages appear to contain no words."


def _ocr_was_capped(pages: list[PageContent]) -> bool:
    """True when the document had more pages needing OCR than the cap allows."""
    return (
        settings.ocr_enabled
        and any(page.from_ocr for page in pages)
        and len(pages) > settings.ocr_max_pages
    )


class DocumentIngestionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        document_registry: DocumentRegistry,
        llm_service=None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._document_registry = document_registry
        # Only the propositional strategy and contextual enrichment need this,
        # so it is optional: the default deployment ingests documents without
        # an API key, and asking for one it never uses would be a poor trade.
        self._llm_service = llm_service

    def ingest_document(self, filename: str, file_bytes: bytes) -> IngestResult:
        """
        Extract text from a PDF, split it into metadata-tagged chunks,
        embed each chunk, and persist them to the vector store.

        Pipeline: upload -> extract text -> split into chunks -> generate
        embeddings -> store inside ChromaDB. The document is registered as
        PROCESSING before any of that runs, then flipped to READY (chunks
        produced) or FAILED (zero extractable text, e.g. a scanned PDF with
        no OCR layer) once it completes - so a document_id always resolves
        to an honest status, even for documents that ingest with 0 chunks.

        Re-uploading bytes that are already ingested returns the existing
        document instead of creating a second copy. Without this, every
        re-upload minted a fresh document_id, so the chunk ids never
        collided, `upsert` had nothing to overwrite, and the collection
        accumulated duplicates - which then compete for the same top-k
        slots at retrieval time with identical scores.

        Raises:
            ValueError: if the PDF is empty, unreadable, or has no pages
                (propagated from rag.loaders.pdf).
        """
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = self._document_registry.find_ready_by_content_hash(content_hash)
        if existing is not None:
            stored = self._vector_store.get_by_document_id(existing.document_id)
            # An empty result means the registry and Chroma disagree - the
            # row survived but its chunks did not. Falling through re-ingests
            # rather than handing back a document with no content.
            if stored:
                return IngestResult(
                    document_id=existing.document_id,
                    filename=existing.filename,
                    page_count=existing.page_count,
                    status=existing.status,
                    chunks=[
                        IngestedChunk(
                            chunk_id=record.chunk_id,
                            document_id=existing.document_id,
                            filename=record.metadata.get("filename", existing.filename),
                            page_number=record.metadata.get("page_number", 0),
                            chunk_index=record.metadata.get("chunk_index", 0),
                            text=record.text,
                            section=record.metadata.get("section", ""),
                        )
                        for record in stored
                    ],
                    deduplicated=True,
                    metadata=DocumentMetadata(
                        title=existing.title,
                        author=existing.author,
                        subject=existing.subject,
                        document_date=existing.document_date,
                    ),
                )

        document_id = DocumentRegistry.new_document_id()
        self._document_registry.create_document(
            document_id=document_id, filename=filename, content_hash=content_hash
        )

        try:
            pages = load_document(filename, file_bytes)
        except ValueError as exc:
            # Extraction failed outright (empty/corrupt PDF, no pages) -
            # record it as FAILED rather than leaving a PROCESSING row
            # stuck forever, then re-raise so the HTTP layer still returns
            # its 400 as before.
            self._document_registry.mark_failed(document_id, reason=str(exc))
            raise

        # Recorded before chunking, and whatever the eventual status: a scan
        # that ends up FAILED still has a title and an author worth listing.
        metadata = extract_metadata(filename, file_bytes, pages)
        self._document_registry.set_metadata(
            document_id,
            title=metadata.title,
            author=metadata.author,
            subject=metadata.subject,
            document_date=metadata.document_date,
        )

        measure = self._embedding_service.count_tokens
        budget = self._chunk_budget()
        strategy = self._build_strategy(pages)
        # Contextual enrichment prepends a generated line to the embedded text,
        # so its allowance comes out of the budget before splitting - the same
        # arithmetic as the section header, and the same overflow if skipped.
        if settings.contextual_retrieval:
            budget -= settings.contextual_reserved_tokens
        overlap = min(settings.chunk_overlap_tokens, max(0, budget - 1))

        chunks: list[IngestedChunk] = []
        for page in pages:
            # Chunk one section at a time rather than the page as a whole, so
            # no chunk ever straddles a heading and every chunk has exactly
            # one section to be labelled with.
            chunk_index = 0
            for section_path, section_blocks in _page_sections(page):
                section = section_label(section_path)
                # The header is reserved *out of* the budget rather than added
                # on top of it: it is part of the text that gets embedded, so
                # a chunk sized without it would overflow by exactly the
                # header's length.
                header = self._section_header(section_path, budget)
                body_budget = budget - measure(header) if header else budget
                drafts = strategy.split(
                    section_blocks,
                    ChunkBudget(
                        size=body_budget,
                        overlap=min(overlap, max(0, body_budget - 1)),
                        measure=measure,
                    ),
                )
                for draft in drafts:
                    chunk_id = f"{document_id}::p{page.page_number}::c{chunk_index}"
                    chunks.append(
                        IngestedChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            filename=filename,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            text=header + draft.text,
                            section=section,
                            embed_text=header + draft.embedded(),
                            # Namespaced per page and section so two sections
                            # that both call their first parent "p0" don't
                            # collapse into one at retrieval.
                            parent_id=(
                                f"{document_id}::p{page.page_number}::s{section}::{draft.parent_key}"
                                if draft.parent_key
                                else ""
                            ),
                        )
                    )
                    chunk_index += 1

        if chunks:
            # Embeds `embed_text`, stores `text`. For most strategies they are
            # the same string; where they differ, that difference *is* the
            # strategy - see rag/chunking/base.py.
            embedded = self._embedding_service.embed_texts(
                [chunk.embed_text for chunk in chunks]
            )
            self._vector_store.store_chunks(
                [
                    StoredChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        embedding=embedded_chunk.embedding,
                        metadata={
                            "document_id": chunk.document_id,
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            # Stored as a joined string because Chroma
                            # metadata values must be scalars, and kept
                            # alongside the text (which already carries the
                            # header) so a result can be filtered or grouped
                            # by section without re-parsing it.
                            "section": chunk.section,
                            # Copied onto every chunk, not just the document
                            # row, because Chroma filters at query time on
                            # chunk metadata - a "only documents by X" search
                            # has nothing else to match against. Empty string
                            # rather than None: Chroma metadata values must be
                            # scalars, and None is not one.
                            "title": metadata.title or "",
                            "author": metadata.author or "",
                            "document_date": metadata.document_date or "",
                            # Read back at retrieval to collapse several
                            # chunks that point at the same passage.
                            "parent_id": chunk.parent_id,
                        },
                    )
                    for chunk, embedded_chunk in zip(chunks, embedded)
                ]
            )
            self._document_registry.mark_ready(
                document_id, page_count=len(pages), chunk_count=len(chunks)
            )
            result_status = "READY"
            failure_reason = None
            if _ocr_was_capped(pages):
                # The document still ingested, so there's no FAILED row to
                # carry an explanation - but pages beyond the cap contributed
                # nothing, and silently returning a partial document as READY
                # would be indistinguishable from a complete one.
                logger.warning(
                    "Document %s: only the first %d scanned pages were OCR'd "
                    "(OCR_MAX_PAGES); %d pages were left empty.",
                    document_id,
                    settings.ocr_max_pages,
                    len(pages) - settings.ocr_max_pages,
                )
        else:
            # Valid PDF, but no extractable text - most often a scan whose
            # pages are images. A real outcome, not a crash, so it's recorded
            # as FAILED rather than silently returning an empty success, and
            # with the reason attached so the next step is obvious.
            failure_reason = _no_text_reason(pages)
            self._document_registry.mark_failed(
                document_id, page_count=len(pages), reason=failure_reason
            )
            result_status = "FAILED"

        return IngestResult(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            status=result_status,
            chunks=chunks,
            failure_reason=failure_reason,
            metadata=metadata,
        )

    def _build_strategy(self, pages: list[PageContent]):
        """
        Build the configured chunking strategy, wrapped in contextual
        enrichment when that is switched on.

        A strategy the deployment can't support - `semantic` with no embedding
        service, `propositional` with no API key - raises rather than quietly
        falling back to `recursive`. Silently chunking a corpus a different way
        than configured is the kind of thing nobody notices until the search
        results are inexplicably worse.
        """
        strategy = build_strategy(
            settings.chunking_strategy,
            embedding_service=self._embedding_service,
            llm_service=self._llm_service,
        )
        if not settings.contextual_retrieval:
            return strategy
        if self._llm_service is None:
            raise ValueError("CONTEXTUAL_RETRIEVAL needs an LLM service.")
        return ContextualEnrichment(
            strategy,
            self._llm_service,
            reserved_tokens=settings.contextual_reserved_tokens,
            document_context=document_context(pages),
        )

    def _chunk_budget(self) -> int:
        """
        The largest chunk, in tokens, that this embedding model will actually
        read in full.

        The configured size is a request, not a promise: exceeding the model's
        window doesn't fail, it silently truncates, so a CHUNK_SIZE_TOKENS
        larger than the model can read would quietly produce chunks whose tails
        never reach an embedding. Clamping here means the misconfiguration
        costs a log line instead of a subtly broken index.
        """
        limit = self._embedding_service.max_tokens
        if settings.chunk_size_tokens > limit:
            logger.warning(
                "CHUNK_SIZE_TOKENS is %d but the embedding model reads only %d tokens; "
                "using %d so chunks are not silently truncated.",
                settings.chunk_size_tokens,
                limit,
                limit,
            )
            return limit
        return settings.chunk_size_tokens

    def _section_header(self, section_path: tuple[str, ...], budget: int) -> str:
        """
        Build the "[2. Methods > 2.1 Sampling]" prefix a chunk is embedded with.

        The prefix goes into the text that gets *embedded*, which is the whole
        point: a paragraph reading "we collected 500 samples" only competes for
        "how was the data collected" once its own text says Methods. It also
        reaches the model at answer time, since rag/context_builder.py passes
        chunk text through verbatim.

        Falls back to the innermost heading alone when the full path would eat
        too much of the budget - the leaf is the most specific and most useful
        part of the path, so it is the right thing to keep when only one part
        fits.
        """
        if not section_path or not settings.chunk_section_headers:
            return ""

        allowance = budget * MAX_HEADER_BUDGET_SHARE
        for candidate in (section_label(section_path), section_path[-1]):
            header = f"[{candidate}]\n\n"
            if self._embedding_service.count_tokens(header) <= allowance:
                return header
        return ""

    def search(
        self,
        query: str,
        top_k: int = 5,
        author: str | None = None,
        title: str | None = None,
        document_date: str | None = None,
    ) -> list[SearchResult]:
        """
        Embed `query` and return the `top_k` most similar stored chunks,
        optionally restricted by document metadata.

        The filters are exact matches, not fuzzy ones: "written by this
        author" is a fact about the document, and a near-miss on a name is a
        wrong answer rather than a slightly worse one. Semantic closeness is
        what the query itself is for.

        Raises:
            ValueError: if `query` is empty or top_k is not positive
                (propagated from embedding_service / vector_store).
        """
        query_embedding = self._embedding_service.embed_text(query)
        results = self._vector_store.search(
            query_embedding,
            # Over-fetched because collapsing duplicates below removes
            # results: without the extra candidates, a parent matched by four
            # of its children would leave top_k three results short.
            top_k=top_k * settings.retrieval_overfetch,
            where={"author": author, "title": title, "document_date": document_date},
        )
        return collapse_duplicate_passages(results)[:top_k]

    def list_documents(self) -> list[DocumentRecord]:
        """Return every uploaded document (most recent first) for the documents list UI."""
        return self._document_registry.list_documents()

    def delete_document(self, document_id: str) -> DeleteResult | None:
        """
        Forget a document completely: its chunks in Chroma and its registry
        row. Returns None if no such document exists.

        Chroma is cleared first. If that succeeds and the registry delete
        then fails, the leftover row is visible and can be retried; the
        reverse order would leave orphaned chunks that nothing lists and so
        nothing can ever remove - still retrievable, attributed to a
        document the UI says is gone.
        """
        record = self._document_registry.get_document(document_id)
        if record is None:
            return None

        chunks_deleted = self._vector_store.delete_by_document_id(document_id)
        self._document_registry.delete_document(document_id)

        return DeleteResult(
            document_id=document_id,
            filename=record.filename,
            chunks_deleted=chunks_deleted,
        )


def get_document_ingestion_service() -> DocumentIngestionService:
    """
    FastAPI dependency that builds a DocumentIngestionService.

    Wraps construction in try/except (consistent with chat_service /
    vision_service) so any configuration error - e.g. the embedding model
    failing to load - surfaces as a clean HTTPException instead of a raw 500.
    """
    try:
        return DocumentIngestionService(
            embedding_service=get_embedding_service(),
            vector_store=get_vector_store(),
            document_registry=get_document_registry(),
            # Built only when something is actually going to call it, so a
            # default deployment ingests documents with no API key at all.
            llm_service=(
                get_llm_service()
                if settings.contextual_retrieval or settings.chunking_strategy == "propositional"
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Document ingestion service is not configured: {exc}",
        ) from exc
