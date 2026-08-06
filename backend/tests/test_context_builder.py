from rag.context_builder import build_context
from rag.retriever import RetrievedChunk


def _chunk(chunk_id: str, text: str, filename: str, page: int, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=text, filename=filename, page=page, score=score)


def test_build_context_labels_each_chunk_with_document_and_page():
    chunks = [
        _chunk("c0", "Employees receive 25 vacation days per year.", "employee_policy.pdf", 12),
    ]

    context = build_context(chunks)

    assert "Document: employee_policy.pdf" in context
    assert "Page: 12" in context
    assert "Content: Employees receive 25 vacation days per year." in context


def test_build_context_joins_multiple_chunks():
    chunks = [
        _chunk("c0", "first chunk text", "a.pdf", 1),
        _chunk("c1", "second chunk text", "b.pdf", 5),
    ]

    context = build_context(chunks)

    assert "Document: a.pdf" in context
    assert "Document: b.pdf" in context
    assert context.index("a.pdf") < context.index("b.pdf")


def test_build_context_returns_empty_string_for_no_chunks():
    assert build_context([]) == ""
