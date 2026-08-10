import pytest

from rag.vector_store import StoredChunk, VectorStore


@pytest.fixture
def store(tmp_path):
    """Fresh, isolated Chroma collection per test (temp dir, unique name)."""
    return VectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_docs")


def _chunk(chunk_id: str, text: str, embedding: list[float], **metadata) -> StoredChunk:
    return StoredChunk(chunk_id=chunk_id, text=text, embedding=embedding, metadata=metadata)


def test_store_chunks_rejects_empty_list(store):
    with pytest.raises(ValueError):
        store.store_chunks([])


def test_search_returns_empty_list_when_store_is_empty(store):
    assert store.search(query_embedding=[1.0, 0.0], top_k=5) == []


def test_search_rejects_invalid_arguments(store):
    with pytest.raises(ValueError):
        store.search(query_embedding=[], top_k=5)
    with pytest.raises(ValueError):
        store.search(query_embedding=[1.0], top_k=0)


def test_store_and_search_ranks_by_similarity(store):
    # Three orthogonal-ish 2D vectors so similarity ranking is unambiguous.
    store.store_chunks(
        [
            _chunk("a", "refund policy chunk", [1.0, 0.0], filename="policy.pdf", page_number=1),
            _chunk("b", "unrelated chunk", [0.0, 1.0], filename="policy.pdf", page_number=2),
            _chunk("c", "near match chunk", [0.9, 0.1], filename="policy.pdf", page_number=3),
        ]
    )

    results = store.search(query_embedding=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    # "a" is an exact match (similarity ~1.0), "c" is close, "b" is far -
    # so the top 2 results should be "a" then "c", not "b".
    assert results[0].chunk_id == "a"
    assert results[1].chunk_id == "c"
    assert results[0].similarity > results[1].similarity


def test_store_chunks_upserts_on_repeated_chunk_id(store):
    store.store_chunks([_chunk("a", "original text", [1.0, 0.0], filename="doc.pdf", page_number=1)])
    store.store_chunks([_chunk("a", "updated text", [1.0, 0.0], filename="doc.pdf", page_number=1)])

    assert store.count() == 1
    results = store.search(query_embedding=[1.0, 0.0], top_k=1)
    assert results[0].text == "updated text"


def test_count_reflects_stored_chunks(store):
    assert store.count() == 0
    store.store_chunks([_chunk("a", "text a", [1.0, 0.0], filename="doc.pdf", page_number=1)])
    assert store.count() == 1


def test_get_by_document_id_returns_only_that_document_in_page_order(store):
    store.store_chunks(
        [
            _chunk("d1::p2::c0", "second page", [0.0, 1.0],
                   document_id="d1", filename="a.pdf", page_number=2, chunk_index=0),
            _chunk("d1::p1::c1", "first page, later", [1.0, 0.0],
                   document_id="d1", filename="a.pdf", page_number=1, chunk_index=1),
            _chunk("d1::p1::c0", "first page, earlier", [1.0, 0.0],
                   document_id="d1", filename="a.pdf", page_number=1, chunk_index=0),
            _chunk("d2::p1::c0", "other document", [0.5, 0.5],
                   document_id="d2", filename="b.pdf", page_number=1, chunk_index=0),
        ]
    )

    records = store.get_by_document_id("d1")

    # Chroma's get() gives no ordering guarantee, so the sort is ours.
    assert [r.chunk_id for r in records] == ["d1::p1::c0", "d1::p1::c1", "d1::p2::c0"]
    assert records[0].text == "first page, earlier"
    assert records[0].metadata["filename"] == "a.pdf"


def test_get_by_document_id_is_empty_for_unknown_document(store):
    assert store.get_by_document_id("nope") == []


def test_get_by_document_id_rejects_empty_id(store):
    with pytest.raises(ValueError):
        store.get_by_document_id("")


def test_delete_by_document_id_removes_only_that_document(store):
    store.store_chunks(
        [
            _chunk("d1::p1::c0", "doomed", [1.0, 0.0], document_id="d1", page_number=1, chunk_index=0),
            _chunk("d1::p1::c1", "also doomed", [1.0, 0.0], document_id="d1", page_number=1, chunk_index=1),
            _chunk("d2::p1::c0", "survivor", [0.0, 1.0], document_id="d2", page_number=1, chunk_index=0),
        ]
    )

    deleted = store.delete_by_document_id("d1")

    assert deleted == 2
    assert store.count() == 1
    assert store.get_by_document_id("d1") == []
    assert [r.chunk_id for r in store.get_by_document_id("d2")] == ["d2::p1::c0"]


def test_delete_by_document_id_reports_zero_for_unknown_document(store):
    assert store.delete_by_document_id("never-existed") == 0


def test_delete_by_document_id_rejects_empty_id(store):
    with pytest.raises(ValueError):
        store.delete_by_document_id("   ")


def test_all_chunks_returns_the_whole_corpus(store):
    """rag/keyword_index.py builds its inverted index from this."""
    store.store_chunks(
        [
            _chunk("a", "refund policy", [1.0, 0.0], filename="policy.pdf", page_number=1),
            _chunk("b", "office hours", [0.0, 1.0], filename="handbook.pdf", page_number=2),
        ]
    )

    records = store.all_chunks()

    assert {record.chunk_id for record in records} == {"a", "b"}
    assert {record.text for record in records} == {"refund policy", "office hours"}
    assert next(r for r in records if r.chunk_id == "a").metadata["filename"] == "policy.pdf"


def test_all_chunks_is_empty_for_an_empty_store(store):
    assert store.all_chunks() == []


def test_fingerprint_changes_on_write(store):
    before = store.fingerprint()

    store.store_chunks([_chunk("a", "text", [1.0, 0.0], document_id="d1")])

    assert store.fingerprint() != before


def test_fingerprint_changes_on_delete(store):
    store.store_chunks([_chunk("a", "text", [1.0, 0.0], document_id="d1")])
    before = store.fingerprint()

    store.delete_by_document_id("d1")

    assert store.fingerprint() != before


def test_fingerprint_is_stable_across_reads(store):
    """A cached keyword index must not be rebuilt on every query."""
    store.store_chunks([_chunk("a", "text", [1.0, 0.0], document_id="d1")])

    first = store.fingerprint()
    store.search(query_embedding=[1.0, 0.0], top_k=1)
    store.all_chunks()

    assert store.fingerprint() == first
