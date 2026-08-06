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
