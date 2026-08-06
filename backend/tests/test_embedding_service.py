import pytest

from rag.embedding_service import EmbeddingService

# Loading the real sentence-transformers model is slow (downloads weights on
# first run, then loads them into memory) - done once per test module via
# this module-level fixture-like instance rather than per test.
_service = EmbeddingService(model_name="all-MiniLM-L6-v2")


def test_embed_text_returns_vector_of_expected_dimension():
    vector = _service.embed_text("Retrieval-Augmented Generation combines search and generation.")

    assert isinstance(vector, list)
    assert len(vector) == _service.dimension
    assert all(isinstance(value, float) for value in vector)


def test_embed_text_rejects_empty_string():
    with pytest.raises(ValueError):
        _service.embed_text("")
    with pytest.raises(ValueError):
        _service.embed_text("   ")


def test_embed_texts_batches_and_preserves_order():
    texts = ["chunk one about cats", "chunk two about dogs", "chunk three about birds"]
    results = _service.embed_texts(texts)

    assert len(results) == 3
    assert [r.text for r in results] == texts
    assert all(len(r.embedding) == _service.dimension for r in results)


def test_embed_texts_rejects_empty_list():
    with pytest.raises(ValueError):
        _service.embed_texts([])


def test_embed_texts_rejects_blank_entry():
    with pytest.raises(ValueError):
        _service.embed_texts(["valid text", ""])


def test_similar_meanings_produce_closer_vectors_than_unrelated_ones():
    """
    Sanity check for the actual point of embeddings: semantically similar
    text should have a smaller distance (higher cosine similarity) than
    semantically unrelated text.
    """
    anchor = _service.embed_text("The cat sat on the mat.")
    similar = _service.embed_text("A cat was resting on a rug.")
    unrelated = _service.embed_text("The stock market crashed yesterday.")

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b)

    sim_similar = cosine_similarity(anchor, similar)
    sim_unrelated = cosine_similarity(anchor, unrelated)

    assert sim_similar > sim_unrelated
