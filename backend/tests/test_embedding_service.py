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


# --- Token counting ---------------------------------------------------------


def test_max_tokens_reports_the_models_real_limit():
    assert _service.max_tokens == 256  # all-MiniLM-L6-v2


def test_count_tokens_includes_the_special_tokens():
    """
    So the count is directly comparable to max_tokens with no off-by-two -
    [CLS] and [SEP] occupy two of the model's 256 slots like anything else.
    """
    assert _service.count_tokens("hello") == 3


def test_characters_are_a_poor_proxy_for_tokens():
    """
    The reason chunk sizing moved to tokens: the same number of characters is
    a wildly different number of tokens depending on what the text is.
    """
    prose = ("The quick brown fox jumps over the lazy dog. " * 20)[:800]
    table = ("| Bergen | 4412 | 2026-03-02 |\n" * 40)[:800]

    assert len(prose) == len(table) == 800
    assert _service.count_tokens(prose) < _service.max_tokens
    # ...while the same 800 characters of table rows overflow it.
    assert _service.count_tokens(table) > _service.max_tokens


def test_oversized_input_is_reported_rather_than_silently_truncated(caplog):
    """
    Truncation is the worst kind of bug - the call succeeds and returns a
    vector describing only the start of the text. With token-sized chunks this
    should never happen, so if it does it needs to be audible.
    """
    with caplog.at_level("WARNING"):
        _service.embed_texts(["word " * 400])

    assert "truncated" in caplog.text
    assert "256-token limit" in caplog.text


def test_normal_input_logs_nothing(caplog):
    with caplog.at_level("WARNING"):
        _service.embed_texts(["a short chunk of text"])

    assert caplog.text == ""


def test_truncation_is_what_makes_oversizing_dangerous():
    """
    Pins the failure itself: past the limit, the tail of a chunk has no effect
    on its embedding at all, so two different chunks get the same vector.
    """
    shared = "| Bergen | 4412 | 2026-03-02 |\n" * 40
    with_refunds = shared + " The refund policy allows returns within 30 days."
    with_zebras = shared + " Zebras migrate across the Serengeti each spring."

    first, second = _service.embed_texts([with_refunds, with_zebras])
    cosine = sum(a * b for a, b in zip(first.embedding, second.embedding))

    assert cosine == pytest.approx(1.0, abs=1e-6)
