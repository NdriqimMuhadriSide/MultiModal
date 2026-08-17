"""
The embedding cache, and the service's use of it.

The service tests use a fake model rather than loading MiniLM: what is
under test is which texts reach the forward pass, and a real model would
make that slow to assert and no more true.
"""
import pytest

from rag.embedding_cache import EmbeddingCache


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(db_path=str(tmp_path / "embeddings.sqlite3"))


# ---- The store -------------------------------------------------------------


def test_a_vector_round_trips(cache):
    cache.put_many("model-a", {"hello": [0.5, -0.25, 0.125]})

    assert cache.get_many("model-a", ["hello"]) == {"hello": [0.5, -0.25, 0.125]}


def test_an_unknown_text_is_simply_absent(cache):
    assert cache.get_many("model-a", ["never seen"]) == {}


def test_a_partial_hit_returns_only_what_is_there(cache):
    cache.put_many("model-a", {"known": [1.0]})

    assert cache.get_many("model-a", ["known", "unknown"]) == {"known": [1.0]}


def test_the_model_is_part_of_the_key(cache):
    cache.put_many("model-a", {"hello": [1.0]})

    # Swapping embedding_model_name must not hand back the old model's
    # vectors - they are not comparable, and a silently mixed collection is
    # unrecoverable without a full re-ingest.
    assert cache.get_many("model-b", ["hello"]) == {}


def test_writing_the_same_text_twice_does_not_raise(cache):
    cache.put_many("model-a", {"hello": [1.0]})
    cache.put_many("model-a", {"hello": [1.0]})

    assert cache.count() == 1


def test_empty_calls_are_no_ops(cache):
    assert cache.get_many("model-a", []) == {}
    cache.put_many("model-a", {})
    assert cache.count() == 0


def test_vectors_survive_a_reopen(tmp_path):
    path = str(tmp_path / "embeddings.sqlite3")
    EmbeddingCache(db_path=path).put_many("model-a", {"hello": [0.25, 0.75]})

    assert EmbeddingCache(db_path=path).get_many("model-a", ["hello"]) == {
        "hello": [0.25, 0.75]
    }


def test_an_unreadable_cache_is_a_miss_not_a_crash(tmp_path):
    cache = EmbeddingCache(db_path=str(tmp_path / "embeddings.sqlite3"))
    cache.put_many("model-a", {"hello": [1.0]})
    # Corrupt the file underneath it. Embedding the text again is the right
    # answer here; failing the request is not.
    (tmp_path / "embeddings.sqlite3").write_bytes(b"not a database")

    assert cache.get_many("model-a", ["hello"]) == {}


# ---- The service's use of it -----------------------------------------------


class FakeModel:
    """Stands in for SentenceTransformer, recording what it was asked to embed."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True):
        import numpy

        if isinstance(texts, str):
            self.batches.append([texts])
            return numpy.array([float(len(texts))], dtype=numpy.float32)

        self.batches.append(list(texts))
        return numpy.array([[float(len(text))] for text in texts], dtype=numpy.float32)

    def get_sentence_embedding_dimension(self):
        return 1

    @property
    def max_seq_length(self):
        return 256

    class _Tokenizer:
        def encode(self, text, add_special_tokens=True, verbose=False):
            return [0] * len(text.split())

    tokenizer = _Tokenizer()


def _service(cache, model=None):
    from rag.embedding_service import EmbeddingService

    service = EmbeddingService.__new__(EmbeddingService)
    service._model = model or FakeModel()
    service._model_name = "fake-model"
    service._cache = cache
    return service


def test_a_repeated_query_skips_the_model(cache):
    model = FakeModel()
    service = _service(cache, model)

    first = service.embed_text("what is the refund window?")
    second = service.embed_text("what is the refund window?")

    assert first == second
    assert len(model.batches) == 1


def test_only_the_changed_chunks_reach_the_model(cache):
    model = FakeModel()
    service = _service(cache, model)

    service.embed_texts(["chunk one", "chunk two", "chunk three"])
    service.embed_texts(["chunk one", "chunk two", "CHANGED"])

    # The re-ingest costs one forward pass, not three.
    assert model.batches[1] == ["CHANGED"]


def test_the_batch_result_keeps_input_order_and_length(cache):
    service = _service(cache)

    service.embed_texts(["b"])
    result = service.embed_texts(["a", "b", "c"])

    assert [chunk.text for chunk in result] == ["a", "b", "c"]
    # "b" came from the cache, the others from the model; the caller cannot
    # tell, which is the point.
    assert all(chunk.embedding for chunk in result)


def test_duplicates_in_a_batch_cost_one_forward_pass(cache):
    model = FakeModel()
    service = _service(cache, model)

    result = service.embed_texts(["same", "same", "other"])

    assert model.batches[0] == ["same", "other"]
    assert len(result) == 3
    assert result[0].embedding == result[1].embedding


def test_a_service_without_a_cache_always_embeds(cache):
    model = FakeModel()
    service = _service(None, model)

    service.embed_text("hello")
    service.embed_text("hello")

    assert len(model.batches) == 2


def test_cached_vectors_match_what_the_model_produced(cache):
    model = FakeModel()
    service = _service(cache, model)

    fresh = service.embed_texts(["a text"])[0].embedding
    from_cache = service.embed_texts(["a text"])[0].embedding

    # float32 in, float32 out - the packed storage is exact for what the
    # model emits, not a lossy approximation of it.
    assert fresh == from_cache
