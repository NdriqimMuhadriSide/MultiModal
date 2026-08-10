"""
Keyword index (BM25).

Responsibility: the *lexical* half of retrieval - find chunks that contain
the words the question actually used, rather than chunks whose meaning is
nearby in embedding space.

This exists because dense retrieval has a specific, well-known blind spot:
an embedding is a lossy summary of meaning, so a rare literal token carries
almost no weight in it. "ERR-4021", "Regulation (EU) 2016/679", a part
number, a surname, a version string - these are exactly the terms a user
types when they know what they are looking for, and exactly the terms a
384-dimension MiniLM vector smooths away. BM25 has the opposite bias: it
knows nothing about meaning, but a term that appears in one chunk out of a
thousand dominates its ranking. Running both and merging (see rag/fusion.py)
is what "hybrid retrieval" means.

Two pieces live here:

    BM25          the scoring structure - a list of chunks in, a ranked list
                  out. No knowledge of where the chunks came from.
    KeywordIndex  keeps a BM25 built from the vector store's current
                  contents, rebuilding when the store changes.

The corpus is read back out of the vector store rather than kept in a second
database of its own. There is then only one place a chunk can exist, so an
ingest or a delete cannot leave the two halves of retrieval disagreeing about
what the corpus is - which is the failure mode a separate keyword store would
introduce, and it is a silent one: the sparse half would keep returning
chunks the dense half no longer has.

Implemented directly rather than pulling in `rank_bm25` or standing up
Elasticsearch/OpenSearch, for the same reason the Markdown and HTML loaders
are hand-written: the whole thing is one scoring formula and an inverted
index, it adds no dependency and no service to run, and the constants that
matter (k1, b) stay visible and tunable instead of buried in a library.

Trade-off worth stating plainly: the index lives in memory in one process
and is rebuilt from scratch on the first query after a restart or a write.
For a corpus of a few thousand chunks that is milliseconds. For a corpus in
the millions this file is the wrong answer and a real search engine is the
right one - the `search()` interface is what a swap would preserve.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from rag.vector_store import ChunkRecord, VectorStore

# Okapi BM25's two knobs.
#
# k1 controls how fast term frequency saturates: with k1=1.2, the fifth
# occurrence of a word adds far less than the second. This is the whole
# reason BM25 beats raw TF-IDF - a chunk that says "refund" twenty times is
# not twenty times more about refunds than one that says it once.
#
# b controls how hard long chunks are penalised (0 = not at all, 1 = fully
# normalised by length). Some length correction is needed because a long
# chunk contains more of every word by accident; full correction over-favours
# short ones. 1.2 / 0.75 are the values the original TREC experiments
# settled on and what Lucene still ships.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

# Words split on anything that isn't a letter or a digit, so "ERR-4021"
# indexes as "err" and "4021" and matches a question that writes it either
# way. Digits are deliberately kept: version numbers, years, article numbers
# and error codes are the queries this index exists to serve.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A deliberately short list. BM25's IDF term already suppresses words that
# appear everywhere - a word in nine chunks out of ten scores near zero on
# its own - so this is not load-bearing for ranking quality. It earns its
# place at the other end: without it, every one of "what is the ..." is a
# posting list the length of the corpus, so a five-word question walks the
# whole index to score terms that then contribute nothing.
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have how i if in into is it its
    of on or that the their there these they this to was were what when where
    which who why will with you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """
    Split text into the terms this index matches on.

    Lowercased and stripped of punctuation, so the query and the document go
    through exactly the same transformation - a term only ever matches a term
    that was produced the same way. No stemming: "run"/"running" stay
    distinct, which costs some recall and buys the guarantee that an exact
    literal a user typed is matched exactly, which is the point of having a
    lexical half at all.
    """
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


@dataclass
class KeywordMatch:
    """
    A chunk matched by keyword search.

    Deliberately shaped like rag/vector_store.py's SearchResult minus the
    similarity, because `score` here is not one: BM25 scores are unbounded
    and only comparable within a single query's results. A 14.2 means "the
    best match for this question", not "86% relevant".
    """

    chunk_id: str
    text: str
    metadata: dict
    score: float


@dataclass
class _Posting:
    """Where one term occurs: document index -> how many times."""

    frequency: dict[int, int] = field(default_factory=dict)


class BM25:
    """
    An inverted index over a fixed list of chunks, ranked with Okapi BM25.

    Immutable once built. Adding a chunk means building a new one - which is
    what KeywordIndex does, and is cheap enough at this corpus size that
    incremental update isn't worth the bookkeeping (and the bugs) it costs.
    """

    def __init__(
        self,
        records: list[ChunkRecord],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._records = records
        self._k1 = k1
        self._b = b
        self._postings: dict[str, _Posting] = {}
        self._lengths: list[int] = []

        for index, record in enumerate(records):
            tokens = tokenize(record.text)
            self._lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, _Posting()).frequency[index] = count

        # Guarded against an empty corpus so scoring never divides by zero;
        # with no documents there is nothing to score anyway.
        self._average_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self._records)

    def _idf(self, term: str) -> float:
        """
        How much seeing `term` should count for.

        The `ln(1 + ...)` form rather than the textbook `ln((N - df + 0.5) /
        (df + 0.5))`: the textbook version goes *negative* once a term appears
        in more than half the corpus, so on a small collection a common word
        actively subtracts from a chunk's score - a chunk could rank lower for
        containing one of the query's words. This variant (Lucene's) is
        always positive and asymptotically identical for rare terms, which are
        the ones that decide the ranking.
        """
        posting = self._postings.get(term)
        if posting is None:
            return 0.0
        document_frequency = len(posting.frequency)
        total = len(self._records)
        return math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[KeywordMatch]:
        """
        Return up to `top_k` chunks ranked by BM25, best first.

        Chunks that share no term with the query score zero and are dropped
        rather than returned at the bottom: a zero here means "this chunk
        contains none of these words", which is not a weak match but a
        non-match, and passing it on would put text in front of the model
        that nothing selected it for.

        Raises:
            ValueError: if `query` is empty or `top_k` is not positive - the
                same contract as rag/vector_store.py's search, so a caller
                handling one handles both.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        terms = tokenize(query)
        if not terms or not self._records:
            return []

        scores: dict[int, float] = {}
        # Iterated per *term* rather than per document: only chunks that
        # actually contain a query term are ever touched, so cost scales with
        # how rare the query's words are, not with the size of the corpus.
        for term, query_count in Counter(terms).items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            idf = self._idf(term)
            for index, count in posting.frequency.items():
                length_norm = (
                    1 - self._b + self._b * (self._lengths[index] / self._average_length)
                    if self._average_length
                    else 1.0
                )
                saturated = (count * (self._k1 + 1)) / (count + self._k1 * length_norm)
                # A term repeated in the question counts more than once, so
                # "refund refund policy" leans further toward refunds than
                # "refund policy" does.
                scores[index] = scores.get(index, 0.0) + idf * saturated * query_count

        ranked = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        return [
            KeywordMatch(
                chunk_id=self._records[index].chunk_id,
                text=self._records[index].text,
                metadata=self._records[index].metadata,
                score=score,
            )
            for index, score in ranked
            if score > 0
        ]


class KeywordIndex:
    """
    A BM25 index kept in step with the vector store's contents.

    Built lazily on first search and rebuilt whenever the store's fingerprint
    changes, so an upload or a delete is reflected on the next question with
    nothing to wire up at the ingestion site. Pulling rather than being
    pushed to is the point: a push would have to be added to every future
    write path, and forgetting one would leave the keyword half quietly
    serving a stale corpus.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self._vector_store = vector_store
        self._k1 = k1
        self._b = b
        self._bm25: BM25 | None = None
        self._fingerprint: tuple[int, int] | None = None

    def search(self, query: str, top_k: int = 5) -> list[KeywordMatch]:
        """
        Return up to `top_k` chunks whose text matches `query`'s terms.

        Raises:
            ValueError: if `query` is empty or `top_k` is not positive.
        """
        self._refresh_if_stale()
        assert self._bm25 is not None  # set by _refresh_if_stale
        return self._bm25.search(query, top_k=top_k)

    def _refresh_if_stale(self) -> None:
        """Rebuild the index when the store has changed since it was built."""
        fingerprint = self._vector_store.fingerprint()
        if self._bm25 is not None and fingerprint == self._fingerprint:
            return
        self._bm25 = BM25(self._vector_store.all_chunks(), k1=self._k1, b=self._b)
        self._fingerprint = fingerprint


_keyword_index_instance: KeywordIndex | None = None


def get_keyword_index() -> KeywordIndex:
    """
    Return a process-wide KeywordIndex.

    Shared rather than per-request for the same reason the embedding model is:
    the expensive part is building it, and a per-request instance would pay
    that cost on every question. Not decorated with @lru_cache, matching
    get_vector_store() - tests point it at a temporary store and need no
    cached instance surviving across runs.
    """
    global _keyword_index_instance
    if _keyword_index_instance is None:
        from app.core.config import settings
        from rag.vector_store import get_vector_store

        _keyword_index_instance = KeywordIndex(
            get_vector_store(), k1=settings.bm25_k1, b=settings.bm25_b
        )
    return _keyword_index_instance
