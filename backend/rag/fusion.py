"""
Rank fusion.

Responsibility: merge several ranked lists of chunk ids into one, without
needing their scores to mean the same thing.

That last clause is the whole problem. Dense search returns cosine
similarity, bounded in [-1, 1], where 0.62 is a decent match. BM25 returns an
unbounded sum of IDF terms, where 0.62 is nothing and 14 is a good match on
one corpus and mediocre on another. There is no constant that converts one
into the other, and normalising each list to [0, 1] doesn't fix it either -
min-max normalisation makes the best result of a list of garbage score 1.0,
so a question no chunk answers still produces a confident-looking top hit.

Reciprocal Rank Fusion sidesteps this by throwing the scores away and using
only the *order*:

    score(chunk) = sum over lists of  weight / (k + rank)

A chunk ranked 1st by one retriever and unranked by the other scores
1/(60+1) = 0.0164. A chunk ranked 3rd and 5th scores 1/63 + 1/65 = 0.0313 -
so agreement between two retrievers outweighs a single first place. That is
exactly the behaviour hybrid retrieval is after: the chunks both halves
found are the ones most likely to be right.

`k` (60 by convention, from Cormack et al. 2009) sets how quickly rank
matters less. It is large relative to the ranks in play, which flattens the
difference between 1st and 2nd - deliberately, because a retriever's
confidence in *which* of its top few is best is worth much less than the
fact that both retrievers surfaced the same chunk at all.
"""
from dataclasses import dataclass

# The conventional constant. Higher flattens the curve further (rank matters
# less, agreement matters more); lower sharpens it toward "whatever was 1st".
DEFAULT_RRF_K = 60


@dataclass
class Ranking:
    """
    One retriever's opinion: chunk ids, best first.

    Ids rather than whole results, because fusion has no business knowing
    what a chunk contains - and because the two inputs carry different
    payload types (SearchResult and KeywordMatch) that would otherwise have
    to be unified here rather than at the caller, which is where the chunks
    are being assembled anyway.
    """

    name: str
    chunk_ids: list[str]
    # Scales this list's contribution. 1.0 / 1.0 treats both halves as equally
    # trustworthy; raising the keyword weight suits a corpus of identifier-
    # heavy technical text, raising the dense weight suits prose where
    # questions rarely reuse the document's own vocabulary.
    weight: float = 1.0


def reciprocal_rank_fusion(
    rankings: list[Ranking], k: int = DEFAULT_RRF_K
) -> list[tuple[str, float]]:
    """
    Fuse `rankings` into one list of (chunk_id, fused_score), best first.

    Ties are broken by first appearance across the input lists, in the order
    the lists were given - so the result is deterministic rather than
    dependent on dict iteration luck, and a tie resolves toward the retriever
    the caller listed first.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking.chunk_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + ranking.weight / (k + rank)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = position
                position += 1

    return sorted(fused.items(), key=lambda item: (-item[1], first_seen[item[0]]))
