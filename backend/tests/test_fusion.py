from rag.fusion import DEFAULT_RRF_K, Ranking, reciprocal_rank_fusion


def test_a_single_ranking_is_returned_in_its_own_order():
    fused = reciprocal_rank_fusion([Ranking(name="dense", chunk_ids=["a", "b", "c"])])

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]


def test_scores_follow_the_reciprocal_rank_formula():
    fused = dict(reciprocal_rank_fusion([Ranking(name="dense", chunk_ids=["a", "b"])]))

    assert fused["a"] == 1 / (DEFAULT_RRF_K + 1)
    assert fused["b"] == 1 / (DEFAULT_RRF_K + 2)


def test_agreement_beats_a_single_first_place():
    """
    The property the whole technique exists for: two retrievers both ranking
    something 2nd is stronger evidence than one ranking it 1st.
    """
    fused = reciprocal_rank_fusion(
        [
            Ranking(name="dense", chunk_ids=["dense-only", "agreed"]),
            Ranking(name="keyword", chunk_ids=["keyword-only", "agreed"]),
        ]
    )

    assert fused[0][0] == "agreed"


def test_weights_scale_a_ranking_s_contribution():
    """A heavily weighted list's first place should outrank a light list's."""
    fused = reciprocal_rank_fusion(
        [
            Ranking(name="dense", chunk_ids=["d"], weight=0.2),
            Ranking(name="keyword", chunk_ids=["k"], weight=5.0),
        ]
    )

    assert [chunk_id for chunk_id, _ in fused] == ["k", "d"]


def test_a_zero_weighted_ranking_contributes_nothing_to_scores():
    fused = dict(
        reciprocal_rank_fusion(
            [
                Ranking(name="dense", chunk_ids=["a"], weight=1.0),
                Ranking(name="keyword", chunk_ids=["a"], weight=0.0),
            ]
        )
    )

    assert fused["a"] == 1 / (DEFAULT_RRF_K + 1)


def test_ties_break_toward_the_ranking_listed_first():
    """Deterministic output matters: the same query must not reorder run to run."""
    fused = reciprocal_rank_fusion(
        [
            Ranking(name="dense", chunk_ids=["d"]),
            Ranking(name="keyword", chunk_ids=["k"]),
        ]
    )

    assert [chunk_id for chunk_id, _ in fused] == ["d", "k"]


def test_a_smaller_k_sharpens_the_gap_between_ranks():
    sharp = dict(reciprocal_rank_fusion([Ranking(name="d", chunk_ids=["a", "b"])], k=1))
    flat = dict(reciprocal_rank_fusion([Ranking(name="d", chunk_ids=["a", "b"])], k=1000))

    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


def test_empty_rankings_fuse_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([Ranking(name="dense", chunk_ids=[])]) == []
