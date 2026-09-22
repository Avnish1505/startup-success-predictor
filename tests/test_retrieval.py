import numpy as np

from src.advisor.chunking import Chunk
from src.advisor.retrieval import build_index, mmr_rerank, retrieve


def _toy_chunks():
    return [
        Chunk("doc_a::1", "doc_a", "Funding", "Startups raise pre-seed seed and series A funding rounds from investors.", "https://a"),
        Chunk("doc_a::2", "doc_a", "Funding Stages", "Series B and Series C rounds fund later stage growth and expansion.", "https://a"),
        Chunk("doc_b::1", "doc_b", "Failure", "Startups fail when they run out of cash or have no product market fit.", "https://b"),
        Chunk("doc_c::1", "doc_c", "Calibration", "Probability calibration adjusts a classifier so predicted scores match observed rates.", "https://c"),
    ]


def test_retrieve_ranks_relevant_chunk_first():
    index = build_index(_toy_chunks())
    results = retrieve("why do startups run out of cash and fail", index, top_k=4)
    assert results[0][0].chunk_id == "doc_b::1"


def test_retrieve_returns_scores_descending():
    index = build_index(_toy_chunks())
    results = retrieve("funding rounds for startups", index, top_k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_mmr_rerank_avoids_single_document_domination():
    index = build_index(_toy_chunks())
    results = retrieve("funding rounds", index, top_k=4)
    candidates = [(chunk, score, index.vectors[index.chunks.index(chunk)]) for chunk, score in results]
    query_vector = index.vectorizer.transform(["funding rounds"]).toarray()[0]
    reranked = mmr_rerank(query_vector, candidates, k=2, lambda_param=0.6)
    assert len(reranked) == 2


def test_save_and_load_index_round_trips(tmp_path):
    from src.advisor.retrieval import load_index, save_index

    index = build_index(_toy_chunks())
    save_index(index, tmp_path)
    loaded = load_index(tmp_path)
    assert len(loaded.chunks) == len(index.chunks)
    assert np.allclose(loaded.vectors, index.vectors)
    results = retrieve("probability calibration classifier", loaded, top_k=1)
    assert results[0][0].chunk_id == "doc_c::1"
