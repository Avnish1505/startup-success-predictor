"""TF-IDF retrieval over the local corpus (chosen over a dense embedding
after measuring real memory - see the implementation plan / README). Vectors
are a plain dense numpy array on disk; no vector database, no network."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.advisor.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")
_SUFFIX_RULES = [("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")]


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer - not a full Porter stemmer, but
    enough to close the plural/singular gap (e.g. "startups" -> "startup")
    that caused a real zero-score retrieval failure found during manual
    testing (see RETRIEVAL_EVAL.md), without pulling in a stemming
    dependency for a corpus this small."""
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _tokenize_and_stem(text: str) -> list[str]:
    # Stopwords are filtered on the raw token here, not passed as
    # TfidfVectorizer's own stop_words= - that filters against raw English
    # words, which would silently stop matching once tokens are stemmed
    # (sklearn warns about exactly this if the two are combined).
    raw_tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in raw_tokens if t not in ENGLISH_STOP_WORDS]


@dataclass
class AdvisorIndex:
    vectorizer: TfidfVectorizer
    vectors: np.ndarray
    chunks: list[Chunk]


def build_index(chunks: list[Chunk]) -> AdvisorIndex:
    vectorizer = TfidfVectorizer(tokenizer=_tokenize_and_stem, token_pattern=None)
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return AdvisorIndex(vectorizer=vectorizer, vectors=matrix.toarray(), chunks=chunks)


def save_index(index: AdvisorIndex, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(index.vectorizer, out_dir / "vectorizer.joblib")
    np.save(out_dir / "vectors.npy", index.vectors)
    chunks_payload = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "heading": c.heading, "text": c.text, "source_url": c.source_url}
        for c in index.chunks
    ]
    (out_dir / "chunks.json").write_text(json.dumps(chunks_payload, indent=2))


def load_index(out_dir: Path) -> AdvisorIndex:
    out_dir = Path(out_dir)
    vectorizer = joblib.load(out_dir / "vectorizer.joblib")
    vectors = np.load(out_dir / "vectors.npy")
    chunks_payload = json.loads((out_dir / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_payload]
    return AdvisorIndex(vectorizer=vectorizer, vectors=vectors, chunks=chunks)


def retrieve(query: str, index: AdvisorIndex, top_k: int = 10) -> list[tuple[Chunk, float]]:
    query_vector = index.vectorizer.transform([query]).toarray()
    scores = cosine_similarity(query_vector, index.vectors)[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [(index.chunks[i], float(scores[i])) for i in order]


def mmr_rerank(
    query_vector: np.ndarray,
    candidates: list[tuple[Chunk, float, np.ndarray]],
    k: int = 3,
    lambda_param: float = 0.6,
) -> list[Chunk]:
    remaining = list(candidates)
    selected: list[tuple[Chunk, float, np.ndarray]] = []

    while remaining and len(selected) < k:
        best_idx, best_score = None, -np.inf
        for i, (chunk, relevance, vector) in enumerate(remaining):
            if selected:
                diversity_penalty = max(
                    float(cosine_similarity([vector], [s_vector])[0, 0]) for _, _, s_vector in selected
                )
            else:
                diversity_penalty = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, i
        selected.append(remaining.pop(best_idx))

    return [chunk for chunk, _, _ in selected]
