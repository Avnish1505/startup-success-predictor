"""Orchestrates Layer 1 (facts) + Layer 2 (retrieval) into one rendered
response. No network call anywhere in this module."""
from __future__ import annotations

from src.advisor.facts import render_facts_bundle
from src.advisor.retrieval import AdvisorIndex, mmr_rerank, retrieve

RETRIEVAL_TOP_K = 8
RERANK_K = 3
# Measured, not guessed: across the 20-question eval set (RETRIEVAL_EVAL.md),
# every in-corpus question's top score was >= 0.167 and every deliberately
# out-of-corpus question's top score was <= 0.150 - a real gap. 0.15 sits in
# that gap with a small margin. This is calibrated against a 20-question set,
# not a law of nature; it may not generalize to every possible off-topic
# query, but it beats returning a confident-looking answer to "what's the
# capital of France" with no cutoff at all.
MIN_RETRIEVAL_SCORE = 0.15


def _retrieve_and_rerank(question: str, index: AdvisorIndex) -> list:
    candidates_scored = retrieve(question, index, top_k=RETRIEVAL_TOP_K)
    candidates_scored = [(c, s) for c, s in candidates_scored if s >= MIN_RETRIEVAL_SCORE]
    if not candidates_scored:
        return []
    query_vector = index.vectorizer.transform([question]).toarray()[0]
    candidates = [
        (chunk, score, index.vectors[index.chunks.index(chunk)])
        for chunk, score in candidates_scored
    ]
    return mmr_rerank(query_vector, candidates, k=RERANK_K)


def build_advisor_response(question: str, facts_bundle: dict | None, index: AdvisorIndex) -> str:
    sections = []

    if facts_bundle is not None:
        sections.append(render_facts_bundle(facts_bundle))
    else:
        sections.append(
            "## What the model says\n"
            "No prediction yet this session - run one in the Predictor tab first "
            "for calibrated-probability, SHAP, and percentile context."
        )

    retrieved = _retrieve_and_rerank(question, index)
    if retrieved:
        lines = ["## Related context"]
        for chunk in retrieved:
            lines.append(f"- **{chunk.heading}** ({chunk.doc_id}): {chunk.text}")
            lines.append(f"  Source: {chunk.source_url}")
        sections.append("\n".join(lines))
    else:
        sections.append("## Related context\nNothing in the local corpus matched this question.")

    if facts_bundle is None and not retrieved:
        sections.append(
            "\nThis question falls outside both the facts core and the local corpus - "
            "I don't have a grounded answer for it."
        )

    return "\n\n".join(sections)


def generate_response(facts_bundle: dict | None, retrieved_chunks: list, question: str) -> str:
    """Optional generation seam - NOT IMPLEMENTED.

    This is exactly the boundary where an opt-in LLM call could later turn
    the facts bundle + retrieved passages into flowing prose, without
    touching any retrieval or facts-assembly logic above. Left unimplemented
    deliberately: this project's hard constraint is fully offline operation,
    and wiring a real generator here would violate it by default. If this is
    ever implemented, it must remain strictly additive/optional (e.g. a
    user-toggled "polish this answer" button that still shows the raw
    grounded response as a fallback), never a replacement for it.
    """
    raise NotImplementedError(
        "generate_response is an intentionally unimplemented seam - see docstring."
    )
