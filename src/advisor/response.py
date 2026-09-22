"""Orchestrates the router (Layer 0) + Layer 1 (facts) + Layer 2 (retrieval)
into one rendered response. No network call anywhere in this module."""
from __future__ import annotations

from src.advisor.facts import render_facts_bundle
from src.advisor.retrieval import AdvisorIndex, mmr_rerank, retrieve
from src.advisor.router import classify_intent, find_named_feature

RETRIEVAL_TOP_K = 8
RERANK_K = 3
# Measured, not guessed: across the 20-question eval set (RETRIEVAL_EVAL.md),
# every in-corpus question's top score was >= 0.167 and every deliberately
# out-of-corpus question's top score was <= 0.150 - a real gap. 0.15 sits in
# that gap with a small margin.
MIN_RETRIEVAL_SCORE = 0.15

CAPABILITY_STATEMENT = (
    "## What I can answer\n"
    "- Questions about your last prediction (\"why is my score low\", \"what hurts my score\").\n"
    "- General questions about startup fundraising and failure (\"why do startups fail\", \"what is a Series A\").\n"
    "- Questions about this model's own limitations (leakage, calibration, survivorship bias).\n"
    "I don't generate free-form advice - answers are grounded in your prediction's real numbers or in the "
    "cited local corpus, never invented."
)


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


def _render_retrieved(retrieved: list) -> str:
    if not retrieved:
        return "## Related context\nNothing in the local corpus matched this question."
    lines = ["## Related context"]
    for chunk in retrieved:
        lines.append(f"- **{chunk.heading}** ({chunk.doc_id}): {chunk.text}")
        lines.append(f"  Source: {chunk.source_url}")
    return "\n".join(lines)


def build_advisor_response(question: str, facts_bundle: dict | None, index: AdvisorIndex) -> str:
    intent = classify_intent(question)

    if intent == "greeting":
        return CAPABILITY_STATEMENT

    if intent == "about_prediction":
        if facts_bundle is None:
            return (
                "## No prediction yet\n"
                "I don't have a prediction to explain yet this session. Go to the **Predictor** tab, "
                "fill in a profile, and click Predict - then come back and ask again."
            )
        foreground = find_named_feature(question)
        return render_facts_bundle(facts_bundle, foreground_feature=foreground)

    if intent == "general":
        retrieved = _retrieve_and_rerank(question, index)
        return _render_retrieved(retrieved)

    # out_of_scope
    return (
        "## Outside what I can answer\n"
        "That doesn't match a prediction question, a startup/fundraising topic in the local corpus, "
        "or a question about this model's limitations.\n\n" + CAPABILITY_STATEMENT
    )


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
