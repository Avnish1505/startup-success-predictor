# Retrieval Evaluation

Measured against 20 hand-written questions (17 with an expected chunk, 3 deliberately out-of-corpus) over the real built index (`models/production/advisor_index/`, 21 chunks from `data/corpus/*.md`).

| Metric | Value |
|---|---|
| recall@3 | 0.882 |
| recall@5 | 0.941 |

recall@k = fraction of questions with a known expected chunk where at least one expected chunk ID appeared in the top-k retrieved results (before MMR reranking - MMR reorders/limits the final k shown to the user, but retrieval quality itself is what this measures).

Run `python3 scripts/run_retrieval_eval.py` to reproduce - these numbers are generated, not hand-typed.

## Design notes and misses, reported honestly

**Stemming.** An initial version of the retrieval pipeline scored *every single chunk at 0.0000* for
the query "Why do startups fail?" - a real bug found by manual testing before this eval was even run,
not a hypothetical. Cause: plain TF-IDF has no stemming, so the query's "startups" never matched the
corpus's "startup," and "fail" never matched "failure"/"failures" (the only forms used in the corpus
body text). Fixed with a minimal suffix-stripping stemmer (`src/advisor/retrieval.py::_stem` - not a
full Porter stemmer, just enough to close common plural/singular gaps) applied consistently to both
corpus and query text, plus one corpus edit (`why_startups_fail.md`) to use "fail" as a bare verb in
the body, not just the document title. The numbers above are measured *after* that fix.

**Minimum-score cutoff.** Across all 20 questions, every in-corpus question's top score was >= 0.167
and every deliberately out-of-corpus question's top score was <= 0.150 - a real, measured gap (not
assumed). `MIN_RETRIEVAL_SCORE = 0.15` sits in that gap, so `build_advisor_response` correctly returns
"nothing in the local corpus matched this question" for the weather/pizza/France questions rather than
a confident-looking but spurious match. This threshold is calibrated against a 20-question set, not a
law of nature, and may not generalize to every possible off-topic query.

**The one persistent miss.** Both recall@3 and recall@5 have exactly one shared culprit:
*"What are the root causes behind startup failure besides running out of money?"* expected
`why_startups_fail::the_root_causes_behind_the_cash_crunch` but the top hit is the closely related
`why_startups_fail::running_out_of_cash_is_the_proximate_cause,_not_the_root_cause` chunk (0.176 vs.
a much lower score for the expected chunk, which doesn't even make the top 5). This is a genuine
TF-IDF vocabulary-mismatch limitation, not a bug: the question's phrasing shares more surface words
with the sibling "proximate cause" chunk than with the chunk that actually answers it (phrased around
"product-market fit," "bad timing," "unit economics" - the source's own terms, not a paraphrase of the
question). A dense embedding model would likely handle this specific case better; it's one of the real
costs of the TF-IDF choice made for memory-budget reasons (see the README's AI Advisor section).

**recall@3 vs. recall@5.** One additional question - *"Why is survivorship bias especially a problem
in venture capital and startup data?"* - lands at rank 4 (score 0.109, just behind three higher-scoring
chunks from other documents), missing k=3 but recovered at k=5. This is ordinary top-k ranking noise on
a 21-chunk corpus, not a separate bug.
