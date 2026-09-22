# Retrieval Evaluation

Measured against 20 hand-written questions (17 with an expected chunk, 3 deliberately out-of-corpus) over the real built index (`models/production/advisor_index/`, 21 chunks from `data/corpus/*.md`).

| Metric | Value |
|---|---|
| recall@3 | 0.941 |
| recall@5 | 0.941 |

recall@k = fraction of questions with a known expected chunk where at least one expected chunk ID appeared in the top-k retrieved results (before MMR reranking - MMR reorders/limits the final k shown to the user, but retrieval quality itself is what this measures).

Run `python3 scripts/run_retrieval_eval.py` to reproduce - these numbers are generated, not hand-typed.

## The one miss

Both metrics land at 16/17 (the same question fails at k=3 and k=5, so widening k doesn't help). The
miss: *"What are the root causes behind startup failure besides running out of money?"* expected
`why_startups_fail::the_root_causes_behind_the_cash_crunch`, but the top-5 results were dominated by
the closely related `why_startups_fail::running_out_of_cash_is_the_proximate_cause,_not_the_root_cause`
chunk (cosine 0.119 vs. the expected chunk's much lower score) plus chunks from other documents that
happen to share terms like "root cause." This is a genuine TF-IDF limitation, not a retrieval bug: the
question's phrasing ("besides running out of money") shares more surface vocabulary with the *sibling*
chunk about running out of cash than with the chunk that actually answers it (which is phrased around
"product-market fit," "bad timing," "unit economics" - the CB Insights source's own terms, not a
paraphrase of the question). A dense embedding model would likely handle this specific case better,
since it captures semantic similarity beyond shared words - one of the real costs of the TF-IDF choice
made for memory-budget reasons (see the README's AI Advisor section).
