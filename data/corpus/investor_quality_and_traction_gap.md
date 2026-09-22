---
title: "What Crunchbase-Style Data Doesn't Capture: Investor Quality and Traction"
source_url: "https://link.springer.com/article/10.1007/s11187-025-01100-8"
retrieved: "2026-09-22"
---

## Investor and Network Reputation as a Signal

Research published in *Small Business Economics* (Springer Nature) on how third parties assess reputation signals of entrepreneurial teams finds that, under the kind of uncertainty that surrounds a young company, outside observers lean heavily on the prominence and track record of that company's affiliates - its lead investors, board members, and network - to judge its likely quality. Companies backed by prominent, well-regarded investors tend to be assessed more favorably (and, the research suggests, to perform better) than otherwise-comparable companies without such backing. This is a well-established idea in entrepreneurship research: who is willing to put their name and capital behind a company is itself informative, independent of anything about the company's own product or financials.

## A Signal That Doesn't Transfer Cleanly to Outcomes

The same line of research also finds an important nuance: entrepreneurial-team and investor reputation signals matter more for *investment decisions* (whether a company raises money, and from whom) than they do for *actual commercial success* down the line. In other words, a prominent-investor signal reliably predicts who gets funded, but is a weaker, noisier predictor of who ultimately builds a durable, successful business - a mismatch worth keeping in mind whenever "getting funded by well-known investors" is used as an implicit stand-in for "will succeed."

## What This Dataset Has, and What It Doesn't

This project's dataset (`DATA_CARD.md`) has `funding_total_usd`, `funding_rounds`, and dates - it has no field identifying *which* investors participated in any round, their track record, or their network position. It similarly has no field capturing product usage, revenue, user growth, or any other direct measure of commercial traction - the closest available proxy is `funding_total_usd` itself, which reflects what investors were willing to commit, not what customers were willing to pay or use. Both investor quality and product traction are, per the research above, meaningfully predictive of startup outcomes and are entirely absent from this dataset's feature set. Any prediction this project's model makes should be read as conditioned only on funding trajectory, timing, and geography/category - not on the two categories of signal (who's backing the company, and how customers are actually responding to it) that a more complete picture of "is this startup going to succeed" would need.
