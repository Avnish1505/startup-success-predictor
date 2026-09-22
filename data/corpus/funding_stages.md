---
title: "Startup Funding Stages"
source_url: "https://www.startups.com/articles/series-funding-a-b-c-d-e"
retrieved: "2026-09-22"
---

## Pre-Seed and Seed

Pre-seed is the earliest stage of outside capital, typically from founders themselves, friends and family, and occasionally an angel investor or accelerator - it funds building toward a minimum viable product, not scaling a proven one. Seed is usually the first *official* equity round: the company has something closer to a working product, and the round is raised to find product-market fit and initial paying customers rather than to scale an already-validated model.

## Series A Through C

Series A is typically the first round led by an institutional venture capital firm, with a formal valuation and often a board seat for the lead investor; it is generally raised to scale a business model that has shown early signs of working, not to discover whether it works at all. Series B is an expansion round, raised once the model is proven and the task shifts to scaling operations and market reach. Series C is later-stage capital, often earmarked for international expansion, acquisitions, or preparing for an IPO. Round sizes climb substantially at each stage, though the exact dollar ranges vary a great deal by sector and era and shouldn't be treated as fixed thresholds.

## Beyond Series C

Companies don't follow a fixed checklist through these stages. Many raise through Series D, E, or beyond; many stop at Series B and are acquired; some skip stages entirely via a large early round; some never raise institutional venture capital at all and are invisible to a Crunchbase-style dataset by construction. The labels "Series A," "Series B," and so on describe a round's position in a company's fundraising history, not a guarantee about where that company is headed next.

## Relevance to This Project's Dataset

This project's dataset (see `DATA_CARD.md`) has no column recording which named stage (seed, Series A, Series B, ...) any individual funding round belongs to - it only has `funding_rounds` (a count) and `funding_total_usd`/`funding_span_days` (aggregate totals and elapsed time). A company with two rounds could be a seed-and-Series-A story or two large late rounds close together; the dataset cannot distinguish these, which is part of why `funding_rounds` and `funding_total_usd` are, at best, a coarse proxy for "how far along" a company's fundraising journey actually is.
