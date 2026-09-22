---
title: "Why Startups Fail"
source_url: "https://www.cbinsights.com/research/report/startup-failure-reasons-top/"
retrieved: "2026-09-22"
---

## Running Out of Cash Is the Proximate Cause, Not the Root Cause

CB Insights' analysis of venture-backed company shutdowns (its 2024 update looked at 431 VC-backed companies that shut down since 2023) found that running out of capital is the single most common *reported* reason, showing up in roughly 70% of post-mortems. But CB Insights itself is careful to frame this as the final event, not the underlying problem: a company runs out of cash because something upstream of the cash position went wrong first. Treating "ran out of money" as an explanation is like treating "the patient stopped breathing" as a cause of death - technically true, and not what anyone actually wants to know.

## The Root Causes Behind the Cash Crunch

Digging past the proximate cause, CB Insights' more recent analysis attributes failures to poor product-market fit (about 43% of cases), bad timing or macro conditions (about 29%), and unsustainable unit economics (about 19%) - and it explicitly notes that these overlap, so the percentages don't sum to 100% and a single company's failure usually has more than one contributing cause. An earlier CB Insights study, looking across a broader historical sample, put "no market need" as the single largest identified cause at around 42%.

## Timing and Sector Effects

CB Insights also found that bad timing or unfavorable macro conditions drove a disproportionate share of failures in sectors that attracted heavy capital during a hype cycle that didn't pan out - climate & energy, food & agriculture, and blockchain companies funded heavily in 2021-2022 are cited as examples. This matters for interpreting any dataset of startup outcomes: a company's failure is not purely a function of its own execution, but also of when and into what market conditions it happened to be founded and funded.

## Relevance to This Project's Dataset

This project's `closed` label (see `DATA_CARD.md`) is a blunt instrument next to CB Insights' taxonomy: it records that a company stopped operating, not *why*. None of "poor product-market fit," "bad timing," or "unsustainable unit economics" are features anywhere in this dataset - the model can only see funding history, timing, and geography/category, which are, at best, weak proxies for some of these causes and silent on others (product-market fit and unit economics in particular are not observable from Crunchbase-style data at all).
