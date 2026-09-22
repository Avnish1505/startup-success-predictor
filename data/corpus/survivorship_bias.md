---
title: "Survivorship Bias in Startup and VC Data"
source_url: "https://en.wikipedia.org/wiki/Survivorship_bias"
retrieved: "2026-09-22"
---

## What Survivorship Bias Is

Survivorship bias is the logical error of concentrating on the people, companies, or data points that "survived" some selection process while overlooking those that didn't, typically because the ones that didn't survive are much less visible. The canonical example is analyzing which WWII aircraft returned from missions with bullet damage in certain places, and concluding those places should be reinforced - when the correct conclusion is the opposite, because the planes hit in *other* places are the ones that didn't come back to be counted at all.

## Why It Bites Especially Hard in Startup and VC Data

Startup and venture capital data is particularly exposed to this failure mode because reporting is largely voluntary and self-selected. Successful companies and funds have every incentive to publicize their numbers; companies that shut down quietly often just stop updating their public profiles rather than formally recording a failure, and funds with weak returns are far less eager to publish them than funds with strong ones. The practical effect is that any dataset built by scraping public company/fund profiles - rather than by tracking a cohort from the moment it was founded, regardless of outcome - will systematically under-represent failures relative to their true frequency.

## The Specific Mechanism in This Project's Dataset

This project's own `DATA_CARD.md` documents exactly this pattern in the underlying Crunchbase-derived dataset: of 66,368 raw company records, 53,034 (74.1%) are `operating` (unresolved, and correctly excluded from the label rather than mislabeled as failures - see `why_startups_fail.md`), and only 6,238 are explicitly `closed`. Given how survivorship bias typically operates, it is very likely that this `closed` count is itself an undercount of true failures, not just a smaller-but-representative sample of them - a company that quietly stops operating without a formal shutdown announcement, or that was never prominent enough to get its status updated, has no natural way to show up as `closed` in this kind of scraped dataset. The practical consequence: any success rate computed from this dataset's `closed` vs. `acquired`/`ipo` split should be read as optimistic relative to the true rate among all companies ever founded, not as a calibrated real-world base rate.
