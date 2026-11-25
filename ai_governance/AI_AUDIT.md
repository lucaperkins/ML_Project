# AI audit – "Catch & Correct"

At least 4 concrete flaws found in AI outputs, **at least one per student**. Claiming the AI made no error scores 0/5.
Typical categories from the rubric: data leakage, wrong metric for imbalanced data, violated statistical assumptions, misuse of the test set, wrong or invented facts, code that runs but computes the wrong thing.

---

## Flaw model
- **Found by:**
- **Date:**
- **Where:** file / notebook cell / report section
- **What the AI produced:** short quote or code excerpt
- **Why it is wrong:** the concrete consequence (e.g. inflated validation score, biased estimate)
- **How we detected it:** check, test or reasoning used
- **Fix:** what we changed (link to commit or cell)
- **Impact:** metric before → after, if measurable

---

## Flaw 1 – Incorrect duplicate statistics

- **Found by:** Faizan NORAT
- **Date:** 2026-11-25
- **Where:** `notebooks/01_eda_simple.ipynb` – Data preparation
- **What the AI produced:** The initial documentation stated that 62 groups of duplicate texts had different emotion annotations.
- **Why it is wrong:** The figure was not supported by an explicit calculation.
- **How we detected it:** We grouped comments by normalized text and counted distinct label combinations. The calculation returned 61 groups.
- **Fix:** Added a reproducible check in Section 11.2 and corrected the documentation.
- **Impact:** The reported statistic is now verified and reproducible.
