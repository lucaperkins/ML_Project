# AI audit – "Catch & Correct"

At least 4 concrete flaws found in AI outputs, **at least one per student**. Claiming the AI made no error scores 0/5.
Typical categories from the rubric: data leakage, wrong metric for imbalanced data, violated statistical assumptions, misuse of the test set, wrong or invented facts, code that runs but computes the wrong thing.

---

## Flaw 1

- **Found by:** Océane de Gentile
- **Date:** 2026-11-25
- **Where:** `notebooks/04_transformer.ipynb`, section 5 (choice of the tokenizer)
- **What the AI produced:** the justification "Cased keeps capital letters, so the signal is kept", used to choose `distilbert-base-cased`.
- **Why it is wrong:** the claim is only half true. The cased vocabulary contains "why" and "Why" but not "WHY", so words fully written in capitals are split into meaningless sub-tokens ("WHY" -> W, ##H, ##Y; "FUCK" -> F, ##UC, ##K). The model sees that the text is shouted but partly loses the meaning of the word.
- **How we detected it:** we printed the tokenization of examples taken from the EDA instead of trusting the explanation.
- **Fix:** the justification is rewritten as a trade-off (keeps the capital-letter signal, fragments full-capital words), and cased vs uncased will be compared experimentally in step 3.
- **Impact:** no metric yet; it prevents an unsupported claim in the report and adds an experiment to the plan.

---

## Flaw 2

- **Found by:** Myriam SAAID
- **Date:** 2026-11-25
- **Where:** `src/evaluation.py` and `notebooks/02_evaluation_tools.ipynb`, check 2.3
- **What the AI produced:** a helper named `_f1_from_counts`, imported in the notebook with `from evaluation import *`.
- **Why it is wrong:** Python's `import *` skips names starting with `_`, so the check comparing our bootstrap F1 with scikit-learn crashed with a `NameError`. Without this check, an error in the bootstrap F1 would have gone unnoticed and given wrong confidence intervals.
- **How we detected it:** by running the notebook end to end before using the functions.
- **Fix:** the function was renamed `f1_from_counts`.
- **Impact:** no metric change; the check now confirms the bootstrap F1 equals scikit-learn's exactly.

---

## Flaw 3 – Incorrect duplicate statistics

- **Found by:** Faizan NORAT
- **Date:** 2026-11-25
- **Where:** `notebooks/01_eda_simple.ipynb` – Data preparation
- **What the AI produced:** The initial documentation stated that 62 groups of duplicate texts had different emotion annotations.
- **Why it is wrong:** The figure was not supported by an explicit calculation.
- **How we detected it:** We grouped comments by normalized text and counted distinct label combinations. The calculation returned 61 groups under the previous lowercase-only normalization policy. The updated text-cleaning pipeline is checked separately.
- **Fix:** Added a reproducible check in Section 12.2 (previously 11.2) and corrected the documentation.
- **Impact:** The reported statistic is now verified and reproducible.
