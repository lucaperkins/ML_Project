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

---

## Flaw 4 – Text cleaning that silently changed 2,700 texts

- **Owner:** Luca PERKINS
- **Date:** 2026-11-25
- **Where:** `notebooks/01_eda_simple.ipynb`, section 11 (`clean_text`)
- **What the AI produced:** a cleaning function that converted emojis to text, shortened repeated letters, then ran `re.sub(r" {2,}", " ", text).strip()` on **every** text, to remove the double spaces created by the emoji conversion.
- **Why it is wrong:** the agreed cleaning was only emojis and repeated letters. The space step also rewrote texts that contain neither (original double spaces, leading or trailing spaces), so the cleaning changed more data than documented. It is harmless for the tokenizers, but the report would have stated the wrong number of modified texts and the step was not agreed by the team.
- **How we detected it:** the notebook counts the texts changed by the cleaning. It reported 3,892 changed train texts, while only about 2.8% of train comments (about 1,200) contain an emoji or a repeated letter. The gap pointed to a third, unintended change.
- **Fix:** the space clean-up now runs only when an emoji was converted. The count fell to 1,212 train, 153 validation and 158 test texts, consistent with the share of comments with emojis (1.94%) or repeated letters (0.88%).
- **Impact:** 2,680 train texts no longer modified; the documented cleaning now matches what the code does.

---

## Flaw 5 – Regex that crashes under pandas 3

- **Owner:** Luca PERKINS
- **Date:** 2026-11-25
- **Where:** `notebooks/01_eda_simple.ipynb`, section 11 (statistics of the cleaning)
- **What the AI produced:** `df["text_raw"].str.contains(r"([A-Za-z])\1{3,}")` to count comments with a letter repeated more than 3 times.
- **Why it is wrong:** since pandas 3, text columns use the PyArrow string engine, whose regex engine (RE2) does not support back-references such as `\1`. The cell raised `ArrowInvalid: invalid escape sequence: \1`. The same pattern works with Python's `re`, so the code looked correct and would have worked with pandas 2: an API assumption that is no longer true for our pinned version (`pandas==3.1.1`).
- **How we detected it:** running the notebook end to end before pushing it.
- **Fix:** `df["text_raw"].apply(lambda t: re.search(r"([A-Za-z])\1{3,}", t) is not None)`, with a comment explaining why `str.contains` is not used.
- **Impact:** no metric change; the notebook runs with the pinned versions of `requirements.txt`.

### Flaw 6 — Annotation file overwritten on rerun

**Author:** Faizan NORAT
**AI tool:** ChatGPT
**Week:** 3

**Problem:**
The initial AI-generated qualitative error analysis
code recreated error_examples.csv every time the
notebook was executed.

**Risk:**
Running all cells after annotating the comments
could overwrite the existing annotations.

**Correction:**
An os.path.exists() condition was added to preserve
the existing annotation file.

An empty-data condition was also added to prevent
plotting errors when no annotations are available.

**Verification:**
The notebook was executed again, and the existing
50 annotations were preserved.

**Lesson learned:**
AI-generated file export operations must be checked
for unintended overwriting and reproducibility issues.

---

## Flaw 7 – Random forest configuration that runs out of memory

- **Owner:** Luca PERKINS
- **Date:** 2026-11-27
- **Where:** `notebooks/05_ensembles.ipynb`, section 3.2 (grid search on the random forest)
- **What the AI produced:** a random forest with fully grown trees (`min_samples_leaf = 1`, 100 trees) on the TF-IDF features, the scikit-learn default, proposed as a candidate of the grid search.
- **Why it is wrong:** a fully grown tree keeps one leaf per few comments, and in a multi-label forest every node stores a value for each of the 28 labels. On 43,188 comments and 15,265 sparse features, the forest took several GB of memory: the process was killed on a 7 GB machine, so it would also crash an 8 GB laptop. It was also the worst setting: validation macro-F1 0.447 against 0.461 with `min_samples_leaf = 5` (single fit), i.e. more variance for no gain.
- **How we detected it:** the prototype run crashed with an out-of-memory kill (exit code 137) when the fitted forest was measured.
- **Fix:** `min_samples_leaf = 1` was removed from the grid, which now tests leaves of 3, 5 and 10 (about 60 MB for leaves of 5); the notebook explains why. The same bias–variance effect is shown on a single decision tree (section 3.1), where it is cheap to measure.
- **Impact:** the notebook runs on the team's laptops; the retained forest (leaves of 3) scores 0.447 alone and improves the soft vote (0.488 without it, 0.495 with it).
