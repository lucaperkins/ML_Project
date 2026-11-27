# Contributions – ownership matrix

One owner per module (the brief requires 1 owner per student), plus a second member who supports the module and reviews its key decisions. The owner is responsible for the module and defends it at the oral. Allocation agreed by the team on 25 November 2026.

| Module | Owner | Support and review | Main files |
|---|---|---|---|
| 1. Data & EDA (loading, quality, duplicates, data preparation, error analysis, feature selection and documentation of choices) | Faizan NORAT | Océane de Gentile | `notebooks/01_eda_simple.ipynb`, `notebooks/07_error_analysis.ipynb`, `notebooks/08_feature_selection.ipynb`, `results/error_analysis/`, `results/feature_selection/` |
| 2. Baselines & AutoML (TF-IDF, logistic regression, SVM, cross-validation, thresholds) | Luca PERKINS | Faizan NORAT (data preparation) | |
| 3. Transformer & improvements (BERT-type fine-tuning, imbalance, comparison with classical models) | Océane de Gentile | Luca PERKINS (comparative experiments) | |
| 4. Evaluation & demo (metrics, Ekman/sentiment groupings, error analysis, demo, report coordination) | Myriam SAAID | everyone (report) | |

## Shared work

| Task | People |
|---|---|
| Report (LaTeX) | coordinated by Myriam, written by everyone |
| PROMPT_LOG.md / AI_AUDIT.md (≥ 1 entry each per student) | everyone |
| Oral defense | everyone |

## Log

| Date | Who | What |
|---|---|---|
| 2026-11-25 | Team | Module allocation agreed |
| 2026-11-25 | Faizan NORAT | Reviewed data preparation, added leakage assertions, investigated duplicate label disagreements and Neutral co-occurrences, and validated the exported Parquet files. |
| 2026-11-27 | Faizan NORAT | Developed the multi-label error analysis notebook (`07_error_analysis.ipynb`), including per-label false positives and false negatives, normalized error association matrices and interpretation of confusion patterns. |
| 2026-11-27 | Faizan NORAT | Conducted an AI-assisted qualitative analysis of 50 misclassified validation comments and documented potential sources of error, including ambiguity, implicit emotions and annotation uncertainty. |
| 2026-11-27 | Faizan NORAT | Compared baseline and ensemble prediction errors. Identified that Soft Voting reduces the number of misclassified comments by 197 and improves F1 for 22 of the 28 labels. |
| 2026-11-27 | Faizan NORAT | Reviewed the stacking implementation for potential data leakage and documented the methodological limitation related to hyperparameter selection before OOF prediction generation. |
| 2026-11-27 | Faizan NORAT | Implemented multi-label chi-square feature selection (`08_feature_selection.ipynb`) and evaluated five configurations. Selecting 5,000 features achieved a validation macro-F1 of 0.4825 while reducing the number of features by 67.2%. |
| 2026-11-27 | Faizan NORAT | Regenerated the final preprocessed datasets, verified the expected split sizes and re-executed both Week 3 notebooks. Exported experimental results and visualizations. |
