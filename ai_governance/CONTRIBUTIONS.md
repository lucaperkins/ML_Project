# Contributions – ownership matrix

One owner per module (the brief requires 1 owner per student), plus a second member who supports the module and reviews its key decisions. The owner is responsible for the module and defends it at the oral. Allocation agreed by the team on 25 November 2026.

| Module | Owner | Support and review | Main files |
|---|---|---|---|
| 1. Data & EDA (loading, quality, duplicates, data preparation, error analysis, feature selection and documentation of choices) | Faizan NORAT | Océane de Gentile | `notebooks/01_eda_simple.ipynb`, `notebooks/07_error_analysis.ipynb`, `notebooks/08_feature_selection.ipynb`, `results/error_analysis/`, `results/feature_selection/` |
| 2. Baselines & AutoML (TF-IDF, logistic regression, SVM, cross-validation, thresholds, grid search on trees, ensembles) | Luca PERKINS | Faizan NORAT (data preparation) | `notebooks/03_baseline.ipynb`, `notebooks/05_ensembles.ipynb`, `results/baselines/`, `results/ensembles/` |
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
| 2026-11-19 | Luca PERKINS | Set up the repository structure required by the brief (`requirements.txt`, `notebooks/`, `report/`, `ai_governance/`), README and first EDA notebook (`01_eda_simple.ipynb`). |
| 2026-11-25 | Luca PERKINS | Added the agreed text cleaning to the data preparation (emojis converted to text, repeated letters shortened, before duplicate removal) and the `emoji` dependency. |
| 2026-11-25 | Luca PERKINS | Built the baselines (`03_baseline.ipynb`): trivial references, TF-IDF + logistic regression and linear SVM with a tokenizer keeping punctuation, capitals and emojis, AutoML v0.1 grid search (LR, SVM, naive Bayes) by cross-validation on train, cross-fitted per-label thresholds. Retained baseline: validation macro-F1 0.477. Wrote section 11 of the week 2 report. |
| 2026-11-27 | Luca PERKINS | Lab 6 part II (`05_ensembles.ipynb`): multi-label stratified folds, grid search on decision tree and random forest (bias–variance analysis), soft voting, bagging and stacking with out-of-fold predictions, cost and interpretability table. Best classical model: soft voting, macro-F1 0.495. |
| 2026-11-27 | Luca PERKINS | Combined the transformer of module 3 with the classical models (`05_ensembles.ipynb`, section 8): transformer + baseline reaches macro-F1 0.538, the best model of the project. Saved the robustness-suite predictions of the classical models. |
| 2026-11-27 | Myriam SAAID | Covered module 4: `09_evaluation.ipynb` with the comparison of the four main models at the three label levels, calibration (Brier score, ECE, reliability diagrams, cross-fitted isotonic recalibration) and the robustness suite (Wilson intervals). Wrote the week 3 report sections. |
