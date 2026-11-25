"""
Evaluation tools for the GoEmotions project (module 4 - Evaluation & demo).

Every model (TF-IDF baselines, transformer, ensemble) is scored with the same
functions so that the results are comparable.

Conventions
- y_true, y_pred: 0/1 arrays of shape (n_comments, n_labels) (multi-hot).
- y_proba: float array of the same shape, one probability per label.
- Thresholds are tuned on the VALIDATION set only. The test set is used once,
  for the final numbers.

Usage from a notebook in notebooks/:
    import sys; sys.path.append("../src")
    from evaluation import *
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Labels at the three levels
# ---------------------------------------------------------------------------

# same order as the Hugging Face dataset (label id = position in this list)
LABELS_28 = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
             "confusion", "curiosity", "desire", "disappointment", "disapproval",
             "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
             "joy", "love", "nervousness", "optimism", "pride", "realization",
             "relief", "remorse", "sadness", "surprise", "neutral"]

# official mappings (google-research/goemotions: ekman_mapping.json, sentiment_mapping.json)
EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
SENTIMENT_MAPPING = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring",
                 "pride", "admiration", "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment",
                 "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
}

LABELS_EKMAN = sorted(EKMAN_MAPPING) + ["neutral"]
LABELS_SENTIMENT = sorted(SENTIMENT_MAPPING) + ["neutral"]

# column name in data/processed/*.parquet -> list of classes
LEVELS = {
    "labels_28": LABELS_28,
    "labels_ekman": LABELS_EKMAN,
    "labels_sentiment": LABELS_SENTIMENT,
}


def to_multi_hot(label_lists, classes):
    """List of label-name lists -> 0/1 array (n_comments, n_classes), columns in `classes` order."""
    index = {c: j for j, c in enumerate(classes)}
    y = np.zeros((len(label_lists), len(classes)), dtype = int)
    for i, labels in enumerate(label_lists):
        for lab in labels:
            y[i, index[lab]] = 1
    return y


def group_matrix(mapping, classes_from = LABELS_28):
    """0/1 matrix (n_classes_from, n_groups): 1 when the emotion belongs to the group. Neutral stays neutral."""
    groups = sorted(mapping) + ["neutral"]
    M = np.zeros((len(classes_from), len(groups)), dtype = int)
    for i, emotion in enumerate(classes_from):
        if emotion == "neutral":
            M[i, groups.index("neutral")] = 1
        for g, members in mapping.items():
            if emotion in members:
                M[i, groups.index(g)] = 1
    return M


def regroup_predictions(y_28, mapping):
    """28-label 0/1 predictions -> group predictions. A group is on when at least one of its emotions is on
    (same rule as the data preparation in 01_eda_simple.ipynb)."""
    return (np.asarray(y_28) @ group_matrix(mapping) > 0).astype(int)


def regroup_proba(proba_28, mapping):
    """28-label probabilities -> group scores, taking the max over the emotions of each group.
    The result is a score, not a calibrated probability: tune new thresholds on validation for it."""
    M = group_matrix(mapping).astype(bool)
    return np.column_stack([np.asarray(proba_28)[:, M[:, g]].max(axis = 1) for g in range(M.shape[1])])


# ---------------------------------------------------------------------------
# From probabilities to labels
# ---------------------------------------------------------------------------

def tune_thresholds(y_val, proba_val, grid = np.arange(0.05, 0.96, 0.01)):
    """One threshold per class, chosen to maximise that class's F1 on the VALIDATION set."""
    y_val = np.asarray(y_val)
    proba_val = np.asarray(proba_val)
    thresholds = np.zeros(y_val.shape[1])
    for j in range(y_val.shape[1]):
        best_f1, best_t = -1, 0.5
        for t in grid:
            f1 = f1_score(y_val[:, j], (proba_val[:, j] >= t).astype(int), zero_division = 0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


def apply_thresholds(proba, thresholds, at_least_one = True):
    """Probabilities -> 0/1 predictions.
    at_least_one: every GoEmotions comment has at least one label, so when no class passes its
    threshold we predict the class with the highest probability."""
    proba = np.asarray(proba)
    y_pred = (proba >= thresholds).astype(int)
    if at_least_one:
        empty = y_pred.sum(axis = 1) == 0
        y_pred[empty, proba[empty].argmax(axis = 1)] = 1
    return y_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Main metrics. Macro-F1 is the primary metric (every emotion counts the same)."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average = "macro", zero_division = 0),
        "micro_f1": f1_score(y_true, y_pred, average = "micro", zero_division = 0),
        "weighted_f1": f1_score(y_true, y_pred, average = "weighted", zero_division = 0),
        "exact_match": float((np.asarray(y_true) == np.asarray(y_pred)).all(axis = 1).mean()),
    }


def per_class_report(y_true, y_pred, classes):
    """Precision, recall, F1 and support for every class, sorted by F1."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average = None, zero_division = 0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index = classes)
    return report.sort_values("f1", ascending = False).round(3)


def f1_from_counts(tp, fp, fn, average):
    if average == "micro":
        tp, fp, fn = tp.sum(axis = -1), fp.sum(axis = -1), fn.sum(axis = -1)
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(axis = -1)


def _bootstrap_counts(y_true, y_pred, n_boot, seed):
    """TP / FP / FN per class for each bootstrap resample of the comments, shape (n_boot, n_classes)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    # weights[b, i] = how many times comment i is drawn in resample b
    weights = np.stack([np.bincount(rng.randint(0, n, n), minlength = n) for _ in range(n_boot)])
    tp = weights @ (y_true & y_pred)
    fp = weights @ (~y_true & y_pred)
    fn = weights @ (y_true & ~y_pred)
    return tp, fp, fn


def bootstrap_ci(y_true, y_pred, average = "macro", n_boot = 1000, seed = 0, alpha = 0.05):
    """F1 with a bootstrap (1 - alpha) confidence interval, resampling comments with replacement.
    Returns (value on the full set, lower bound, upper bound)."""
    tp, fp, fn = _bootstrap_counts(y_true, y_pred, n_boot, seed)
    scores = f1_from_counts(tp, fp, fn, average)
    value = f1_score(y_true, y_pred, average = average, zero_division = 0)
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return value, low, high


def paired_bootstrap(y_true, y_pred_a, y_pred_b, average = "macro", n_boot = 1000, seed = 0, alpha = 0.05):
    """Is model B better than model A on the same comments?
    The same resamples are used for both models (paired), which is fairer than comparing two separate CIs.
    Returns a dict: difference B - A on the full set, its CI, and the share of resamples where B <= A."""
    tp_a, fp_a, fn_a = _bootstrap_counts(y_true, y_pred_a, n_boot, seed)
    tp_b, fp_b, fn_b = _bootstrap_counts(y_true, y_pred_b, n_boot, seed)  # same seed -> same resamples
    diff = f1_from_counts(tp_b, fp_b, fn_b, average) - f1_from_counts(tp_a, fp_a, fn_a, average)
    value = (f1_score(y_true, y_pred_b, average = average, zero_division = 0)
             - f1_score(y_true, y_pred_a, average = average, zero_division = 0))
    low, high = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff_b_minus_a": float(value), "ci_low": float(low), "ci_high": float(high),
            "share_b_not_better": float((diff <= 0).mean())}


def evaluate(name, y_true, y_pred, n_boot = 1000, seed = 0):
    """One row of the results table: main metrics + bootstrap 95% CI on macro- and micro-F1."""
    row = {"model": name, **compute_metrics(y_true, y_pred)}
    _, row["macro_f1_low"], row["macro_f1_high"] = bootstrap_ci(y_true, y_pred, "macro", n_boot, seed)
    _, row["micro_f1_low"], row["micro_f1_high"] = bootstrap_ci(y_true, y_pred, "micro", n_boot, seed)
    return row


def evaluate_all_levels(name, y_pred_28, y_true_by_level, n_boot = 1000, seed = 0):
    """Score 28-label predictions at the three levels (28, Ekman, sentiment).
    y_true_by_level: dict {"labels_28": ..., "labels_ekman": ..., "labels_sentiment": ...}.
    The Ekman / sentiment predictions are derived from the 28-label ones (regroup_predictions)."""
    preds = {
        "labels_28": np.asarray(y_pred_28),
        "labels_ekman": regroup_predictions(y_pred_28, EKMAN_MAPPING),
        "labels_sentiment": regroup_predictions(y_pred_28, SENTIMENT_MAPPING),
    }
    rows = []
    for level, y_pred in preds.items():
        row = evaluate(name, y_true_by_level[level], y_pred, n_boot, seed)
        row["level"] = level.replace("labels_", "")
        rows.append(row)
    return pd.DataFrame(rows)


def format_results(df):
    """Readable results table: 'macro-F1 [low, high]'."""
    out = df.copy()
    for m in ["macro_f1", "micro_f1"]:
        if f"{m}_low" in out:
            out[m] = out.apply(lambda r: f"{r[m]:.3f} [{r[m + '_low']:.3f}, {r[m + '_high']:.3f}]", axis = 1)
            out = out.drop(columns = [f"{m}_low", f"{m}_high"])
    return out.round(3)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def brier_per_label(y_true, y_proba, classes):
    """Brier score per label (mean squared error between probability and 0/1 label, lower is better)."""
    brier = ((np.asarray(y_proba) - np.asarray(y_true)) ** 2).mean(axis = 0)
    return pd.Series(brier, index = classes, name = "brier").sort_values(ascending = False).round(4)


def expected_calibration_error(y_true, y_proba, n_bins = 10):
    """ECE per label (average gap between predicted probability and observed frequency, weighted by bin size)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = np.zeros(y_true.shape[1])
    for j in range(y_true.shape[1]):
        idx = np.clip(np.digitize(y_proba[:, j], bins) - 1, 0, n_bins - 1)
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                ece[j] += mask.mean() * abs(y_proba[mask, j].mean() - y_true[mask, j].mean())
    return ece


def plot_reliability(y_true, y_proba, classes, labels_to_plot, n_bins = 10, path = None):
    """Reliability diagram for a few labels: predicted probability vs observed frequency."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    fig, ax = plt.subplots(figsize = (5, 5))
    ax.plot([0, 1], [0, 1], "--", color = "grey", label = "perfect calibration")
    for lab in labels_to_plot:
        j = classes.index(lab)
        frac_pos, mean_pred = calibration_curve(y_true[:, j], y_proba[:, j], n_bins = n_bins, strategy = "quantile")
        ax.plot(mean_pred, frac_pos, marker = "o", label = lab)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)
    return fig


def fit_calibrators(y_val, proba_val, method = "isotonic"):
    """One calibrator per label, fitted on the VALIDATION set ('isotonic' or 'platt').
    Warning: if the thresholds are also tuned on validation, tune them AFTER calibration, and keep the
    test set untouched for the final check."""
    y_val = np.asarray(y_val)
    proba_val = np.asarray(proba_val)
    calibrators = []
    for j in range(y_val.shape[1]):
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds = "clip").fit(proba_val[:, j], y_val[:, j])
        else:
            cal = LogisticRegression().fit(proba_val[:, [j]], y_val[:, j])
        calibrators.append(cal)
    return calibrators


def apply_calibrators(calibrators, proba):
    proba = np.asarray(proba)
    cols = []
    for j, cal in enumerate(calibrators):
        if isinstance(cal, IsotonicRegression):
            cols.append(cal.predict(proba[:, j]))
        else:
            cols.append(cal.predict_proba(proba[:, [j]])[:, 1])
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_per_class_f1(report, title, path = None):
    """Horizontal bar chart of F1 per class (report from per_class_report)."""
    report = report.sort_values("f1")
    fig, ax = plt.subplots(figsize = (6, max(3, 0.25 * len(report))))
    ax.barh(report.index, report["f1"], color = "#2a78d6")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    return fig


def plot_f1_vs_support(report, title, path = None):
    """F1 against the number of examples of each class: shows how much rare classes suffer."""
    fig, ax = plt.subplots(figsize = (6, 4))
    ax.scatter(report["support"], report["f1"], color = "#2a78d6")
    for lab, r in report.iterrows():
        ax.annotate(lab, (r["support"], r["f1"]), fontsize = 7)
    ax.set_xscale("log")
    ax.set_xlabel("Support (log scale)")
    ax.set_ylabel("F1")
    ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    return fig


# ---------------------------------------------------------------------------
# Robustness suite (objective 5)
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z = 1.96):
    """Wilson 95% interval for a proportion (better than the normal one for small n, e.g. 20 messages)."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def evaluate_robustness(suite, predicted_labels):
    """suite: DataFrame with columns text, category, expected (labels separated by ';').
    predicted_labels: one list of predicted label names per row of the suite.
    A message is a hit when at least one expected label is predicted.
    Returns the per-message table and a summary per category with a Wilson 95% CI."""
    detail = suite.copy()
    detail["predicted"] = [";".join(p) for p in predicted_labels]
    detail["hit"] = [bool(set(e.split(";")) & set(p)) for e, p in zip(suite["expected"], predicted_labels)]
    rows = []
    for cat, g in list(detail.groupby("category")) + [("ALL", detail)]:
        low, high = wilson_ci(g["hit"].sum(), len(g))
        rows.append({"category": cat, "n": len(g), "hit_rate": g["hit"].mean(), "ci_low": low, "ci_high": high})
    return detail, pd.DataFrame(rows).round(3)
