"""Evaluation metrics and JSON-ready reports for NumPyForge."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from src.math import clip_probabilities
from src.types import FloatArray, IntArray, RawArrayLike
from src.validation import check_target_vector

AverageMode = Literal["binary", "macro", "micro", "weighted"]


def accuracy_score(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return classification accuracy."""
    targets, predictions = _classification_arrays(y_true, y_pred)
    return float(np.mean(targets == predictions))


def confusion_matrix(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    *,
    labels: RawArrayLike | None = None,
) -> IntArray:
    """Return a confusion matrix with rows as true labels and columns as predictions."""
    targets, predictions = _classification_arrays(y_true, y_pred)
    class_labels = (
        np.asarray(labels, dtype=np.int_)
        if labels is not None
        else np.unique(np.concatenate((targets, predictions)))
    )
    matrix = np.zeros((class_labels.shape[0], class_labels.shape[0]), dtype=np.int_)
    index = {int(label): position for position, label in enumerate(class_labels)}

    for target, prediction in zip(targets, predictions, strict=True):
        matrix[index[int(target)], index[int(prediction)]] += 1

    return matrix


def precision_score(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    *,
    positive_label: int = 1,
    average: AverageMode = "binary",
    zero_division: float = 0.0,
) -> float:
    """Return precision for binary or multiclass classification."""
    stats = _per_class_stats(y_true, y_pred)
    return _average_metric(
        stats,
        average=average,
        positive_label=positive_label,
        zero_division=zero_division,
        metric="precision",
    )


def recall_score(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    *,
    positive_label: int = 1,
    average: AverageMode = "binary",
    zero_division: float = 0.0,
) -> float:
    """Return recall for binary or multiclass classification."""
    stats = _per_class_stats(y_true, y_pred)
    return _average_metric(
        stats,
        average=average,
        positive_label=positive_label,
        zero_division=zero_division,
        metric="recall",
    )


def f1_score(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    *,
    positive_label: int = 1,
    average: AverageMode = "binary",
    zero_division: float = 0.0,
) -> float:
    """Return F1-score for binary or multiclass classification."""
    stats = _per_class_stats(y_true, y_pred)
    if average not in {"binary", "macro", "micro", "weighted"}:
        raise ValueError("average must be 'binary', 'macro', 'micro', or 'weighted'.")

    if average == "binary":
        if positive_label not in stats:
            raise ValueError("positive_label is not present in y_true or y_pred.")
        return _f1_from_stats(stats[positive_label], zero_division=zero_division)

    if average == "micro":
        precision = _average_metric(
            stats,
            average="micro",
            positive_label=positive_label,
            zero_division=zero_division,
            metric="precision",
        )
        recall = _average_metric(
            stats,
            average="micro",
            positive_label=positive_label,
            zero_division=zero_division,
            metric="recall",
        )
        if precision + recall == 0.0:
            return zero_division
        return 2.0 * precision * recall / (precision + recall)

    scores = [_f1_from_stats(values, zero_division=zero_division) for values in stats.values()]
    if average == "macro":
        return float(np.mean(scores))

    supports = np.array([values["support"] for values in stats.values()], dtype=np.float64)
    if float(np.sum(supports)) == 0.0:
        return zero_division
    return float(np.average(np.array(scores, dtype=np.float64), weights=supports))


def log_loss(
    y_true: RawArrayLike, y_proba: RawArrayLike, *, labels: RawArrayLike | None = None
) -> float:
    """Return cross-entropy log loss for binary or multiclass probabilities."""
    targets = cast(IntArray, check_target_vector(y_true).astype(np.int_))
    probabilities = np.asarray(y_proba, dtype=np.float64)

    if probabilities.ndim == 1:
        probabilities = np.column_stack((1.0 - probabilities, probabilities))

    class_labels = np.asarray(labels, dtype=np.int_) if labels is not None else np.unique(targets)
    label_to_index = {int(label): index for index, label in enumerate(class_labels)}
    target_indices = np.array([label_to_index[int(label)] for label in targets], dtype=np.int_)
    clipped = clip_probabilities(cast(FloatArray, probabilities))
    return float(-np.mean(np.log(clipped[np.arange(targets.shape[0]), target_indices])))


def roc_curve(
    y_true: RawArrayLike,
    y_score: RawArrayLike,
    *,
    positive_label: int = 1,
) -> dict[str, list[float]]:
    """Return binary ROC curve points as JSON-ready lists."""
    targets, scores = _binary_curve_arrays(y_true, y_score, positive_label=positive_label)
    order = np.argsort(-scores, kind="mergesort")
    targets = targets[order]
    scores = scores[order]
    positives = float(np.sum(targets == 1))
    negatives = float(np.sum(targets == 0))
    if positives == 0.0 or negatives == 0.0:
        raise ValueError("ROC curve requires both positive and negative samples.")

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [_initial_curve_threshold(scores)]
    tp = 0.0
    fp = 0.0

    for target, score in zip(targets, scores, strict=True):
        if target == 1:
            tp += 1.0
        else:
            fp += 1.0
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        thresholds.append(float(score))

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def roc_auc_score(y_true: RawArrayLike, y_score: RawArrayLike, *, positive_label: int = 1) -> float:
    """Return binary ROC-AUC using trapezoidal integration."""
    curve = roc_curve(y_true, y_score, positive_label=positive_label)
    return float(np.trapezoid(np.asarray(curve["tpr"]), np.asarray(curve["fpr"])))


def precision_recall_curve(
    y_true: RawArrayLike,
    y_score: RawArrayLike,
    *,
    positive_label: int = 1,
) -> dict[str, list[float]]:
    """Return binary precision-recall curve points as JSON-ready lists."""
    targets, scores = _binary_curve_arrays(y_true, y_score, positive_label=positive_label)
    order = np.argsort(-scores, kind="mergesort")
    targets = targets[order]
    scores = scores[order]
    positives = float(np.sum(targets == 1))
    if positives == 0.0:
        raise ValueError("Precision-recall curve requires at least one positive sample.")

    precision = [1.0]
    recall = [0.0]
    thresholds = [_initial_curve_threshold(scores)]
    tp = 0.0
    fp = 0.0

    for target, score in zip(targets, scores, strict=True):
        if target == 1:
            tp += 1.0
        else:
            fp += 1.0
        precision.append(tp / (tp + fp))
        recall.append(tp / positives)
        thresholds.append(float(score))

    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def pr_auc_score(y_true: RawArrayLike, y_score: RawArrayLike, *, positive_label: int = 1) -> float:
    """Return binary PR-AUC using recall-ordered trapezoidal integration."""
    curve = precision_recall_curve(y_true, y_score, positive_label=positive_label)
    return float(np.trapezoid(np.asarray(curve["precision"]), np.asarray(curve["recall"])))


def mean_absolute_error(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return mean absolute error."""
    targets, predictions = _regression_arrays(y_true, y_pred)
    return float(np.mean(np.abs(targets - predictions)))


def mean_squared_error(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return mean squared error."""
    targets, predictions = _regression_arrays(y_true, y_pred)
    return float(np.mean((targets - predictions) ** 2))


def root_mean_squared_error(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return coefficient of determination, R2."""
    targets, predictions = _regression_arrays(y_true, y_pred)
    residual_sum = float(np.sum((targets - predictions) ** 2))
    total_sum = float(np.sum((targets - np.mean(targets)) ** 2))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return 1.0 - residual_sum / total_sum


def adjusted_r2_score(y_true: RawArrayLike, y_pred: RawArrayLike, *, n_features: int) -> float:
    """Return adjusted R2."""
    targets = check_target_vector(y_true)
    n_samples = targets.shape[0]
    if n_samples <= n_features + 1:
        raise ValueError("adjusted R2 requires n_samples > n_features + 1.")
    score = r2_score(y_true, y_pred)
    return 1.0 - (1.0 - score) * (n_samples - 1) / (n_samples - n_features - 1)


def explained_variance_score(y_true: RawArrayLike, y_pred: RawArrayLike) -> float:
    """Return explained variance score."""
    targets, predictions = _regression_arrays(y_true, y_pred)
    target_variance = float(np.var(targets))
    if target_variance == 0.0:
        return 1.0 if float(np.var(targets - predictions)) == 0.0 else 0.0
    return 1.0 - float(np.var(targets - predictions)) / target_variance


def classification_report_dict(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    y_score: RawArrayLike | None = None,
) -> dict[str, object]:
    """Return JSON-ready classification metrics."""
    report: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        report["roc_auc"] = roc_auc_score(y_true, y_score)
        report["pr_auc"] = pr_auc_score(y_true, y_score)
        report["roc_curve"] = roc_curve(y_true, y_score)
        report["precision_recall_curve"] = precision_recall_curve(y_true, y_score)
    return report


def regression_report_dict(
    y_true: RawArrayLike,
    y_pred: RawArrayLike,
    *,
    n_features: int | None = None,
) -> dict[str, float]:
    """Return JSON-ready regression metrics."""
    report = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "explained_variance": explained_variance_score(y_true, y_pred),
    }
    if n_features is not None:
        report["adjusted_r2"] = adjusted_r2_score(y_true, y_pred, n_features=n_features)
    return report


def _classification_arrays(y_true: RawArrayLike, y_pred: RawArrayLike) -> tuple[IntArray, IntArray]:
    targets = cast(IntArray, check_target_vector(y_true).astype(np.int_))
    predictions = cast(
        IntArray, check_target_vector(y_pred, n_samples=targets.shape[0]).astype(np.int_)
    )
    return targets, predictions


def _regression_arrays(y_true: RawArrayLike, y_pred: RawArrayLike) -> tuple[FloatArray, FloatArray]:
    targets = cast(FloatArray, check_target_vector(y_true).astype(np.float64))
    predictions = cast(
        FloatArray, check_target_vector(y_pred, n_samples=targets.shape[0]).astype(np.float64)
    )
    return targets, predictions


def _binary_curve_arrays(
    y_true: RawArrayLike,
    y_score: RawArrayLike,
    *,
    positive_label: int,
) -> tuple[IntArray, FloatArray]:
    targets = cast(IntArray, check_target_vector(y_true).astype(np.int_))
    scores = cast(
        FloatArray, check_target_vector(y_score, n_samples=targets.shape[0]).astype(np.float64)
    )
    binary_targets = (targets == positive_label).astype(np.int_)
    if np.unique(binary_targets).shape[0] != 2:
        raise ValueError("Binary curve metrics require both classes.")
    return cast(IntArray, binary_targets), scores


def _initial_curve_threshold(scores: FloatArray) -> float:
    """Return a finite threshold just above the highest score."""
    max_score = float(np.max(scores))
    return max_score + max(1.0, abs(max_score)) * float(np.finfo(np.float64).eps)


def _per_class_stats(y_true: RawArrayLike, y_pred: RawArrayLike) -> dict[int, dict[str, float]]:
    targets, predictions = _classification_arrays(y_true, y_pred)
    labels = np.unique(np.concatenate((targets, predictions)))
    stats: dict[int, dict[str, float]] = {}
    for label in labels:
        label_int = int(label)
        true_positive = float(np.sum((targets == label) & (predictions == label)))
        false_positive = float(np.sum((targets != label) & (predictions == label)))
        false_negative = float(np.sum((targets == label) & (predictions != label)))
        support = float(np.sum(targets == label))
        stats[label_int] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "support": support,
        }
    return stats


def _average_metric(
    stats: dict[int, dict[str, float]],
    *,
    average: AverageMode,
    positive_label: int,
    zero_division: float,
    metric: Literal["precision", "recall"],
) -> float:
    if average not in {"binary", "macro", "micro", "weighted"}:
        raise ValueError("average must be 'binary', 'macro', 'micro', or 'weighted'.")

    if average == "binary":
        if positive_label not in stats:
            raise ValueError("positive_label is not present in y_true or y_pred.")
        return _precision_or_recall_from_stats(
            stats[positive_label], metric=metric, zero_division=zero_division
        )

    if average == "micro":
        tp = sum(values["tp"] for values in stats.values())
        denominator = sum(
            values["tp"] + (values["fp"] if metric == "precision" else values["fn"])
            for values in stats.values()
        )
        return zero_division if denominator == 0.0 else tp / denominator

    scores = [
        _precision_or_recall_from_stats(values, metric=metric, zero_division=zero_division)
        for values in stats.values()
    ]
    if average == "macro":
        return float(np.mean(scores))

    supports = np.array([values["support"] for values in stats.values()], dtype=np.float64)
    if float(np.sum(supports)) == 0.0:
        return zero_division
    return float(np.average(np.array(scores, dtype=np.float64), weights=supports))


def _precision_or_recall_from_stats(
    values: dict[str, float],
    *,
    metric: Literal["precision", "recall"],
    zero_division: float,
) -> float:
    denominator = (
        values["tp"] + values["fp"] if metric == "precision" else values["tp"] + values["fn"]
    )
    if denominator == 0.0:
        return zero_division
    return values["tp"] / denominator


def _f1_from_stats(values: dict[str, float], *, zero_division: float) -> float:
    precision = _precision_or_recall_from_stats(
        values, metric="precision", zero_division=zero_division
    )
    recall = _precision_or_recall_from_stats(values, metric="recall", zero_division=zero_division)
    if precision + recall == 0.0:
        return zero_division
    return 2.0 * precision * recall / (precision + recall)
