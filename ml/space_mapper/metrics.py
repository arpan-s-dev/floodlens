from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _as_bool(array: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.bool_:
        return array
    return array.astype(np.float32) > threshold


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> ConfusionCounts:
    """Compute binary segmentation confusion counts."""

    true = _as_bool(y_true, threshold=threshold).ravel()
    pred = _as_bool(y_pred, threshold=threshold).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true has {true.shape}, y_pred has {pred.shape}")

    tp = int(np.logical_and(true, pred).sum())
    fp = int(np.logical_and(~true, pred).sum())
    tn = int(np.logical_and(~true, ~pred).sum())
    fn = int(np.logical_and(true, ~pred).sum())
    return ConfusionCounts(tp, fp, tn, fn)


def _safe_divide(numerator: float, denominator: float, *, empty_value: float = 1.0) -> float:
    return empty_value if denominator == 0 else numerator / denominator


def iou_from_counts(counts: ConfusionCounts) -> float:
    return _safe_divide(counts.true_positive, counts.true_positive + counts.false_positive + counts.false_negative)


def dice_from_counts(counts: ConfusionCounts) -> float:
    return _safe_divide(2 * counts.true_positive, 2 * counts.true_positive + counts.false_positive + counts.false_negative)


def precision_from_counts(counts: ConfusionCounts) -> float:
    return _safe_divide(counts.true_positive, counts.true_positive + counts.false_positive)


def recall_from_counts(counts: ConfusionCounts) -> float:
    return _safe_divide(counts.true_positive, counts.true_positive + counts.false_negative)


def binary_iou(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    return iou_from_counts(confusion_counts(y_true, y_pred, threshold=threshold))


def dice_f1(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    return dice_from_counts(confusion_counts(y_true, y_pred, threshold=threshold))


def precision(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    return precision_from_counts(confusion_counts(y_true, y_pred, threshold=threshold))


def recall(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> float:
    return recall_from_counts(confusion_counts(y_true, y_pred, threshold=threshold))


def metrics_from_counts(counts: ConfusionCounts) -> dict[str, float | int]:
    return {
        **counts.as_dict(),
        "iou": iou_from_counts(counts),
        "dice_f1": dice_from_counts(counts),
        "precision": precision_from_counts(counts),
        "recall": recall_from_counts(counts),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int]:
    return metrics_from_counts(confusion_counts(y_true, y_pred, threshold=threshold))
