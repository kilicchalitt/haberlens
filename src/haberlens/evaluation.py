"""Model değerlendirme metrikleri."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

__all__ = ["evaluate_predictions", "format_report"]


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, object]:
    """Doğruluk, makro F1, sınıf bazlı rapor ve karışıklık matrisi üretir."""
    labels = list(labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), 4),
            "recall": round(float(report[label]["recall"]), 4),
            "f1": round(float(report[label]["f1-score"]), 4),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": np.asarray(matrix).tolist()},
    }


def format_report(metrics: dict[str, object]) -> str:
    """Metrikleri terminalde okunabilir bir tablo olarak biçimlendirir."""
    lines = [
        f"Doğruluk (accuracy) : %{metrics['accuracy'] * 100:.2f}",
        f"Makro F1            : %{metrics['f1_macro'] * 100:.2f}",
        "",
        f"{'Kategori':<12} {'Kesinlik':>9} {'Duyarlılık':>11} {'F1':>7} {'Örnek':>7}",
        "-" * 50,
    ]
    for label, row in metrics["per_class"].items():
        lines.append(
            f"{label:<12} {row['precision']:>9.3f} {row['recall']:>11.3f} "
            f"{row['f1']:>7.3f} {row['support']:>7d}"
        )
    return "\n".join(lines)
