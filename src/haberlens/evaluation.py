"""Model değerlendirme metrikleri."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

__all__ = ["evaluate_model", "evaluate_predictions", "format_report", "top_k_accuracy"]


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
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted")), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": np.asarray(matrix).tolist()},
    }


def top_k_accuracy(
    y_true: Sequence[str], probabilities: np.ndarray, classes: Sequence[str], k: int = 2
) -> float:
    """Doğru kategorinin en olası ``k`` tahmin arasında olma oranı."""
    classes = np.asarray(classes)
    top = classes[np.argsort(probabilities, axis=1)[:, -k:]]
    hits = [truth in row for truth, row in zip(y_true, top, strict=True)]
    return round(float(np.mean(hits)), 4)


def _subset_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    # Makro F1 yalnızca alt kümede gerçekten bulunan kategoriler üzerinden alınır;
    # aksi hâlde o kaynakta hiç olmayan kategoriler ortalamayı yapay olarak düşürür.
    present = sorted(set(y_true))
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=present, average="macro")), 4),
    }


def evaluate_model(
    model: Any,
    texts: pd.Series,
    y_true: pd.Series,
    labels: Sequence[str],
    *,
    sources: pd.Series | None = None,
    lead_words: int = 12,
) -> dict[str, Any]:
    """Eğitilmiş bir pipeline'ı kapsamlı olarak değerlendirir.

    Temel metriklere ek olarak:

    * ``top2_accuracy``: doğru kategorinin ilk iki öneride olma oranı
    * ``per_source``: her veri kaynağı için ayrı doğruluk ve makro F1
    * ``headline``: uzun test haberlerinin yalnızca ilk ``lead_words`` kelimesi
      verildiğinde (başlık benzeri kısa girdi) elde edilen başarı
    """
    texts = texts.reset_index(drop=True)
    y_true_arr = y_true.reset_index(drop=True).to_numpy()
    probabilities = model.predict_proba(texts)
    classes = np.asarray(model.classes_)
    predicted = classes[probabilities.argmax(axis=1)]

    metrics: dict[str, Any] = evaluate_predictions(y_true_arr, predicted, labels)
    metrics["top2_accuracy"] = top_k_accuracy(y_true_arr, probabilities, classes, k=2)

    if sources is not None:
        source_arr = sources.reset_index(drop=True).to_numpy()
        metrics["per_source"] = {
            str(name): _subset_scores(y_true_arr[source_arr == name], predicted[source_arr == name])
            for name in sorted(set(source_arr))
        }

    words = texts.str.split()
    long_mask = (words.str.len() > 2 * lead_words).to_numpy()
    if long_mask.sum() >= 10:
        leads = words[long_mask].str[:lead_words].str.join(" ")
        metrics["headline"] = {
            "lead_words": lead_words,
            **_subset_scores(y_true_arr[long_mask], np.asarray(model.predict(leads))),
        }
    return metrics


def format_report(metrics: dict[str, object]) -> str:
    """Metrikleri terminalde okunabilir bir tablo olarak biçimlendirir."""
    lines = [
        f"Doğruluk (accuracy) : %{metrics['accuracy'] * 100:.2f}",
        f"Makro F1            : %{metrics['f1_macro'] * 100:.2f}",
    ]
    if "top2_accuracy" in metrics:
        lines.append(f"İlk-2 doğruluk      : %{metrics['top2_accuracy'] * 100:.2f}")
    lines += [
        "",
        f"{'Kategori':<13} {'Kesinlik':>9} {'Duyarlılık':>11} {'F1':>7} {'Örnek':>7}",
        "-" * 51,
    ]
    for label, row in metrics["per_class"].items():
        lines.append(
            f"{label:<13} {row['precision']:>9.3f} {row['recall']:>11.3f} "
            f"{row['f1']:>7.3f} {row['support']:>7d}"
        )
    if metrics.get("per_source"):
        lines += ["", "Kaynak bazında:"]
        for name, row in metrics["per_source"].items():
            lines.append(
                f"  {name:<12} n={row['n']:<6d} doğruluk=%{row['accuracy'] * 100:.2f} "
                f"makro F1=%{row['f1_macro'] * 100:.2f}"
            )
    if metrics.get("headline"):
        row = metrics["headline"]
        lines += [
            "",
            f"Başlık benzeri kısa metin (ilk {row['lead_words']} kelime, n={row['n']}): "
            f"doğruluk=%{row['accuracy'] * 100:.2f} makro F1=%{row['f1_macro'] * 100:.2f}",
        ]
    return "\n".join(lines)
