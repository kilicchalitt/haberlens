from haberlens.evaluation import evaluate_predictions, format_report


def test_perfect_predictions() -> None:
    metrics = evaluate_predictions(["A", "B", "A"], ["A", "B", "A"], ["A", "B"])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["confusion_matrix"] == {"labels": ["A", "B"], "matrix": [[2, 0], [0, 1]]}


def test_per_class_scores() -> None:
    metrics = evaluate_predictions(["A", "A", "B", "B"], ["A", "B", "B", "B"], ["A", "B"])
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["A"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "support": 2,
    }
    assert metrics["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]


def test_format_report_lists_every_class() -> None:
    metrics = evaluate_predictions(["Spor", "Ekonomi"], ["Spor", "Spor"], ["Ekonomi", "Spor"])
    report = format_report(metrics)
    assert "Doğruluk" in report and "Makro F1" in report
    assert "Ekonomi" in report and "Spor" in report
