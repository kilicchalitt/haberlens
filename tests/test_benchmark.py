import json
from pathlib import Path

from haberlens.benchmark import run_benchmark, write_report
from haberlens.cli import main


def test_benchmark_ranks_methods(sample_csv: Path) -> None:
    results = run_benchmark(data_path=sample_csv, folds=2, methods=["bow", "tfidf"])
    assert {r.method for r in results} == {"bow", "tfidf"}
    scores = [r.f1_macro_mean for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < r.accuracy_mean <= 1 for r in results)


def test_report_files(sample_csv: Path, tmp_path: Path) -> None:
    results = run_benchmark(data_path=sample_csv, folds=2, methods=["tfidf"])
    report = write_report(results, tmp_path, folds=2, n_samples=420)
    text = report.read_text(encoding="utf-8")
    assert "| 1 | TF-IDF |" in text
    assert json.loads((tmp_path / "benchmark.json").read_text())[0]["method"] == "tfidf"


def test_benchmark_cli(sample_csv: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "benchmark",
            "--data",
            str(sample_csv),
            "--folds",
            "2",
            "--methods",
            "bow",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "benchmark.md").is_file()
    assert "Bag-of-Words" in capsys.readouterr().out
