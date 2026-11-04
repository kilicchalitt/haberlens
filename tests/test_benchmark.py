import json
from pathlib import Path

import pandas as pd
import pytest

from haberlens.benchmark import run_benchmark, write_report
from haberlens.cli import main


@pytest.fixture(scope="module")
def multi_source_csv(sample_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    frame = pd.read_csv(sample_csv)
    frame["KAYNAK"] = ["a" if i % 2 else "b" for i in range(len(frame))]
    path = tmp_path_factory.mktemp("bench") / "multi.csv"
    frame.to_csv(path, index=False)
    return path


def test_holdout_ranking(sample_csv: Path) -> None:
    report = run_benchmark(data_path=sample_csv, methods=["bow", "tfidf"])
    assert {r.method for r in report.holdout} == {"bow", "tfidf"}
    scores = [r.f1_macro for r in report.holdout]
    assert scores == sorted(scores, reverse=True)
    assert all(r.top2_accuracy >= r.accuracy for r in report.holdout)
    assert report.cross_source == []  # kaynak sütunu yok → atlanır


def test_cross_source_evaluation(multi_source_csv: Path) -> None:
    report = run_benchmark(
        data_path=multi_source_csv, methods=["tfidf"], cross_source_methods=["tfidf"]
    )
    assert {r.held_out_source for r in report.cross_source} == {"a", "b"}
    assert set(report.sources) == {"a", "b"}
    assert all(r.n_test > 0 for r in report.cross_source)


def test_report_files(multi_source_csv: Path, tmp_path: Path) -> None:
    report = run_benchmark(
        data_path=multi_source_csv, methods=["tfidf"], cross_source_methods=["tfidf"]
    )
    path = write_report(report, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "| 1 | TF-IDF |" in text
    assert "Görülmemiş kaynak testi" in text and "→ a" in text
    assert "," in text  # Türkçe ondalık biçimi
    saved = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    assert saved["holdout"][0]["method"] == "tfidf"


def test_benchmark_cli(sample_csv: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "benchmark",
            "--data",
            str(sample_csv),
            "--methods",
            "bow",
            "--no-cross-source",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "benchmark.md").is_file()
    assert "Bag-of-Words" in capsys.readouterr().out


def test_word2vec_benchmark_when_gensim_available(sample_csv: Path) -> None:
    pytest.importorskip("gensim")
    report = run_benchmark(data_path=sample_csv, methods=["tfidf"], include_embeddings=True)
    assert {r.method for r in report.holdout} == {"tfidf", "word2vec"}
