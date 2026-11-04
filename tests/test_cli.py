import json
from pathlib import Path

import pandas as pd
import pytest

from haberlens.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "haberlens" in capsys.readouterr().out


def test_train_command(sample_csv: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "cli.joblib"
    code = main(
        [
            "train",
            "--data",
            str(sample_csv),
            "--output",
            str(output),
            "--architecture",
            "tfidf",
            "--cv-folds",
            "2",
        ]
    )
    assert code == 0 and output.is_file()
    out = capsys.readouterr().out
    assert "Makro F1" in out and "çapraz doğrulama" in out


def test_predict_single_text(trained_model_path: Path, capsys) -> None:
    code = main(["predict", "--model", str(trained_model_path), "Takım derbide galip geldi"])
    assert code == 0
    assert "→" in capsys.readouterr().out


def test_predict_json(trained_model_path: Path, capsys) -> None:
    main(["predict", "--model", str(trained_model_path), "--json", "Borsa yükseldi"])
    payload = json.loads(capsys.readouterr().out)
    assert {"text", "label", "confidence", "scores", "keywords"} <= payload.keys()


def test_predict_from_stdin(trained_model_path: Path, capsys, monkeypatch) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Borsa yükseldi\n\nMaç berabere bitti\n"))
    main(["predict", "--model", str(trained_model_path), "--json"])
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_batch_prediction_csv(trained_model_path: Path, tmp_path: Path, capsys) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"id": [1, 2, 3], "baslik": ["Borsa düştü", "Gol attı", None]}).to_csv(
        source, index=False
    )
    target = tmp_path / "out.csv"
    code = main(
        [
            "predict",
            "--model",
            str(trained_model_path),
            "--file",
            str(source),
            "--text-column",
            "baslik",
            "--output",
            str(target),
        ]
    )
    assert code == 0
    result = pd.read_csv(target)
    assert list(result.columns) == ["id", "baslik", "predicted_label", "confidence", "uncertain"]
    assert bool(result.loc[2, "uncertain"]) is True
    assert "3 metin sınıflandırıldı" in capsys.readouterr().out


def test_batch_prediction_txt_to_jsonl(trained_model_path: Path, tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    source.write_text("Borsa düştü\nYeni telefon tanıtıldı\n", encoding="utf-8")
    target = tmp_path / "out.jsonl"
    main(
        [
            "predict",
            "--model",
            str(trained_model_path),
            "--file",
            str(source),
            "--output",
            str(target),
        ]
    )
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and "predicted_label" in rows[0]


def test_batch_prediction_missing_column(trained_model_path: Path, tmp_path: Path) -> None:
    source = tmp_path / "in.csv"
    pd.DataFrame({"x": ["a"]}).to_csv(source, index=False)
    with pytest.raises(SystemExit, match="sütunu bulunamadı"):
        main(["predict", "--model", str(trained_model_path), "--file", str(source)])


def test_evaluate_command(trained_model_path: Path, sample_csv: Path, capsys) -> None:
    code = main(
        ["evaluate", "--model", str(trained_model_path), "--data", str(sample_csv), "--json"]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["accuracy"] > 0.9


def test_info_command(trained_model_path: Path, capsys) -> None:
    assert main(["info", "--model", str(trained_model_path)]) == 0
    assert "Makro F1" in capsys.readouterr().out


def test_missing_model_returns_error(tmp_path: Path, capsys) -> None:
    code = main(["predict", "--model", str(tmp_path / "yok.joblib"), "test"])
    assert code == 1
    assert "bulunamadı" in capsys.readouterr().err


def test_data_build_command(sample_csv: Path, tmp_path: Path, capsys, monkeypatch) -> None:
    from haberlens import corpus

    monkeypatch.setattr(corpus, "SOURCES", {"haberlens": corpus.SOURCES["haberlens"]})
    output = tmp_path / "corpus.csv"
    code = main(["data", "build", "--raw", str(sample_csv), "--output", str(output)])
    assert code == 0
    assert pd.read_csv(output).columns.tolist() == ["HABERLER", "ETIKET", "KAYNAK"]
    out = capsys.readouterr().out
    assert "Derlem hazır" in out and "haberlens" in out


def test_evaluate_on_label_subset_uses_present_labels(
    trained_model_path: Path, sample_csv: Path, tmp_path: Path, capsys
) -> None:
    subset = pd.read_csv(sample_csv)
    subset = subset[subset["ETIKET"].isin(["Spor", "Ekonomi"])]
    path = tmp_path / "subset.csv"
    subset.to_csv(path, index=False)
    main(["evaluate", "--model", str(trained_model_path), "--data", str(path), "--json"])
    metrics = json.loads(capsys.readouterr().out)
    assert set(metrics["per_class"]) == {"Ekonomi", "Spor"}
    assert metrics["f1_macro"] > 0.9
