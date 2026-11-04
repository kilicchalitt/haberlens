import json
from pathlib import Path

import pytest

from haberlens.artifact import ModelArtifact, ModelLoadError
from haberlens.training import train_model


def test_training_produces_artifact_and_metrics(sample_csv: Path, tmp_path: Path) -> None:
    output = tmp_path / "model.joblib"
    result = train_model(data_path=sample_csv, output_path=output, cv_folds=3)

    assert result.model_path == output and output.is_file()
    assert output.with_suffix(".json").is_file()

    metrics = result.metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["cross_validation"]["folds"] == 3
    assert set(metrics["per_class"]) == set(result.metadata["labels"])
    assert len(metrics["confusion_matrix"]["matrix"]) == 7

    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["architecture"] == "ensemble"
    assert len(sidecar["data"]["sha256"]) == 64


def test_model_learns_signal(trained_model_path: Path) -> None:
    artifact = ModelArtifact.load(trained_model_path)
    # Küçük alt kümede bile rastgele tahminin (≈%14) çok üzerinde olmalı.
    assert artifact.metadata["metrics"]["accuracy"] > 0.6


def test_label_subset_training(sample_csv: Path, tmp_path: Path) -> None:
    result = train_model(
        data_path=sample_csv,
        output_path=tmp_path / "m.joblib",
        labels=["Ekonomi", "Spor", "Teknoloji"],
        architecture="tfidf",
        cv_folds=0,
    )
    assert result.metadata["labels"] == ["Ekonomi", "Spor", "Teknoloji"]
    assert "cross_validation" not in result.metrics


def test_loading_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="bulunamadı"):
        ModelArtifact.load(tmp_path / "yok.joblib")


def test_loading_foreign_file_raises(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "foreign.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelLoadError, match="geçerli bir HaberLens"):
        ModelArtifact.load(path)


def test_augment_with_leads_adds_only_long_texts() -> None:
    import pandas as pd

    from haberlens.training import augment_with_leads

    texts = pd.Series(["kısa haber", " ".join(f"kelime{i}" for i in range(30))], index=[7, 9])
    labels = pd.Series(["Spor", "Ekonomi"], index=[7, 9])
    out_texts, out_labels = augment_with_leads(texts, labels, n_words=5)
    assert len(out_texts) == 3
    assert out_texts.iloc[-1] == "kelime0 kelime1 kelime2 kelime3 kelime4"
    assert out_labels.tolist() == ["Spor", "Ekonomi", "Ekonomi"]
    same_texts, _ = augment_with_leads(texts, labels, n_words=0)
    assert len(same_texts) == 2


def test_training_reports_source_and_headline_metrics(tmp_path: Path) -> None:
    import pandas as pd

    base = pd.read_csv(Path(__file__).resolve().parents[1] / "data/raw/turkish_news.csv")
    sample = base.groupby("ETIKET", group_keys=False).head(40).copy()
    long = sample.copy()
    long["HABERLER"] = (long["HABERLER"] + " ") * 6  # uzun metin benzetimi
    sample["KAYNAK"], long["KAYNAK"] = "kisa", "uzun"
    path = tmp_path / "corpus.csv"
    pd.concat([sample, long]).to_csv(path, index=False)

    result = train_model(
        data_path=path, output_path=tmp_path / "m.joblib", architecture="tfidf", refit_full=False
    )
    metrics = result.metrics
    assert set(metrics["per_source"]) == {"kisa", "uzun"}
    assert 0 < metrics["top2_accuracy"] <= 1
    assert metrics["top2_accuracy"] >= metrics["accuracy"]
    assert metrics["headline"]["n"] > 0
    distribution = result.metadata["data"]["source_distribution"]
    assert set(distribution) == {"kisa", "uzun"}
    assert distribution["kisa"] == distribution["uzun"]  # tekrarlar iki kaynakta da ayıklanır
