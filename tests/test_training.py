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
    assert sidecar["architecture"] == "hybrid"
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
