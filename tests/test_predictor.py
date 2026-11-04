from pathlib import Path

import pytest

from haberlens import NewsClassifier, Prediction
from haberlens.artifact import ModelArtifact
from haberlens.pipelines import build_pipeline


@pytest.fixture(scope="module")
def classifier(trained_model_path: Path) -> NewsClassifier:
    return NewsClassifier.load(trained_model_path)


def test_labels_and_metadata(classifier: NewsClassifier) -> None:
    assert len(classifier.labels) == 7
    assert classifier.metadata["architecture"] == "ensemble"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Galatasaray, Süper Lig'de deplasmanda gol atarak maçı kazandı", "Spor"),
        ("Merkez Bankası faiz kararını açıkladı, dolar ve borsa yükseldi", "Ekonomi"),
    ],
)
def test_obvious_examples(classifier: NewsClassifier, text: str, expected: str) -> None:
    assert classifier.predict(text).label == expected


def test_prediction_structure(classifier: NewsClassifier) -> None:
    prediction = classifier.predict("Apple yeni iPhone modelini tanıttı", top_k=3)
    assert isinstance(prediction, Prediction)
    assert next(iter(prediction.scores)) == prediction.label
    assert len(prediction.scores) == 3
    assert list(prediction.scores.values()) == sorted(prediction.scores.values(), reverse=True)
    assert 0.0 < prediction.confidence <= 1.0
    assert prediction.to_dict()["label"] == prediction.label


def test_keywords_explain_the_decision(classifier: NewsClassifier) -> None:
    prediction = classifier.predict("Teknik direktör maç sonrası futbolcuları tebrik etti")
    assert prediction.keywords, "açıklama boş olmamalı"
    assert all(keyword.weight > 0 for keyword in prediction.keywords)
    weights = [keyword.weight for keyword in prediction.keywords]
    assert weights == sorted(weights, reverse=True)


def test_batch_matches_single(classifier: NewsClassifier) -> None:
    texts = ["Borsa günü düşüşle kapattı", "Yeni akıllı telefon tanıtıldı"]
    batch = classifier.predict_batch(texts)
    singles = [classifier.predict(text, explain=False) for text in texts]
    assert [p.label for p in batch] == [p.label for p in singles]
    assert all(p.keywords == [] for p in batch)


def test_empty_text_is_flagged(classifier: NewsClassifier) -> None:
    prediction = classifier.predict("!!! 123 ve bu")
    assert prediction.label is None
    assert prediction.uncertain is True


def test_uncertainty_threshold(trained_model_path: Path) -> None:
    strict = NewsClassifier.load(trained_model_path, confidence_threshold=1.0)
    assert strict.predict("Hükümet yeni ekonomi paketini açıkladı").uncertain is True


def test_invalid_inputs(classifier: NewsClassifier, trained_model_path: Path) -> None:
    assert classifier.predict_batch([]) == []
    with pytest.raises(TypeError):
        classifier.predict_batch("tek metin")
    with pytest.raises(ValueError):
        NewsClassifier.load(trained_model_path, confidence_threshold=1.5)


@pytest.mark.parametrize(
    ("architecture", "has_keywords"),
    [("tfidf", True), ("char", False), ("svm", True), ("nb", False), ("ensemble", True)],
)
def test_explanations_for_other_architectures(architecture: str, has_keywords: bool) -> None:
    texts = ["Takım maçı kazandı", "Takım gol attı", "Faiz arttı", "Borsa düştü"] * 3
    labels = ["Spor", "Spor", "Ekonomi", "Ekonomi"] * 3
    pipeline = build_pipeline(architecture).fit(texts, labels)
    clf = NewsClassifier(ModelArtifact(pipeline=pipeline, labels=["Ekonomi", "Spor"]))
    prediction = clf.predict("Takım maçı kazandı")
    assert prediction.label == "Spor"
    assert bool(prediction.keywords) is has_keywords


def test_load_uses_environment_variable(
    trained_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HABERLENS_MODEL_PATH", str(trained_model_path))
    assert NewsClassifier.load().labels


def test_warmup_prepares_explanations(trained_model_path: Path) -> None:
    clf = NewsClassifier.load(trained_model_path)
    assert "_explainable_features" not in clf.__dict__
    clf.warmup()
    assert "_explainable_features" in clf.__dict__
