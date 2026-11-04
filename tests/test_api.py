from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from haberlens.api.app import create_app
from haberlens.config import Settings
from haberlens.predictor import NewsClassifier


@pytest.fixture(scope="module")
def classifier(trained_model_path: Path) -> NewsClassifier:
    return NewsClassifier.load(trained_model_path)


def _client(classifier: NewsClassifier | None, **settings: object) -> TestClient:
    config = Settings(model_path=Path("/nonexistent/model.joblib"), **settings)
    return TestClient(create_app(settings=config, classifier=classifier))


@pytest.fixture
def client(classifier: NewsClassifier) -> TestClient:
    with _client(classifier) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["auth_required"] is False
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_categories(client: TestClient) -> None:
    assert len(client.get("/v1/categories").json()) == 7


def test_model_info(client: TestClient) -> None:
    body = client.get("/v1/model").json()
    assert body["architecture"] == "hybrid"
    assert "f1_macro" in body["metrics"]


def test_classify(client: TestClient) -> None:
    response = client.post(
        "/v1/classify", json={"text": "Galatasaray deplasmanda gol atarak kazandı", "top_k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Spor"
    assert len(body["scores"]) == 2
    assert body["keywords"]
    assert body["processing_ms"] >= 0


def test_classify_validation_errors(client: TestClient) -> None:
    assert client.post("/v1/classify", json={"text": ""}).status_code == 422
    assert client.post("/v1/classify", json={"text": "a", "top_k": 0}).status_code == 422
    assert client.post("/v1/classify", json={}).status_code == 422


def test_batch(client: TestClient) -> None:
    texts = ["Borsa günü düşüşle kapattı", "Yeni akıllı telefon tanıtıldı", "..."]
    response = client.post("/v1/classify/batch", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert body["results"][2]["label"] is None
    assert body["uncertain_count"] >= 1


def test_batch_limits(classifier: NewsClassifier) -> None:
    with _client(classifier, max_batch_size=2, max_text_length=10) as limited:
        too_many = limited.post("/v1/classify/batch", json={"texts": ["a", "b", "c"]})
        assert too_many.status_code == 413
        too_long = limited.post("/v1/classify", json={"text": "x" * 11})
        assert too_long.status_code == 422


def test_api_key_protection(classifier: NewsClassifier) -> None:
    with _client(classifier, api_key="gizli-anahtar") as secured:
        assert secured.get("/health").json()["auth_required"] is True
        assert secured.get("/v1/categories").status_code == 401
        wrong = secured.get("/v1/categories", headers={"X-API-Key": "yanlis"})
        assert wrong.status_code == 401
        ok = secured.get("/v1/categories", headers={"X-API-Key": "gizli-anahtar"})
        assert ok.status_code == 200


def test_missing_model_degrades_gracefully() -> None:
    with _client(None) as degraded:
        health = degraded.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert degraded.post("/v1/classify", json={"text": "test"}).status_code == 503


def test_cors_headers(classifier: NewsClassifier) -> None:
    with _client(classifier, cors_origins=("https://haber.example.com",)) as cors:
        response = cors.get("/health", headers={"Origin": "https://haber.example.com"})
        assert response.headers["access-control-allow-origin"] == "https://haber.example.com"


def test_openapi_schema(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "HaberLens API"
    assert "/v1/classify/batch" in schema["paths"]


def test_web_ui_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "HaberLens" in response.text
