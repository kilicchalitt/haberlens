"""FastAPI uygulaması.

Uç noktalar:
    GET  /                    Web arayüzü
    GET  /health              Canlılık / hazır olma kontrolü
    GET  /v1/model            Model kartı
    GET  /v1/categories       Desteklenen kategoriler
    POST /v1/classify         Tek metin sınıflandırma
    POST /v1/classify/batch   Toplu sınıflandırma

``HABERLENS_API_KEY`` tanımlıysa ``/v1/*`` uç noktaları ``X-API-Key`` başlığı ister.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader

from haberlens import __version__
from haberlens.api.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionOut,
)
from haberlens.artifact import ModelLoadError
from haberlens.config import Settings, get_settings
from haberlens.predictor import NewsClassifier

logger = logging.getLogger("haberlens.api")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(
    settings: Settings | None = None,
    classifier: NewsClassifier | None = None,
) -> FastAPI:
    """Uygulama fabrikası. Testlerde hazır bir ``classifier`` enjekte edilebilir."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.classifier is None:
            try:
                app.state.classifier = NewsClassifier.load(settings.model_path)
                logger.info("Model yüklendi: %s", settings.model_path)
            except ModelLoadError as exc:
                logger.error("Model yüklenemedi: %s", exc)
        yield

    app = FastAPI(
        title="HaberLens API",
        version=__version__,
        description=(
            "Türkçe haber metinlerini **Ekonomi, Magazin, Sağlık, Siyaset, Spor, Teknoloji, "
            "Yaşam** kategorilerine ayıran, kararını anahtar kelimelerle açıklayan REST API."
        ),
        contact={"name": "Halit Kılıç", "url": "https://github.com/kilicchalitt"},
        license_info={"name": "MIT"},
        lifespan=lifespan,
    )
    app.state.classifier = classifier
    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def add_request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "%s %s -> %s (%.1f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response

    # ------------------------------------------------------------- bağımlılıklar
    def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
        expected = settings.api_key
        if expected and not (api_key and secrets.compare_digest(api_key, expected)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Geçersiz veya eksik API anahtarı (X-API-Key).",
            )

    def get_classifier(request: Request) -> NewsClassifier:
        clf = request.app.state.classifier
        if clf is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model yüklü değil. `haberlens train` ile modeli oluşturun.",
            )
        return clf

    def check_lengths(texts: list[str]) -> None:
        too_long = [i for i, t in enumerate(texts) if len(t) > settings.max_text_length]
        if too_long:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Metin uzunluğu {settings.max_text_length} karakteri aşamaz "
                    f"(indeksler: {too_long[:10]})."
                ),
            )

    protected = [Depends(require_api_key)]

    # ------------------------------------------------------------------ rotalar
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return resources.files("haberlens.api").joinpath("static/index.html").read_text("utf-8")

    @app.get("/health", response_model=HealthResponse, tags=["Sistem"])
    def health(request: Request, response: Response) -> HealthResponse:
        loaded = request.app.state.classifier is not None
        if not loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ok" if loaded else "degraded",
            version=__version__,
            model_loaded=loaded,
            auth_required=bool(settings.api_key),
        )

    @app.get("/v1/model", response_model=ModelInfoResponse, dependencies=protected, tags=["Model"])
    def model_info(clf: NewsClassifier = Depends(get_classifier)) -> ModelInfoResponse:
        meta = clf.metadata
        metrics = meta.get("metrics", {})
        return ModelInfoResponse(
            version=meta.get("version"),
            architecture=meta.get("architecture"),
            architecture_description=meta.get("architecture_description"),
            trained_at=meta.get("trained_at"),
            labels=clf.labels,
            metrics={k: metrics[k] for k in ("accuracy", "f1_macro", "per_class") if k in metrics},
            confidence_threshold=clf.confidence_threshold,
        )

    @app.get("/v1/categories", response_model=list[str], dependencies=protected, tags=["Model"])
    def categories(clf: NewsClassifier = Depends(get_classifier)) -> list[str]:
        return clf.labels

    @app.post(
        "/v1/classify",
        response_model=ClassifyResponse,
        dependencies=protected,
        tags=["Sınıflandırma"],
        summary="Tek bir haber metnini sınıflandır",
    )
    def classify(
        payload: ClassifyRequest, clf: NewsClassifier = Depends(get_classifier)
    ) -> ClassifyResponse:
        check_lengths([payload.text])
        started = time.perf_counter()
        prediction = clf.predict(payload.text, top_k=payload.top_k, explain=payload.explain)
        return ClassifyResponse(
            **prediction.to_dict(),
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    @app.post(
        "/v1/classify/batch",
        response_model=BatchClassifyResponse,
        dependencies=protected,
        tags=["Sınıflandırma"],
        summary="Birden çok haber metnini tek istekte sınıflandır",
    )
    def classify_batch(
        payload: BatchClassifyRequest, clf: NewsClassifier = Depends(get_classifier)
    ) -> BatchClassifyResponse:
        if len(payload.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Tek istekte en fazla {settings.max_batch_size} metin gönderilebilir.",
            )
        check_lengths(payload.texts)
        started = time.perf_counter()
        predictions = clf.predict_batch(payload.texts, top_k=payload.top_k, explain=payload.explain)
        results = [PredictionOut(**p.to_dict()) for p in predictions]
        return BatchClassifyResponse(
            count=len(results),
            uncertain_count=sum(r.uncertain for r in results),
            results=results,
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    return app
