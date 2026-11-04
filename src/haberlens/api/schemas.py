"""API istek ve yanıt şemaları."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Sınıflandırılacak haber metni.")
    top_k: int = Field(3, ge=1, le=20, description="Döndürülecek en olası kategori sayısı.")
    explain: bool = Field(True, description="Karara etki eden anahtar kelimeleri döndür.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Merkez Bankası politika faizini yüzde 50'de sabit tuttu.",
                    "top_k": 3,
                    "explain": True,
                }
            ]
        }
    )


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Sınıflandırılacak metinler.")
    top_k: int = Field(1, ge=1, le=20)
    explain: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        "Fenerbahçe deplasmanda 2-1 kazandı.",
                        "Yeni nesil akıllı telefon tanıtıldı.",
                    ],
                    "top_k": 1,
                }
            ]
        }
    )


class KeywordOut(BaseModel):
    term: str
    weight: float


class PredictionOut(BaseModel):
    label: str | None = Field(..., description="En olası kategori; metin boşsa null.")
    confidence: float = Field(..., ge=0, le=1)
    scores: dict[str, float]
    uncertain: bool = Field(..., description="Güven eşiğin altındaysa insan onayı önerilir.")
    keywords: list[KeywordOut] = []


class ClassifyResponse(PredictionOut):
    processing_ms: float


class BatchClassifyResponse(BaseModel):
    count: int
    uncertain_count: int
    results: list[PredictionOut]
    processing_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    auth_required: bool


class ModelInfoResponse(BaseModel):
    version: str | None = None
    architecture: str | None = None
    architecture_description: str | None = None
    trained_at: str | None = None
    labels: list[str]
    metrics: dict[str, object] = {}
    confidence_threshold: float
