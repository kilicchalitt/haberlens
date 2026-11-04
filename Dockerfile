# syntax=docker/dockerfile:1.7

# ---------- 1. aşama: bağımlılıkları kur ve modeli eğit ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# full: çok kaynaklı derlemi indirip doğrular ve üretim modelini eğitir (varsayılan)
# raw : yalnızca depodaki özgün veriyle hızlı derleme (CI duman testi için)
ARG HABERLENS_CORPUS=full
RUN apt-get update \
 && apt-get install -y --no-install-recommends libarchive-tools \
 && rm -rf /var/lib/apt/lists/*

COPY data/raw ./data/raw
RUN if [ "$HABERLENS_CORPUS" = "full" ]; then \
      haberlens data build && haberlens train --output /app/models/haberlens.joblib; \
    else \
      haberlens train --data data/raw/turkish_news.csv --output /app/models/haberlens.joblib; \
    fi

# ---------- 2. aşama: yalın çalışma imajı ----------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="HaberLens" \
      org.opencontainers.image.description="Türkçe haber metni sınıflandırma API'si" \
      org.opencontainers.image.source="https://github.com/kilicchalitt/haberlens" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Halit Kılıç"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HABERLENS_MODEL_PATH=/app/models/haberlens.joblib

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app/models /app/models

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["haberlens", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
