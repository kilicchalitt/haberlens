.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install dev data train serve test lint format benchmark docker-build docker-run clean

help: ## Komutları listele
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Paketi API desteğiyle kur
	$(PYTHON) -m pip install -e ".[api]"

dev: ## Geliştirme bağımlılıklarını kur
	$(PYTHON) -m pip install -e ".[dev]"

data: ## Çok kaynaklı eğitim derlemini indir ve oluştur
	haberlens data build

train: ## Modeli eğit (models/haberlens.joblib); önce `make data`
	haberlens train

serve: ## API + web arayüzünü başlat (http://127.0.0.1:8000)
	haberlens serve --reload

test: ## Testleri kapsam raporuyla çalıştır
	$(PYTHON) -m pytest --cov

lint: ## Kod kalitesi kontrolleri
	ruff check src tests
	ruff format --check src tests

format: ## Kodu otomatik biçimlendir
	ruff check --fix src tests
	ruff format src tests

benchmark: ## Yöntem karşılaştırması (reports/benchmark.md)
	haberlens benchmark

docker-build: ## Docker imajını oluştur
	docker build -t haberlens:latest .

docker-run: ## Docker imajını çalıştır
	docker run --rm -p 8000:8000 $(if $(wildcard .env),--env-file .env,) haberlens:latest

clean: ## Geçici dosyaları temizle
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
