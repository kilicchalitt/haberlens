"""HaberLens — Türkçe haber metinleri için açıklanabilir sınıflandırma servisi."""

__version__ = "1.1.0"

from haberlens.predictor import Keyword, NewsClassifier, Prediction

__all__ = ["Keyword", "NewsClassifier", "Prediction", "__version__"]
