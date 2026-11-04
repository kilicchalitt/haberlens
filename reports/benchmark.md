# Karşılaştırmalı Değerlendirme (Benchmark)

- **Veri:** 30435 benzersiz haber, 9 kategori
- **Kaynaklar:** kemik42k (24099), haberlens (4086), ttc3600 (2250)
- **Bölme:** katmanlı %80 eğitim / %20 test, sabit tohum (42)
- **Başlık artırımı:** eğitimde uzun haberlerin ilk 12 kelimesi
- **Üretim:** `haberlens benchmark` · 2026-11-04

## 1. Ayrılmış test kümesi

| Sıra | Yöntem | Doğruluk | Makro F1 | İlk-2 doğruluk | Başlık F1¹ | Eğitim (sn) |
| :-: | --- | :-: | :-: | :-: | :-: | :-: |
| 1 | Hibrit (kelime + karakter) + Lojistik Regresyon | %91,18 | %87,16 | %97,59 | %78,51 | 241,5 |
| 2 | Hibrit + Doğrusal SVM (kalibre) | %91,19 | %87,16 | %97,26 | %78,88 | 148,5 |
| 3 | **HaberLens Ensemble** (LR + SVM + NB) | %91,08 | %86,98 | %97,54 | %79,80 | 330,8 |
| 4 | TF-IDF | %90,44 | %86,12 | %97,08 | %76,94 | 61,5 |
| 5 | Karakter n-gram | %90,37 | %85,98 | %97,04 | %76,99 | 168,9 |
| 6 | Bag-of-Words | %89,39 | %84,60 | %96,44 | %74,81 | 38,0 |
| 7 | Hibrit + Complement Naive Bayes | %90,08 | %84,60 | %97,06 | %78,42 | 65,2 |
| 8 | Word2Vec (ortalama vektör) | %87,96 | %82,56 | %95,88 | %73,34 | 187,8 |
| 9 | N-Gram (bigram) | %85,92 | %79,27 | %92,69 | %64,32 | 81,5 |

¹ Uzun test haberlerinin yalnızca ilk 12 kelimesi verildiğinde (başlık benzeri kısa girdi) makro F1.

## 2. Görülmemiş kaynak testi (leave-one-source-out)

Her sütunda o kaynak eğitimden tamamen çıkarılmış ve model yalnızca diğer kaynaklarla eğitilmiştir. Değerler makro F1'dir (yalnızca o kaynakta bulunan kategoriler üzerinden).

| Yöntem | → haberlens | → kemik42k | → ttc3600 | Ortalama |
| --- | :-: | :-: | :-: | :-: |
| Hibrit (kelime + karakter) + Lojistik Regresyon | %86,93 | %77,93 | %88,92 | %84,59 |
| Hibrit + Doğrusal SVM (kalibre) | %87,42 | %78,75 | %89,99 | %85,39 |
| Hibrit + Complement Naive Bayes | %84,19 | %77,54 | %86,50 | %82,74 |
| **HaberLens Ensemble** (LR + SVM + NB) | %87,49 | %78,74 | %89,63 | %85,29 |

> Varsayılan üretim mimarisi: `ensemble`.
