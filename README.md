# NLP ile Türkçe Haber Metni Sınıflandırma

Bu proje, Türkçe haber metinlerini **Ekonomi, Spor ve Teknoloji** kategorilerine ayırmak için geliştirilmiş, geleneksel ve modern Doğal Dil İşleme (NLP) tekniklerini kıyaslayan bir uçtan uca sınıflandırma sistemidir.

Projenin temel amacı, farklı metin temsil (text representation) yöntemlerinin model başarısı üzerindeki etkisini bilimsel bir yaklaşımla analiz etmektir.

---

## 🚀 Öne Çıkan Özellikler

* **Veri Dengeleme (Balancing):** Modelin yanlılığını (bias) önlemek için her sınıftan eşit sayıda (550) örneklem kullanılmıştır.
* **Gelişmiş Ön İşleme:** Türkçe dil yapısına uygun karakter dönüşümü, noktalama temizliği ve özel stop-words ayıklama işlemleri uygulanmıştır.
* **Karşılaştırmalı Analiz:** 5 farklı vektörleştirme yöntemi (BoW'dan BERT'e kadar) aynı veri seti ve sınıflayıcı üzerinde test edilmiştir.
* **Canlı Demo Modülü:** Eğitilen modeller üzerinden kullanıcıdan alınan metinlerle anlık tahmin ve güven skoru (confidence score) üretilmektedir.

---

## 🛠 Kullanılan Teknolojiler

* **Dil:** Python 3.x
* **Veri Analizi:** `Pandas`, `NumPy`
* **Makine Öğrenmesi:** `Scikit-Learn` (Logistic Regression, Label Encoding, Metrics)
* **NLP & Embeddings:** * `Gensim` (Word2Vec)
* `HuggingFace Transformers` (BERT - dbmdz/bert-base-turkish-cased)
* `PyTorch` (BERT inference için)



---

## 🏗 Proje Mimarisi ve İş Akışı

Proje beş ana aşamadan oluşmaktadır:

1. **Data Preprocessing:** Metinlerin küçük harfe çevrilmesi, sayısal verilerin ve noktalama işaretlerinin kaldırılması, etkisiz kelimelerin (stop-words) ayıklanması.
2. **Feature Extraction:** Metinlerin sayısal vektörlere dönüştürülmesi:
* **İstatistiksel:** Bag-of-Words, TF-IDF, N-Grams (Bigram).
* **Anlamsal:** Word2Vec (Average Pooling), BERT (CLS Token Embedding).


3. **Model Training:** Tüm temsiller üzerinde stabiliteyi ölçmek adına Lojistik Regresyon sınıflayıcısının eğitilmesi.
4. **Evaluation:** Accuracy metrikleri ile yöntemlerin kıyaslanması.
5. **Inference:** Kullanıcı girdisiyle tüm modellerin eş zamanlı olarak test edilmesi.

---

## 📊 Performans Değerlendirmesi

| Yöntem | Doğruluk (Accuracy) | Karakteristik |
| --- | --- | --- |
| **Bag-of-Words** | %... | Frekans tabanlı, basit |
| **TF-IDF** | %... | Kelime önemine odaklı |
| **N-Grams** | %... | Bağlamsal (İkili) kelime grupları |
| **Word2Vec** | %... | Anlamsal vektör uzayı |
| **BERT** | %... | Derin bağlamsal temsil |

---

## 🎯 Sonuç

Bu çalışma, karmaşık NLP görevlerinde **BERT** gibi transformatör modellerinin bağlamı anlama konusundaki üstünlüğünü kanıtlarken, **TF-IDF** gibi geleneksel yöntemlerin kısıtlı kaynaklarla ne kadar efektif sonuçlar verebileceğini göstermektedir.
