# Deney Günlüğü — v1.1 Model Seçimi

Bu belge, v1.1'deki model ve veri kararlarının hangi ölçümlere dayandığını kaydeder.
Ana sonuçlar `haberlens benchmark` ile yeniden üretilebilir (bkz.
[`reports/benchmark.md`](../reports/benchmark.md)); aşağıdaki ablasyon deneyleri
keşif amaçlıdır ve aynı derlem, aynı bölme ve sabit tohum (`42`) ile yapılmıştır.

## 1. Problem: tek kaynaklı model genelleşmiyordu

v1.0 modeli tek bir veri setiyle eğitilmişti ve kendi test bölmesinde %98,5 makro F1
veriyordu. Aynı model **hiç görmediği yayın organlarının** haberleriyle sınandığında:

| Eğitim | Test (görülmemiş kaynak) | Makro F1 |
| --- | --- | :-: |
| haberlens (v1.0 verisi) | haberlens test bölmesi | %98,5 |
| haberlens | Kemik 42K (7 ortak kategori) | **%76,5** |
| haberlens | TTC-3600 (5 ortak kategori) | %84,8 |

Sonuç: v1.0'ın yüksek skoru veri setine özgüydü; gerçek dünyada başka bir sitenin
haberinde başarı belirgin şekilde düşüyordu. Çözüm daha çok ve daha çeşitli veriydi.

## 2. Veri: üç kaynaklı derlem

| Kaynak | İçerik | Kullanılan |
| --- | --- | --- |
| haberlens | Kısa haber metinleri ve başlıklar | 4.086 |
| Kemik 42K (YTÜ) | Tam metin haberler, 12 etiket | 24.099 (9 etiket) |
| TTC-3600 | Tam metin haberler, 6 kategori | 2.250 |

Önemli veri kararları:

- **Konu belirtmeyen etiketler çıkarıldı.** Kemik'teki `genel`, `güncel`, `türkiye`
  etiketleri yayın bölümü adlarıdır; içlerinde spor, siyaset ve asayiş haberleri
  karışıktır (örnekleme ile doğrulandı). Bu ~14 bin haber eğitime alınmadı.
- **Kaynak sızıntısı temizlendi.** TTC-3600'de `RADİKAL -` ön eki siyaset haberlerinde
  114, teknoloji haberlerinde 7 kez geçiyordu; model yayın organını öğrenebilirdi.
- **Kaynaklar arası tekrarlar ayıklandı.** TTC-3600'ün 1.254 metni Kemik derleminde de
  bulunuyordu. Tekrarlar ayıklanmasaydı test kümesine eğitim verisi sızardı.
- **TTC-3600 katkısı ölçüldü.** Kemik → haberlens: %81,6; Kemik + TTC-3600 → haberlens:
  %83,5 (+1,9 puan). Ek bağımlılık (RAR açıcı) gerektirdiği hâlde tutuldu; araç yoksa
  derleme bu kaynağı atlayarak devam eder.

## 3. Ölçüm hatası ve düzeltmesi

İlk ölçümde kaynak bazlı makro F1, o kaynakta **hiç bulunmayan** kategorileri de sıfır
puanla ortalamaya katıyordu (ör. haberlens'te Dünya kategorisi yok). Bu, %97'lik bir
skoru %75 gibi gösteriyordu. Kaynak ve alt küme skorları artık yalnızca o alt kümede
bulunan kategoriler üzerinden hesaplanıyor (`evaluation.evaluate_model`).

## 4. Model karşılaştırması

Öznitelikler tüm modellerde aynı: kelime 1-2 gram + karakter 2-5 gram TF-IDF, ilk 1.500
karakter.

| Model | Karışık test makro F1 | Görülmemiş kaynak¹ makro F1 | Görülmemiş kaynak¹ ilk-2 |
| --- | :-: | :-: | :-: |
| Lojistik Regresyon | %87,8 | %78,7 | %87,1 |
| Doğrusal SVM (kalibre) | %87,8 | %81,2 | %87,7 |
| Complement Naive Bayes | %85,2 | — | — |
| fastText (alt-kelime gömmeli sinir ağı) | %85,0 | %80,9 | %90,2 |
| LR + SVM | %88,0 | %80,5 | %87,9 |
| **LR + SVM + NB (1 : 1 : 0,5)** | **%88,0** | **%83,1** | **%90,4** |
| LR + SVM + fastText | %87,2 | %83,5 | %89,9 |
| LR + SVM + fastText + NB | %87,5 | %84,2 | %91,4 |

¹ Kemik + TTC-3600 ile eğitilip haberlens başlıklarında test (başlık artırımı olmadan).

**Karar:** Varsayılan model **LR + SVM + NB** topluluğu. Aynı dağılımdaki testte tek
modellerle aynı düzeyde, görülmemiş kaynakta ise Lojistik Regresyondan 4,4 puan iyi.
Modeller farklı türde hatalar yaptığı için birleşim genellemeyi artırıyor.

**fastText neden eklenmedi:** Topluluğa +1,1 puan görülmemiş kaynak kazancı sağladı,
ancak karışık testte −0,5 puan kaybettirdi. Ayrıca derleme gerektiren bir C++
bağımlılığı ve scikit-learn ile uyumsuz ayrı bir kayıt biçimi getiriyor. Bu maliyet,
belirsiz net kazanç için gerekçelendirilemedi.

**BERT neden yok:** Türkçe BERT ağırlıkları Hugging Face Hub'dan indirilmelidir; bu
çalışmanın yürütüldüğü ortamda Hub'a erişim yoktu ve doğrulanamayan bir model
eklenmedi. İnce ayarlı (fine-tuned) Türkçe transformer yol haritasındadır.

## 5. Başlık artırımı

Kullanıcılar çoğunlukla yalnızca başlık gönderir; derlem ise büyük ölçüde tam
metindir. Her uzun eğitim haberinin ilk N kelimesi ek örnek olarak eklendi
(yalnızca eğitim bölümüne):

| Artırım | Görülmemiş kaynak başlıkları | Karışık test (tam metin) | Karışık test (ilk 12 kelime) |
| --- | :-: | :-: | :-: |
| Yok | %83,5 | %87,7 | %71,4 |
| **İlk 12 kelime** | **%87,9** | %87,0 | **%78,7** |
| İlk 20 kelime | %88,4 | %87,1 | %78,3 |

(Doğrusal SVM ile ölçülmüştür.) **Karar:** 12 kelime. Başlık uzunluğuna daha yakın;
20 kelimeyle fark ölçüm gürültüsü düzeyinde.

## 6. Metin kesme uzunluğu

| İlk N karakter | Görülmemiş kaynak başlıkları | Karışık test |
| --- | :-: | :-: |
| **1.500** | %83,5 | %87,7 |
| 3.000 | %83,3 | %88,1 |

**Karar:** 1.500 karakter. Doğruluk farkı ihmal edilebilir; eğitim ve tahmin daha hızlı.
