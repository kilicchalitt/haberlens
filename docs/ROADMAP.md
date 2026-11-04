# Yol Haritası

HaberLens, bir öğrenci projesi olarak başlayan Türkçe haber sınıflandırma çalışmasının
kurumların günlük iş akışlarında kullanabileceği bir ürüne dönüştürülmesidir.

## ✅ v1.0 — Üretime hazır temel (tamamlandı)

**Karar:** Eski tek dosyalık betik yamalanmak yerine aynı veri ve fikir korunarak
modüler bir paket olarak **yeniden yapılandırıldı**. Gerekçeler:

| Eski durum | Sorun | v1.0 çözümü |
| --- | --- | --- |
| 7 kategorilik veri setinin yalnızca 3'ü kullanılıyordu | Gerçek hayatta işe yaramaz | 7 kategorinin tamamı |
| Vektörleştiriciler bölmeden önce tüm veriyle eğitiliyordu | Veri sızıntısı, iyimser skorlar | Tek `Pipeline`, katmanlı bölme |
| 85 tekrar eden metin hem eğitimde hem testte olabiliyordu | Ezberleme ölçülüyordu | Tekrarlar ayıklanıyor |
| Her çalıştırmada BERT dahil 5 model sıfırdan eğitiliyordu | Dakikalarca bekleme | Bir kez eğit, sürümlü dosyadan yükle |
| Yalnızca `input()` ile terminal demosu | Entegrasyon imkânsız | CLI, REST API, web arayüzü, Python API |
| Yalnızca doğruluk (accuracy) | Zayıf sınıflar görünmez | Makro F1, sınıf bazlı rapor, karışıklık matrisi |
| Test, CI, paketleme yok | Güvenilmez | %95 test kapsamı, GitHub Actions, Docker |

## 🔜 v1.1 — Kurumsal kullanım

- [ ] Tahmin geri bildirimi uç noktası (`POST /v1/feedback`) ile editör düzeltmelerinin toplanması
- [ ] Toplanan geri bildirimlerle periyodik yeniden eğitim komutu
- [ ] Prometheus metrikleri (`/metrics`): istek sayısı, gecikme, kategori dağılımı
- [ ] İstek hız sınırlama (rate limiting)
- [ ] Web arayüzünde CSV yükleyip toplu sınıflandırma ve sonuç indirme

## 🔭 v2.0 — Derin öğrenme ve çoklu etiket

- [ ] Türkçe BERT/ELECTRA modelinin ince ayarı (fine-tuning) ve ONNX ile hızlı CPU çıkarımı
- [ ] Çoklu etiket desteği (ör. *Ekonomi + Teknoloji*)
- [ ] Kategori dışı (out-of-distribution) haber tespiti
- [ ] Daha büyük ve güncel veri setleriyle eğitim
- [ ] Hugging Face Hub üzerinde model yayını ve canlı demo (Spaces)
