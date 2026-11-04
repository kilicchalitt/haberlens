# Güvenlik Politikası

## Desteklenen sürümler

| Sürüm | Destek |
| --- | --- |
| 1.x | ✅ |
| < 1.0 (eski `main.py` betiği) | ❌ |

## Güvenlik açığı bildirme

Bir güvenlik açığı bulduysanız lütfen **herkese açık bir issue açmayın**.
Bunun yerine GitHub üzerinden
[özel güvenlik bildirimi](https://github.com/kilicchalitt/haberlens/security/advisories/new)
oluşturun. Bildiriminiz en geç 72 saat içinde yanıtlanır.

## Üretim ortamı için öneriler

- **API anahtarı kullanın.** `HABERLENS_API_KEY` ortam değişkenine güçlü ve rastgele
  bir değer verin (`openssl rand -hex 32`). Anahtarı asla depoya eklemeyin;
  `.env` dosyası `.gitignore` tarafından zaten yok sayılır.
- **HTTPS arkasında çalıştırın.** Servisi doğrudan internete açmak yerine TLS
  sonlandıran bir ters vekil sunucu (Nginx, Traefik, bulut yük dengeleyici) kullanın.
- **CORS'u daraltın.** `HABERLENS_CORS_ORIGINS` yalnızca güvendiğiniz alan adlarını içermelidir.
- **Yalnızca güvendiğiniz model dosyalarını yükleyin.** Model dosyaları `joblib`
  (pickle) biçimindedir; güvenilmeyen bir kaynaktan gelen `.joblib` dosyası
  yüklendiğinde rastgele kod çalıştırabilir. Modelleri her zaman kendi
  verinizle `haberlens train` komutu ile üretin.
- **Kaynak sınırlarını koruyun.** `HABERLENS_MAX_BATCH_SIZE` ve
  `HABERLENS_MAX_TEXT_LENGTH` değerleri, aşırı büyük isteklere karşı servisi korur.
