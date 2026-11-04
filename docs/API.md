# REST API Referansı

Sunucuyu başlatın:

```bash
haberlens serve --host 0.0.0.0 --port 8000
```

Etkileşimli dokümantasyon: **`/docs`** (Swagger UI) ve **`/redoc`**.

## Kimlik doğrulama

`HABERLENS_API_KEY` tanımlıysa tüm `/v1/*` uç noktaları `X-API-Key` başlığı ister.
`/` ve `/health` her zaman herkese açıktır.

```bash
export HABERLENS_API_KEY="$(openssl rand -hex 32)"
```

## Uç noktalar

| Yöntem | Yol | Açıklama |
| --- | --- | --- |
| `GET` | `/` | Web arayüzü |
| `GET` | `/health` | Canlılık ve hazır olma kontrolü (model yoksa `503`) |
| `GET` | `/v1/model` | Model kartı: mimari, eğitim tarihi, metrikler |
| `GET` | `/v1/categories` | Desteklenen kategoriler |
| `POST` | `/v1/classify` | Tek metin sınıflandırma |
| `POST` | `/v1/classify/batch` | Toplu sınıflandırma (varsayılan en fazla 256 metin) |

### `POST /v1/classify`

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $HABERLENS_API_KEY" \
  -d '{"text": "Merkez Bankası politika faizini sabit tuttu", "top_k": 3, "explain": true}'
```

```json
{
  "label": "Ekonomi",
  "confidence": 0.9799,
  "scores": { "Ekonomi": 0.9799, "Magazin": 0.0055, "Yaşam": 0.0036 },
  "uncertain": false,
  "keywords": [
    { "term": "bankası", "weight": 0.4522 },
    { "term": "merkez", "weight": 0.4136 },
    { "term": "merkez bankası", "weight": 0.3637 }
  ],
  "processing_ms": 2.1
}
```

| Alan | Tip | Açıklama |
| --- | --- | --- |
| `text` | `string` | Zorunlu. En fazla `HABERLENS_MAX_TEXT_LENGTH` karakter |
| `top_k` | `int` | 1-20, varsayılan `3` |
| `explain` | `bool` | Anahtar kelimeleri döndür, varsayılan `true` |

`label` metin anlamlı içerik barındırmıyorsa `null` olur. `uncertain: true` ise
model kararsızdır; bu kayıtların bir editör tarafından kontrol edilmesi önerilir.

### `POST /v1/classify/batch`

```bash
curl -X POST http://localhost:8000/v1/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Fenerbahçe deplasmanda kazandı", "Yeni akıllı telefon tanıtıldı"]}'
```

```json
{
  "count": 2,
  "uncertain_count": 0,
  "results": [
    { "label": "Spor", "confidence": 0.99, "scores": { "Spor": 0.99 }, "uncertain": false, "keywords": [] },
    { "label": "Teknoloji", "confidence": 0.98, "scores": { "Teknoloji": 0.98 }, "uncertain": false, "keywords": [] }
  ],
  "processing_ms": 4.3
}
```

## Hata kodları

| Kod | Durum |
| --- | --- |
| `401` | API anahtarı eksik veya hatalı |
| `413` | Toplu istekte metin sayısı sınırı aşıldı |
| `422` | Geçersiz istek gövdesi veya metin uzunluğu sınırı aşıldı |
| `503` | Model yüklü değil |

## Başlıklar

| Başlık | Açıklama |
| --- | --- |
| `X-Request-ID` | İstekte gönderilirse aynen döner, yoksa üretilir; günlüklerde izleme için |
| `X-Process-Time-Ms` | Sunucu tarafı işlem süresi |

## İstemci örnekleri

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/classify",
    json={"text": "Galatasaray derbiyi 2-1 kazandı"},
    headers={"X-API-Key": "..."},
    timeout=10,
)
print(response.json()["label"])  # Spor
```

### JavaScript

```js
const res = await fetch("http://localhost:8000/v1/classify", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": "..." },
  body: JSON.stringify({ text: "Yapay zekâ destekli yeni telefon tanıtıldı" }),
});
const { label, confidence } = await res.json();
```
