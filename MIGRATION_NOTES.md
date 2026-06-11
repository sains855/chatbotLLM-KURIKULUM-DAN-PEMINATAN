# Migration: MySQL → JSON Storage

## Yang Berubah

| File | Perubahan |
|------|-----------|
| `database.py` | Ditulis ulang penuh — MySQL dihapus, diganti JSON file storage |
| `requirements.txt` | `mysql-connector-python` dihapus |
| `.env.example` | Variabel `DB_*` dihapus |
| `Procfile` | Baru — gunicorn untuk Railway |
| `railway.toml` | Baru — konfigurasi deploy Railway |

## Cara Kerja Storage JSON

- Data disimpan di `/data/knowledge_base.json` (Railway Volume)
- Fallback ke `./data/knowledge_base.json` saat lokal/dev
- Write atomik via temp-file + rename → aman dari korupsi
- Thread-safe dengan `threading.Lock()`

## Setup di Railway

1. Push repo ke GitHub
2. Buat project baru di Railway → Deploy from GitHub
3. Tambah **Volume** di Railway: mount path `/data`
4. Set environment variables (hanya `GEMINI_API_KEY*` + `FLASK_*`)
5. Deploy — tidak perlu MySQL plugin sama sekali
