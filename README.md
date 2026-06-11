# 🤖 Chatbot RAG — Peminatan Ilmu Komputer UHO

Sistem chatbot berbasis **Retrieval-Augmented Generation (RAG)** untuk membantu mahasiswa Program Studi Ilmu Komputer FMIPA Universitas Halu Oleo (UHO) mendapatkan informasi akademik secara cepat dan akurat.

Dibangun dengan **Flask**, **FAISS**, **Google Gemini**, dan **MySQL**.

---

## Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Prasyarat](#prasyarat)
- [Instalasi & Konfigurasi](#instalasi--konfigurasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [API Reference](#api-reference)
- [Struktur Proyek](#struktur-proyek)
- [Cara Kerja Sistem](#cara-kerja-sistem)
- [Optimasi & Penghematan Token](#optimasi--penghematan-token)

---

## Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **RAG berbasis FAISS** | Pencarian konteks semantik menggunakan vector similarity |
| **Semantic Chunking** | Dokumen dipotong di batas paragraf/kalimat, bukan karakter mentah |
| **FAISS Persistent Cache** | Index vector disimpan ke disk — `0 token API` saat server restart |
| **Multi API Key + Rotasi** | Otomatis beralih ke key cadangan saat kena rate limit 429 |
| **LRU Answer Cache** | Pertanyaan berulang dijawab dari cache tanpa memanggil API |
| **Relevance Threshold** | Chunk dengan cosine similarity < 0.30 dibuang — mengurangi noise |
| **Validasi Jawaban** | Deteksi `finish_reason=MAX_TOKENS` agar jawaban terpotong tidak dicache |
| **Fallback Model** | Jika `gemini-2.5-flash` gagal, otomatis coba `gemini-1.5-flash` |
| **Knowledge Base Management** | Upload, list, dan hapus dokumen via REST API |
| **Format Dokumen** | Mendukung PDF, DOCX, dan TXT |

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                     Browser / Client                     │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│                    Flask App (app.py)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  GET /      │  │ POST /chat   │  │ /api/knowledge │  │
│  │  (UI)       │  │ (Chatbot)    │  │ (CRUD Dokumen) │  │
│  └─────────────┘  └──────┬───────┘  └───────┬────────┘  │
└─────────────────────────┼───────────────────┼───────────┘
                          │                   │
         ┌────────────────▼───────────────────▼─────────┐
         │              RAG Engine (rag_engine.py)        │
         │                                               │
         │  ┌─────────────┐    ┌──────────────────────┐ │
         │  │  LRU Cache  │    │  Semantic Splitter   │ │
         │  │  (256 slot) │    │  (paragraf/kalimat)  │ │
         │  └─────────────┘    └──────────────────────┘ │
         │                                               │
         │  ┌─────────────────────────────────────────┐ │
         │  │           FAISS Index (dim=768)          │ │
         │  │     Load dari disk / Rebuild dari DB     │ │
         │  └─────────────────────────────────────────┘ │
         │                                               │
         │  ┌─────────────────────────────────────────┐ │
         │  │   Gemini API  (Embedding + Generation)  │ │
         │  │   Key Rotation: KEY_1 → KEY_2 → KEY_3   │ │
         │  └─────────────────────────────────────────┘ │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │            MySQL Database (database.py)        │
         │         Tabel: knowledge_base (chunks)         │
         └───────────────────────────────────────────────┘
```

---

## Teknologi yang Digunakan

- **Python 3.10+**
- **Flask** — web framework
- **Google Gemini API** — embedding (`gemini-embedding-001`) + generation (`gemini-2.5-flash`)
- **FAISS** — vector similarity search
- **MySQL** — penyimpanan chunk dokumen
- **pypdf** — ekstraksi teks dari PDF
- **docx2txt** — ekstraksi teks dari DOCX
- **python-dotenv** — manajemen environment variable

---

## Prasyarat

Pastikan hal berikut sudah tersedia sebelum instalasi:

1. **Python 3.10** atau lebih baru
2. **MySQL Server** (lokal atau remote)
3. **Google Gemini API Key** — buat di [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Instalasi & Konfigurasi

### 1. Clone / ekstrak proyek

```bash
# Jika dari ZIP
unzip chatbotLLM.zip
cd chatbotLLM
```

### 2. Buat virtual environment & install dependensi

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install flask google-genai faiss-cpu mysql-connector-python \
            pypdf docx2txt python-dotenv numpy
```

> **Catatan:** Gunakan `faiss-gpu` jika memiliki GPU CUDA untuk performa lebih cepat.

### 3. Konfigurasi file `.env`

Salin template dan isi sesuai konfigurasi Anda:

```env
# ── Gemini API Key ─────────────────────────────────────────
# Key utama (wajib)
GEMINI_API_KEY=your_api_key_here

# Key cadangan — uncomment untuk aktifkan rotasi otomatis
# GEMINI_API_KEY_2=your_second_key_here
# GEMINI_API_KEY_3=your_third_key_here

# ── Database MySQL ──────────────────────────────────────────
DB_HOST=127.0.0.1
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=uho_rag
DB_PORT=3306

# ── Flask ───────────────────────────────────────────────────
FLASK_ENV=development
FLASK_APP=app.py
FLASK_RUN_PORT=5000
```

> Database dan tabel dibuat **otomatis** saat aplikasi pertama kali dijalankan.

### 4. (Opsional) Tambahkan dokumen base knowledge

Letakkan file PDF/DOCX/TXT ke folder `documents/`. Dokumen akan otomatis diindeks saat server pertama kali dinyalakan.

```
documents/
├── Buku_Kurikulum_Ilkom_2022.pdf
├── Label_MK_Peminatan.pdf
└── ... (file lainnya)
```

---

## Menjalankan Aplikasi

```bash
python app.py
```

Atau menggunakan Flask CLI:

```bash
flask run
```

Server berjalan di `http://localhost:5000`

Saat startup, log berikut akan muncul secara berurutan:

```
--> [DATABASE INFO] Database 'uho_rag' dan tabel 'knowledge_base' siap digunakan.
--> [CONFIG] 2 API Key Gemini berhasil dimuat.
--> [LOCAL DOCS] Mengindeks dokumen baru dari disk: 'Kurikulum.pdf'...
--> [FAISS CACHE] Di-load dari disk: 1842 vectors (0 token API)
--> [VECTOR INDEX] Cache valid. Skip embedding, 0 token API digunakan.
```

---

## API Reference

### `POST /api/chat`

Mengirim pertanyaan dan menerima jawaban dari chatbot.

**Request Body:**
```json
{
  "message": "Apa saja mata kuliah peminatan RPL?"
}
```

**Response:**
```json
{
  "status": "success",
  "response": "Peminatan Rekayasa Perangkat Lunak (RPL) mencakup..."
}
```

---

### `GET /api/knowledge`

Mengambil daftar dokumen yang sudah terindeks.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "file_name": "Kurikulum_Ilkom_2022.pdf",
      "total_chunks": 142,
      "uploaded_at": "2025-01-15 10:30:00"
    }
  ]
}
```

---

### `POST /api/knowledge`

Upload satu atau lebih dokumen ke knowledge base.

**Request:** `multipart/form-data`

| Field | Tipe | Keterangan |
|---|---|---|
| `files` | File (multiple) | PDF, DOCX, atau TXT |

**Response:**
```json
{
  "message": "Proses batch upload selesai. 2 file berhasil di-indeks.",
  "success_files": [
    { "file_name": "MK_Baru.pdf", "total_chunks": 38 }
  ],
  "errors": []
}
```

---

### `DELETE /api/knowledge`

Menghapus dokumen dari knowledge base secara permanen.

**Request Body:**
```json
{
  "file_name": "MK_Baru.pdf"
}
```

**Response:**
```json
{
  "message": "Dokumen 'MK_Baru.pdf' berhasil dihapus secara permanen dari sistem RAG."
}
```

---

## Struktur Proyek

```
chatbotLLM/
│
├── app.py                  # Entry point Flask — routing & bootstrap
├── rag_engine.py           # Core RAG: chunking, embedding, retrieval, generation
├── database.py             # DatabaseManager — CRUD ke MySQL
│
├── .env                    # Konfigurasi API key & database (jangan di-commit)
│
├── documents/              # Folder base knowledge (PDF/DOCX/TXT)
│   ├── Buku_Kurikulum_Ilkom_2022.pdf
│   └── ...
│
├── templates/
│   └── index.html          # Antarmuka chatbot
│
├── my_index.faiss          # File cache FAISS index (auto-generated)
├── my_index_docs.pkl       # File cache dokumen teks (auto-generated)
│
└── README.md
```

> `my_index.faiss` dan `my_index_docs.pkl` dibuat otomatis. Tidak perlu di-commit ke Git.

---

## Cara Kerja Sistem

### Alur Saat Server Start

```
Server start
│
├─ load_local_documents_to_db()
│    └─ Scan documents/ → file baru → ekstrak teks → chunking → simpan ke MySQL
│
└─ reload_vector_index()
     ├─ Ada cache FAISS di disk?
     │    ├─ YA & jumlah chunk sama dengan DB → load dari disk (0 token API) ✓
     │    └─ TIDAK / berbeda → embed semua chunk dari DB → simpan cache baru
     └─ force_rebuild=True (setelah upload/delete) → skip cache, rebuild penuh
```

### Alur Saat Menerima Pertanyaan

```
Pertanyaan masuk
│
├─ 1. Cek LRU Cache → HIT? → return jawaban (0 token API) ✓
│
├─ 2. Embed pertanyaan → cari 6 chunk terdekat di FAISS
│
├─ 3. Filter threshold (score ≥ 0.30) → buang chunk tidak relevan
│    └─ 0 chunk lolos? → return pesan "data tidak tersedia" (0 generation token) ✓
│
├─ 4. Susun prompt sintesis → kirim ke Gemini API
│    └─ Rate limit? → rotasi API key → coba ulang
│    └─ Model gagal? → fallback ke model berikutnya
│
├─ 5. Validasi finish_reason
│    ├─ STOP → jawaban lengkap → simpan ke cache → return ✓
│    └─ MAX_TOKENS → jawaban terpotong → TIDAK dicache → coba model lain
│
└─ 6. Semua model gagal → return pesan fallback
```

---

## Deploy ke Railway

### Persiapan Awal

Pastikan sudah punya akun di [railway.app](https://railway.app) dan GitHub.

### Langkah 1 — Push ke GitHub

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/username/nama-repo.git
git push -u origin main
```

> File `documents/*.pdf` dan `my_index.faiss` sudah dikecualikan via `.gitignore` — tidak akan ikut push.

### Langkah 2 — Buat Project di Railway

1. Login ke [railway.app](https://railway.app)
2. Klik **New Project** → **Deploy from GitHub repo**
3. Pilih repository yang baru di-push
4. Railway otomatis mendeteksi Python dan mulai build

### Langkah 3 — Tambahkan MySQL Database

1. Di dashboard project, klik **+ New** → **Database** → **MySQL**
2. Tunggu hingga MySQL selesai provisioning
3. Klik service MySQL → tab **Variables** → catat nilai:
   - `MYSQL_HOST`
   - `MYSQL_PORT`
   - `MYSQL_USER`
   - `MYSQL_PASSWORD`
   - `MYSQL_DATABASE`

### Langkah 4 — Set Environment Variables

Di service Flask (bukan MySQL), buka tab **Variables** → tambahkan satu per satu:

| Variable | Nilai |
|---|---|
| `GEMINI_API_KEY` | API key Gemini Anda |
| `GEMINI_API_KEY_2` | (opsional) key cadangan |
| `DB_HOST` | nilai `MYSQL_HOST` dari step 3 |
| `DB_PORT` | nilai `MYSQL_PORT` dari step 3 |
| `DB_USER` | nilai `MYSQL_USER` dari step 3 |
| `DB_PASSWORD` | nilai `MYSQL_PASSWORD` dari step 3 |
| `DB_NAME` | nilai `MYSQL_DATABASE` dari step 3 |
| `FLASK_ENV` | `production` |

> Setelah semua variabel di-set, Railway otomatis redeploy.

### Langkah 5 — Dapatkan URL Publik

1. Di service Flask → tab **Settings** → **Networking** → klik **Generate Domain**
2. Railway memberi URL seperti `https://nama-app.up.railway.app`

### Langkah 6 — Upload Dokumen Knowledge Base

Karena filesystem Railway bersifat ephemeral (file hilang saat redeploy), dokumen **tidak boleh** disimpan di disk Railway. Sebagai gantinya, upload via API setelah deploy — chunk akan tersimpan permanen di MySQL.

Gunakan `curl` atau Postman:

```bash
# Upload satu file
curl -X POST https://nama-app.up.railway.app/api/knowledge \
  -F "files=@/path/lokal/Kurikulum_Ilkom_2022.pdf"

# Upload beberapa file sekaligus
curl -X POST https://nama-app.up.railway.app/api/knowledge \
  -F "files=@/path/lokal/MK11_Machine_Learning.pdf" \
  -F "files=@/path/lokal/MK46_Data_Mining.pdf"
```

> Setelah upload, FAISS index dibangun di memori. Saat redeploy berikutnya, index otomatis dibangun ulang dari MySQL — tidak perlu upload ulang dokumen.

### Catatan Penting Railway

**Ephemeral Filesystem** — File yang dibuat di runtime (termasuk `my_index.faiss`) akan hilang saat redeploy. Ini normal. Index selalu dibangun ulang dari MySQL saat startup.

**Startup Time** — Jika MySQL sudah berisi banyak chunk (ribuan), proses embedding saat startup bisa memakan waktu beberapa menit. Railway health check timeout sudah diset 300 detik di `railway.json`.

**Worker = 1** — Gunicorn dikonfigurasi dengan 1 worker karena FAISS index disimpan di memori proses. Lebih dari 1 worker akan membuat setiap worker punya index sendiri dan boros RAM.

**Free Tier** — Railway free tier memiliki batasan 500 jam/bulan. Untuk aplikasi yang selalu aktif, pertimbangkan plan berbayar atau gunakan sleep mode.

---

## Optimasi & Penghematan Token

Sistem ini dirancang dengan beberapa lapis optimasi untuk meminimalkan penggunaan token API:

**FAISS Persistent Cache** — Index vector disimpan ke disk. Setiap kali server restart, embedding tidak perlu diulang dari awal. Penghematan: `N chunk × ~300 token` per restart.

**LRU Answer Cache (256 slot)** — Jawaban akhir dari Gemini disimpan dalam memori. Pertanyaan yang sama atau mirip (case-insensitive, whitespace-normalized) langsung dijawab dari cache tanpa menyentuh API sama sekali.

**Relevance Threshold (0.30)** — Hanya chunk dengan cosine similarity ≥ 0.30 yang masuk ke prompt. Chunk tidak relevan dibuang sebelum dikirim ke API, sehingga input token berkurang signifikan.

**Validasi Jawaban Terpotong** — Jawaban dengan `finish_reason=MAX_TOKENS` tidak dicache dan tidak dikembalikan ke user. Cache juga secara otomatis menginvalidasi jawaban lama yang terdeteksi terpotong (heuristik akhir kalimat).

**`max_output_tokens=8192`** — Cukup besar untuk mencegah pemotongan jawaban, namun penghematan tetap terjaga karena konteks input sudah difilter lebih ketat di sisi retrieval.
