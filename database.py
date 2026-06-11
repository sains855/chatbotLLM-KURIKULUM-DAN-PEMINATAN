# database.py — JSON-based storage (pengganti MySQL)
# Drop-in replacement: semua method signature identik dengan versi MySQL.
# Data disimpan di data/knowledge_base.json agar persisten di Railway Volume.

import os
import json
import threading
from datetime import datetime, timezone

# Lokasi file JSON — Railway menyediakan persistent volume di /data
# Fallback ke direktori proyek jika /data tidak tersedia (lokal/dev)
_DATA_DIR  = "/data" if os.path.isdir("/data") else os.path.join(os.path.dirname(__file__), "data")
_DB_FILE   = os.path.join(_DATA_DIR, "knowledge_base.json")
_LOCK      = threading.Lock()   # thread-safety untuk concurrent request Flask


def _load() -> dict:
    """Baca seluruh data dari file JSON. Kembalikan dict kosong jika belum ada."""
    if not os.path.exists(_DB_FILE):
        return {"records": []}
    try:
        with open(_DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Jika file korup atau kosong, kembalikan struktur default
        return {"records": []}


def _save(data: dict) -> None:
    """Tulis ulang seluruh data ke file JSON secara atomik."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    # Tulis ke file tmp dulu, lalu rename — mencegah korupsi jika proses tiba-tiba mati
    tmp_path = _DB_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _DB_FILE)


class DatabaseManager:
    def __init__(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        # Inisialisasi file jika belum ada
        if not os.path.exists(_DB_FILE):
            _save({"records": []})
        print(f"--> [DATABASE INFO] JSON storage siap digunakan: {_DB_FILE}")

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------
    def save_chunks(self, file_name: str, chunks: list) -> None:
        """[CREATE] Simpan banyak chunks dokumen ke JSON storage."""
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            data = _load()
            # Tentukan id berikutnya secara incremental
            next_id = (max((int(r["id"]) for r in data["records"]), default=0)) + 1
            
            for index, chunk in enumerate(chunks):
                data["records"].append({
                    "id":          next_id,
                    "file_name":   str(file_name),
                    "chunk_index": int(index),
                    "content":     str(chunk),
                    "created_at":  now
                })
                next_id += 1
            _save(data)

    # ------------------------------------------------------------------
    # READ — metadata
    # ------------------------------------------------------------------
    def get_all_documents_metadata(self) -> list:
        """[READ] Kembalikan list file unik beserta jumlah chunk & waktu upload."""
        with _LOCK:
            data = _load()

        aggregated = {}
        for r in data["records"]:
            fn = r["file_name"]
            if fn not in aggregated:
                aggregated[fn] = {
                    "file_name": fn, 
                    "total_chunks": 0, 
                    "uploaded_at": r["created_at"]
                }
            aggregated[fn]["total_chunks"] += 1
            # Simpan timestamp terbaru jika ada bentrokan
            if r["created_at"] > aggregated[fn]["uploaded_at"]:
                aggregated[fn]["uploaded_at"] = r["created_at"]

        return list(aggregated.values())

    # ------------------------------------------------------------------
    # READ — semua chunk (untuk rebuild FAISS)
    # ------------------------------------------------------------------
    def get_all_chunks(self) -> list:
        """
        [READ] Kembalikan semua record chunk berurutan untuk inisialisasi FAISS.
        Mengembalikan list of dict agar app.py bisa memetakan teks ke metadata file aslinya.
        """
        with _LOCK:
            data = _load()
            
        # Urutkan berdasarkan ID untuk menjaga konsistensi urutan indeks FAISS
        sorted_records = sorted(data["records"], key=lambda r: int(r["id"]))
        
        return [
            {
                "id": int(r["id"]),
                "file_name": str(r["file_name"]),
                "chunk_index": int(r["chunk_index"]),
                "content": str(r["content"]),
                "created_at": r["created_at"]
            }
            for r in sorted_records
        ]

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------
    def delete_document(self, file_name: str) -> bool:
        """[DELETE] Hapus semua chunk milik file_name. Return True jika ada yang terhapus."""
        with _LOCK:
            data    = _load()
            before  = len(data["records"])
            # Filter hanya dokumen yang namanya TIDAK sama dengan file_name
            data["records"] = [r for r in data["records"] if r["file_name"] != file_name]
            after   = len(data["records"])
            
            if before != after:
                _save(data)
                
        return before != after