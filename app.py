# app.py
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pypdf
import docx2txt

# 1. Load file .env di awal sebelum inisialisasi class lain
load_dotenv()

from rag_engine import RAGEngine
from database import DatabaseManager

app = Flask(__name__)

# =====================================================================
# MULTI API KEY LOADER
# Baca GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, dst. dari .env
# =====================================================================
def load_api_keys() -> list:
    """
    Kumpulkan semua Gemini API key yang tersedia dari .env.
    Key utama: GEMINI_API_KEY
    Key cadangan: GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
    """
    keys = []
    # Key utama
    primary = os.environ.get("GEMINI_API_KEY")
    if primary:
        keys.append(primary)
    # Key cadangan (indeks mulai dari 2)
    i = 2
    while True:
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1

    if not keys:
        raise ValueError("CRITICAL ERROR: Tidak ada GEMINI_API_KEY yang ditemukan di file .env!")

    print(f"--> [CONFIG] {len(keys)} API Key Gemini berhasil dimuat.")
    return keys


# 2. Inisialisasi Instance Service
GEMINI_API_KEYS = load_api_keys()
rag_engine      = RAGEngine(api_keys=GEMINI_API_KEYS)
db_manager      = DatabaseManager()

# Folder lokal untuk base knowledge permanen
DOCUMENTS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)


# =====================================================================
# HELPER: EKSTRAK TEKS
# =====================================================================

def extract_text_from_file(file) -> str:
    """Ekstrak teks dari file PDF, DOCX, atau TXT (objek file Flask)."""
    filename       = file.filename.lower()
    extracted_text = ""

    if filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

    elif filename.endswith(".docx"):
        extracted_text = docx2txt.process(file)

    elif filename.endswith(".txt"):
        extracted_text = file.read().decode("utf-8")

    return extracted_text.strip()


def extract_text_from_path(filepath: str) -> str:
    """Ekstrak teks dari file di disk (path lokal)."""
    extracted_text = ""
    filename       = filepath.lower()

    try:
        if filename.endswith(".pdf"):
            with open(filepath, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

        elif filename.endswith(".docx"):
            extracted_text = docx2txt.process(filepath)

        elif filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()

    except Exception as e:
        print(f"    [WARNING] Gagal membaca file '{filepath}': {e}")

    return extracted_text.strip()


# =====================================================================
# BOOTSTRAP: SYNC DOKUMEN LOKAL KE DB
# =====================================================================

def load_local_documents_to_db():
    """
    Scan folder documents/ dan indexkan file yang BELUM ada di database.
    Dipanggil saat bootstrap supaya base knowledge lokal selalu tersinkron.
    """
    supported_ext = (".pdf", ".docx", ".txt")
    files_in_folder = [
        f for f in os.listdir(DOCUMENTS_FOLDER)
        if f.lower().endswith(supported_ext)
    ]

    if not files_in_folder:
        print("--> [LOCAL DOCS] Folder documents/ kosong, tidak ada file lokal untuk di-load.")
        return

    existing_docs      = db_manager.get_all_documents_metadata()
    existing_filenames = {doc["file_name"] for doc in existing_docs}
    new_files_indexed  = 0

    for filename in files_in_folder:
        if filename in existing_filenames:
            print(f"--> [LOCAL DOCS] '{filename}' sudah ada di database, dilewati.")
            continue

        filepath = os.path.join(DOCUMENTS_FOLDER, filename)
        print(f"--> [LOCAL DOCS] Mengindeks dokumen baru dari disk: '{filename}'...")

        content = extract_text_from_path(filepath)
        if not content:
            print(f"    [WARNING] File '{filename}' kosong atau tidak bisa diekstrak, dilewati.")
            continue

        chunks = rag_engine.split_text(content)
        db_manager.save_chunks(filename, chunks)
        print(f"    [OK] '{filename}' berhasil disimpan ke DB ({len(chunks)} chunks).")
        new_files_indexed += 1

    if new_files_indexed > 0:
        print(f"--> [LOCAL DOCS] {new_files_indexed} dokumen lokal baru berhasil diindeks ke database.")
    else:
        print("--> [LOCAL DOCS] Semua dokumen lokal sudah sinkron dengan database.")

    return new_files_indexed  # return jumlah file baru agar bootstrap tahu perlu rebuild


# =====================================================================
# BOOTSTRAP: RELOAD VECTOR INDEX (DENGAN FAISS CACHE)
# =====================================================================

def reload_vector_index(force_rebuild: bool = False):
    """
    Bangun/muat ulang FAISS index dengan strategi hemat token:

    1. Cek apakah file cache FAISS sudah ada di disk.
    2. Jika ada DAN jumlah chunk sama dengan DB → load dari disk (0 token API).
    3. Jika tidak ada / jumlah berbeda / force_rebuild=True → embed ulang dari DB.

    Args:
        force_rebuild: Paksa rebuild meski cache tersedia (e.g. setelah delete dokumen).
    """
    all_chunks      = db_manager.get_all_chunks()
    db_chunk_count  = len(all_chunks)

    if db_chunk_count == 0:
        print("--> [VECTOR INDEX] Database kosong. Index FAISS dikosongkan.")
        rag_engine.reset_index()
        return

    # === COBA LOAD CACHE DULU ===
    if not force_rebuild:
        cache_loaded = rag_engine.load_index_from_disk()
        if cache_loaded and rag_engine.is_index_synced(db_chunk_count):
            print(
                f"--> [VECTOR INDEX] Cache valid ({rag_engine.index.ntotal} vectors == {db_chunk_count} chunks di DB). "
                f"Skip embedding, 0 token API digunakan."
            )
            return
        elif cache_loaded:
            print(
                f"--> [VECTOR INDEX] Cache tidak sinkron "
                f"(cache={rag_engine.index.ntotal} vectors, DB={db_chunk_count} chunks). "
                f"Rebuild diperlukan."
            )

    # === REBUILD DARI DB ===
    print(f"--> [VECTOR INDEX] Membangun ulang index dari {db_chunk_count} chunks database...")
    try:
        rag_engine.reset_index()
        rag_engine.add_documents(all_chunks, save_cache=True)  # simpan cache setelah selesai
        print(f"--> [VECTOR INDEX] Rebuild selesai. {rag_engine.index.ntotal} vectors aktif.")
    except Exception as e:
        print(f"--> [VECTOR INDEX ERROR] Gagal rebuild index: {e}")
        raise


# =====================================================================
# BOOTSTRAPPING SAAT SERVER FLASK PERTAMA KALI DINYALAKAN
# =====================================================================

with app.app_context():
    # LANGKAH 1: Scan folder documents/ dan masukkan file baru ke DB
    new_docs_count = load_local_documents_to_db()

    # LANGKAH 2: Load/rebuild FAISS index
    # force_rebuild=True hanya jika ada dokumen baru yang baru saja dimasukkan ke DB
    reload_vector_index(force_rebuild=bool(new_docs_count))


# =====================================================================
# RENDER TEMPLATE ROUTE
# =====================================================================

@app.route("/")
def index():
    """Menampilkan halaman utama dashboard dan antarmuka chatbot."""
    return render_template("index.html")


# =====================================================================
# MANAGEMENT KNOWLEDGE BASE API
# =====================================================================

@app.route("/api/knowledge", methods=["POST"])
def create_knowledge():
    """[CREATE] Upload banyak file, ekstrak teks, chunking, simpan ke MySQL & perbarui FAISS."""
    if "files" not in request.files:
        return jsonify({"error": "Tidak ada berkas yang diunggah"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "Daftar file unggahan kosong"}), 400

    existing_docs      = db_manager.get_all_documents_metadata()
    existing_filenames = {doc["file_name"] for doc in existing_docs}

    success_files = []
    errors        = []

    for file in files:
        file_name = file.filename

        if file_name in existing_filenames:
            errors.append(f"Berkas '{file_name}' dilewati karena sudah terdaftar di sistem.")
            continue

        try:
            content = extract_text_from_file(file)
            if not content:
                errors.append(f"Gagal mengekstrak teks dari '{file_name}' (file kosong/corrupt).")
                continue

            chunks = rag_engine.split_text(content)

            # Simpan ke DB
            db_manager.save_chunks(file_name, chunks)

            # Tambahkan ke FAISS index di memori (tanpa save cache dulu)
            rag_engine.add_documents(chunks, save_cache=False)

            # Simpan salinan fisik ke folder documents/
            save_path = os.path.join(DOCUMENTS_FOLDER, file_name)
            if not os.path.exists(save_path):
                try:
                    file.stream.seek(0)
                    with open(save_path, "wb") as out_f:
                        out_f.write(file.stream.read())
                    print(f"--> [UPLOAD] Salinan fisik '{file_name}' disimpan ke folder documents/.")
                except Exception as save_err:
                    print(f"--> [UPLOAD WARNING] Gagal menyimpan salinan fisik '{file_name}': {save_err}")

            success_files.append({"file_name": file_name, "total_chunks": len(chunks)})

        except Exception as e:
            errors.append(f"Gagal memproses file '{file_name}': {str(e)}")

    # Simpan cache FAISS ke disk setelah semua file selesai diproses
    if success_files:
        rag_engine.save_index_to_disk()

    return jsonify({
        "message":      f"Proses batch upload selesai. {len(success_files)} file berhasil di-indeks.",
        "success_files": success_files,
        "errors":        errors
    }), 200


@app.route("/api/knowledge", methods=["GET"])
def read_knowledge():
    """[READ] Ambil list metadata dokumen yang terindeks."""
    try:
        documents = db_manager.get_all_documents_metadata()
        return jsonify({"status": "success", "data": documents}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/knowledge", methods=["DELETE"])
def delete_knowledge():
    """[DELETE] Hapus dokumen dari MySQL dan rebuild FAISS index."""
    data = request.get_json()
    if not data or "file_name" not in data:
        return jsonify({"error": "Parameter file_name wajib disertakan"}), 400

    file_name = data.get("file_name")
    try:
        is_deleted = db_manager.delete_document(file_name)
        if not is_deleted:
            return jsonify({"error": f"Dokumen '{file_name}' tidak ditemukan."}), 404

        local_path = os.path.join(DOCUMENTS_FOLDER, file_name)
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"--> [DELETE] Salinan fisik '{file_name}' dihapus dari folder documents/.")

        # Delete = data berubah → force rebuild + simpan cache baru
        reload_vector_index(force_rebuild=True)

        return jsonify({"message": f"Dokumen '{file_name}' berhasil dihapus secara permanen dari sistem RAG."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================================
# CHATBOT CORE API
# =====================================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    """[CHAT API] Proses pertanyaan user menggunakan FAISS + Gemini LLM."""
    data         = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Pesan dari mahasiswa tidak boleh kosong"}), 400

    try:
        bot_response = rag_engine.generate_answer(user_message)
        return jsonify({"status": "success", "response": bot_response}), 200
    except Exception as e:
        return jsonify({"error": f"Gagal memproses pembuatan jawaban: {str(e)}"}), 500


if __name__ == "__main__":
    flask_port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host="0.0.0.0", port=flask_port, debug=True)
