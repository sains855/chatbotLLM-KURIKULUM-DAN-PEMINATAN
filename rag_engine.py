# rag_engine.py  ─ v3: Semantic Chunking + Threshold Filtering + Query Cache
import os
import re
import pickle
import hashlib
import numpy as np
import faiss
from collections import OrderedDict
from google import genai
from google.genai import types

FAISS_INDEX_PATH = "my_index.faiss"
FAISS_DOCS_PATH  = "my_index_docs.pkl"

# ---------------------------------------------------------------------------
# LRU Cache sederhana (tidak butuh library eksternal)
# Menyimpan hasil generate_answer agar pertanyaan identik/mirip tidak memanggil
# API lagi → hemat token drastis untuk FAQ berulang.
# ---------------------------------------------------------------------------
class _LRUCache:
    def __init__(self, maxsize: int = 128):
        self._cache   = OrderedDict()
        self._maxsize = maxsize

    def _key(self, text: str) -> str:
        # Normalize: lowercase + strip whitespace → hash
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def _looks_truncated(answer: str) -> bool:
        """
        Deteksi apakah jawaban yang tersimpan di cache kemungkinan terpotong.
        Heuristik: jawaban yang berakhir tanpa tanda baca penutup yang wajar
        (titik, tanda tanya, tanda seru, kurung tutup, petik) dianggap terpotong.
        """
        text = answer.strip()
        if not text:
            return True
        # Tanda akhir yang valid
        valid_endings = ('.', '!', '?', ')', '"', "'", '»', '…', ':')
        # Juga terima jika baris terakhir adalah item list / header pendek
        last_line = text.splitlines()[-1].strip()
        if last_line.endswith(valid_endings):
            return False
        # Jika baris terakhir adalah list bullet tanpa tanda baca, toleransi
        if re.match(r'^[-•*\d]+[.)]\s+\S', last_line):
            return False
        return True  # kemungkinan terpotong

    def get(self, text: str):
        k = self._key(text)
        if k in self._cache:
            cached_val = self._cache[k]
            # Invalidasi otomatis jika jawaban tersimpan terdeteksi terpotong
            if self._looks_truncated(cached_val):
                del self._cache[k]
                print("--> [CACHE INVALIDATE] Jawaban lama terpotong dihapus dari cache.")
                return None
            self._cache.move_to_end(k)       # refresh LRU order
            return cached_val
        return None

    def set(self, text: str, value):
        k = self._key(text)
        self._cache[k] = value
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # buang item terlama


# ---------------------------------------------------------------------------
# SEMANTIC TEXT SPLITTER
# Prioritas batas: paragraf → kalimat → kata
# Jauh lebih baik dari split karakter mentah karena konteks tidak terpotong.
# ---------------------------------------------------------------------------
class _SemanticSplitter:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap    = overlap

    def _split_sentences(self, text: str) -> list:
        """Pecah teks menjadi kalimat menggunakan regex (tanpa NLTK)."""
        # Tangani singkatan umum agar tidak dianggap akhir kalimat
        text = re.sub(r'\b(Prof|Dr|Mr|Mrs|Ms|No|Jl|Vol|hal|dst|dll|dsb|hlm|dkk)\.',
                      r'\1<ABBR>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.replace('<ABBR>', '.') for s in sentences if s.strip()]

    def split(self, text: str) -> list:
        # Bersihkan whitespace berlebih
        text = re.sub(r'\n{3,}', '\n\n', text.strip())

        # Pecah per paragraf dulu
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        chunks   = []
        current  = ""

        for para in paragraphs:
            # Jika paragraf sendiri > chunk_size, pecah per kalimat
            if len(para) > self.chunk_size:
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= self.chunk_size:
                        current = (current + " " + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        # Overlap: ambil akhir chunk sebelumnya
                        overlap_text = current[-self.overlap:] if len(current) > self.overlap else current
                        current = (overlap_text + " " + sent).strip()
            else:
                if len(current) + len(para) + 2 <= self.chunk_size:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current:
                        chunks.append(current)
                    overlap_text = current[-self.overlap:] if len(current) > self.overlap else current
                    current = (overlap_text + "\n\n" + para).strip()

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c.strip()) > 30]  # buang serpihan


# ---------------------------------------------------------------------------
# RAG ENGINE UTAMA
# ---------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("Minimal satu GEMINI_API_KEY harus disediakan.")

        self.api_keys            = api_keys
        self._current_key_index  = 0
        self.client              = genai.Client(api_key=self.api_keys[0])

        self.embedding_model = "gemini-embedding-001"
        self.embedding_dim   = 768   # hemat 75% vs 3072 default

        self.index     = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []

        # Splitter semantik (gantikan split karakter naif)
        self._splitter = _SemanticSplitter(chunk_size=800, overlap=150)

        # Cache hasil jawaban (hemat token untuk pertanyaan berulang)
        self._answer_cache = _LRUCache(maxsize=256)

        # Threshold minimum cosine similarity untuk menerima chunk
        # Chunk dengan score < threshold dianggap tidak relevan → dibuang
        self.RELEVANCE_THRESHOLD = 0.30

    # ── API KEY ROTATION ────────────────────────────────────────────────────

    def _rotate_key(self):
        next_idx = (self._current_key_index + 1) % len(self.api_keys)
        if next_idx == self._current_key_index:
            raise RuntimeError(
                "Rate limit & tidak ada API Key cadangan. "
                "Tambahkan GEMINI_API_KEY_2, GEMINI_API_KEY_3 ke .env"
            )
        self._current_key_index = next_idx
        self.client = genai.Client(api_key=self.api_keys[self._current_key_index])
        print(f"--> [KEY ROTATION] Beralih ke key #{self._current_key_index + 1}/{len(self.api_keys)}")

    def _is_rate_limit(self, e: Exception) -> bool:
        msg = str(e).lower()
        return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg

    # ── FAISS PERSISTENT CACHE ──────────────────────────────────────────────

    def save_index_to_disk(self, index_path=FAISS_INDEX_PATH, docs_path=FAISS_DOCS_PATH):
        try:
            faiss.write_index(self.index, index_path)
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"--> [FAISS CACHE] Disimpan: {self.index.ntotal} vectors → '{index_path}'")
        except Exception as e:
            print(f"--> [FAISS CACHE WARNING] Gagal simpan: {e}")

    def load_index_from_disk(self, index_path=FAISS_INDEX_PATH, docs_path=FAISS_DOCS_PATH) -> bool:
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            return False
        try:
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print(f"--> [FAISS CACHE] Di-load dari disk: {self.index.ntotal} vectors (0 token API)")
            return True
        except Exception as e:
            print(f"--> [FAISS CACHE WARNING] Cache rusak, rebuild: {e}")
            return False

    def is_index_synced(self, db_chunk_count: int) -> bool:
        return self.index.ntotal == db_chunk_count

    # ── TEXT SPLITTING ───────────────────────────────────────────────────────

    def split_text(self, text: str, chunk_size: int = None, overlap: int = None) -> list:
        """
        Semantic splitting: potong di batas paragraf/kalimat, bukan karakter.
        chunk_size & overlap diabaikan (untuk kompatibilitas mundur); gunakan
        nilai default _SemanticSplitter di __init__.
        """
        return self._splitter.split(text)

    # ── EMBEDDING ────────────────────────────────────────────────────────────

    def get_embedding(self, text: str, max_retries: int = None) -> list:
        if max_retries is None:
            max_retries = max(len(self.api_keys), 1)

        last_error = None
        for _ in range(max_retries):
            try:
                resp = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=self.embedding_dim)
                )
                emb  = np.array(resp.embeddings[0].values, dtype="float32")
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb.tolist()
            except Exception as e:
                last_error = e
                if self._is_rate_limit(e) and len(self.api_keys) > 1:
                    print(f"--> [EMBEDDING] Rate limit key #{self._current_key_index + 1}: {e}")
                    try:
                        self._rotate_key()
                    except RuntimeError as re_err:
                        raise RuntimeError(str(re_err)) from e
                else:
                    raise

        raise RuntimeError(f"Embedding gagal setelah {max_retries} coba: {last_error}")

    def add_documents(self, doc_list: list, save_cache: bool = False):
        if not doc_list:
            return
        embeddings = []
        for i, doc in enumerate(doc_list):
            embeddings.append(self.get_embedding(doc))
            self.documents.append(doc)
            if (i + 1) % 50 == 0:
                print(f"--> [EMBEDDING] {i + 1}/{len(doc_list)} chunks selesai")
        self.index.add(np.array(embeddings, dtype="float32"))
        if save_cache:
            self.save_index_to_disk()

    def reset_index(self):
        self.index     = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []

    # ── RETRIEVAL DENGAN THRESHOLD ───────────────────────────────────────────

    def retrieve(self, query: str, k: int = 6) -> list:
        """
        Retrieve k kandidat chunk, lalu filter dengan RELEVANCE_THRESHOLD.
        Mengembalikan list of (chunk_text, score) — hanya yang benar-benar relevan.
        """
        if self.index.ntotal == 0:
            return []

        query_emb = np.array([self.get_embedding(query)], dtype="float32")
        actual_k  = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, actual_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            if float(score) >= self.RELEVANCE_THRESHOLD:
                results.append((self.documents[idx], float(score)))

        # Urutkan score tertinggi dulu
        results.sort(key=lambda x: x[1], reverse=True)
        return results  # list of (text, score)

    # ── GENERATION DENGAN PROMPT SYNTHESIS ──────────────────────────────────

    def generate_answer(self, query: str, max_retries: int = None) -> str:
        """
        Pipeline lengkap:
          1. Cek LRU cache → jika hit, langsung return (0 token)
          2. Retrieve + filter threshold
          3. Bangun prompt sintesis (reasoning, bukan copy-paste)
          4. Generate via Gemini dengan fallback model & key rotation
          5. Simpan hasil ke cache
        """
        # === STEP 1: CACHE CHECK ===
        cached = self._answer_cache.get(query)
        if cached:
            print(f"--> [CACHE HIT] Jawaban diambil dari cache, 0 token API digunakan.")
            return cached

        if max_retries is None:
            max_retries = max(len(self.api_keys), 1)

        # === STEP 2: RETRIEVE + FILTER ===
        retrieved = self.retrieve(query, k=6)

        if not retrieved:
            no_data_msg = (
                "Maaf, saya belum memiliki informasi yang cukup relevan untuk menjawab "
                "pertanyaan ini. Silakan hubungi bagian akademik atau dosen pembimbing "
                "untuk informasi lebih lanjut."
            )
            return no_data_msg

        # Susun blok konteks dengan skor relevansi (membantu LLM prioritas)
        context_blocks = []
        for i, (chunk, score) in enumerate(retrieved):
            context_blocks.append(
                f"[Konteks {i+1} | Relevansi: {score:.2f}]\n{chunk}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        # === STEP 3: PROMPT SINTESIS ===
        # Prompt ini mendorong LLM untuk:
        # a) Memahami & menyimpulkan, bukan meniru teks
        # b) Menggabungkan info dari beberapa konteks
        # c) Menjawab meski phrasing berbeda dari dokumen asli
        # d) Jujur jika info tidak ada (anti-hallusinasi)
        system_instruction = """\
Anda adalah Asisten Akademik cerdas untuk Program Studi Ilmu Komputer FMIPA \
Universitas Halu Oleo (UHO), Kendari.

KEMAMPUAN ANDA:
• Memahami maksud pertanyaan meskipun kalimatnya berbeda dari dokumen
• Menyimpulkan dan mensintesis informasi dari beberapa sumber konteks
• Menjelaskan dengan bahasa yang mudah dipahami mahasiswa
• Menghubungkan informasi antar konteks untuk jawaban yang komprehensif

ATURAN MENJAWAB:
1. PAHAMI dulu intent pertanyaan, bukan hanya kata kuncinya
2. SINTESIS: gabungkan informasi dari semua konteks yang relevan — jangan \
hanya copy-paste satu konteks
3. INFERENSI: jika jawabannya bisa disimpulkan dari konteks meski tidak \
disebutkan eksplisit, jelaskan dengan logis
4. JUJUR: jika setelah membaca semua konteks informasinya benar-benar tidak \
ada, katakan dengan sopan dan sarankan menghubungi pihak jurusan
5. GAYA: gunakan bahasa Indonesia yang ramah, terstruktur, dan informatif
6. JANGAN: mengarang fakta di luar konteks yang diberikan"""

        prompt = f"""Berikut adalah konteks dari knowledge base akademik UHO yang \
relevan dengan pertanyaan mahasiswa. Setiap blok konteks sudah diurutkan \
berdasarkan relevansi (1.00 = sangat relevan).

=== KONTEKS ===
{context_text}

=== PERTANYAAN MAHASISWA ===
{query}

=== INSTRUKSI ===
Berdasarkan konteks di atas, berikan jawaban yang:
- Langsung menjawab inti pertanyaan
- Menyintesis informasi dari semua konteks yang relevan
- Menggunakan penalaran jika perlu menyimpulkan
- Terstruktur dan mudah dipahami mahasiswa

Jawaban:"""

        # === STEP 4: GENERATE + FALLBACK ===
        # max_output_tokens sengaja besar (8192) agar jawaban tidak terpotong.
        # Hemat token tetap dijaga di sisi retrieval (hanya chunk relevan masuk)
        # dan cache (pertanyaan sama tidak generate ulang).
        MAX_OUTPUT_TOKENS = 8192
        models_to_try     = ["gemini-2.5-flash", "gemini-1.5-flash"]

        for model_name in models_to_try:
            for attempt in range(max_retries):
                try:
                    print(f"--> [AI] {model_name} | key #{self._current_key_index + 1} | attempt {attempt + 1}")
                    resp = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        )
                    )

                    # ── VALIDASI finish_reason sebelum menerima jawaban ──────
                    # Gemini mengembalikan finish_reason di candidate pertama.
                    # STOP   = selesai normal → aman dicache
                    # MAX_TOKENS = terpotong  → JANGAN dicache, coba lagi
                    finish_reason = None
                    try:
                        finish_reason = resp.candidates[0].finish_reason
                        # finish_reason bisa berupa enum atau string tergantung SDK
                        finish_str = str(finish_reason).upper()
                    except Exception:
                        finish_str = "STOP"  # fallback aman jika atribut tidak ada

                    if "MAX_TOKEN" in finish_str:
                        # Terpotong karena token limit — ini seharusnya tidak
                        # terjadi dengan MAX_OUTPUT_TOKENS=8192, tapi jaga-jaga.
                        print(
                            f"--> [GENERATION WARNING] Jawaban terpotong (MAX_TOKENS) "
                            f"pada {model_name}. Tidak dicache, coba model berikutnya."
                        )
                        break  # lanjut ke model berikutnya

                    answer = resp.text

                    # Validasi tambahan: jawaban tidak boleh kosong
                    if not answer or not answer.strip():
                        print(f"--> [GENERATION WARNING] Jawaban kosong dari {model_name}, coba lagi.")
                        continue

                    # === STEP 5: SIMPAN KE CACHE (hanya jawaban lengkap) ===
                    self._answer_cache.set(query, answer)
                    print(f"--> [CACHE SET] Jawaban lengkap disimpan ke cache (finish: {finish_str}).")

                    return answer

                except Exception as e:
                    if self._is_rate_limit(e) and len(self.api_keys) > 1:
                        print(f"--> [GENERATION] Rate limit {model_name} key #{self._current_key_index + 1}")
                        try:
                            self._rotate_key()
                        except RuntimeError:
                            break
                    else:
                        print(f"--> [GENERATION] {model_name} error: {e} → coba model berikutnya")
                        break

        return (
            "Mohon maaf, server AI sedang mengalami lonjakan permintaan. "
            "Silakan coba lagi beberapa saat."
        )
