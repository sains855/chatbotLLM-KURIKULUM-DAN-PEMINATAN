# database.py
import os
import mysql.connector

class DatabaseManager:
    def __init__(self):
        # 1. Konfigurasi awal diserap dari file .env
        self.host = os.environ.get('DB_HOST', '127.0.0.1')
        self.user = os.environ.get('DB_USER', 'root')
        self.password = os.environ.get('DB_PASSWORD', 'password')
        self.database_name = os.environ.get('DB_NAME', 'uho_rag_db')
        self.port = int(os.environ.get('DB_PORT', 3306))

        # 2. Jalankan inisialisasi otomatis Database & Tabel saat objek dibuat
        self._auto_initialize_db_and_tables()

        # 3. Konfigurasi final untuk koneksi operasional CRUD
        self.config = {
            'host': self.host,
            'user': self.user,
            'password': self.password,
            'database': self.database_name,
            'port': self.port
        }

    def _auto_initialize_db_and_tables(self):
        """Mekanisme otomatis untuk membuat database dan tabel jika belum eksis"""
        try:
            # Koneksi awal ke MySQL server tanpa memilih database terlebih dahulu
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port
            )
            cursor = conn.cursor()

            # A. Buat Database jika belum ada
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name}")
            
            # Pilih database yang baru saja dibuat/dipastikan ada
            cursor.execute(f"USE {self.database_name}")

            # B. Buat Tabel knowledge_base jika belum ada
            create_table_query = """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            cursor.execute(create_table_query)
            
            conn.commit()
            cursor.close()
            conn.close()
            print(f"--> [DATABASE INFO] Database '{self.database_name}' dan tabel 'knowledge_base' siap digunakan.")
            
        except mysql.connector.Error as err:
            print(f"--> [DATABASE ERROR] Gagal melakukan inisialisasi otomatis: {err}")

    def get_connection(self):
        """Membuka koneksi operasional ke database"""
        return mysql.connector.connect(**self.config)

    def save_chunks(self, file_name: str, chunks: list):
        """[CREATE] Menyimpan banyak chunks dokumen sekaligus ke MySQL"""
        conn = self.get_connection()
        cursor = conn.cursor()
        query = "INSERT INTO knowledge_base (file_name, chunk_index, content) VALUES (%s, %s, %s)"
        data_to_insert = [(file_name, index, chunk) for index, chunk in enumerate(chunks)]
        cursor.executemany(query, data_to_insert)
        conn.commit()
        cursor.close()
        conn.close()

    def get_all_documents_metadata(self):
        """[READ] Mengambil daftar file unik yang sudah di-upload"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT file_name, COUNT(*) as total_chunks, MAX(created_at) as uploaded_at FROM knowledge_base GROUP BY file_name"
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results

    def get_all_chunks(self):
        """[READ] Mengambil seluruh teks chunk secara berurutan untuk inisialisasi ulang FAISS"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT content FROM knowledge_base ORDER BY id ASC"
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row['content'] for row in results]

    def delete_document(self, file_name: str):
        """[DELETE] Menghapus dokumen berdasarkan nama file dari MySQL"""
        conn = self.get_connection()
        cursor = conn.cursor()
        query = "DELETE FROM knowledge_base WHERE file_name = %s"
        cursor.execute(query, (file_name,))
        conn.commit()
        affected_rows = cursor.rowcount
        cursor.close()
        conn.close()
        return affected_rows > 0