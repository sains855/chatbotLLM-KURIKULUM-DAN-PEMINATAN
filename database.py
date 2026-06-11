import os
import mysql.connector

class DatabaseManager:
    def __init__(self):
        # 1. Konfigurasi disesuaikan dengan Environment Variables bawaan Railway
        # Jika tidak ada, baru dia akan fallback ke nilai lokal (untuk development)
        self.host = os.environ.get('MYSQLHOST', os.environ.get('DB_HOST', '127.0.0.1'))
        self.user = os.environ.get('MYSQLUSER', os.environ.get('DB_USER', 'root'))
        self.password = os.environ.get('MYSQLPASSWORD', os.environ.get('DB_PASSWORD', 'password'))
        self.database_name = os.environ.get('MYSQLDATABASE', os.environ.get('DB_NAME', 'uho_rag_db'))
        
        # Railway memberikan port dalam bentuk string, wajib di-cast ke int
        port_env = os.environ.get('MYSQLPORT', os.environ.get('DB_PORT', '3306'))
        self.port = int(port_env)

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