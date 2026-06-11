import os
import time  # <-- Tambahkan import ini
import mysql.connector

class DatabaseManager:
    def __init__(self):
        # 1. Konfigurasi disesuaikan dengan Environment Variables bawaan Railway
        self.host = os.environ.get('MYSQLHOST', os.environ.get('DB_HOST', '127.0.0.1'))
        self.user = os.environ.get('MYSQLUSER', os.environ.get('DB_USER', 'root'))
        self.password = os.environ.get('MYSQLPASSWORD', os.environ.get('DB_PASSWORD', 'password'))
        self.database_name = os.environ.get('MYSQLDATABASE', os.environ.get('DB_NAME', 'uho_rag_db'))
        
        port_env = os.environ.get('MYSQLPORT', os.environ.get('DB_PORT', '3306'))
        self.port = int(port_env)

        # Simpan konfigurasi ke self.config DULU sebelum dipakai di fungsi inisialisasi
        self.config = {
            'host': self.host,
            'user': self.user,
            'password': self.password,
            'database': self.database_name,
            'port': self.port
        }

        # 2. Jalankan inisialisasi otomatis dengan mekanisme "Sabar Menunggu"
        self._safe_auto_initialize()

    def _safe_auto_initialize(self):
        """Mencoba menjalankan inisialisasi tabel dengan toleransi waktu tunggu (Retry)"""
        max_retries = 5
        delay = 5  # Jeda waktu dalam detik setiap kali gagal
        
        for i in range(max_retries):
            try:
                print(f"[*] Mencoba menginisialisasi database... (Percobaan {i+1}/{max_retries})")
                self._auto_initialize_db_and_tables()
                print("[+] Database & Tabel berhasil diinisialisasi!")
                return  # Keluar dari fungsi jika sukses
            except mysql.connector.errors.DatabaseError as err:
                # Jika error-nya karena masalah koneksi (seperti error 111 atau 2003)
                print(f"[!] MySQL belum siap atau koneksi ditolak: {err}")
                if i < max_retries - 1:
                    print(f"[*] Menunggu {delay} detik sebelum mencoba kembali...")
                    time.sleep(delay)
                else:
                    print("[-] Sudah mencoba 5 kali dan tetap gagal. Menghentikan aplikasi.")
                    raise err # Lempar error asli jika sudah mentok gagal terus

    def _auto_initialize_db_and_tables(self):
        # Di sini isi fungsi kamu yang lama untuk membuat DB dan tabel.
        # Pastikan fungsi ini menggunakan koneksi biasa.
        pass

    def get_connection(self):
        # Tambahkan juga retry mini di sini untuk operasional sehari-hari jika diperlukan
        return mysql.connector.connect(**self.config)