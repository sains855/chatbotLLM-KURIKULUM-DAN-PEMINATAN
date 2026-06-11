# Folder Base Knowledge Dokumen

Letakkan file dokumen knowledge base di folder ini.
Format yang didukung: .pdf, .docx, .txt

Saat server Flask dinyalakan, semua dokumen di folder ini
akan otomatis diindeks ke database dan FAISS vector index.

Contoh isi folder:
  - kurikulum_ilmu_komputer.pdf
  - jadwal_akademik_2024.docx
  - info_beasiswa.txt

Catatan:
- Dokumen yang sudah pernah diindeks tidak akan diproses ulang
- Hapus dokumen dari UI dashboard atau lewat API DELETE jika ingin menghapus
