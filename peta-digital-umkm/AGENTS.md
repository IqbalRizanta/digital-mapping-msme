# AGENTS — Panduan untuk Agen AI dan Kontributor

> **Status:** Final Terverifikasi · 28 Oktober 2026  
> **Rujukan Resmi Arsitektur & Flowchart:**
> 1. `diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json` (`bd-ai-archi` dan `flowchart-bd-ai`)
> 2. `diagram-architecture/flowchart-mandiri.json` (`flowchart-mandiri-gmaps-realworld`)

Dokumen ini adalah pedoman operasional wajib bagi agen AI asisten pengkodean maupun pengembang yang bekerja di repositori **Peta Digital UMKM Kuliner Surabaya Timur**.

---

## 1. Ringkasan Arsitektur & Peran Data Mandiri
Platform ini mengimplementasikan satu alur pipeline analitik big data multimodal terpadu yang memadukan:
1. **Data Spasial & Tabular:** Metadata 190 UMKM kuliner di 6 kecamatan Surabaya Timur.
2. **Data Teks Ulasan:** 730 ulasan pelanggan teranonimkan dari Google Places API.
3. **Data Citra Kurasi Ketat:** 274 foto stan fisik dan indikator digital (`qris`, `edc`, `pos`, `merchant_online`). Foto makanan, minuman, menu, dan ilustrasi digital AI disaring 100% menggunakan filter zero-shot dual-gate OpenAI CLIP.
4. **Data Dokumen:** Dokumen regulasi perizinan dan insentif dinas.

Alur data mandiri (*Google Maps API + Survei Lapangan SWK*) memvalidasi kesepakatan indikator visual menggunakan statistik **Cohen's Kappa ($\kappa \ge 0.70$)** sebelum digabungkan ke skema tabel Gold kelompok.

---

## 2. Aturan Mutlak Tata Kelola (Non-Negotiable Guardrails)
1. **Nol Identitas Personal (Zero Student Identity):**
   - File notebook (`eda_3_*.ipynb`, `model_3_*.ipynb`, `bda_3_*.ipynb`) **DILARANG KERAS** memuat identitas personal (nama mahasiswa, NIM, maupun nama dosen pembimbing) di dalam konten markdown, sel teks, maupun komentar kode.
2. **Kepatuhan Privasi Publik (GDPR / UU PDP):**
   - 100% wajah pengunjung/kasir yang tertangkap kamera wajib disanitasi menggunakan OpenCV Gaussian Face Blurring sebelum masuk zona Clean.
   - Nomor kontak dan identitas pengulas teks disanitasi menggunakan ekspresi reguler (regex).
3. **Integritas Cloud Storage (Google Drive):**
   - Penyimpanan utama antar-tahap adalah Google Drive (`G:\My Drive\` / Google Colab).
   - Seluruh output otomatis tersimpan di direktori target:
     - `G:\My Drive\clean\` (tabel Parquet dan foto bersih terburamkan)
     - `G:\My Drive\Iqbal-EDA\` (notebook EDA dan 6 grafik visual)
     - `G:\My Drive\Iqbal-Model-Evaluation\` (notebook evaluasi, model weights, dan deliverables visual YOLOv8)
     - `G:\My Drive\Iqbal-BDA\` (notebook BDA dan peta kesenjangan spasial)
4. **Validasi Anti-Kebocoran Data (Anti-Leakage):**
   - Pembagian data latih/uji wajib menggunakan `GroupKFold` ($K=5$) berdasarkan `umkm_id`.
   - Scaler dan encoder hanya dipasang pada data latih dalam Pipeline.
   - Label kelas kematangan berasal dari survei lapangan nyata di SWK, **bukan** dari hasil klastering.
5. **Kurasi Citra Bersih (Zero Non-Target):**
   - Foto pada zona Clean **HANYA** memuat stan/etalase/kasir dan instrumen pembayaran. Tidak ada foto piring makanan atau minuman. Usaha waralaba besar tanpa foto kasir/pembayaran wajib dipangkas.

---

## 3. Pembagian Tugas Agen Khusus (Specialist Agent Roles)

| Peran Agen | Tugas & Tanggung Jawab | Output & Kontrak Data |
| :--- | :--- | :--- |
| **Ingestion Agent** | Mengambil data Places API, survei SWK, dan dokumen regulasi; memeriksa gerbang volume data ($\ge 200\text{ MB}$, $\ge 2.000$ baris). | Raw Data di Google Drive & Data Collection Schema |
| **Data Quality & Filter Agent** | Menjalankan filter dual-gate OpenAI CLIP, memangkas usaha rantai besar, mendeteksi dan memburamkan 315 wajah via OpenCV, serta membersihkan teks ulasan. | `clean/*.parquet` dan `clean/photos/` |
| **Vision Multi-Task Agent** | Menjalankan 4 task Ultralytics YOLOv8: Deteksi bounding box, Segmentasi poligon, Estimasi pose skeleton, dan Klasifikasi kematangan visual; mengekspor log metrik. | `reports/yolo/` dan `G:\My Drive\Iqbal-Model-Evaluation\yolo_outputs/` |
| **Cross-Validation Agent** | Menghitung koefisien kesepakatan silang Cohen's Kappa antara Google Maps dan Survei SWK ($\kappa \ge 0.70$); menyusun skema tabel Gold. | Matriks kesepakatan dan `gold_fitur` |
| **Modeling & BDA Agent** | Menjalankan K-Means clustering ($k=3$), evaluasi GroupKFold Random Forest, agregasi spasial 6 kecamatan, dan Digital Gap Index. | Notebook dieksekusi via `run_notebook.py` & grafik di Google Drive |
| **Documentation Agent** | Menjaga keselarasan seluruh file markdown di `peta-digital-umkm/` dengan file diagram JSON di `diagram-architecture/`. | Dokumentasi terverifikasi |

---

## 4. Perintah Eksekusi Terstandar

```powershell
# 1. Regenerasi Notebook Bebas Identitas
python scripts/create_student_notebooks.py

# 2. Eksekusi Headless & Sinkronisasi Otomatis ke Google Drive
python scripts/run_notebook.py notebooks/eda_3_09040624091.ipynb
python scripts/run_notebook.py notebooks/model_3_09040624091.ipynb
python scripts/run_notebook.py notebooks/bda_3_09040624091.ipynb

# 3. Menjalankan Pipeline Visi Komputer Lengkap (Bila Perlu)
python src/umkm/vision/yolo_full_pipeline.py
```
