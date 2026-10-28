# PRD — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 28 Oktober 2026  
> **Rujukan Resmi Arsitektur & Flowchart:**
> 1. `diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json` (Halaman: `bd-ai-archi` dan `flowchart-bd-ai`)
> 2. `diagram-architecture/flowchart-mandiri.json` (Halaman: `flowchart-mandiri-gmaps-realworld`)

---

## 1. Ringkasan Eksekutif
**Peta Digital UMKM** adalah platform analitik dan aplikasi AI berbasis pendekatan modern big data analytics yang memetakan tingkat maturitas digitalisasi UMKM kuliner di 6 kecamatan wilayah Surabaya Timur (Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo).

Sistem memadukan **satu alur data multimodal terpadu** (citra visual stan/indikator, teks ulasan pelanggan, metadata spasial/tabular, dan dokumen regulasi) untuk mengklasifikasikan UMKM ke dalam 3 tingkat kematangan digital:
1. **Konvensional:** Transaksi tunai tradisional tanpa perangkat pembayaran digital atau kehadiran toko daring.
2. **Berkembang:** Memiliki $\ge 1$ saluran pembayaran digital mandiri (QRIS statis/dinamis atau mesin EDC) namun belum terintegrasi ke sistem kasir point-of-sale modern.
3. **Maju:** Mengintegrasikan sistem pembayaran digital lengkap (QRIS + POS) dan terhubung dengan platform pemesanan online (*Merchant Online* GoFood/GrabFood/ShopeeFood).

---

## 2. Latar Belakang & Masalah
- **Disparitas Digital Spasial:** Adopsi pembayaran digital pada sektor kuliner mikro, kecil, dan menengah di Surabaya Timur sangat timpang antarkecamatan dan antarsentra kuliner.
- **Data Multimodal Terfragmentasi:** Bukti digital tersebar dalam ulasan teks publik Google Maps, foto fisik etalase/kasir, dokumen perizinan, dan survei lapangan di Sentra Wisata Kuliner (SWK).
- **Ketiadaan Validasi Silang Objektif:** Sering kali data digital di Google Maps bias terhadap usaha menengah-ke-atas. Dibutuhkan validasi silang (*Cross-Source Validation*) menggunakan survei lapangan nyata di SWK dan uji statistik kesepakatan **Cohen's Kappa ($\kappa$)**.

---

## 3. Tujuan Proyek & Kriteria Keberhasilan
1. **Pipeline Data End-to-End:** Mengimplementasikan alur lengkap sesuai arsitektur kelompok dan flowchart mandiri:
   $$\text{Raw Data (Google Drive)} \longrightarrow \text{Clean Parquet} \longrightarrow \text{Gold Parquet} \longrightarrow \text{Modeling} \longrightarrow \text{BDA Gap Analysis}$$
2. **Kurasi Citra Bersih Mutlak (Zero Non-Target Policy):**
   - Mengeliminasi 100% foto piring makanan, gelas minuman, buku menu, dan ilustrasi digital hasil generate AI menggunakan filter zero-shot dual-gate OpenAI CLIP.
   - Dataset foto bersih **HANYA** memuat stan fisik dan 4 indikator target klasifikasi (`qris`, `edc`, `pos`, `merchant_online`).
3. **Output Visual Lengkap Ultralytics YOLOv8:**
   - Deteksi Bounding Box Berwarna + Nama Label + Skor Confidence ([`yolov8n_indicator_detect.pt`](file:///C:/big-data-analytic/models/yolov8n_indicator_detect.pt)).
   - Segmentasi Instance Poligon Mask Transparan ([`yolov8n-seg.pt`](file:///C:/big-data-analytic/reports/yolo/segment/)).
   - Estimasi Pose Kerangka Tubuh Kasir/Pedagang ([`yolov8n-pose.pt`](file:///C:/big-data-analytic/reports/yolo/pose/)).
   - Klasifikasi Kematangan 3-Kelas ([`yolov8n_maturity_cls.pt`](file:///C:/big-data-analytic/models/yolov8n_maturity_cls.pt)).
4. **Validasi Silang Antar-Sumber (Cohen's Kappa):**
   - Tingkat kesepakatan indikator Google Maps vs Survei Lapangan SWK mencapai derajat *substantial* ($\kappa \ge 0.70$).
5. **Evaluasi Anti-Leakage (GroupKFold):**
   - Pembagian data latih/uji dikelompokkan per unit usaha (`umkm_id`) menggunakan `GroupKFold` ($K=5$) sehingga tidak ada kebocoran foto/ulasan satu UMKM ke data uji.
   - Evaluasi klasifikasi menghasilkan Macro-F1 yang terukur dan dapat dipertanggungjawabkan.
6. **Integritas Cloud Storage Google Drive:**
   - Seluruh artefak data bersih (`clean/`), notebook eksekusi, grafik evaluasi, model weights, dan laporan visual tersimpan dan ter-sync otomatis di Google Drive (`G:\My Drive\`).
7. **Privasi & Anonimitas Total:**
   - 100% wajah manusia (315 wajah) disanitasi menggunakan OpenCV Gaussian Blur.
   - Teks ulasan dibersihkan dari nomor telepon dan nama pribadi via regex.
   - Konten notebook bersih dari identitas pribadi (tanpa nama mahasiswa, NIM, atau dosen).

---

## 4. Ruang Lingkup & Batasan
- **Wilayah Studi:** 6 Kecamatan di Surabaya Timur: Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo.
- **Fokus Khusus:** Sentra Wisata Kuliner (SWK Convention Hall & SWK Deles Sukolilo) sebagai pusat percontohan digitalisasi UMKM binaan pemerintah kota.
- **Entitas Data:** 190 profil UMKM, 730 ulasan pelanggan terkurasi, 274 foto stan bersih, dan dokumen regulasi perizinan.

---

## 5. Fitur Utama Platform
| ID Fitur | Modul | Deskripsi Implementasi |
| :--- | :--- | :--- |
| **F-01** | Ingestion & Sync GDrive | Pengambilan data Places API + survei SWK dan penyimpanan ke `G:\My Drive\raw\` & `clean\`. |
| **F-02** | Dual-Gate CLIP Filter | Zero-shot gate untuk menyaring foto non-target (makanan/minuman/menu/AI-generated) secara otomatis. |
| **F-03** | Multi-Task YOLOv8 | Eksekusi 4 task visi: Bounding Box Detect, Poligon Seg, Pose Skeleton, dan Visual Cls. |
| **F-04** | Cohen's Kappa Validator | Komparasi statistik kesepakatan indikator online vs lapangan untuk mendeteksi bias pengamatan. |
| **F-05** | Multimodal Fusion | Penggabungan matriks tabular, ulasan teks (IndoBERT/TF-IDF), dan deteksi citra ke tabel Gold. |
| **F-06** | Unsupervised Clustering | Pengelompokan K-Means ($k=3$) dievaluasi dengan Silhouette Score dan Davies-Bouldin Index. |
| **F-07** | Supervised Classification | Model Random Forest dengan validasi `GroupKFold` ($K=5$) anti-leakage. |
| **F-08** | Spatial Gap Analysis | Pemetaan penetrasi digital antarkecamatan, Digital Gap Index, dan formulasi rekomendasi kebijakan. |
| **F-09** | Automated Notebook Sync | Pipeline `run_notebook.py` yang mengeksekusi dan mengunggah notebook secara headless ke Google Drive. |
