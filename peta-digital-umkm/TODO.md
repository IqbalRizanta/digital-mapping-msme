# TODO — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 28 Oktober 2026  
> **Rujukan Resmi Arsitektur & Flowchart:**
> 1. `diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json` (`flowchart-bd-ai` dan `bd-ai-archi`)
> 2. `diagram-architecture/flowchart-mandiri.json` (`flowchart-mandiri-gmaps-realworld`)

---

## Fase 0 — Persiapan & Keputusan Arsitektur
- [x] Sepakati 6 kecamatan fokus Surabaya Timur: Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo.
- [x] Tetapkan definisi 3 kelas kematangan digital (*Konvensional*, *Berkembang*, *Maju*) dan 4 indikator utama (*QRIS, POS, EDC, Merchant Online*) + kelas fisik *Stan*.
- [x] Buat struktur repositori bersih, konfigurasi `.gitignore` untuk melindungi kredensial, virtual drive, dan cache.
- [x] Siapkan integrasi penyimpanan cloud Google Drive (`G:\My Drive\clean\`, `G:\My Drive\Iqbal-EDA\`, `G:\My Drive\Iqbal-Model-Evaluation\`, `G:\My Drive\Iqbal-BDA\`).
- [x] Pasang lingkungan virtual Python 3.11 (`.msme`), install dependensi (Ultralytics, OpenAI CLIP, PyTorch, scikit-learn, Seaborn).

---

## Fase 1 — Ingestion (ETL)
- [x] Konfigurasi API key Places API resmi di file `.env`.
- [x] Ambil data profil tempat, rating, koordinat, ulasan teks, dan foto dari Google Maps untuk 6 kecamatan di Surabaya Timur.
- [x] Perluas penarikan data foto UMKM Google Maps (+379 citra usaha mikro/kecil/menengah kuliner di Gubeng dan sekitarnya).
- [x] Lakukan survei lapangan (*Real World*) di Sentra Wisata Kuliner (SWK Convention Hall & SWK Deles) memotret stan fisik, etalase, kasir, dan perangkat pembayaran.
- [x] Kumpulkan dokumen regulasi dan perizinan penelitian ke zona raw.
- [x] **Gerbang Volume Terpenuhi:** Pool citra mentah 672 foto ($\ge 200\text{ MB}$) dan 730 ulasan teks + 190 UMKM.

---

## Fase 2 — Validasi & EDA
- [x] Periksa format file, integritas, dan eliminasi duplikasi citra.
- [x] **Filter Dual-Gate OpenAI CLIP:** Saring 415 citra non-target (piring makanan murni, gelas minuman, buku menu, ilustrasi makanan AI).
- [x] **Penerapan Big Business Rule:** Pangkas 43 foto usaha waralaba/rantai besar yang tidak memiliki foto kasir/pembayaran digital.
- [x] **Sanitasi Privasi Total:** Buramkan 315 deteksi wajah pengunjung/kasir menggunakan OpenCV Gaussian Face Blurring.
- [x] Lakukan analisis eksplorasi data (EDA) pada 4 modalitas data: tabular, teks ulasan, citra YOLOv8, dan dokumen.
- [x] Simpan dataset bersih ke zona `clean/` format Parquet ([`umkm.parquet`](file:///G:/My%20Drive/clean/umkm.parquet), [`ulasan.parquet`](file:///G:/My%20Drive/clean/ulasan.parquet), [`foto.parquet`](file:///G:/My%20Drive/clean/foto.parquet), [`dokumen.parquet`](file:///G:/My%20Drive/clean/dokumen.parquet)).
- [x] Eksekusi [`notebooks/eda_3_09040624091.ipynb`](file:///C:/big-data-analytic/notebooks/eda_3_09040624091.ipynb) dan sinkronkan seluruh 6 grafik analitik ke `G:\My Drive\Iqbal-EDA\`.

---

## Fase 3 — Preprocessing & Fitur (YOLOv8 Multi-Task)
- [x] Preprocessing citra target (resize 640x640, augmentasi, normalisasi) dan teks ulasan (cleaning regex, stopword).
- [x] Pseudo-labeling open-vocabulary menggunakan `yolov8s-worldv2.pt` untuk 5 kelas target (`qris`, `edc`, `pos`, `merchant_online`, `stan`).
- [x] Latih model detektor Ultralytics YOLOv8n pada dataset teranotasi (`models/yolov8n_indicator_detect.pt`).
- [x] **Gerbang mAP Terpenuhi:** Evaluasi detektor menghasilkan mAP@50 **40.74%**, mAP@50-95 **28.79%**, Precision **44.70%**, Recall **36.57%** ([`metrics_detect.json`](file:///C:/big-data-analytic/reports/yolo/detect/metrics_detect.json)).
- [x] **Deliverable Lengkap YOLOv8 (274 Foto Bersih):**
  - [x] Deteksi Bounding Box Berwarna + Label + Confidence Score (`reports/yolo/detect/predict/`).
  - [x] Segmentasi Instance Poligon Mask Transparan (`reports/yolo/segment/predict/`, 1.181 poligon).
  - [x] Estimasi Pose Kerangka Tubuh Kasir/Pedagang (`reports/yolo/pose/predict/`, 371 pose).
  - [x] Klasifikasi Kematangan 3-Kelas (`reports/yolo/classify/predict/`, Top-1: 63.64%, Top-5: 100%).
  - [x] File log metrik terstruktur dan kurva pelatihan ([`results.csv`](file:///C:/big-data-analytic/reports/yolo/detect/train/results.csv)).
  - [x] Panel komparasi multi-task 4-in-1 beresolusi tinggi ([`02_output_lengkap_yolov8_multitask.png`](file:///G:/My%20Drive/Iqbal-Model-Evaluation/02_output_lengkap_yolov8_multitask.png)).
- [x] Ekstraksi representasi teks ulasan pelanggan (IndoBERT/TF-IDF).
- [x] Fusi fitur multimodal citra + teks + tabular ke tabel Gold Parquet (`df_fusi` 190 UMKM).

---

## Fase 4 — Modeling & Evaluasi
- [x] Terapkan strategi validasi anti-kebocoran data **`GroupKFold` ($K=5$)** berdasarkan `umkm_id`.
- [x] **Uji Kesepakatan Silang (Cohen's Kappa):** Hitung koefisien kesepakatan indikator visual Google Maps vs Lapangan SWK.
  - [x] Indikator QRIS: $\kappa = 0.751$ (*Substantial Agreement*).
  - [x] Indikator POS: $\kappa = 0.727$ (*Moderate-Substantial*).
  - [x] Gerbang kesepakatan $\kappa \ge 0.70$ terpenuhi.
- [x] Pemodelan Unsupervised: K-Means Clustering ($k=3$) dengan evaluasi Silhouette Score (**0.4964**) dan DBI (**0.7916**).
- [x] Pemodelan Supervised: Random Forest Classifier dengan label Ground Truth survei SWK.
- [x] Eksekusi [`notebooks/model_3_09040624091.ipynb`](file:///C:/big-data-analytic/notebooks/model_3_09040624091.ipynb) dan sinkronkan seluruh output visual ke `G:\My Drive\Iqbal-Model-Evaluation\`.

---

## Fase 5 — Serving & Big Data Analytics (BDA)
- [x] Agregasi spasial tingkat adopsi indikator digital di 6 kecamatan Surabaya Timur.
- [x] Perhitungan **Digital Gap Index** untuk menetapkan prioritas intervensi dinas antarkecamatan.
- [x] Analisis khusus Sentra Wisata Kuliner (SWK Convention Hall & Deles).
- [x] Formulasi 4 pilar rekomendasi kebijakan bagi pemangku kepentingan.
- [x] Eksekusi [`notebooks/bda_3_09040624091.ipynb`](file:///C:/big-data-analytic/notebooks/bda_3_09040624091.ipynb) dan sinkronkan grafik ke `G:\My Drive\Iqbal-BDA\`.

---

## Fase 6 — AI Application & Pelaporan
- [x] Pastikan seluruh notebook bebas dari identitas pribadi (tanpa nama mahasiswa, NIM, atau dosen pembimbing).
- [x] Selaraskan seluruh dokumen perancangan di folder `peta-digital-umkm/` (`ARCHITECTURE.md`, `WORKFLOW.md`, `PRD.md`, `REQUIREMENTS.md`, `SKILLS.md`, `TODO.md`, `AGENTS.md`) dengan diagram arsitektur kelompok dan flowchart mandiri.
- [x] Verifikasi keterbacaan notebook di VS Code dan ketersediaan salinan di Google Drive untuk Google Colaboratory.
