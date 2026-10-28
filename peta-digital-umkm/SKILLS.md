# SKILLS — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 28 Oktober 2026  
> **Rujukan Resmi Arsitektur & Flowchart:**
> 1. `diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json` (`flowchart-bd-ai` dan `bd-ai-archi`)
> 2. `diagram-architecture/flowchart-mandiri.json` (`flowchart-mandiri-gmaps-realworld`)

Kumpulan prosedur terstandar (*Standard Operating Procedures*) yang diimplementasikan pada pipeline analitik multimodal Peta Digital UMKM Kuliner Surabaya Timur.

---

### S01 `ingest-places-api`
- **Tujuan:** Mengambil metadata UMKM, ulasan teks, dan foto etalase dari Google Maps secara legal dan terstruktur.
- **Masukan:** Grid koordinat 6 kecamatan Surabaya Timur; API key tersimpan di file `.env`.
- **Keluaran:** Respons JSON mentah dan foto disimpan ke Google Drive (`raw/google_maps/`).
- **Alat:** Python `requests`, Google Places API, Pandas.

### S04 `field-survey-intake`
- **Tujuan:** Mengumpulkan data ground truth lapangan di Sentra Wisata Kuliner (SWK Convention Hall & Deles).
- **Masukan:** Foto stan fisik, meja kasir, instrumen pembayaran (QRIS standee, mesin EDC, display POS), dan formulir survei.
- **Keluaran:** Foto lapangan terorganisir di `raw/survei/` dan tabel label kematangan digital.
- **Guardrail:** Wajib memprioritaskan privasi pemilik dan pengunjung; foto difokuskan pada kasir dan stan.

### S05 `volume-gate`
- **Tujuan:** Memvalidasi ambang batas volume data sebelum data masuk ke tahap processing.
- **Aturan:** Total ukuran foto $\ge 200\text{ MB}$ dan total baris data multimodal $\ge 2.000$ entitas.

### S06 `validate-and-dedupe`
- **Tujuan:** Memeriksa integritas file citra, menghitung image hash (MD5/Perceptual Hash), dan menduplikasi data duplikat.
- **Keluaran:** Manifest data bersih sementara dan daftar citra tereliminasi.

### S07 `anonymize-and-blur`
- **Tujuan:** Menjamin privasi publik dengan menyamarkan identitas pribadi sebelum masuk zona Clean.
- **Langkah:** (1) Sanitasi nomor telepon/kontak pada ulasan teks menggunakan regex; (2) Deteksi wajah menggunakan Haar Cascade / DNN OpenCV dan pemburaman Gaussian Blur ($k=51$).
- **Keluaran:** 315 wajah berhasil disanitasi tanpa mengubah detail instrumen kasir atau etalase.

### S07b `clip-dual-gate-filtering` (Kurasi Citra Bersih Mutlak)
- **Tujuan:** Menjamin dataset foto bersih HANYA memuat stan fisik dan target klasifikasi.
- **Langkah:** Evaluasi zero-shot OpenAI CLIP (`ViT-B/32`) dengan dual-gate:
  1. *Prompt Gate:* Komparasi probabilitas indikator (`storefront`, `cashier`, `qris`, `payment machine`) vs non-target (`food dish`, `drink cup`, `printed menu`, `digital food illustration AI`).
  2. *Place-Dominance Gate:* Mensyaratkan skor dominasi tempat $p_{\text{place}} \ge 0.60$.
  3. *Big Business Rule:* Mengeliminasi jaringan restoran besar yang tidak memuat foto kasir/pembayaran.
- **Keluaran:** 415 foto makanan/minuman/menu/AI disaring keluar; 274 foto stan target diterima di `data/clean/photos/`.

### S08 `eda-multimodal`
- **Tujuan:** Melakukan analisis eksplorasi data komprehensif pada 4 modalitas:
  1. Tabular: Distribusi UMKM per kecamatan, rating, dan tipe lokasi.
  2. Teks: Frekuensi kata kunci indikator digital dan sebaran panjang ulasan.
  3. Citra: Proporsi deteksi indikator YOLOv8 pada stan dan sebaran kelas survei SWK.
  4. Dokumen: Ringkasan perizinan penelitian dan regulasi insentif dinas.
- **Keluaran:** 6 file grafik beresolusi tinggi tersimpan di `G:\My Drive\Iqbal-EDA\`.

### S11 `autolabel-yolo-world`
- **Tujuan:** Menghasilkan anotasi bounding box awal (pseudo-labeling) secara obyektif tanpa fabrikasi koordinat.
- **Alat:** `yolov8s-worldv2.pt` open-vocabulary prompt detection untuk 5 kelas target (`qr code`, `card payment terminal`, `cash register`, `delivery app sticker`, `food stall`).
- **Keluaran:** Format anotasi standar YOLO di `data/yolo/`.

### S12 `train-yolo-detector`
- **Tujuan:** Melatih model deteksi objek indikator target kematangan digital.
- **Alat:** Ultralytics YOLOv8 (`yolov8n.pt`), PyTorch.
- **Keluaran:** Model weights [`models/yolov8n_indicator_detect.pt`](file:///C:/big-data-analytic/models/yolov8n_indicator_detect.pt), kurva hasil [`results.csv`](file:///C:/big-data-analytic/reports/yolo/detect/train/results.csv), dan log metrik [`metrics_detect.json`](file:///C:/big-data-analytic/reports/yolo/detect/metrics_detect.json) (mAP@50: **40.74%**).

### S12b `multitask-yolo-inference` (Deliverable Lengkap YOLOv8)
- **Tujuan:** Menghasilkan seluruh deliverable visual YOLOv8 pada seluruh 274 citra bersih:
  1. **Deteksi Objek:** 274 citra bounding box berwarna + label kelas + confidence di `reports/yolo/detect/predict/`.
  2. **Segmentasi Poligon:** 274 citra polygon mask transparan stan/etalase di `reports/yolo/segment/predict/`.
  3. **Estimasi Pose:** 274 citra skeleton keypoints kasir/pedagang di `reports/yolo/pose/predict/`.
  4. **Klasifikasi Kematangan:** 274 citra berlabel kematangan visual di `reports/yolo/classify/predict/`.
  5. **Panel Komparasi 4-in-1:** Panel komparatif visual beresolusi tinggi di `02_output_lengkap_yolov8_multitask.png`.
- **Keluaran:** Tersimpan lokal di `reports/yolo/` dan tersinkronisasi ke `G:\My Drive\Iqbal-Model-Evaluation\yolo_outputs/`.

### S13 `build-gold-features`
- **Tujuan:** Menggabungkan fitur ekstraksi citra YOLOv8, teks ulasan, dan tabular ke dalam tabel fusi terpadu.
- **Keluaran:** Tabel fusi multimodal (190 UMKM) disimpan ke format Parquet.

### S14 `group-split`
- **Tujuan:** Pembagian data cross-validation anti-kebocoran data (*anti-leakage*).
- **Metode:** `GroupKFold` ($K=5$) berdasarkan `umkm_id` sehingga seluruh foto/ulasan dari satu UMKM berada pada fold yang sama.

### S14b `cohens-kappa-agreement` (Uji Kesepakatan Silang)
- **Tujuan:** Mengukur derajat kesepakatan indikator visual Google Maps vs Ground Truth survei lapangan di SWK.
- **Metode:** Menghitung statistik Cohen's Kappa ($\kappa$).
- **Hasil:** $\kappa_{\text{QRIS}} = 0.751$ (*Substantial Agreement*), $\kappa_{\text{POS}} = 0.727$ (*Moderate-Substantial*). Lolos gerbang kesepakatan ($\kappa \ge 0.70$).

### S15 `train-classifier`
- **Tujuan:** Melatih model klasifikasi kematangan digital 3-kelas (*Konvensional*, *Berkembang*, *Maju*).
- **Alat:** Random Forest Classifier dengan validasi `GroupKFold` ($K=5$).
- **Keluaran:** Confusion matrix dan laporan klasifikasi di `04_confusion_matrix_klasifikasi.png`.

### S16 `cluster-validation`
- **Tujuan:** Mengelompokkan tingkat digitalisasi alami melalui pendekatan tanpa supervisi (*unsupervised*).
- **Alat:** K-Means Clustering ($k=3$) pada fitur multimodal terskala.
- **Keluaran:** Evaluasi Silhouette Score (**0.4964**) dan Davies-Bouldin Index (**0.7916**), serta peta panas profil klaster di `03_profil_klaster_kmeans.png`.

### S20 `bda-spatial-gap`
- **Tujuan:** Menganalisis penetrasi pembayaran digital di 6 kecamatan Surabaya Timur dan menyusun indeks kesenjangan digital.
- **Keluaran:** Peta penetrasi digital, Digital Gap Index, dan formulasi rekomendasi kebijakan di `G:\My Drive\Iqbal-BDA\`.

### S21 `headless-notebook-sync`
- **Tujuan:** Menjalankan notebook analitik secara otomatis, menyematkan grafik base64, menjamin nol identitas pribadi, dan menyinkronkan salinan notebook ke Google Drive (`G:\My Drive\`).
- **Alat:** Python headless runner [`scripts/run_notebook.py`](file:///C:/big-data-analytic/scripts/run_notebook.py).
