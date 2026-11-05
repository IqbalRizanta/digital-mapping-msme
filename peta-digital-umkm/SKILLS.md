# SKILLS — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 4 November 2026  
> **Rujukan Resmi Arsitektur:** *Arsitektur Big Data & AI - Peta Digital UMKM Kuliner Surabaya Timur* (`mxGraphModel`)

Kumpulan prosedur operasional terstandar (*Standard Operating Procedures*) yang memetakan setiap modul, library, dan framework pada diagram arsitektur ke dalam langkah eksekusi teknis.

---

### S01 `ingest-multimodal-etl` (`DS: s0–s3` $\longrightarrow$ `ETL: e0–e5` $\longrightarrow$ `ST: c0`)
- **Tujuan:** Mengakuisisi 4 sumber data utama dan menyimpannya ke tier `Raw Data (c0)` di Google Drive.
- **Komponen Arsitektur:**
  - **`s0` $\rightarrow$ `e0` (Google Places API & REST HTTP):** Mengambil data lokasi, rating, teks ulasan, dan foto UMKM dari Google Maps.
  - **`s1` $\rightarrow$ `e1` (KaggleHub CLI):** Mengunduh dataset numerik & gambar pendukung dari Kaggle.
  - **`s2` $\rightarrow$ `e2` (Google Drive API & Manual Download):** Mengunduh dokumen PDF jurnal & regulasi.
  - **`s3` $\rightarrow$ `e3` (Manual Foto & Google Forms):** Mengumpulkan foto stan & survei lapangan di SWK Surabaya Timur.
  - **`e4` (Pandas & PySpark):** Menggabungkan batch data mentah berformat JSON & CSV.
  - **`e5` (Google Sheet):** Mencatat katalog & skema data.
- **Keluaran:** `Raw Data (c0)` di Google Drive (JSON, CSV, PDF, Foto).

### S02 `text-and-pdf-processing` (`DT: PR — p0, p1`)
- **Tujuan:** Membersihkan teks ulasan pelanggan dan mengekstrak informasi dari dokumen PDF.
- **Komponen Arsitektur:**
  - **`p0` (Regex & NLTK):** Pembersihan teks ulasan, penghapusan *stopwords* bahasa Indonesia/daerah, dan penyensoran informasi pribadi (PII seperti nomor telepon/nama).
  - **`p1` (PyMuPDF & pdfplumber):** Ekstraksi teks naratif dan tabel regulasi/jurnal dari dokumen PDF.
- **Keluaran:** Korpus teks ulasan bersih dan potongan teks/tabel dokumen terstruktur.

### S03 `image-sanitization-and-augmentation` (`DT: PR — p2, p3, p5`)
- **Tujuan:** Menjamin kualitas visual, privasi publik, dan variasi citra stan UMKM.
- **Komponen Arsitektur:**
  - **`p2` (OpenCV):** Pengecekan foto duplikat, pemburaman wajah (*Gaussian Face Blur* pada 315 wajah), dan *resize* dimensi gambar.
  - **`p3` (Albumentations):** Augmentasi gambar untuk memperkaya variasi pencahayaan dan sudut pandang pelatihan.
  - **`p5` (Pillow):** Pembacaan dan konversi format foto ke standar RGB `.jpg`.
- **Keluaran:** 274 foto stan bersih bebas wajah dan bebas duplikat di `Processed Data (c1)`.

### S04 `tabular-cleaning-and-validation` (`DT: PR — p4` $\longleftrightarrow$ `ST: c1`)
- **Tujuan:** Membersihkan data numerik/tabular dan memvalidasi atribut indikator digital.
- **Komponen Arsitektur:**
  - **`p4` (Pandas & SciPy):** Imputasi nilai kosong, deteksi dan penanganan outlier berbasis *Interquartile Range* (IQR), serta validasi silang indikator `QRIS`/`EDC`/`POS`.
- **Keluaran:** Dataset `Parquet bersih` tersimpan pada tier **`Processed Data (c1)`** di Google Drive (`umkm.parquet`, `ulasan.parquet`, `foto.parquet`, `dokumen.parquet`).

### S05 `exploratory-data-analysis` (`DT: EDA — d0, d1, d2`)
- **Tujuan:** Memahami karakteristik distribusi data dan membandingkan kondisi daring vs lapangan.
- **Komponen Arsitektur:**
  - **`d0` (ydata-profiling):** Pembuatan profil statistik otomatis untuk dataset Google Maps, Kaggle, dan survei SWK.
  - **`d1` (Seaborn & Matplotlib):** Visualisasi sebaran rating & jumlah ulasan per wilayah kecamatan di Surabaya Timur.
  - **`d2` (Matplotlib):** Komparasi visual berdampingan antara foto Google Maps vs *Real World* (survei SWK).
- **Keluaran:** Laporan profil dan 6 grafik analitik EDA di [`notebooks/eda_3_09040624091.ipynb`](file:///C:/big-data-analytic/notebooks/eda_3_09040624091.ipynb) serta `G:\My Drive\Iqbal-EDA\`.

### S06 `feature-extraction-and-fusion` (`DT: FE — f0–f5` $\longleftrightarrow$ `FS: fs0–fs3`)
- **Tujuan:** Mengekstrak fitur dari keempat modalitas data dan menggabungkannya ke dalam *Feature Store*.
- **Komponen Arsitektur:**
  - **`f0` (IndoBERT & TF-IDF):** Ekstraksi vektor representasi semantik dan bobot kata dari teks ulasan.
  - **`f1` (Label Studio & Ultralytics YOLO):** Anotasi dan deteksi objek indikator `QRIS`, `EDC`, `POS`, dan stiker merchant.
  - **`f2` (ResNet-50):** Ekstraksi fitur visual merchant berdimensi tinggi.
  - **`f3` (Binary/Label Encoding):** Transformasi indikator `QRIS`/`POS`/`EDC` menjadi fitur biner `0/1`.
  - **`f4` (Sentence-Transformers):** Pembuatan *embedding* vektor untuk *chunk* dokumen PDF.
  - **`f5` (scikit-learn Pipeline):** Standarisasi dengan `StandardScaler` & reduksi dimensi `PCA` untuk fusi fitur multimodal.
- **Penyimpanan (`FS`):** Disimpan ke **Google Drive (`fs0`)** dalam format **Apache Parquet via PyArrow (`fs1`)**, array **NumPy `.npy/.npz` (`fs2`)**, dan **Pandas DataFrame (`fs3`)**.

### S07 `ml-and-dl-modeling` (`AOS: ML m0–m3` & `DL l0–l2`)
- **Tujuan:** Melatih model klastering tanpa supervisi, klasifikasi kematangan 3 kelas, dan detektor *deep learning*.
- **Komponen Arsitektur:**
  - **Machine Learning (`ML`):**
    - **`m0` (scikit-learn):** Klastering `K-Means` & `K-Prototypes`, serta klasifikasi `Random Forest` & `Logistic Regression` dengan validasi `GroupKFold` ($K=5$).
    - **`m1` (XGBoost):** Klasifikasi *gradient boosting* untuk 3 kelas kematangan digital (*Konvensional*, *Berkembang*, *Maju*).
    - **`m2` (NumPy):** Optimasi titik pusat klaster (*Centroid optimization / ACO*).
    - **`m3` (Joblib):** Penyimpanan artefak model & pipeline preprocessing.
  - **Deep Learning Frameworks (`DL`):**
    - **`l0` (PyTorch):** Backend komputasi tensor deep learning.
    - **`l1` (Hugging Face):** Eksekusi model Transformers (`IndoBERT`).
    - **`l2` (Ultralytics):** Pelatihan & inferensi detektor objek `YOLO`.

### S08 `model-evaluation-and-xai` (`ME: me0–me5`)
- **Tujuan:** Mengevaluasi kualitas model secara menyeluruh dan menjelaskan dasar prediksi model.
- **Komponen Arsitektur:**
  - **`me0` (scikit-learn Metrics):** Perhitungan `macro-F1`, `accuracy`, `precision`, dan `recall`.
  - **`me1` (Silhouette Score):** Pengukuran kualitas kepadatan dan pemisahan klaster `K-Means` (capaian: **0.4964**).
  - **`me2` (Cohen's Kappa):** Pengukuran kesepakatan indikator antara Google Maps vs survei lapangan SWK ($\kappa_{\text{QRIS}} = 0.751$, $\kappa_{\text{POS}} = 0.727$).
  - **`me3` (Ultralytics mAP):** Evaluasi akurasi kotak deteksi YOLO (capaian mAP@50: **40.74%**).
  - **`me4` (SHAP):** Analisis nilai kontribusi fitur terhadap output klasifikasi.
  - **`me5` (Grad-CAM):** Visualisasi peta aktivasi area fokus gambar pada deteksi visual.

### S09 `model-registry-and-inferencing` (`REG: rg0–rg3` $\longrightarrow$ `INF: in0–in3`)
- **Tujuan:** Membekukan versi model terbaik dan menyajikannya sebagai layanan prediksi siap pakai.
- **Komponen Arsitektur:**
  - **Registry (`REG`):** **`rg0` (MLflow Model Registry)** untuk versi & stage model, **`rg1` (ONNX `.onnx`)** untuk bobot model portabel, **`rg2` (Joblib `.joblib`)** untuk model scikit-learn, dan **`rg3` (YOLO Weights `.pt`)** untuk bobot detektor.
  - **Inferencing (`INF`):** **`in0` (FastAPI)** sebagai endpoint Inference API, **`in1` (ONNX Runtime)** untuk prediksi model ONNX, **`in2` (scikit-learn Predict)** untuk prediksi klaster & kelas UMKM baru, serta **`in3` (Ultralytics YOLO Predict)** untuk deteksi indikator pada foto baru.

### S10 `spatial-bda-visualization-and-app` (`BDA: bd0–bd3` $\longrightarrow$ `VIS: vz0–vz3` $\longrightarrow$ `APP: a0–a2`)
- **Tujuan:** Menganalisis kesenjangan digitalisasi secara spasial di Surabaya Timur, memvisualisasikannya, dan menyajikannya ke pemangku kepentingan.
- **Komponen Arsitektur:**
  - **Big Data Analytics Spasial (`BDA`):**
    - **`bd0` (Pandas & PySpark):** Agregasi metrik adopsi digital per kecamatan.
    - **`bd1` (GeoPandas & Shapely):** Pemetaan titik koordinat UMKM ke poligon kecamatan Surabaya Timur.
    - **`bd2` (PySAL, SciPy & statsmodels):** Analisis statistik spasial dan evaluasi bias cakupan wilayah.
    - **`bd3` (Pandas):** Perhitungan dan analisis *gap* digitalisasi wilayah (*Digital Gap Index*).
  - **Visualization (`VIS`):**
    - **`vz0` (Folium & Plotly):** Peta interaktif sebaran 3 kelas kematangan digital.
    - **`vz1` (Seaborn & Matplotlib):** Grafik analitik statis untuk laporan EDA & BDA.
    - **`vz2` (PCA):** Visualisasi proyeksi 2D/3D klaster kematangan UMKM.
    - **`vz3` (YOLO Bounding Box):** Visualisasi kotak deteksi indikator pada foto stan.
  - **AI Application (`APP`):**
    - **`a0` (Streamlit & FastAPI):** Web App interaktif untuk mengecek status digital UMKM.
    - **`a1` (LangChain & FAISS):** Mesin rekomendasi intervensi & pencarian regulasi otomatis berbasis *Retrieval-Augmented Generation* (RAG).
    - **`a2` (Google Docs):** Penyusunan laporan akhir & rekomendasi intervensi kebijakan *E-Government*.

### S11 `mlops-and-headless-sync` (`OPS: o0–o2` & `ST: c2`)
- **Tujuan:** Menjaga reproduksibilitas eksperimen, versi data/kode, dan sinkronisasi artefak ke Google Drive.
- **Komponen Arsitektur:**
  - **`o0` (MLflow & DVC):** Pencatatan log eksperimen dan pengelolaan versi dataset.
  - **`o1` (Git / GitHub):** Pengelolaan versi kode sumber.
  - **`o2` (Google Colab + VS Code):** Lingkungan pengembangan dan eksekusi CPU/GPU.
  - **Headless Runner ([`scripts/run_notebook.py`](file:///C:/big-data-analytic/scripts/run_notebook.py)):** Mengeksekusi seluruh notebook bebas identitas personal dan menyinkronkan hasilnya ke `Output Data (c2)` di Google Drive (`G:\My Drive\`).
