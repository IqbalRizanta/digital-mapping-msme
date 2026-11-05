# AGENTS — Panduan untuk Agen AI dan Kontributor

> **Status:** Final Terverifikasi · 4 November 2026  
> **Rujukan Resmi Arsitektur:** *Arsitektur Big Data & AI - Peta Digital UMKM Kuliner Surabaya Timur* (`mxGraphModel`)

Dokumen ini adalah pedoman operasional wajib bagi agen AI asisten pengkodean maupun pengembang yang bekerja di repositori **Peta Digital UMKM Kuliner Surabaya Timur**.

---

## 1. Ringkasan Arsitektur & Alur Sistem
Platform ini mengimplementasikan arsitektur **Big Data & AI** berlapis dengan pemisahan tegas antara **alur data utama satu arah ($\longrightarrow$)** dan **pertukaran data/artefak/log dua arah ($\longleftrightarrow$)**:

1. **DATA SOURCES (`DS`):**
   - `s0` **Google Maps:** Teks Ulasan & Foto.
   - `s1` **Kaggle:** Data Numerik & Gambar.
   - `s2` **Journal:** Data Dokumen (PDF).
   - `s3` **Real World:** Foto & Survei Lapangan SWK Surabaya Timur.
2. **ETL (INGESTION) (`ETL`):**
   - `e0` Google Places API & REST HTTP, `e1` KaggleHub CLI, `e2` Google Drive API & Manual Download, `e3` Manual Foto & Google Forms, `e4` Pandas & PySpark (gabung batch JSON & CSV), `e5` Google Sheet (katalog & skema data).
3. **DATA STORAGE (`ST`):**
   - Berbasis Google Drive yang terbagi menjadi 3 tier: `c0` **Raw Data** (JSON, CSV, PDF, Foto), `c1` **Processed Data** (Parquet bersih), dan `c2` **Output Data** (Model, peta, laporan).
4. **Data Transformation (`DT`):**
   - **Processing (`PR`):** `p0` Regex & NLTK, `p1` PyMuPDF & pdfplumber, `p2` OpenCV (cek duplikat, blur wajah, resize), `p3` Albumentations, `p4` Pandas & SciPy (imputasi, outlier IQR, validasi QRIS/EDC/POS), `p5` Pillow.
   - **Exploratory Data Analysis (`EDA`):** `d0` ydata-profiling, `d1` Seaborn & Matplotlib (sebaran per wilayah), `d2` Matplotlib (komparasi foto Gmaps vs Real World).
   - **Feature Extraction & Engineering (`FE`):** `f0` IndoBERT & TF-IDF, `f1` Label Studio & Ultralytics YOLO, `f2` ResNet-50, `f3` Binary/Label Encoding (`0/1`), `f4` Sentence-Transformers, `f5` scikit-learn Pipeline (`StandardScaler` & `PCA`).
5. **Feature Store (`FS`):**
   - `fs0` Google Drive, `fs1` Apache Parquet (PyArrow), `fs2` NumPy (`.npy` / `.npz`), `fs3` Pandas DataFrame.
6. **Analytics Operating System (`AOS`):**
   - **Machine Learning (`ML`):** `m0` scikit-learn (K-Means, K-Prototypes, Random Forest, Logistic Regression), `m1` XGBoost (klasifikasi 3 kelas kematangan digital), `m2` NumPy (Centroid optimization / ACO), `m3` Joblib.
   - **Deep Learning Frameworks (`DL`):** `l0` PyTorch, `l1` Hugging Face Transformers (IndoBERT), `l2` Ultralytics YOLO.
7. **Model Evaluation (`ME`), Registry (`REG`), & Inferencing (`INF`):**
   - Evaluasi komprehensif dengan `me0` scikit-learn Metrics, `me1` Silhouette Score, `me2` Cohen's Kappa, `me3` Ultralytics mAP, `me4` SHAP, dan `me5` Grad-CAM.
   - Penyimpanan model di `rg0` MLflow Model Registry, `rg1` ONNX (`.onnx`), `rg2` Joblib (`.joblib`), dan `rg3` YOLO Weights (`.pt`), serta penyajian via `in0` FastAPI, `in1` ONNX Runtime, `in2` scikit-learn Predict, dan `in3` Ultralytics YOLO Predict.
8. **Big Data Analytics Spasial (`BDA`), Visualization (`VIS`), & AI Application (`APP`):**
   - Analisis spasial (`bd0` Pandas & PySpark, `bd1` GeoPandas & Shapely, `bd2` PySAL/SciPy/statsmodels, `bd3` Pandas analisis gap), visualisasi (`vz0` Folium & Plotly, `vz1` Seaborn & Matplotlib, `vz2` PCA, `vz3` YOLO Bounding Box), hingga aplikasi akhir (`a0` Streamlit & FastAPI, `a1` LangChain & FAISS RAG, `a2` Google Docs laporan E-Gov).
9. **AI/ML Ops (`OPS`):**
   - `o0` MLflow & DVC, `o1` Git / GitHub, dan `o2` Google Colab + VS Code (CPU/GPU) yang bertukar artefak/log dua arah dengan `DT`, `REG`, dan `INF`.

---

## 2. Aturan Mutlak Tata Kelola (Non-Negotiable Guardrails)
1. **Nol Identitas Personal (Zero Student Identity):**
   - File notebook (`eda_3_*.ipynb`, `model_3_*.ipynb`, `bda_3_*.ipynb`) **DILARANG KERAS** memuat identitas personal (nama mahasiswa, NIM, maupun nama dosen pembimbing) di dalam konten markdown, sel teks, maupun komentar kode.
2. **Kepatuhan Privasi Publik (GDPR / UU PDP — `p0` & `p2`):**
   - 100% wajah pengunjung/kasir yang tertangkap kamera wajib disanitasi menggunakan **OpenCV (`p2`)** sebelum masuk ke `Processed Data (c1)`.
   - Nomor kontak dan PII pada teks ulasan disensor menggunakan **Regex & NLTK (`p0`)**.
3. **Integritas Cloud Storage & Feature Store (`ST` & `FS`):**
   - Penyimpanan utama adalah Google Drive (`G:\My Drive\` / Google Colab) yang mencakup `Raw Data (c0)`, `Processed Data (c1)`, `Output Data (c2)`, dan `Feature Store (fs0-fs3)`.
   - Seluruh output otomatis tersimpan di direktori target:
     - `G:\My Drive\clean\` (tabel Parquet bersih `c1` dan foto stan terburamkan)
     - `G:\My Drive\Iqbal-EDA\` (notebook EDA dan grafik visual `d0-d2`, `vz1`)
     - `G:\My Drive\Iqbal-Model-Evaluation\` (notebook evaluasi, model weights `rg1-rg3`, dan deliverables visual `me0-me5`, `vz2-vz3`)
     - `G:\My Drive\Iqbal-BDA\` (notebook BDA dan peta kesenjangan spasial `bd0-bd3`, `vz0-vz1`)
4. **Validasi Anti-Kebocoran Data (Anti-Leakage):**
   - Pembagian data latih/uji wajib dikelompokkan berdasarkan `umkm_id` (`GroupKFold` $K=5$).
   - Normalisasi `StandardScaler` & `PCA` (`f5`) hanya dipasang pada data latih di dalam `scikit-learn Pipeline`.
   - Label kelas kematangan berasal dari validasi survei lapangan di SWK (`s3`, `e3`, `me2`), **bukan** dari hasil klastering.
5. **Kurasi Citra Bersih (Zero Non-Target):**
   - Foto pada `Processed Data (c1)` **HANYA** memuat stan/etalase/kasir dan instrumen pembayaran (`QRIS`, `EDC`, `POS`, stiker merchant).

---

## 3. Pembagian Tugas Agen Khusus (Specialist Agent Roles)

| Peran Agen | Blok Arsitektur Terkait | Tugas & Tanggung Jawab | Output & Kontrak Data |
| :--- | :--- | :--- | :--- |
| **Ingestion & Storage Agent** | `DS`, `ETL`, `ST` (`s0-s3`, `e0-e5`, `c0`) | Mengambil data Places API (`e0`), KaggleHub (`e1`), PDF (`e2`), dan survei SWK (`e3`); menggabungkan batch dengan Pandas/PySpark (`e4`) dan mencatat skema di Google Sheet (`e5`). | `Raw Data (c0)` di Google Drive & Katalog Skema |
| **Processing & EDA Agent** | `DT: PR & EDA` (`p0-p5`, `d0-d2`, `c1`) | Menjalankan sensor PII Regex/NLTK (`p0`), ekstraksi PDF PyMuPDF/pdfplumber (`p1`), cek duplikat & blur wajah OpenCV (`p2`), augmentasi Albumentations (`p3`), imputasi/IQR Pandas & SciPy (`p4`), konversi Pillow (`p5`), serta profil EDA (`d0-d2`). | `Processed Data (c1)` Parquet bersih & grafik EDA |
| **Feature & Vision Agent** | `DT: FE`, `FS`, `DL` (`f0-f5`, `fs0-fs3`, `l0-l2`) | Mengekstraksi fitur teks IndoBERT/TF-IDF (`f0`), deteksi objek Label Studio & YOLO (`f1`), fitur visual ResNet-50 (`f2`), encoding biner (`f3`), embedding dokumen Sentence-Transformers (`f4`), serta fusi fitur StandardScaler & PCA (`f5`) ke Feature Store (`fs0-fs3`). | Matriks fitur `.parquet` (`fs1`), `.npy/.npz` (`fs2`), dan `DataFrame` (`fs3`) |
| **Modeling & Evaluation Agent** | `AOS: ML`, `ME`, `REG` (`m0-m3`, `me0-me5`, `rg0-rg3`) | Melatih model scikit-learn (`m0`), XGBoost (`m1`), optimasi centroid ACO NumPy (`m2`), menyimpan dengan Joblib (`m3`); mengevaluasi macro-F1 (`me0`), Silhouette (`me1`), Cohen's Kappa (`me2`), mAP (`me3`), SHAP (`me4`), Grad-CAM (`me5`); mendaftarkan ke Registry (`rg0-rg3`). | Bobot model (`.onnx`, `.joblib`, `.pt`) & laporan evaluasi |
| **Serving, BDA & App Agent** | `INF`, `BDA`, `VIS`, `APP` (`in0-in3`, `bd0-bd3`, `vz0-vz3`, `a0-a2`) | Menyajikan Inference API FastAPI & ONNX Runtime (`in0-in3`), agregasi spasial kecamatan (`bd0-bd3`), visualisasi peta Folium/Plotly & YOLO bbox (`vz0-vz3`), Web App Streamlit (`a0`), RAG LangChain/FAISS (`a1`), dan laporan E-Gov Google Docs (`a2`). | `Output Data (c2)`, Peta Interaktif, Web App & Laporan Akhir |
| **MLOps & Governance Agent** | `OPS` (`o0-o2`) | Mengelola log eksperimen & versi data dengan MLflow & DVC (`o0`), repositori Git/GitHub (`o1`), dan eksekusi lintas Google Colab + VS Code (`o2`). | Sinkronisasi artefak/log dua arah (`E7`, `E8`, `E9`) |

---

## 4. Perintah Eksekusi Terstandar

```powershell
# 1. Regenerasi Notebook Bebas Identitas
python scripts/create_student_notebooks.py

# 2. Eksekusi Headless & Sinkronisasi Otomatis ke Google Drive (ST: Output Data c2)
python scripts/run_notebook.py notebooks/eda_3_09040624091.ipynb
python scripts/run_notebook.py notebooks/model_3_09040624091.ipynb
python scripts/run_notebook.py notebooks/bda_3_09040624091.ipynb

# 3. Menjalankan Pipeline Visi Komputer Lengkap (Bila Perlu)
python src/umkm/vision/yolo_full_pipeline.py
```
