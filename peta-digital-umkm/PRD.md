# PRD — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 4 November 2026  
> **Rujukan Resmi Arsitektur:** *Arsitektur Big Data & AI - Peta Digital UMKM Kuliner Surabaya Timur* (`mxGraphModel`)

---

## 1. Ringkasan Eksekutif
**Peta Digital UMKM Kuliner Surabaya Timur** adalah sistem analitik *Big Data & AI* terintegrasi yang memetakan tingkat kematangan digitalisasi UMKM kuliner di 6 kecamatan wilayah Surabaya Timur (Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo).

Sistem mengolah **4 sumber data multimodal (`DATA SOURCES`)**—yakni **Google Maps (`s0`)**, **Kaggle (`s1`)**, **Journal PDF (`s2`)**, dan **Real World Survei Lapangan SWK (`s3`)**—untuk mengklasifikasikan dan memetakan UMKM ke dalam **3 kelas kematangan digital**:
1. **Konvensional:** Transaksi tunai tradisional tanpa perangkat pembayaran digital (`QRIS`/`EDC`/`POS`) atau stiker *merchant* daring.
2. **Berkembang:** Memiliki saluran pembayaran digital dasar (seperti `QRIS` statis/dinamis atau mesin `EDC`) namun belum terintegrasi penuh dengan sistem kasir *Point-of-Sale* (`POS`) dan ekosistem *merchant online*.
3. **Maju:** Mengintegrasikan pembayaran digital lengkap (`QRIS`, `EDC`, `POS`) serta terhubung aktif dengan platform pesan-antar daring (*stiker merchant* GoFood/GrabFood/ShopeeFood).

---

## 2. Latar Belakang & Masalah
- **Disparitas Digital Spasial:** Tingkat adopsi instrumen pembayaran digital pada sektor kuliner mikro, kecil, dan menengah di Surabaya Timur belum merata antarkecamatan dan antarsentra kuliner.
- **Data Multimodal Terfragmentasi:** Bukti adopsi digital tersebar dalam ulasan teks dan foto Google Maps (`s0`), dataset numerik & gambar publik Kaggle (`s1`), dokumen regulasi & jurnal PDF (`s2`), serta kondisi fisik nyata di Sentra Wisata Kuliner / SWK (`s3`).
- **Kebutuhan Validasi Silang & Penanganan Bias Cakupan:** Representasi usaha di Google Maps sering kali memiliki bias cakupan geografis. Oleh karena itu, diperlukan komparasi visual (`d2`), uji kesepakatan **Cohen's Kappa (`me2`)** antara data Maps vs survei lapangan SWK, serta analisis statistik spasial & bias cakupan menggunakan **PySAL, SciPy & statsmodels (`bd2`)**.

---

## 3. Tujuan Proyek & Kriteria Keberhasilan
1. **Pipeline Arsitektur End-to-End Terstandar:**
   Mengimplementasikan alur lengkap dari `DATA SOURCES` $\longrightarrow$ `ETL (INGESTION)` $\longrightarrow$ `DATA STORAGE` $\longleftrightarrow$ `Data Transformation` $\longleftrightarrow$ `Analytics Operating System` $\longrightarrow$ `Model Evaluation` $\longrightarrow$ `Registry` $\longrightarrow$ `Inferencing` $\longrightarrow$ `Big Data Analytics (Spasial)` $\longrightarrow$ `Visualization` $\longrightarrow$ `AI Application`, dengan dukungan `AI/ML Ops` dan `Feature Store`.
2. **Transformasi & Pembersihan Multimodal (`PR`):**
   - Pembersihan teks ulasan, penghapusan *stopwords*, dan sensor PII dengan **Regex & NLTK (`p0`)**.
   - Ekstraksi teks dan tabel dokumen PDF dengan **PyMuPDF & pdfplumber (`p1`)**.
   - Pengecekan foto duplikat, pemburaman wajah (*face blur*), dan *resize* menggunakan **OpenCV (`p2`)**, konversi format dengan **Pillow (`p5`)**, serta augmentasi citra dengan **Albumentations (`p3`)**.
   - Imputasi data, deteksi outlier IQR, dan validasi atribut `QRIS`/`EDC`/`POS` dengan **Pandas & SciPy (`p4`)**.
3. **Ekstraksi Fitur & Fusi Multimodal (`FE` & `FS`):**
   - Vektorisasi teks ulasan dengan **IndoBERT & TF-IDF (`f0`)** dan *embedding chunk* dokumen dengan **Sentence-Transformers (`f4`)**.
   - Deteksi indikator visual (`QRIS`, `EDC`, `POS`, stiker merchant) dengan **Label Studio & Ultralytics YOLO (`f1`)** serta ekstraksi fitur visual merchant dengan **ResNet-50 (`f2`)**.
   - Pengodean indikator biner `0/1` (**Binary/Label Encoding `f3`**) dan fusi fitur menggunakan **scikit-learn Pipeline (`StandardScaler` & `PCA` `f5`)** yang disimpan ke **Feature Store (`fs0-fs3`)**.
4. **Pemodelan & Evaluasi Kuantitatif + Explainable AI (`AOS` & `ME`):**
   - Pemodelan *unsupervised* (`K-Means`, `K-Prototypes` `m0` + optimasi centroid `ACO` via `NumPy` `m2`) yang dievaluasi dengan **Silhouette Score (`me1`)**.
   - Pemodelan *supervised* (`Random Forest`, `Logistic Regression` `m0`, serta `XGBoost` `m1`) yang dievaluasi dengan **scikit-learn Metrics (`macro-F1`, `accuracy`, `precision`, `recall` `me0`)**.
   - Validasi kesepakatan Google Maps vs lapangan dengan **Cohen's Kappa (`me2`, $\kappa \ge 0.70$)** dan akurasi deteksi objek dengan **Ultralytics mAP (`me3`)**.
   - Interpretabilitas model menggunakan **SHAP (`me4`)** untuk pengaruh fitur dan **Grad-CAM (`me5`)** untuk area fokus gambar.
5. **Registrasi Model, Inferensi & Analitik Spasial (`REG`, `INF`, `BDA`, `VIS`, `APP`):**
   - Penyimpanan artefak model di **MLflow Model Registry (`rg0`)**, **ONNX (`.onnx` `rg1`)**, **Joblib (`.joblib` `rg2`)**, dan **YOLO Weights (`.pt` `rg3`)**.
   - Penyajian prediksi melalui **FastAPI (`in0`)**, **ONNX Runtime (`in1`)**, **scikit-learn Predict (`in2`)**, dan **Ultralytics YOLO Predict (`in3`)**.
   - Agregasi spasial per kecamatan (`bd0`), pemetaan titik ke poligon Surabaya Timur (`bd1`), statistik spasial & bias cakupan (`bd2`), analisis gap digitalisasi wilayah (`bd3`), peta interaktif 3 kelas (`vz0`), hingga Web App **Streamlit & FastAPI (`a0`)**, sistem rekomendasi RAG **LangChain & FAISS (`a1`)**, dan laporan intervensi E-Gov di **Google Docs (`a2`)**.

---

## 4. Ruang Lingkup & Batasan
- **Wilayah Studi:** 6 Kecamatan di Surabaya Timur: Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo.
- **Lokasi Survei Lapangan (`s3`):** Sentra Wisata Kuliner (SWK Convention Hall & SWK Deles Sukolilo) sebagai tolok ukur validasi lapangan.
- **Entitas Data:** 190 profil UMKM, 730 ulasan pelanggan teranonimkan, 274 foto stan bersih, dataset pendukung Kaggle, serta dokumen jurnal & regulasi PDF.

---

## 5. Fitur Utama Platform Sesuai Blok Arsitektur
| ID Fitur | Blok Arsitektur | Modul & Teknologi | Deskripsi Implementasi |
| :--- | :--- | :--- | :--- |
| **F-01** | `DS` & `ETL` | Places API (`e0`), KaggleHub (`e1`), Drive API (`e2`), Forms (`e3`), Pandas/PySpark (`e4`), GSheet (`e5`) | Akuisisi 4 sumber data, penggabungan batch JSON & CSV, dan pencatatan skema katalog data. |
| **F-02** | `ST` | Google Drive (`c0`, `c1`, `c2`) | Penyimpanan terstruktur untuk *Raw Data*, *Processed Data* (Parquet bersih), dan *Output Data*. |
| **F-03** | `DT: Processing` | Regex/NLTK (`p0`), PyMuPDF/pdfplumber (`p1`), OpenCV (`p2`), Albumentations (`p3`), Pandas/SciPy (`p4`), Pillow (`p5`) | Sensor PII teks, ekstraksi tabel PDF, cek duplikat & blur wajah, augmentasi citra, dan validasi IQR/indikator. |
| **F-04** | `DT: EDA` | ydata-profiling (`d0`), Seaborn & Matplotlib (`d1`, `d2`) | Profil otomatis dataset, analisis sebaran rating/ulasan wilayah, dan komparasi foto Maps vs lapangan. |
| **F-05** | `DT: Feature Eng.` | IndoBERT/TF-IDF (`f0`), Label Studio/YOLO (`f1`), ResNet-50 (`f2`), Encoding (`f3`), Sentence-Transformers (`f4`), Pipeline PCA (`f5`) | Ekstraksi fitur teks, deteksi indikator visual, embedding dokumen, dan fusi fitur multimodal. |
| **F-06** | `FS` | Google Drive (`fs0`), Parquet/PyArrow (`fs1`), NumPy `.npy/.npz` (`fs2`), Pandas DataFrame (`fs3`) | Penyimpanan matriks fitur terpadu, vektor embedding, dan tabel label siap pakai. |
| **F-07** | `AOS: ML & DL` | scikit-learn (`m0`), XGBoost (`m1`), NumPy ACO (`m2`), Joblib (`m3`), PyTorch (`l0`), Hugging Face (`l1`), Ultralytics (`l2`) | Pelatihan klastering (K-Means, K-Prototypes, ACO), klasifikasi 3 kelas kematangan, dan model deep learning. |
| **F-08** | `ME` | sklearn Metrics (`me0`), Silhouette (`me1`), Cohen's Kappa (`me2`), mAP (`me3`), SHAP (`me4`), Grad-CAM (`me5`) | Evaluasi metrik performa model, kesepakatan sumber data, dan *Explainable AI* (XAI). |
| **F-09** | `REG` & `INF` | MLflow Registry (`rg0`), ONNX (`rg1`, `in1`), Joblib (`rg2`), YOLO `.pt` (`rg3`, `in3`), FastAPI (`in0`), sklearn Predict (`in2`) | Manajemen versi bobot model portabel dan penyajian endpoint prediksi *real-time* / batch. |
| **F-10** | `BDA` & `VIS` | Pandas/PySpark (`bd0`), GeoPandas/Shapely (`bd1`), PySAL/SciPy/statsmodels (`bd2`), Folium/Plotly (`vz0`), PCA (`vz2`), YOLO BBox (`vz3`) | Agregasi kecamatan, pemetaan poligon, statistik spasial, analisis gap wilayah, dan visualisasi interaktif. |
| **F-11** | `APP` | Streamlit & FastAPI (`a0`), LangChain & FAISS (`a1`), Google Docs (`a2`) | Web App cek status digital UMKM, rekomendasi otomatis berbasis RAG, dan laporan intervensi E-Gov. |
| **F-12** | `OPS` | MLflow & DVC (`o0`), Git/GitHub (`o1`), Google Colab + VS Code (`o2`) | Tata kelola eksperimen, versi data dan kode, serta lingkungan komputasi CPU/GPU. |
