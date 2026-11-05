# REQUIREMENTS — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 4 November 2026  
> **Rujukan Resmi Arsitektur:** *Arsitektur Big Data & AI - Peta Digital UMKM Kuliner Surabaya Timur* (`mxGraphModel`)

---

## 1. Persyaratan Fungsional (Functional Requirements)

### 1.1 Sumber Data (`DS`), Ingesti (`ETL`), & Penyimpanan (`ST`)
| ID | Blok & Kode Node | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- | :--- |
| **FR-01** | `s0`, `e0` | Sistem **WAJIB** mengakuisisi data lokasi, rating, teks ulasan, dan foto UMKM di Surabaya Timur dari **Google Maps (`s0`)** menggunakan **Google Places API & REST HTTP (`e0`)**. | WAJIB | Terpenuhi (190 UMKM, 730 ulasan) |
| **FR-02** | `s1`, `e1` | Sistem **WAJIB** mengunduh dataset numerik & gambar pendukung dari **Kaggle (`s1`)** melalui **KaggleHub CLI (`e1`)**. | WAJIB | Terpenuhi |
| **FR-03** | `s2`, `e2` | Sistem **WAJIB** mengumpulkan dokumen PDF jurnal & regulasi (**Journal `s2`**) melalui **Google Drive API & Manual Download (`e2`)**. | WAJIB | Terpenuhi |
| **FR-04** | `s3`, `e3` | Sistem **WAJIB** mengakuisisi foto & data survei lapangan di SWK (**Real World `s3`**) menggunakan **Manual Foto & Google Forms (`e3`)**. | WAJIB | Terpenuhi (44 stan survei SWK) |
| **FR-05** | `e4`, `e5` | Sistem **WAJIB** menggabungkan batch JSON & CSV menggunakan **Pandas & PySpark (`e4`)** dan mendokumentasikan katalog & skema data di **Google Sheet (`e5`)**. | WAJIB | Terpenuhi |
| **FR-06** | `c0`, `c1`, `c2` | Sistem **WAJIB** menyimpan seluruh artefak ke 3 tier **Google Drive (`ST`)**: **Raw Data (`c0` — JSON, CSV, PDF, Foto)**, **Processed Data (`c1` — Parquet bersih)**, dan **Output Data (`c2` — Model, peta, laporan)**. | WAJIB | Terpenuhi (`G:\My Drive\`) |

### 1.2 Transformasi Data: Processing (`PR`), EDA (`EDA`), & Feature Engineering (`FE`)
| ID | Blok & Kode Node | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- | :--- |
| **FR-07** | `p0`, `p1` | Sistem **WAJIB** melakukan pembersihan teks ulasan, penghapusan *stopwords*, dan penyensoran PII menggunakan **Regex & NLTK (`p0`)**, serta mengekstrak teks & tabel PDF dengan **PyMuPDF & pdfplumber (`p1`)**. | WAJIB | Terpenuhi |
| **FR-08** | `p2`, `p3`, `p5` | Sistem **WAJIB** memeriksa foto duplikat, memburamkan seluruh wajah manusia (*face blur*), dan melakukan *resize* dengan **OpenCV (`p2`)**, membaca/konversi format foto dengan **Pillow (`p5`)**, serta melakukan augmentasi citra dengan **Albumentations (`p3`)**. | WAJIB | Terpenuhi (315 wajah diburamkan, 274 foto bersih) |
| **FR-09** | `p4` | Sistem **WAJIB** menjalankan imputasi nilai hilang, penanganan outlier berbasis IQR, dan validasi silang atribut `QRIS`/`EDC`/`POS` menggunakan **Pandas & SciPy (`p4`)**. | WAJIB | Terpenuhi |
| **FR-10** | `d0`, `d1`, `d2` | Sistem **WAJIB** menyusun profil otomatis dataset dengan **ydata-profiling (`d0`)**, memvisualisasikan sebaran rating & ulasan per wilayah dengan **Seaborn & Matplotlib (`d1`)**, dan membandingkan foto Google Maps vs Real World dengan **Matplotlib (`d2`)**. | WAJIB | Terpenuhi di `eda_3_09040624091.ipynb` |
| **FR-11** | `f0`, `f4` | Sistem **WAJIB** mengekstraksi vektor teks ulasan dengan **IndoBERT & TF-IDF (`f0`)** dan membuat *embedding* potongan dokumen dengan **Sentence-Transformers (`f4`)**. | WAJIB | Terpenuhi |
| **FR-12** | `f1`, `f2`, `f3` | Sistem **WAJIB** mendeteksi indikator `QRIS`, `EDC`, `POS`, dan stiker merchant menggunakan **Label Studio & Ultralytics YOLO (`f1`)**, mengekstrak fitur visual merchant dengan **ResNet-50 (`f2`)**, serta mengodekan indikator ke bentuk biner `0/1` (**Binary/Label Encoding `f3`**). | WAJIB | Terpenuhi |
| **FR-13** | `f5`, `fs0..fs3` | Sistem **WAJIB** melakukan fusi fitur dengan **scikit-learn Pipeline (`StandardScaler` & `PCA` `f5`)** dan menyimpannya ke **Feature Store (`FS`)** yang mencakup **Google Drive (`fs0`)**, **Apache Parquet / PyArrow (`fs1`)**, **NumPy `.npy/.npz` (`fs2`)**, dan **Pandas DataFrame (`fs3`)**. | WAJIB | Terpenuhi |

### 1.3 Analytics Operating System (`AOS`), Evaluasi (`ME`), Registry (`REG`), & Inferensi (`INF`)
| ID | Blok & Kode Node | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- | :--- |
| **FR-14** | `m0`, `m1`, `m2`, `m3` | Sistem **WAJIB** mendukung pelatihan model **scikit-learn (`m0` — K-Means, K-Prototypes, Random Forest, Logistic Regression)**, **XGBoost (`m1` — klasifikasi 3 kelas kematangan digital)**, optimasi centroid ACO dengan **NumPy (`m2`)**, dan penyimpanan pipeline dengan **Joblib (`m3`)**. | WAJIB | Terpenuhi |
| **FR-15** | `l0`, `l1`, `l2` | Sistem **WAJIB** mengintegrasikan **PyTorch (`l0`)** sebagai backend deep learning, **Hugging Face (`l1`)** untuk Transformers (IndoBERT), dan **Ultralytics (`l2`)** untuk deteksi objek YOLO. | WAJIB | Terpenuhi |
| **FR-16** | `me0`, `me1`, `me2`, `me3` | Sistem **WAJIB** mengevaluasi performa model menggunakan **scikit-learn Metrics (`me0` — macro-F1, accuracy, precision, recall)**, **Silhouette Score (`me1`)**, **Cohen's Kappa (`me2` — kesepakatan Maps vs lapangan $\kappa \ge 0.70$)**, dan **Ultralytics mAP (`me3`)**. | WAJIB | Terpenuhi ($\kappa_{\text{QRIS}}=0.751$, Silhouette $0.4964$, mAP@50 $40.74\%$) |
| **FR-17** | `me4`, `me5` | Sistem **WAJIB** menyediakan *explainability* model menggunakan **SHAP (`me4`)** untuk pengaruh fitur tabular/multimodal dan **Grad-CAM (`me5`)** untuk area fokus gambar. | WAJIB | Terpenuhi |
| **FR-18** | `rg0..rg3`, `in0..in3` | Sistem **WAJIB** mencatat model di **MLflow Model Registry (`rg0`)** dalam format **ONNX (`.onnx` `rg1`)**, **Joblib (`.joblib` `rg2`)**, dan **YOLO Weights (`.pt` `rg3`)**, serta menyajikannya melalui **FastAPI (`in0`)**, **ONNX Runtime (`in1`)**, **scikit-learn Predict (`in2`)**, dan **Ultralytics YOLO Predict (`in3`)**. | WAJIB | Terpenuhi |

### 1.4 Big Data Analytics Spasial (`BDA`), Visualisasi (`VIS`), Aplikasi AI (`APP`), & MLOps (`OPS`)
| ID | Blok & Kode Node | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- | :--- |
| **FR-19** | `bd0`, `bd1`, `bd2`, `bd3` | Sistem **WAJIB** melakukan agregasi per kecamatan dengan **Pandas & PySpark (`bd0`)**, pemetaan titik ke poligon Surabaya Timur dengan **GeoPandas & Shapely (`bd1`)**, statistik spasial & evaluasi bias cakupan dengan **PySAL, SciPy & statsmodels (`bd2`)**, serta analisis gap digitalisasi wilayah dengan **Pandas (`bd3`)**. | WAJIB | Terpenuhi di `bda_3_09040624091.ipynb` |
| **FR-20** | `vz0`, `vz1`, `vz2`, `vz3` | Sistem **WAJIB** menghasilkan peta interaktif 3 kelas dengan **Folium & Plotly (`vz0`)**, grafik EDA & BDA dengan **Seaborn & Matplotlib (`vz1`)**, visualisasi klaster dengan **PCA (`vz2`)**, dan visualisasi deteksi dengan **YOLO Bounding Box (`vz3`)**. | WAJIB | Terpenuhi |
| **FR-21** | `a0`, `a1`, `a2` | Sistem **WAJIB** menyediakan antarmuka **Streamlit & FastAPI (`a0`)** untuk cek status digital, rekomendasi otomatis berbasis RAG dengan **LangChain & FAISS (`a1`)**, serta laporan akhir & rekomendasi intervensi E-Gov di **Google Docs (`a2`)**. | WAJIB | Terpenuhi |
| **FR-22** | `o0`, `o1`, `o2` | Sistem **WAJIB** menerapkan tata kelola **AI/ML Ops (`OPS`)** menggunakan **MLflow & DVC (`o0`)** untuk log eksperimen & versi data, **Git / GitHub (`o1`)** untuk versi kode, dan **Google Colab + VS Code (`o2`)** untuk eksekusi CPU/GPU. | WAJIB | Terpenuhi |

---

## 2. Persyaratan Non-Fungsional (Non-Functional Requirements)

| ID | Kategori | Persyaratan |
| :--- | :--- | :--- |
| **NFR-01** | **Bebas Identitas Personal** | Seluruh file notebook (`eda_3_*.ipynb`, `model_3_*.ipynb`, `bda_3_*.ipynb`) **TIDAK BOLEH** memuat identitas personal (tanpa nama mahasiswa, NIM, maupun nama dosen pembimbing) di dalam sel markdown atau komentar kode. |
| **NFR-02** | **Kepatuhan Privasi (GDPR/UU PDP)** | Seluruh PII teks ulasan (`p0`) dan wajah manusia pada foto (`p2`) wajib disensor/diburamkan sebelum disimpan ke `Processed Data (c1)`. |
| **NFR-03** | **Validasi Anti-Leakage** | Evaluasi model wajib memisahkan entitas berdasarkan `umkm_id` (`GroupKFold` $K=5$) agar tidak terjadi kebocoran sampel antar-tahap pelatihan dan pengujian. |
| **NFR-04** | **Interoperabilitas & Portabilitas Model** | Bobot model wajib disimpan dalam format standar yang portabel (`.onnx` `rg1`, `.joblib` `rg2`, `.pt` `rg3`) agar dapat dijalankan secara konsisten oleh mesin inferensi (`in0-in3`). |
| **NFR-05** | **Efisiensi Penyimpanan Kolumnar & Matriks** | Dataset bersih dan matriks fitur wajib menggunakan format **Apache Parquet via PyArrow (`c1`, `fs1`)** serta array terkompresi **NumPy `.npy/.npz` (`fs2`)**. |
| **NFR-06** | **Reproduksibilitas MLOps** | Seluruh alur dua arah (`E3`, `E5`, `E6`, `E7`, `E8`, `E9`) wajib tercatat secara konsisten antara repositori lokal, Git/GitHub (`o1`), MLflow/DVC (`o0`), dan Google Drive (`ST`/`FS`). |
