# TODO — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Siap Eksekusi End-to-End · 4 November 2026  
> **Rujukan Resmi:**
> 1. *Arsitektur Big Data & AI - Peta Digital UMKM Kuliner Surabaya Timur* (`mxGraphModel` Arsitektur)
> 2. *Flowchart Alur Pengerjaan Mandiri 6 Tahap* (`mxGraphModel` `B1`–`B6`)

---

## 1. INGESTION (ETL) (`B1: n1–n8` | Arsitektur: `DS`, `ETL`, `ST: c0`)
- [x] **`n1` (Start):** Inisialisasi pipeline mandiri untuk 6 kecamatan di Surabaya Timur (Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo).
- [x] **`n2` (Tentukan kelas & indikator):** Tetapkan 3 kelas kematangan digital (*Konvensional*, *Berkembang*, *Maju*) dan indikator target (`QRIS`, `EDC`, `POS`, stiker merchant).
- [x] **`n3` (Ambil data Google Maps API — `s0`, `e0`):** Tarik data lokasi, rating, teks ulasan, dan foto menggunakan Google Places API & REST HTTP.
- [x] **`n4` (Survei & foto lapangan SWK — `s3`, `e3`):** Ambil foto stan, kasir, dan instrumen pembayaran di SWK Surabaya Timur.
- [x] **`n5` (Catat label survei Sheets — `e3`, `e5`):** Dokumentasikan label ground truth hasil survei lapangan ke Google Sheets.
- [x] **`n6` (Gabung data Batch — `e4`):** Gabungkan batch JSON & CSV menggunakan Pandas & PySpark.
- [x] **`n7` (Catat Data Schema — `e5`):** Catat katalog dan skema data pada Google Sheet.
- [x] **`n8` (Simpan Raw Data — `c0`):** Simpan data mentah (`JSON`, `CSV`, `PDF`, `Foto`) ke Google Drive (`Raw Data`).

---

## 2. VALIDASI & EDA (`B2: vol–simpan` | Arsitektur: `DT: PR & EDA`, `ST: c1`)
- [x] **`vol` (Volume cukup?):** Periksa kecukupan volume ($\ge 200\text{ MB}$ & $\ge 2.000$ baris). Bila kurang (`vn`), jalankan **`tambah` (Tambah data $\ge 200\text{ MB}$ & $\ge 2.000$ baris)** kembali ke `n3`.
- [x] **`cek` (Cek format file — `p1`, `p5`):** Validasi dan konversi format foto dengan Pillow serta ekstraksi teks/tabel PDF dengan PyMuPDF & pdfplumber.
- [x] **`dup` (Hapus duplikat — `p2`, `p4`):** Cek dan hapus foto duplikat menggunakan OpenCV serta deduplikasi baris tabular.
- [x] **Kurasi Foto Keep vs Reject (`p2`):** Saring foto non-target (makanan/minuman/menu/ilustrasi) agar hanya menyisakan foto stan/kasir/indikator pembayaran (*Keep*).
- [x] **`anon` (Anonimisasi: blur wajah, sensor PII — `p0`, `p2`):** Buramkan wajah manusia menggunakan OpenCV dan sensor PII teks ulasan menggunakan Regex & NLTK.
- [x] **`eda1` (EDA statistik: rating, ulasan — `d0`, `d1`):** Buat profil otomatis dataset (`ydata-profiling`) dan visualisasi sebaran rating & ulasan per wilayah (`Seaborn & Matplotlib`).
- [x] **`eda2` (EDA visualisasi: foto Maps vs Real — `d2`):** Bandingkan foto Google Maps vs *Real World* survei SWK menggunakan Matplotlib.
- [x] **`miss` $\rightarrow$ `clean` (Ada missing / outlier? $\rightarrow$ Cleaning: imputasi/hapus — `p4`):** Jalankan imputasi, penanganan outlier IQR, dan validasi `QRIS`/`EDC`/`POS` menggunakan Pandas & SciPy.
- [x] **`simpan` (Simpan Clean Parquet — `c1`):** Simpan dataset bersih (`Parquet bersih`) ke Google Drive (`Processed Data`).

---

## 3. PREPROCESSING & FITUR (`B3: pre–fs` | Arsitektur: `DT: PR & FE`, `FS`)
- [x] **`pre` (Preprocessing teks & foto — `p0`, `p2`, `p3`):** Normalisasi teks ulasan, *resize* foto dengan OpenCV, dan augmentasi gambar dengan Albumentations.
- [x] **`ann` (Anotasi foto Label Studio — `f1`):** Anotasi bounding box untuk indikator `QRIS`, `EDC`, `POS`, dan stiker merchant.
- [x] **`latih` (Latih deteksi YOLO — `f1`, `l0`, `l2`):** Latih model detektor objek Ultralytics YOLO di atas PyTorch.
- [x] **`dmap` (mAP $\ge$ target? — `me3`):** Evaluasi akurasi kotak deteksi Ultralytics mAP. Bila belum tercapai (`dn`), jalankan **`tun` (Tambah anotasi / tuning)** kembali ke `ann`.
- [x] **`det` (Deteksi indikator pada semua foto — `f1`, `f3`):** Deteksi indikator pada seluruh foto bersih dan konversi ke `0/1` dengan Binary/Label Encoding.
- [x] **`emb` (Embedding teks & gambar — `f0`, `f2`, `f4`, `l1`):** Ekstraksi vektor teks ulasan (`IndoBERT & TF-IDF`), fitur visual merchant (`ResNet-50`), dan embedding chunk dokumen (`Sentence-Transformers`).
- [x] **`fusi` (Fusi fitur & scaling — `f5`):** Gabungkan fitur multimodal menggunakan `scikit-learn Pipeline` (`StandardScaler` & `PCA`).
- [x] **`fs` (Simpan Feature Store Parquet — `fs0–fs3`):** Simpan matriks fitur ke Google Drive (`fs0`) dalam format Apache Parquet / PyArrow (`fs1`), NumPy `.npy/.npz` (`fs2`), dan Pandas DataFrame (`fs3`).

---

## 4. MODELING & EVALUASI (`B4: split–shap` | Arsitektur: `AOS: ML & DL`, `ME`)
- [x] **`split` (Split data per UMKM):** Bagi dataset menggunakan `GroupKFold` ($K=5$) berdasarkan `umkm_id` (anti-leakage).
- **Cabang Unsupervised (Klastering):**
  - [x] **`kl` (Klaster K-Means / ACO — `m0`, `m2`):** Jalankan K-Means / K-Prototypes (`scikit-learn`) dan centroid optimization ACO (`NumPy`).
  - [x] **`sil` $\rightarrow$ `dk` (Hitung Silhouette $\rightarrow$ k optimal? — `me1`):** Evaluasi Silhouette Score. Bila belum optimal (`kno`), jalankan **`atk` (Atur k / ACO)** kembali ke `kl`.
  - [x] **`prof` (Profil klaster — `vz2`):** Susun profil klaster kematangan dan visualisasi PCA.
- **Cabang Supervised (Klasifikasi):**
  - [x] **`lab` (Ambil label survei):** Ambil label ground truth dari survei lapangan SWK.
  - [x] **`rf` (Latih model RF / XGBoost — `m0`, `m1`):** Latih Random Forest / Logistic Regression (`scikit-learn`) dan `XGBoost` untuk klasifikasi 3 kelas kematangan digital.
  - [x] **`f1` $\rightarrow$ `df` (Hitung macro-F1 & Kappa $\rightarrow$ F1 $\ge 70\%$? — `me0`, `me2`):** Evaluasi `macro-F1`, `accuracy`, `precision`, `recall`, serta `Cohen's Kappa` (kesepakatan Maps vs lapangan). Bila $< 70\%$ (`lno`), jalankan **`par` (Atur parameter)** kembali ke `rf`.
  - [x] **`shap` (Jelaskan model SHAP — `me4`, `me5`):** Analisis pengaruh fitur menggunakan `SHAP` dan fokus visual gambar menggunakan `Grad-CAM`.

---

## 5. SERVING & BDA (`B5: s1–s8` | Arsitektur: `REG`, `INF`, `BDA`, `VIS`)
- [x] **`s1` (Simpan model & hasil MLflow — `o0`, `m3`, `rg0–rg3`):** Catat eksperimen ke MLflow Model Registry dan simpan bobot `.onnx`, `.joblib`, serta `.pt`.
- [x] **`s2` (Buat Inference API — `in0–in3`):** Sediakan layanan prediksi FastAPI, ONNX Runtime, scikit-learn Predict, dan Ultralytics YOLO Predict.
- [x] **`s3` (Agregasi per kecamatan — `bd0`):** Agregasi indikator dan kelas kematangan per kecamatan menggunakan Pandas & PySpark.
- [x] **`s4` (Analisis spasial peta — `bd1`):** Petakan titik UMKM ke poligon kecamatan Surabaya Timur menggunakan GeoPandas & Shapely.
- [x] **`s5` (Statistik spasial — `bd2`):** Hitung statistik spasial & analisis bias cakupan menggunakan PySAL, SciPy & statsmodels.
- [x] **`s6` (Cari gap per wilayah — `bd3`):** Hitung analisis *gap* digitalisasi wilayah (*Digital Gap Index*) menggunakan Pandas.
- [x] **`s7` (Buat dashboard — `vz0–vz3`):** Bangun visualisasi peta interaktif 3 kelas (`Folium & Plotly`), grafik EDA & BDA (`Seaborn & Matplotlib`), klaster (`PCA`), dan deteksi (`YOLO Bounding Box`).
- [x] **`s8` (Rekomendasi intervensi):** Susun rekomendasi intervensi per kecamatan dan sentra kuliner.

---

## 6. AI APPLICATION (`B6: web–end` | Arsitektur: `APP`, `ST: c2`)
- [x] **`web` (Bangun web app — `a0`):** Bangun Web App cek status digital UMKM menggunakan Streamlit & FastAPI.
- [x] **`hub` (Hubungkan API & dashboard):** Integrasikan endpoint Inference API (`in0`) dan peta interaktif (`vz0`).
- [x] **`fit` (Fitur cari UMKM opsional — `a1`):** Integrasikan pencarian dan rekomendasi otomatis berbasis RAG menggunakan LangChain & FAISS.
- [x] **`uji` $\rightarrow$ `ok` (Uji aplikasi $\rightarrow$ Lolos uji?):** Uji seluruh alur aplikasi. Bila belum lolos (`on`), jalankan **`per` (Perbaiki)** kembali ke `hub`.
- [x] **`lap` $\rightarrow$ `end` (Tulis laporan $\rightarrow$ End — `a2`, `c2`):** Susun laporan akhir & rekomendasi intervensi E-Gov di Google Docs dan sinkronkan seluruh artefak ke Google Drive (`Output Data`).
