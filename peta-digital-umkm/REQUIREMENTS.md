# REQUIREMENTS — Peta Digital UMKM Kuliner Surabaya Timur

> Status: Draft 0.1 · 26 Oktober 2026 · Rujukan: `PRD.md`, `ARCHITECTURE.md`, diagram `bd-ai-archi` dan `flowchart-bd-ai`.
> Kata kunci: **WAJIB** (must), **SEBAIKNYA** (should), **OPSIONAL** (may). Tanda **[USULAN]** perlu dikonfirmasi.

## 1. Persyaratan fungsional

### 1.1 Data Source dan Ingestion (komponen: DataSource, Ingestion/ETL)
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-01 | Sistem WAJIB mengambil data Google Maps (nama, alamat, koordinat, rating, ulasan, foto) melalui **API resmi** dengan API key dari variabel lingkungan. | WAJIB |
| FR-02 | Sistem WAJIB mengunduh dataset Kaggle melalui kagglehub/Kaggle API dan mencatat lisensinya. | WAJIB |
| FR-03 | Sistem WAJIB dapat memuat data Opendata dan dataset Mendeley Data (CSV/JSON/XLSX). | WAJIB |
| FR-04 | Dokumen jurnal (ScienceDirect, Dinasti Publisher) WAJIB diperoleh dengan cara yang sesuai ToS (unduh manual atau API resmi) dan disimpan sebagai PDF. | WAJIB |
| FR-05 | Sistem WAJIB menyediakan jalur **Survei** (foto lapangan + isian) dan **Real World** yang diunggah ke Google Drive dengan penamaan terstruktur `Kecamatan/Nama_Tempat/file`. | WAJIB |
| FR-06 | Setiap sumber WAJIB dicatat di **Data Collection Schema** (Google Sheets): sumber, tanggal, format, jumlah, lisensi, pemilik data. | WAJIB |
| FR-07 | Ingestion WAJIB menggabungkan sumber melalui jalur Batch; jalur Streaming (PySpark) OPSIONAL sebagai demonstrasi. | WAJIB / OPSIONAL |
| FR-08 | Raw Data WAJIB disimpan tidak berubah (immutable) di Google Drive; hasil olahan disimpan di zona terpisah. | WAJIB |
| FR-09 | Gerbang volume WAJIB: lanjut hanya jika **gambar ≥ 200 MB** dan **teks/numerik ≥ 2.000 baris**; jika tidak, kembali ke pengumpulan data. | WAJIB |

### 1.2 Validasi, EDA, dan Processing (komponen: Data Processing)
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-10 | Sistem WAJIB memeriksa format file, duplikat (hash untuk gambar, kunci unik untuk tabel), dan integritas file. | WAJIB |
| FR-11 | Sistem WAJIB menghasilkan EDA statistik dan visual untuk tiap tipe data (tabel, teks, gambar, dokumen). | WAJIB |
| FR-12 | Sistem WAJIB menangani missing value dan outlier (imputasi/hapus) dan mencatat keputusannya. | WAJIB |
| FR-13 | Sistem WAJIB menganonimkan data pribadi (nama pemilik, kontak, ID pengulas) dan memburamkan wajah pada foto sebelum zona Clean. | WAJIB |
| FR-14 | Preprocessing WAJIB mencakup: gambar (resize, normalisasi, augmentasi), teks (cleaning, normalisasi slang, stopword), dokumen (ekstraksi teks, OCR bila scan, chunking). | WAJIB |
| FR-15 | Sistem WAJIB menyimpan hasil ke zona **Clean** (Parquet) dan **Gold** (fitur per UMKM). | WAJIB |

### 1.3 Deteksi indikator dan fitur
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-16 | Sistem WAJIB menyediakan anotasi foto untuk kelas QRIS, POS, EDC, Merchant Online (Label Studio atau setara) dengan pedoman anotasi tertulis. | WAJIB |
| FR-17 | Sistem WAJIB melatih detektor indikator (Ultralytics YOLO) dan melaporkan mAP per kelas. Satu foto dapat memuat banyak indikator (multi-label). | WAJIB |
| FR-18 | Jika mAP di bawah target, proses WAJIB kembali ke penambahan anotasi/tuning (gerbang "mAP ≥ target?"). | WAJIB |
| FR-19 | Sistem WAJIB mengekstrak embedding teks (IndoBERT/TF-IDF) dan gambar (ResNet-50). | WAJIB |
| FR-20 | Sistem WAJIB menggabungkan fitur (indikator, embedding, numerik) lalu melakukan scaling; reduksi dimensi (PCA) dipakai bila dimensi terlalu tinggi. | WAJIB |
| FR-21 | Sistem WAJIB menangani UMKM dengan modalitas yang hilang (tidak semua memiliki foto/ulasan) tanpa gagal. | WAJIB |

### 1.4 Modeling dan Evaluasi (komponen: Modeling, Evaluation)
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-22 | Pembagian data WAJIB per UMKM (GroupKFold, K = 5 atau 10, stratified bila memungkinkan); resampling hanya pada data latih. | WAJIB |
| FR-23 | Klaster eksploratif WAJIB memakai K-Means (baseline) dan K-Prototypes bila fitur campuran; ACO SEBAIKNYA sebagai pembanding; fitur WAJIB diskalakan. | WAJIB |
| FR-24 | Klasifikasi WAJIB melatih baseline (Logistic Regression) dan minimal satu model kuat (Random Forest atau XGBoost). | WAJIB |
| FR-25 | Label klasifikasi WAJIB berasal dari survei lapangan (Label Ground Truth), **bukan** dari hasil klaster. | WAJIB |
| FR-26 | Evaluasi WAJIB melaporkan macro-F1, balanced accuracy, confusion matrix; klaster: Silhouette dan DBI; deteksi: mAP. | WAJIB |
| FR-27 | Jika metrik di bawah target (macro-F1 < 70%), WAJIB ada loop tuning (Optuna/GridSearchCV) sebelum lanjut. | WAJIB |
| FR-28 | Sistem WAJIB menyertakan baseline pembanding (DummyClassifier dan aturan kata kunci). | WAJIB |
| FR-29 | Sistem WAJIB menyediakan penjelasan model (SHAP; Grad-CAM untuk gambar). | WAJIB |
| FR-30 | Eksperimen WAJIB dicatat (MLflow atau tabel bersama): parameter, metrik, versi data, seed. | WAJIB |

### 1.5 Serving dan BDA
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-31 | Model terpilih WAJIB disimpan di Model Registry (MLflow/ONNX) beserta versi dan metrik. | WAJIB |
| FR-32 | Inference API (FastAPI) WAJIB menerima foto dan/atau teks dan mengembalikan kelas, probabilitas, indikator terdeteksi, dan penjelasan singkat. | WAJIB |
| FR-33 | BDA WAJIB mencakup agregasi per kecamatan, analisis spasial (GeoPandas/Folium), statistik spasial (PySAL/SciPy/statsmodels), analisis gap per wilayah, dan dashboard. | WAJIB |
| FR-34 | Sistem WAJIB menghasilkan rekomendasi intervensi per kecamatan/kelas berdasarkan indikator yang belum dimiliki. | WAJIB |

### 1.6 Aplikasi AI
| ID | Persyaratan | Prioritas |
|---|---|---|
| FR-35 | Web App (Streamlit) WAJIB menampilkan peta sebaran, filter kecamatan/kelas, dan detail UMKM. | WAJIB |
| FR-36 | Web App WAJIB memungkinkan unggah foto/teks untuk cek status digital melalui Inference API. | WAJIB |
| FR-37 | Pencarian dan rekomendasi regulasi/insentif (LangChain + FAISS) OPSIONAL, dengan menampilkan potongan sumber. | OPSIONAL |
| FR-38 | Aplikasi WAJIB lolos uji fungsi (UAT) sebelum laporan final; kegagalan memicu perbaikan. | WAJIB |
| FR-39 | Laporan WAJIB menjelaskan kode dan metode ML/DL dengan bukti (Google Docs). | WAJIB |

## 2. Persyaratan non-fungsional
| ID | Kategori | Persyaratan |
|---|---|---|
| NFR-01 | Reproduksibilitas | Pustaka dipin di `requirements.txt`; seed acak tetap; notebook dapat dijalankan ulang dari Raw Data. |
| NFR-02 | Kinerja | Inference API merespons ≤ 5 detik per permintaan pada CPU untuk 1 foto **[USULAN]**. |
| NFR-03 | Skalabilitas | Pipeline memakai Parquet/DuckDB sehingga dapat diperbesar; PySpark dipakai untuk demonstrasi batch/stream. |
| NFR-04 | Keamanan | API key di `.env`, tidak di repo; akses Drive private; hak akses berbasis peran. |
| NFR-05 | Privasi | Tidak ada data pribadi di zona Clean/Gold, repo, atau laporan; wajah diburamkan; hanya agregat ditampilkan di dashboard publik. |
| NFR-06 | Kepatuhan | Patuhi ToS API dan lisensi dataset; catat lisensi AGPL-3.0 pada Ultralytics bila kode dirilis. |
| NFR-07 | Maintainability | Kode modular per tahap (`ingestion`, `processing`, `features`, `models`, `evaluation`, `bda`, `serving`), ruff/black, uji unit dasar. |
| NFR-08 | Observability | Log metrik dan versi model; pantau drift data baru (opsional). |
| NFR-09 | Usabilitas | Antarmuka berbahasa Indonesia, istilah sederhana, label kelas konsisten. |
| NFR-10 | Dokumentasi | Diagram hitam-putih di draw.io; semua keputusan penting dicatat di laporan. |

## 3. Persyaratan data
### 3.1 Volume dan format
- Gambar ≥ 200 MB; teks/numerik ≥ 2.000 baris; dokumen sebagai pendukung.
- Setiap anggota: ≥ 3 aplikasi online dan 1 sumber lapangan (real world); data ≥ 3 format (teks, gambar, dokumen, idealnya numerik).
- Definisi "baris" harus jelas **[USULAN]**: baris ulasan dan baris UMKM dihitung terpisah; targetkan ≥ 2.000 baris ulasan/teks **dan** daftar UMKM yang cukup untuk pembagian per UMKM.

### 3.2 Skema inti **[USULAN]**
| Tabel | Kolom kunci |
|---|---|
| `umkm` | `umkm_id`, `nama_anonim`, `kecamatan`, `lat`, `lon`, `sumber`, `kategori_kuliner` |
| `ulasan` | `ulasan_id`, `umkm_id`, `rating`, `teks_bersih`, `tanggal`, `sumber` |
| `foto` | `foto_id`, `umkm_id`, `path`, `hash`, `sumber`, `status_keep` |
| `anotasi` | `foto_id`, `kelas_indikator`, `bbox`, `annotator` |
| `dokumen` | `dok_id`, `judul`, `sumber`, `halaman`, `lisensi` |
| `label` | `umkm_id`, `kelas` (Konvensional/Berkembang/Maju), `qris`, `pos`, `edc`, `merchant_online`, `sumber_label=survei` |
| `gold_fitur` | `umkm_id`, fitur gabungan (indikator, embedding tereduksi, numerik terskala), `split_fold` |

### 3.3 Kualitas data
- Kunci unik dan `umkm_id` konsisten di semua tabel.
- Tidak ada UMKM yang sama di train dan test.
- Foto duplikat/hampir sama dibuang; foto tidak relevan masuk `foto_reject`.
- Setiap transformasi dicatat (sumber → Raw → Clean → Gold).

## 4. Persyaratan teknologi (dari arsitektur)
| Lapisan | Teknologi |
|---|---|
| Ingestion | requests, Places API, kagglehub, BeautifulSoup, Pandas, DuckDB, PySpark, Google Sheets |
| Penyimpanan | Google Drive (Raw), Parquet + PyArrow (Clean, Gold) |
| Processing | ydata-profiling, Seaborn, scikit-learn, SciPy, OpenCV, Albumentations, PyMuPDF, Tesseract OCR, NLTK |
| AI | Label Studio, Ultralytics YOLO, IndoBERT, ResNet-50, TF-IDF, scikit-learn Pipeline, PCA |
| Modeling | K-Means, K-Prototypes, ACO, Logistic Regression, Random Forest, XGBoost, Optuna, GridSearchCV |
| Evaluasi | scikit-learn Metrics, GroupKFold, mAP, SHAP, Grad-CAM |
| Serving | MLflow, ONNX, ONNX Runtime, FastAPI |
| BDA | GeoPandas, Folium, PySAL, statsmodels, Plotly, Streamlit |
| Aplikasi | Streamlit, FastAPI, LangChain + FAISS (opsional), Google Docs |
| Pendukung | Google Colab, VS Code, Python 3.10+, PyTorch, Hugging Face, Git/GitHub, DVC (opsional), python-dotenv |

## 5. Kriteria penerimaan
1. Gerbang volume terpenuhi dan terdokumentasi di Data Collection Schema.
2. Seluruh notebook inti berjalan ulang tanpa error dari Raw Data.
3. macro-F1 pada data uji (GroupKFold) ≥ 70% **atau** ada analisis tertulis mengapa tidak tercapai beserta bukti tuning.
4. Detektor indikator melaporkan mAP per kelas; Grad-CAM atau contoh deteksi menunjukkan model melihat objek yang benar.
5. Tidak ada data pribadi atau API key di repo, laporan, atau dashboard.
6. Web App dan Inference API berjalan dan lolos uji fungsi.
7. Diagram (arsitektur, flowchart, C4, flowchart aplikasi) hitam-putih di draw.io dan konsisten dengan kode.
8. Laporan menjelaskan kode dan metode dengan bukti, plus keterbatasan (bias seleksi, ulasan terbatas, ukuran sampel).

## 6. Keterlacakan (ringkas)
| Komponen arsitektur | Requirement |
|---|---|
| DataSource, Ingestion | FR-01–FR-09 |
| Data Processing | FR-10–FR-21 |
| Modeling | FR-22–FR-25, FR-27 |
| Evaluation & Serving | FR-26, FR-28–FR-32 |
| BDA | FR-33–FR-34 |
| AI Application | FR-35–FR-39 |
| MLOps, Governance, Assessment | FR-30, NFR-01, NFR-04–NFR-08 |
