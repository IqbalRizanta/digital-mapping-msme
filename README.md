# Peta Digital Kematangan UMKM Kuliner Surabaya Timur

Sistem analitik Big Data Spasial, Machine Learning, dan Deep Learning untuk memetakan tingkat kematangan digital (**Konvensional**, **Berkembang**, **Maju**) serta rekomendasi intervensi kebijakan UMKM kuliner di 6 kecamatan Surabaya Timur (`Sukolilo`, `Gubeng`, `Rungkut`, `Mulyorejo`, `Tenggilis Mejoyo`, `Gunung Anyar`).

---

## Struktur Direktori Proyek (Clean Directory Architecture)

```text
C:\big-data-analytic\
├── app/                                # Layer Aplikasi (Client & Server)
│   ├── client/                         # Frontend Web Application (Streamlit)
│   │   ├── dashboard.py                # UI 4-Tab: BDA Spasial, Prediksi 6 Model, Direktori, Artefak
│   │   └── api_client.py               # HTTP Client ke FastAPI Server + Fallback Lokal
│   └── server/                         # Backend Inference Server (FastAPI)
│       ├── main.py                     # Endpoint REST (/health, /bda/summary, /umkm/list, /predict/multimodal)
│       ├── inference_service.py        # Pemuat model ML (.pkl) & DL (.pt/.pth/.h5)
│       └── schemas.py                  # Skema validasi request/response Pydantic
│
├── model/                              # Layer Big Data Analytic & Stored Models
│   ├── ingestion/                      # Akuisisi Google Places API, Survei SWK & Filter sample/keep
│   │   ├── places_client.py
│   │   ├── survey_scanner.py
│   │   ├── run_quota_expansion.py
│   │   └── sample_keep_filter.py
│   ├── processing/                     # Pembersihan Multimodal & PySpark Staging ke Google Drive
│   │   ├── pyspark_staging.py          # SparkSession SQL Join & sinkronisasi ke G:\My Drive\staging_spark
│   │   ├── text_processor.py           # Normalisasi teks ulasan & sensor PII
│   │   ├── image_processor.py          # Deduplikasi dHash & OpenCV Gaussian Face Blur
│   │   ├── anonymizer.py
│   │   └── build_clean_dataset.py
│   ├── features/                       # Feature Engineering & Multimodal Embedding
│   │   └── feature_embedding.py        # Ekspor embedding ke .npy, .npz, dan .parquet
│   ├── training/                       # Pelatihan & Evaluasi Model ML dan DL
│   │   ├── ml_pipeline.py              # ML Clustering, Classification, Regression (.pkl)
│   │   ├── dl_pipeline.py              # DL Clustering, Classification, Regression (.pt, .pth, .h5)
│   │   └── yolo_full_pipeline.py       # Deteksi & klasifikasi indikator visual YOLOv8
│   ├── analytics/                      # Big Data Analytics Spasial (BDA)
│   │   └── bda_analytics.py            # Perhitungan Digital Gap Index per Kecamatan
│   ├── notebooks/                      # Notebook Analisis & Evaluasi (Tersinkronisasi ke Google Drive)
│   │   ├── eda_3_09040624091.ipynb
│   │   ├── model_3_09040624091.ipynb
│   │   ├── bda_3_09040624091.ipynb
│   │   ├── build_notebooks.py
│   │   └── run_notebook.py
│   ├── saved/                          # Direktori Penyimpanan Model & Embedding Terlatih
│   │   ├── ml_clustering_kmeans.pkl
│   │   ├── ml_classification_rf.pkl
│   │   ├── ml_classification_xgb.pkl
│   │   ├── ml_regression_rf.pkl
│   │   ├── ml_regression_ridge.pkl
│   │   ├── feature_fusion_pipeline.pkl
│   │   ├── dl_clustering_autoencoder.pt / .pth / .h5
│   │   ├── dl_classification_mlp.pt / .pth / .h5
│   │   ├── dl_regression_mlp.pt / .pth / .h5
│   │   ├── yolov8n_indicator_detect.pt
│   │   ├── yolov8n_maturity_cls.pt
│   │   ├── embeddings_multimodal.npy
│   │   ├── embeddings_multimodal.npz
│   │   └── embeddings_multimodal.parquet
│   └── run_all.py                      # Orkestrator eksekusi seluruh pipeline model/
│
├── data/                               # Penyimpanan Dataset Lokal (Tersinkronisasi ke Google Drive)
│   ├── raw/                            # Data mentah Google Maps, Survei SWK, & Katalog Skema
│   ├── sample/                         # Referensi kurasi visual: keep/ (43 foto) & reject/ (627 foto)
│   ├── staging_spark/                  # Hasil staging PySpark (umkm_spark_staged.parquet)
│   ├── clean/                          # Parquet bersih (umkm, ulasan, foto, dokumen)
│   └── gold/                           # Feature Store (feature_store.parquet)
│
├── peta-digital-umkm/                  # Dokumentasi Rancangan, Arsitektur C4, Flowchart & Laporan
│   ├── ARCHITECTURE.md                 # Spesifikasi Arsitektur Big Data & AI
│   ├── C4_ARCHITECTURE_AND_FLOWCHART_1_4.md # Arsitektur Aplikasi C4 Model & Flowchart Halaman 1-4
│   ├── WORKFLOW.md                     # Alur Pengerjaan 6 Tahap (B1-B6)
│   ├── MONITORING_CODING_40_60.md      # Matriks Proporsi 60% Non-AI vs <=40% AI, Dokumentasi & Log Prompt
│   ├── PRD.md
│   ├── REQUIREMENTS.md
│   ├── SKILLS.md
│   ├── TODO.md
│   ├── AGENTS.md
│   ├── diagrams/                       # File Draw.io (.xml) Arsitektur, C4 & Flowchart
│   └── reports/                        # Metrik Evaluasi Model & Tabel BDA Kecamatan
│
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan Proyek

### 1. Eksekusi Pipeline Big Data Analytic, PySpark Staging, Embedding & Pelatihan 6 Model (`model/`)
```powershell
$env:PYTHONPATH="C:\big-data-analytic"
.\.msme\Scripts\python.exe -m model.run_all
```

### 2. Menjalankan Backend Inference Server (`app/server/`)
```powershell
$env:PYTHONPATH="C:\big-data-analytic"
.\.msme\Scripts\uvicorn.exe app.server.main:app --host 127.0.0.1 --port 8000
```

### 3. Menjalankan Aplikasi Web Client (`app/client/`)
```powershell
$env:PYTHONPATH="C:\big-data-analytic"
.\.msme\Scripts\streamlit.exe run app/client/dashboard.py
```
