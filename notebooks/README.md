# Notebooks Workflow — Peta Digital UMKM Kuliner Surabaya Timur

Urutan eksekusi notebook wajib mengikuti alur pipeline standar:

| No | File Notebook | Deskripsi & Fokus Khusus (Google Maps & Survei) | Output / Target |
|---|---|---|---|
| `01` | `01_ingestion.ipynb` | Ingestion kueri Google Places API (profil, ulasan, foto) + Intake data survei lapangan primer | `data/raw/` (Immutable) + Check Volume Gate |
| `02` | `02_eda.ipynb` | Hash deduplikasi foto, sanitasi regex pada ulasan, blur wajah OpenCV, statistik sebaran awal | Laporan EDA & Sanity Check |
| `03` | `03_preprocessing.ipynb` | Standardisasi foto kasir/tempat (224x224), normalisasi teks ulasan Indonesia, formatting tabel survei | `data/clean/*.parquet` |
| `04` | `04_features.ipynb` | Inferensi detektor YOLO (QRIS/POS/EDC/Online) + IndoBERT teks + ResNet-50 foto + Fusi Fitur | `data/gold/fitur_umkm.parquet` |
| `05` | `05_models.ipynb` | GroupKFold split (berdasarkan `umkm_id`), K-Means clustering, Klasifikasi 3-kelas (RF/XGBoost) | Model Weights & Checkpoints |
| `06` | `06_evaluation.ipynb` | Evaluasi macro-F1 (target ≥ 70%), Confusion Matrix, Feature Importance (SHAP), Grad-CAM detektor | `reports/figures/` & Model Registry |
| `07` | `07_bda_spatial.ipynb` | Agregasi per kecamatan Surabaya Timur (Gubeng, Rungkut, dll.), Spatial Autocorrelation (Moran's I), Gap Analysis | Visualisasi GeoPandas / Folium |
