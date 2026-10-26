# TODO — Peta Digital UMKM Kuliner Surabaya Timur

> Status: Draft 0.1 · 26 Oktober 2026. Centang `[x]` saat selesai. **PJ** = penanggung jawab (isi nama/NIM). Tanggal target belum ditetapkan; isi setelah rapat kelompok.
> Urutan mengikuti flowchart `flowchart-bd-ai`: Ingestion → Validasi & EDA → Preprocessing & Fitur → Modeling & Evaluasi → Serving & BDA → AI Application.

## Fase 0 — Persiapan dan keputusan
- [ ] Sepakati daftar kecamatan Surabaya Timur dan sumber batas wilayah (GeoJSON/shapefile). — PJ: ____
- [ ] Tulis definisi final kelas (Konvensional/Berkembang/Maju) dan indikator (QRIS, POS, EDC, Merchant Online); minta persetujuan dosen. — PJ: ____
- [ ] Tentukan arti "baris" untuk target ≥ 2.000 baris dan pembagian target gambar per anggota. — PJ: ____
- [ ] Buat repositori GitHub, struktur folder, `.gitignore` (data, model besar, `.env`). — PJ: ____
- [ ] Siapkan Google Drive (folder `raw/`, `clean/`, `gold/`, `models/`, `reports/`) dan hak akses. — PJ: ____
- [ ] Siapkan Google Sheet monitoring (tugas, anggota, status, Data Collection Schema). — PJ: ____
- [ ] Pasang lingkungan: Python 3.10+, `requirements.txt` terpin, template notebook Colab. — PJ: ____
- [ ] Putuskan: RAG (F8) dan Streaming (F9) masuk atau tidak. — PJ: ____

## Fase 1 — Ingestion (ETL)
- [ ] Daftar API key (Google Places) dan simpan di `.env`; cek kuota dan harga terbaru. — PJ: ____
- [ ] Buat klien API: ambil data tempat, ulasan, foto Google Maps dengan banyak kueri/grid per kecamatan. — PJ: ____
- [ ] Unduh dataset Kaggle dan catat lisensi. — PJ: ____
- [ ] Ambil data Opendata dan Mendeley Data; catat lisensi. — PJ: ____
- [ ] Kumpulkan dokumen jurnal/regulasi secara sah (unduh manual/API resmi) sebagai PDF. — PJ: ____
- [ ] Rancang form survei dan pedoman foto (kasir, etalase, papan QRIS, mesin EDC, tampilan POS); uji coba. — PJ: ____
- [ ] Lakukan survei lapangan dan unggah foto ke `Kecamatan/Nama_Tempat/`. — PJ: ____
- [ ] Gabungkan sumber (Batch) dan isi Data Collection Schema. — PJ: ____
- [ ] Simpan ke Raw Data (Google Drive), jangan diubah setelahnya.
- [ ] **Gerbang:** gambar ≥ 200 MB dan teks/numerik ≥ 2.000 baris? Jika belum, tambah data.

## Fase 2 — Validasi dan EDA
- [ ] Cek format file, integritas, dan duplikat (hash gambar). — PJ: ____
- [ ] Anonimisasi data pribadi (regex) dan blur wajah (OpenCV). — PJ: ____
- [ ] EDA tabel: distribusi, missing, korelasi/asosiasi (Cramér's V, chi-square). — PJ: ____
- [ ] EDA teks: panjang, kata dominan, bahasa. — PJ: ____
- [ ] EDA gambar: resolusi, kecerahan, distribusi kelas. — PJ: ____
- [ ] EDA dokumen: jumlah halaman, rasio scan, istilah kunci. — PJ: ____
- [ ] Tangani missing/outlier dan catat keputusan. — PJ: ____
- [ ] Simpan Clean (Parquet).

## Fase 3 — Preprocessing dan fitur
- [ ] Preprocessing gambar (resize, normalisasi, augmentasi), teks (cleaning, slang, stopword), dokumen (ekstraksi, OCR, chunking). — PJ: ____
- [ ] Tulis pedoman anotasi; anotasi foto di Label Studio (cek konsistensi antar-anotator). — PJ: ____
- [ ] Latih YOLO deteksi indikator; evaluasi mAP per kelas. — PJ: ____
- [ ] **Gerbang:** mAP ≥ target? Jika belum, tambah anotasi/tuning.
- [ ] Jalankan deteksi pada semua foto → fitur indikator per UMKM. — PJ: ____
- [ ] Ekstraksi embedding teks (IndoBERT/TF-IDF) dan gambar (ResNet-50). — PJ: ____
- [ ] Fusi fitur + scaling (+ PCA bila perlu); tangani modalitas hilang. — PJ: ____
- [ ] Simpan Gold (Parquet).

## Fase 4 — Modeling dan evaluasi
- [ ] Susun label ground truth dari survei (tabel `label`). — PJ: ____
- [ ] Split per UMKM (GroupKFold K = 5/10), resampling hanya data latih. — PJ: ____
- [ ] Baseline: DummyClassifier, aturan kata kunci, Logistic Regression. — PJ: ____
- [ ] Klasifikasi: Random Forest, XGBoost; tuning (Optuna/GridSearchCV). — PJ: ____
- [ ] Klaster: K-Means, K-Prototypes; ACO sebagai pembanding; Silhouette, DBI. — PJ: ____
- [ ] **Gerbang:** macro-F1 ≥ 70%? Jika belum, tuning/perbaikan fitur.
- [ ] Penjelasan model: SHAP; Grad-CAM untuk detektor. — PJ: ____
- [ ] Catat eksperimen (MLflow/tabel) dan simpan model ke registry (ONNX). — PJ: ____

## Fase 5 — Serving dan BDA
- [ ] Bangun Inference API (FastAPI + ONNX Runtime) dan uji. — PJ: ____
- [ ] Agregasi per kecamatan (DuckDB/Pandas; PySpark untuk demonstrasi). — PJ: ____
- [ ] Analisis spasial: join titik ke kecamatan (GeoPandas), peta (Folium). — PJ: ____
- [ ] Statistik spasial: Moran's I/hotspot (PySAL), uji beda antar-kecamatan. — PJ: ____
- [ ] Analisis gap per wilayah dan rekomendasi intervensi. — PJ: ____
- [ ] Dashboard (Plotly/Streamlit). — PJ: ____

## Fase 6 — Aplikasi AI dan laporan
- [ ] Web App Streamlit: peta, filter, detail, unggah foto/teks. — PJ: ____
- [ ] Hubungkan Web App ke Inference API dan dashboard. — PJ: ____
- [ ] (Opsional) Pencarian regulasi/insentif: LangChain + FAISS. — PJ: ____
- [ ] Uji aplikasi (UAT); perbaiki sampai lolos. — PJ: ____
- [ ] Tulis laporan (Google Docs): metodologi, kode, bukti, keterbatasan. — PJ: ____

## Diagram dan dokumen tugas
- [ ] draw.io hlm. 1: arsitektur BDA (hitam-putih) — sinkron dengan kode.
- [ ] draw.io hlm. 2: flowchart BDA (v1 serpentine dan/atau v2 swimlane).
- [ ] draw.io hlm. 3: arsitektur aplikasi AI (C4 Model) — **belum dibuat**.
- [ ] draw.io hlm. 4: flowchart aplikasi AI — **belum dibuat**.
- [ ] Arsitektur mandiri dan flowchart mandiri tiap anggota (selaras dengan versi kelompok).
- [ ] Perbarui Google Sheet monitoring (Data, EDA, Model-Evaluation, BDA, Aplikasi AI): rapikan ID Data, nama kelas, kolom metode.

## Definition of Done (per fase)
- Kode atau notebook jalan ulang dari awal tanpa error.
- Keluaran tersimpan di zona yang benar (Raw/Clean/Gold/models) dan tercatat di Sheet.
- Tidak ada data pribadi atau secret yang ter-commit.
- Metrik dan keputusan gerbang dicatat; pull request telah ditinjau minimal satu anggota lain.

## Risiko aktif untuk dipantau
- [ ] Volume data belum terpenuhi.
- [ ] Anotasi lambat atau tidak konsisten.
- [ ] Kuota GPU Colab tidak cukup.
- [ ] Kelas sangat tidak seimbang.
- [ ] ToS/lisensi sumber data belum diverifikasi.
