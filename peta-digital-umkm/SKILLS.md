# SKILLS — Peta Digital UMKM Kuliner Surabaya Timur

> Status: Draft 0.1 · 26 Oktober 2026. Kumpulan "skill" (prosedur terstandar) yang dipakai agen AI maupun anggota tim. Tiap skill memiliki tujuan, masukan, langkah, keluaran, alat, dan batasan. Nama pustaka mengikuti `ARCHITECTURE.md`. Path kode adalah konvensi **[USULAN]**.

## Format skill
`nama-skill` → **Tujuan** · **Masukan** · **Langkah** · **Keluaran** · **Alat** · **Guardrail**

---
## S01 `ingest-places-api`
- **Tujuan:** mengambil data tempat, ulasan, dan foto Google Maps secara sah.
- **Masukan:** daftar kueri/grid koordinat per kecamatan; API key di `.env`.
- **Langkah:** (1) baca konfigurasi kecamatan; (2) panggil Places API dengan banyak kueri untuk menembus batas hasil per kueri; (3) simpan respons mentah JSON; (4) ambil foto yang diizinkan; (5) catat jumlah, tanggal, dan kuota di Data Collection Schema.
- **Keluaran:** `raw/google_maps/<tanggal>/*.json`, foto di `raw/google_maps/<tanggal>/foto/`.
- **Alat:** requests, Places API, Pandas.
- **Guardrail:** hanya API resmi; jangan menyimpan key di repo; catat batas ulasan (sekitar 5/tempat); jangan menyimpan identitas pengulas.

## S02 `ingest-open-datasets`
- **Tujuan:** memuat dataset Kaggle, Opendata, dan Mendeley Data.
- **Masukan:** tautan/nama dataset; kredensial Kaggle.
- **Langkah:** unduh via kagglehub/URL → verifikasi checksum/ukuran → catat lisensi → simpan ke `raw/<sumber>/`.
- **Keluaran:** berkas mentah + baris di Data Collection Schema.
- **Alat:** kagglehub, requests, Pandas.
- **Guardrail:** hormati lisensi; jangan menimpa Raw.

## S03 `ingest-documents`
- **Tujuan:** menyimpan dokumen jurnal/regulasi sebagai PDF.
- **Masukan:** PDF hasil unduhan manual atau API resmi.
- **Langkah:** beri nama standar → catat metadata (judul, sumber, lisensi) → simpan.
- **Keluaran:** `raw/dokumen/*.pdf` + metadata.
- **Guardrail:** tidak ada pengambilan otomatis dari situs yang melarangnya (mis. ScienceDirect tanpa izin).

## S04 `field-survey-intake`
- **Tujuan:** memasukkan foto dan isian survei lapangan secara rapi.
- **Masukan:** foto, form survei (indikator QRIS/POS/EDC/Online, kelas lapangan).
- **Langkah:** validasi nama folder `Kecamatan/Nama_Tempat/` → cek kelengkapan form → hubungkan ke `umkm_id` → simpan.
- **Keluaran:** `raw/survei/...` + tabel `label` mentah.
- **Guardrail:** minta persetujuan; jangan memotret wajah/dokumen pribadi; label survei adalah sumber kebenaran kelas.

## S05 `volume-gate`
- **Tujuan:** memastikan data memenuhi syarat sebelum diproses.
- **Masukan:** isi folder Raw dan Data Collection Schema.
- **Langkah:** hitung total ukuran gambar (MB) dan jumlah baris teks/numerik → bandingkan dengan 200 MB dan 2.000 baris → hasilkan laporan kekurangan per anggota/sumber.
- **Keluaran:** `reports/volume_check.md`.
- **Guardrail:** hitung duplikat sekali; jelaskan definisi "baris".

## S06 `validate-and-dedupe`
- **Tujuan:** memeriksa format, integritas, dan duplikat.
- **Langkah:** cek file rusak → hash gambar (imagehash/md5) → deduplikasi tabel berdasar kunci → pindahkan foto tak relevan ke `foto_reject/` (jangan hapus).
- **Keluaran:** daftar masalah + berkas bersih sementara.
- **Alat:** Pandas, Pillow, hashlib.

## S07 `anonymize`
- **Tujuan:** menghapus data pribadi sebelum data masuk Clean.
- **Langkah:** regex untuk telepon/email/nama pemilik; hapus ID pengulas; deteksi dan blur wajah pada foto; cek sampel acak secara manual.
- **Keluaran:** data teks dan foto teranonimkan; log jumlah item yang diubah.
- **Alat:** regex, OpenCV.
- **Guardrail:** simpan hanya versi anonim di Clean/Gold; anonimisasi tidak dianggap sempurna, selalu cek sampel.

## S08 `eda-multimodal`
- **Tujuan:** memahami data sebelum modeling.
- **Langkah:** tabel (distribusi, missing, asosiasi kategorikal: chi-square/Cramér's V) → teks (panjang, kata dominan) → gambar (resolusi, kecerahan, distribusi kelas, duplikat) → dokumen (halaman, rasio scan) → simpulkan masalah kualitas.
- **Keluaran:** `reports/eda/*.html|png` + ringkasan temuan.
- **Alat:** Pandas, ydata-profiling, Matplotlib, Seaborn, SciPy.
- **Guardrail:** hindari scatter plot untuk fitur biner/kategorikal; laporkan, jangan menghapus diam-diam.

## S09 `clean-missing-outliers`
- **Tujuan:** menangani missing dan outlier.
- **Langkah:** tentukan aturan per kolom → imputasi/hapus/winsorize → catat sebelum/sesudah.
- **Keluaran:** Clean (Parquet) + log keputusan.
- **Alat:** scikit-learn, SciPy.

## S10 `preprocess-3-formats`
- **Tujuan:** menyiapkan gambar, teks, dan dokumen.
- **Langkah:** gambar → resize 224×224, normalisasi, augmentasi (hanya data latih); teks → case folding, normalisasi slang, stopword; dokumen → PyMuPDF, OCR Tesseract bila scan, chunking.
- **Keluaran:** tensor/teks bersih/chunk di Clean.
- **Alat:** OpenCV, Albumentations, NLTK, PyMuPDF, Tesseract.

## S11 `annotate-indicators`
- **Tujuan:** membuat anotasi QRIS/POS/EDC/Merchant Online.
- **Langkah:** tulis pedoman (contoh positif/negatif) → bagi foto ke anotator → anotasi bounding box di Label Studio → cek konsistensi antar-anotator → ekspor format YOLO.
- **Keluaran:** `data/annotations/` format YOLO.
- **Guardrail:** satu foto boleh memuat banyak kelas; targetkan jumlah instance per kelas sejak awal.

## S12 `train-indicator-detector`
- **Tujuan:** melatih detektor indikator.
- **Langkah:** split per UMKM → latih YOLO (versi kecil di Colab) → evaluasi mAP50 dan mAP50-95 per kelas → simpan contoh deteksi dan Grad-CAM/visualisasi → jika di bawah target, tambah anotasi atau tuning.
- **Keluaran:** bobot model, kurva metrik, contoh deteksi.
- **Alat:** Ultralytics, PyTorch.
- **Guardrail:** catat lisensi AGPL-3.0; jangan evaluasi pada foto dari UMKM yang sama dengan data latih.

## S13 `build-gold-features`
- **Tujuan:** membuat tabel fitur per UMKM.
- **Langkah:** jalankan detektor → indikator per UMKM → embedding teks (IndoBERT/TF-IDF) dan gambar (ResNet-50) → agregasi per UMKM → gabung fitur numerik → scaling dan PCA dalam Pipeline → tangani modalitas hilang (indikator "tidak ada data") → simpan Gold.
- **Keluaran:** `gold/fitur_umkm.parquet`.
- **Alat:** scikit-learn Pipeline, PyArrow.
- **Guardrail:** scaler dipasang hanya di data latih.

## S14 `group-split`
- **Tujuan:** pembagian data tanpa kebocoran.
- **Langkah:** GroupKFold (K = 5 atau 10) berdasarkan `umkm_id`; stratifikasi kelas bila memungkinkan; resampling hanya pada fold latih.
- **Keluaran:** kolom `split_fold`.

## S15 `train-classifier`
- **Tujuan:** melatih klasifikasi 3 kelas.
- **Langkah:** baseline (Dummy, aturan kata kunci, Logistic Regression) → Random Forest/XGBoost → tuning (Optuna/GridSearchCV) → macro-F1 per fold.
- **Keluaran:** model, metrik, confusion matrix.
- **Guardrail:** label dari survei saja; laporkan hasil baseline; jangan hanya melaporkan accuracy.

## S16 `cluster-validation`
- **Tujuan:** segmentasi eksploratif.
- **Langkah:** scaling → K-Means (k = 2..6) dan K-Prototypes bila fitur campuran → ACO sebagai pembanding → Silhouette, DBI, ukuran klaster → bandingkan dengan label survei (ARI/NMI) → tafsirkan profil klaster.
- **Guardrail:** klaster bukan sumber label; waspadai Silhouette tinggi akibat satu fitur dominan atau klaster kecil.

## S17 `explain-model`
- **Tujuan:** menjelaskan prediksi.
- **Langkah:** SHAP global/lokal untuk klasifikasi; Grad-CAM untuk detektor; ringkas fitur terpenting dalam bahasa sederhana.

## S18 `register-model`
- **Tujuan:** menyimpan model terpilih.
- **Langkah:** catat parameter, metrik, versi data, seed di MLflow → ekspor ONNX → uji kesetaraan keluaran → tulis model card singkat.

## S19 `serve-inference-api`
- **Tujuan:** menyediakan layanan prediksi.
- **Langkah:** FastAPI dengan skema Pydantic → muat model ONNX → endpoint prediksi + kesehatan → uji dengan contoh foto/teks.
- **Guardrail:** batas ukuran file; validasi tipe; jangan menyimpan unggahan pengguna tanpa persetujuan.

## S20 `bda-spatial`
- **Tujuan:** analisis spasial dan gap.
- **Langkah:** agregasi per kecamatan (DuckDB/Pandas) → join titik ke batas wilayah (GeoPandas) → peta (Folium) → statistik spasial (Moran's I/hotspot PySAL) → uji beda antar-kecamatan (SciPy/statsmodels) → hitung gap indikator per wilayah → rekomendasi.
- **Guardrail:** tampilkan agregat; jangan menampilkan titik yang dapat mengidentifikasi pemilik tanpa izin.

## S21 `build-streamlit-app`
- **Tujuan:** aplikasi pengguna.
- **Langkah:** halaman peta + filter → halaman cek status (unggah foto/teks → API) → halaman dashboard → penanganan error dan status memuat → uji UAT.
- **Guardrail:** bahasa Indonesia sederhana; tampilkan ketidakpastian (probabilitas) dan keterbatasan model.

## S22 `rag-regulation-search` (opsional)
- **Tujuan:** pencarian regulasi/insentif.
- **Langkah:** chunk dokumen → embedding → FAISS → pencarian → jawaban dengan potongan sumber → evaluasi Recall@k dan jawaban manual.
- **Guardrail:** tampilkan sumber; jangan menjawab di luar dokumen.

## S23 `write-report`
- **Tujuan:** laporan terstruktur.
- **Langkah:** ringkas tujuan → data dan etika → metode → hasil (tabel, grafik) → penjelasan kode dan bukti → keterbatasan → rekomendasi.

## S24 `update-diagrams`
- **Tujuan:** menjaga diagram konsisten.
- **Langkah:** setelah perubahan arsitektur/alur → perbarui draw.io (hitam-putih), `ARCHITECTURE.md`, `WORKFLOW.md`.

---
## Matriks kompetensi tim (isi sendiri)
| Kompetensi | Skill terkait | Anggota |
|---|---|---|
| Python, Pandas, DuckDB | S02, S06, S08, S09, S20 | ____ |
| Computer vision (YOLO, OpenCV) | S07, S10–S12 | ____ |
| NLP (IndoBERT, TF-IDF) | S10, S13 | ____ |
| ML dan evaluasi | S14–S18 | ____ |
| API dan aplikasi | S19, S21, S22 | ____ |
| Dokumentasi dan diagram | S23, S24 | ____ |
