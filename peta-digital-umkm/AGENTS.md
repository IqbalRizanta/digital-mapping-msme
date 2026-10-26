# AGENTS — Panduan untuk Agen AI dan Kontributor

> Status: Draft 0.1 · 26 Oktober 2026. Berlaku untuk agen AI (mis. asisten kode) dan anggota tim yang bekerja di repositori **Peta Digital UMKM** (Pemetaan Digitalisasi UMKM Kuliner Surabaya Timur). Baca `PRD.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `WORKFLOW.md`, `SKILLS.md` sebelum mengubah apa pun.

## 1. Ringkasan proyek
Aplikasi AI multimodal (gambar, teks/numerik, dokumen) yang mengklasifikasikan UMKM kuliner Surabaya Timur ke **Konvensional / Berkembang / Maju** berdasarkan indikator **QRIS, POS, EDC, Merchant Online**, lalu memetakan sebaran dan gap per kecamatan. Alur: Data → EDA → Modeling/Evaluasi → BDA → Aplikasi AI.

## 2. Aturan utama (tidak boleh dilanggar)
1. **Privasi:** jangan menulis, mencetak, atau meng-commit data pribadi (nama pemilik, kontak, ID pengulas, wajah). Pakai versi teranonimkan.
2. **Secret:** API key dan kredensial hanya di `.env` (di-gitignore). Jangan taruh di kode, notebook, log, atau laporan.
3. **ToS dan lisensi:** pakai API resmi; jangan membuat scraper untuk situs yang melarangnya (mis. ScienceDirect, scraping tidak resmi Google Maps). Catat lisensi dataset.
4. **Raw immutable:** jangan mengubah atau menimpa `raw/`. Hasil olahan ke `clean/` dan `gold/`.
5. **Tanpa kebocoran data:** split per `umkm_id` (GroupKFold); scaler/encoder dipasang di data latih saja; label kelas hanya dari survei lapangan, **bukan** dari hasil klaster atau fitur yang sama dengan masukan.
6. **Jujur pada hasil:** laporkan metrik apa adanya beserta baseline; jangan memilih angka terbaik secara selektif; nyatakan keterbatasan.
7. **Jangan mengarang data atau sumber.** Jika data/atribut tidak ada, katakan dan minta keputusan.
8. **Perubahan besar** (arsitektur, definisi kelas, ambang metrik) harus dibahas dulu; jangan diubah diam-diam.

## 3. Struktur repositori yang diharapkan **[USULAN]**
```
peta-digital-umkm/
├── AGENTS.md  PRD.md  REQUIREMENTS.md  TODO.md  WORKFLOW.md  SKILLS.md  ARCHITECTURE.md
├── configs/            # kecamatan.yaml, kelas.yaml, threshold.yaml
├── data/               # raw/ clean/ gold/ (di-gitignore; tersimpan di Google Drive)
├── notebooks/          # 01_ingestion ... 07_bda
├── src/umkm/
│   ├── ingestion/  processing/  features/  models/  evaluation/  bda/  serving/
├── app/                # streamlit_app.py
├── tests/
├── reports/            # grafik, metrik, bukti laporan
├── models/             # registry lokal / ONNX (di-gitignore bila besar)
├── requirements.txt  .env.example  .gitignore
```
Jika struktur repo berbeda, ikuti struktur yang ada dan perbarui bagian ini.

## 4. Lingkungan dan perintah umum **[USULAN, sesuaikan]**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # isi kunci secara lokal, jangan commit
ruff check . && black --check .  # gaya kode
pytest -q                        # uji
uvicorn src.umkm.serving.api:app --reload      # Inference API
streamlit run app/streamlit_app.py             # aplikasi
```
Python 3.10+. GPU hanya lewat Google Colab: simpan hasil ke Drive, jangan bergantung pada sesi.

## 5. Konvensi kode
- Modular per tahap; fungsi berdokumentasi (docstring singkat), tipe bila memungkinkan.
- Konfigurasi (kecamatan, kelas, ambang) di `configs/`, bukan hard-coded.
- Notebook tipis: logika dipindah ke `src/umkm/`; hapus output sebelum commit.
- Seed tetap (`random_state`) dan versi pustaka dipin.
- Nama kelas dan indikator konsisten: `Konvensional`, `Berkembang`, `Maju`; `qris`, `pos`, `edc`, `merchant_online`.
- Bahasa: komentar/laporan boleh Indonesia; nama variabel dan fungsi berbahasa Inggris.

## 6. Definition of Done untuk setiap perubahan
- Jalan ulang tanpa error dari data yang relevan.
- `ruff`, `black`, dan `pytest` lolos (bila ada uji).
- Tidak ada data pribadi/secret baru di diff.
- Metrik dan keputusan gerbang dicatat; dokumen terkait (diagram, `ARCHITECTURE.md`, `WORKFLOW.md`) diperbarui bila alur berubah.
- Pull request kecil dengan deskripsi: apa, mengapa, bagaimana diuji.

## 7. Agen khusus (peran) **[USULAN]**
| Agen | Tanggung jawab | Skill (`SKILLS.md`) | Masukan → Keluaran | Serah-terima ke |
|---|---|---|---|---|
| Data Agent | Ingestion, skema data, Drive, gerbang volume | S01–S06 | Sumber → Raw + Schema | EDA Agent |
| EDA & Quality Agent | Validasi, anonimisasi, EDA, cleaning | S06–S09 | Raw → Clean + laporan EDA | Vision/NLP Agent |
| Vision Agent | Anotasi, YOLO, preprocessing gambar | S10–S12 | Foto → indikator | Feature Agent |
| NLP & Document Agent | Preprocessing teks/dokumen, embedding, RAG | S10, S13, S22 | Teks/PDF → embedding/chunk | Feature Agent |
| Feature Agent | Fusi fitur, scaling, Gold | S13–S14 | Indikator + embedding → Gold | Modeling Agent |
| Modeling Agent | Klaster, klasifikasi, tuning, SHAP, registry | S15–S18 | Gold + label → model + metrik | Serving/BDA Agent |
| BDA Agent | Agregasi, spasial, statistik, gap | S20 | Hasil model → peta + analisis | App Agent |
| App Agent | Inference API, Streamlit | S19, S21 | Model + BDA → aplikasi | Docs Agent |
| Docs Agent | Laporan, diagram, dokumen proyek | S23–S24 | Semua hasil → laporan | — |

### Aturan koordinasi antar-agen
- Satu agen satu tugas dalam satu PR; jangan mengedit area agen lain tanpa koordinasi.
- Serah-terima lewat berkas dan skema di `ARCHITECTURE.md` (kontrak data), bukan asumsi lisan.
- Jika kontrak data berubah, perbarui `ARCHITECTURE.md` dan beri tahu agen hilir.
- Agen tidak memutuskan definisi kelas, ambang, atau lingkup; itu keputusan tim/dosen.

## 8. Yang harus ditanyakan, bukan diasumsikan
- Daftar kecamatan final dan sumber batas wilayah.
- Definisi kelas final.
- Arti "baris" untuk target ≥ 2.000.
- Target mAP.
- Akses dan kuota API.
- Apakah RAG/Streaming masuk cakupan.

## 9. Format laporan hasil kerja agen
Tulis: apa yang dikerjakan · berkas yang berubah · cara menjalankan/menguji · metrik utama · keterbatasan/risiko · langkah berikutnya. Tandai bagian yang belum terverifikasi.
