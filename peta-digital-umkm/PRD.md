# PRD — Peta Digital UMKM Kuliner Surabaya Timur

> Status: Draft 0.1 · Disusun: 26 Oktober 2026 · Sumber: diagram `H7D_5-3_diagram-architecture-2026` (halaman `bd-ai-archi` dan `flowchart-bd-ai`) dan panduan tugas Big Data Analytic (BDA).
> Tanda **[USULAN]** berarti belum disepakati tim/dosen dan harus dikonfirmasi sebelum dikunci.

## 1. Ringkasan
**Peta Digital UMKM** adalah aplikasi AI berbasis pendekatan modern big data analytic yang memetakan tingkat digitalisasi UMKM kuliner di wilayah Surabaya Timur. Sistem menggabungkan **satu alur data multimodal** (gambar, teks/numerik, dokumen) untuk mengklasifikasikan setiap UMKM ke tiga tingkat kematangan digital: **Konvensional**, **Berkembang**, dan **Maju**, berdasarkan empat indikator pembayaran dan penjualan digital: **QRIS, POS, EDC, dan Merchant Online**.

## 2. Latar belakang dan masalah
- UMKM kuliner adalah sektor UMKM yang paling banyak dan tersebar, tetapi tingkat adopsi pembayaran dan penjualan digitalnya tidak merata dan jarang terpetakan secara terstruktur di level kecamatan.
- Data pendukung tersebar di banyak sumber dan format (foto usaha, ulasan, dataset publik, dokumen regulasi, survei lapangan), sehingga sulit dianalisis bersama.
- Pembuat kebijakan dan pendamping UMKM membutuhkan **peta sebaran dan analisis gap** untuk menentukan wilayah dan kelompok yang perlu intervensi.

**Pernyataan masalah:** Bagaimana memetakan tingkat digitalisasi UMKM kuliner Surabaya Timur secara objektif dan dapat direproduksi dengan memadukan foto, ulasan, data numerik, dan dokumen, serta menyajikannya dalam aplikasi yang dapat dipakai pemangku kepentingan?

## 3. Tujuan dan non-tujuan
### Tujuan
1. Membangun pipeline data end-to-end (Data → EDA → Modeling/Evaluasi → BDA → Aplikasi AI) yang terdokumentasi dan reproduksibel.
2. Mengklasifikasikan UMKM ke 3 kelas kematangan digital dengan **macro-F1 ≥ 70%** pada data uji.
3. Mendeteksi keempat indikator (QRIS, POS, EDC, Merchant Online) dari foto usaha.
4. Menyajikan peta sebaran, statistik spasial, dan rekomendasi intervensi per kecamatan.
5. Memenuhi syarat tugas: arsitektur dan flowchart (draw.io hitam-putih), laporan, dan monitoring di Google Sheet.

### Non-tujuan
- Bukan sistem pembayaran atau verifikasi transaksi.
- Bukan penilaian kinerja atau kelayakan kredit individu UMKM.
- Tidak memublikasikan data pribadi pemilik usaha atau pengulas.
- Tidak mencakup seluruh Surabaya atau sektor non-kuliner (ekspansi di luar cakupan).

## 4. Pengguna dan kebutuhan
| Pengguna | Kebutuhan utama |
|---|---|
| Pemangku kebijakan UMKM (dinas, kecamatan) | Peta sebaran status digital, wilayah prioritas, gap terhadap program insentif |
| Pendamping / mentor UMKM | Daftar UMKM per kelas dan indikator yang belum dimiliki |
| Pemilik UMKM | Cek status digital usahanya dan saran langkah berikutnya |
| Tim peneliti (mahasiswa) dan dosen | Pipeline reproduksibel, metrik evaluasi, bukti analisis |

## 5. Cakupan
**Wilayah:** Surabaya Timur. Daftar kecamatan **[USULAN, perlu konfirmasi]**: Gubeng, Rungkut, Tenggilis Mejoyo, Gunung Anyar, Sukolilo, Mulyorejo.
**Objek:** UMKM kuliner (warung, kafe, katering, jajanan, dll.).
**Data minimum:** Gambar ≥ 200 MB; teks/numerik ≥ 2.000 baris; dokumen regulasi/insentif sebagai pendukung. Minimal 3 aplikasi online dan 1 sumber lapangan per anggota.

## 6. Definisi kelas dan indikator
Empat indikator: **QRIS**, **POS**, **EDC**, **Merchant Online** (mis. GoFood/GrabFood/ShopeeFood).

| Kelas | Definisi **[USULAN]** |
|---|---|
| Konvensional | Tidak ada indikator digital (transaksi tunai/manual) |
| Berkembang | Memiliki ≥ 1 pembayaran digital (QRIS atau EDC) tanpa sistem lengkap |
| Maju | QRIS + POS + (EDC atau Merchant Online) |

Aturan label harus ditulis final dan disetujui tim sebelum anotasi. **Label ground truth berasal dari survei lapangan**, bukan dari hasil klaster atau dari fitur yang sama dengan masukan model (mencegah data leakage).

## 7. Fitur
### MVP (wajib)
| ID | Fitur | Deskripsi |
|---|---|---|
| F1 | Pipeline data multimodal | Ingestion API, dokumen, survei dan real world ke Raw → Clean → Gold |
| F2 | Deteksi indikator dari foto | Model YOLO untuk QRIS/POS/EDC/Merchant Online |
| F3 | Klasifikasi kematangan digital | Model 3 kelas pada fitur gabungan |
| F4 | Klaster eksploratif | K-Means / K-Prototypes / ACO sebagai validasi segmentasi |
| F5 | Peta dan dashboard | Sebaran per kecamatan, statistik spasial, gap analysis |
| F6 | Aplikasi web | Cek status UMKM, lihat peta, unduh ringkasan |
| F7 | Laporan | Metodologi, hasil, dan bukti kode (Google Docs) |

### Opsional / tahap lanjut
- F8 Pencarian dan rekomendasi regulasi/insentif (RAG: LangChain + FAISS).
- F9 Streaming simulasi (PySpark Structured Streaming) untuk demonstrasi BDA.
- F10 Ekspor laporan per kecamatan.

## 8. Cerita pengguna (contoh)
- Sebagai pejabat kecamatan, saya ingin melihat peta sebaran UMKM Konvensional/Berkembang/Maju agar dapat menentukan wilayah prioritas pendampingan.
- Sebagai pendamping, saya ingin tahu indikator mana (QRIS/POS/EDC/Online) yang belum dimiliki sebuah UMKM agar saran saya spesifik.
- Sebagai pemilik UMKM, saya ingin mengunggah foto kasir/etalase dan melihat status digital serta langkah berikutnya.
- Sebagai dosen/penguji, saya ingin melihat metrik, confusion matrix, dan penjelasan model agar hasil dapat diaudit.

## 9. Metrik keberhasilan
| Metrik | Target | Catatan |
|---|---|---|
| macro-F1 klasifikasi | ≥ 70% | Evaluasi per UMKM (GroupKFold), bukan per foto |
| mAP deteksi indikator | Ditetapkan setelah baseline **[USULAN: mAP50 ≥ 0,60]** | Dilaporkan per kelas |
| Volume data | Gambar ≥ 200 MB; teks/numerik ≥ 2.000 baris | Dicek di gerbang "Volume cukup?" |
| Silhouette / DBI | Dilaporkan, bukan syarat lulus | Bandingkan K-Means vs K-Means+ACO |
| Reproduksibilitas | Seluruh notebook jalan ulang dari data Raw | Versi pustaka dan seed dicatat |
| Aplikasi | Lolos uji fungsi (UAT) | Keputusan "Lolos uji?" pada flowchart |

## 10. Asumsi dan batasan
- API resmi Google Places membatasi ulasan (sekitar 5 per tempat) dan hasil pencarian per kueri; volume teks harus dicapai lewat banyak kueri, survei, dan sumber lain.
- Sumber seperti ScienceDirect melarang pengambilan otomatis tanpa izin; gunakan unduhan manual atau API resmi.
- GPU hanya tersedia lewat Google Colab (kuota tidak pasti).
- Kelas Konvensional < Berkembang < Maju berurutan (ordinal), tetapi model awal diperlakukan sebagai klasifikasi biasa.

## 11. Risiko
| Risiko | Dampak | Mitigasi |
|---|---|---|
| Kelas tidak seimbang (Maju sedikit) | F1 turun | macro-F1, class_weight, resampling hanya data latih |
| Foto berlabel sedikit | mAP rendah | Transfer learning, augmentasi, prioritaskan anotasi kelas langka |
| Data leakage (foto satu UMKM di train dan test) | Skor palsu tinggi | GroupKFold per UMKM |
| Bias seleksi (UMKM tanpa jejak daring tampak Konvensional) | Peta bias | Label dari survei lapangan; catat sebagai keterbatasan |
| Pelanggaran ToS/privasi | Masalah etika/hukum | API resmi, anonimisasi, blur wajah, tanpa data pribadi |
| Waktu anotasi | Jadwal mundur | Targetkan jumlah gambar per kelas sejak awal, bagi tugas anotasi |

## 12. Milestone **[USULAN]**
| Fase | Keluaran |
|---|---|
| M0 Persiapan | Repo, lingkungan Colab, skema data, definisi kelas disepakati |
| M1 Data | Raw Data lolos gerbang volume; Data Collection Schema terisi |
| M2 Processing | EDA, cleaning, anonimisasi, preprocessing → Clean/Gold |
| M3 Modeling | Detektor indikator, klaster, klasifikasi, evaluasi |
| M4 BDA | Agregasi, analisis spasial, gap, dashboard |
| M5 Aplikasi | Web App + Inference API, uji, laporan, diagram final |

## 13. Pertanyaan terbuka
1. Daftar kecamatan final dan sumber batas wilayah (shapefile/GeoJSON)?
2. Definisi kelas final dan siapa yang menyetujui?
3. Jumlah anggota dan pembagian fase/anotasi?
4. Apakah fitur RAG (F8) diperlukan atau cukup sebagai pengembangan lanjutan?
5. Ambang mAP dan metrik tambahan yang diminta dosen?
