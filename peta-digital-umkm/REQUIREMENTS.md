# REQUIREMENTS — Peta Digital UMKM Kuliner Surabaya Timur

> **Status:** Final Terverifikasi · 28 Oktober 2026  
> **Rujukan Resmi Arsitektur & Flowchart:**
> 1. `diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json` (Halaman: `bd-ai-archi` dan `flowchart-bd-ai`)
> 2. `diagram-architecture/flowchart-mandiri.json` (Halaman: `flowchart-mandiri-gmaps-realworld`)

---

## 1. Persyaratan Fungsional (Functional Requirements)

### 1.1 Ingestion & Penyimpanan Cloud
| ID | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-01** | Sistem **WAJIB** mengambil data UMKM Google Maps di Surabaya Timur melalui Places API resmi dengan API key terproteksi pada file `.env`. | WAJIB | Terpenuhi (190 UMKM) |
| **FR-02** | Sistem **WAJIB** menyediakan jalur survei lapangan nyata (*Real World*) untuk sentra kuliner binaan (SWK Convention Hall & Deles) dengan panduan foto stan, kasir, etalase, dan instrumen pembayaran. | WAJIB | Terpenuhi (44 stan survei) |
| **FR-03** | Seluruh data mentah, data bersih, model weights, dan hasil evaluasi **WAJIB** tersinkronisasi otomatis ke Google Drive (`G:\My Drive\` / Colab). | WAJIB | Terpenuhi (`G:\My Drive\`) |
| **FR-04** | Memenuhi gerbang volume data minimum: gambar $\ge 200\text{ MB}$ dan data teks/baris $\ge 2.000$ baris data multimodal terpadu. | WAJIB | Terpenuhi (Pool 672 citra) |

### 1.2 Kurasi, Sanitasi & Preprocessing Citra (CLIP Dual-Gate)
| ID | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-05** | Sistem **WAJIB** menerapkan filter zero-shot dual-gate OpenAI CLIP (`ViT-B/32`) untuk menyaring citra non-target (piring makanan, gelas minuman, buku menu, dan ilustrasi digital hasil generate AI). | WAJIB | Terpenuhi (415 foto disaring) |
| **FR-06** | Dataset citra pada zona Clean (`data/clean/photos/`) **HANYA BOLEH** memuat citra stan fisik, etalase, kasir, dan indikator target klasifikasi (`qris`, `edc`, `pos`, `merchant_online`). | WAJIB | Terpenuhi (274 foto bersih) |
| **FR-07** | Sistem **WAJIB** menerapkan filter usaha skala besar (*Big Business Rule*): usaha waralaba/rantai besar tanpa foto kasir/pembayaran wajib dieliminasi dari dataset. | WAJIB | Terpenuhi (43 foto rantai besar dipangkas) |
| **FR-08** | Sistem **WAJIB** menganonimkan 100% wajah pengunjung/kasir yang tertangkap kamera menggunakan OpenCV Gaussian Face Blurring sebelum masuk ke zona Clean. | WAJIB | Terpenuhi (315 wajah disanitasi) |

### 1.3 Vision Multi-Task Ultralytics YOLOv8
| ID | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-09** | **Deteksi Objek (Detect):** Model YOLOv8 **WAJIB** mendeteksi 5 kelas target (`qris`, `edc`, `pos`, `merchant_online`, `stan`) dan menghasilkan citra dengan bounding box berwarna + label kelas + confidence score. | WAJIB | Terpenuhi (274 foto ber-bbox, mAP@50: 40.74%) |
| **FR-10** | **Segmentasi Instance (Seg):** Model YOLOv8-seg **WAJIB** menghasilkan polygon mask warna transparan untuk mengidentifikasi geometri fisik stan/etalase. | WAJIB | Terpenuhi (1.181 objek poligon) |
| **FR-11** | **Estimasi Pose (Pose):** Model YOLOv8-pose **WAJIB** memetakan kerangka tubuh manusia (skeleton keypoints) pedagang/kasir di stan kuliner. | WAJIB | Terpenuhi (371 pose manusia) |
| **FR-12** | **Klasifikasi Kematangan (Cls):** Model YOLOv8-cls **WAJIB** mengklasifikasikan foto stan ke dalam tingkat kematangan (*Konvensional*, *Berkembang*, *Maju*). | WAJIB | Terpenuhi (Top-1 Acc: 63.64%, Top-5: 100%) |
| **FR-13** | Sistem **WAJIB** mempublikasikan file log metrik terstruktur ([`metrics_detect.json`](file:///C:/big-data-analytic/reports/yolo/detect/metrics_detect.json), [`results.csv`](file:///C:/big-data-analytic/reports/yolo/detect/train/results.csv), [`summary_segment.json`](file:///C:/big-data-analytic/reports/yolo/segment/summary_segment.json), [`summary_pose.json`](file:///C:/big-data-analytic/reports/yolo/pose/summary_pose.json), [`metrics_classify.json`](file:///C:/big-data-analytic/reports/yolo/classify/metrics_classify.json)). | WAJIB | Terpenuhi di folder reports & GDrive |

### 1.4 Evaluasi Statistik & Validasi Silang (Cohen's Kappa & GroupKFold)
| ID | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-14** | Sistem **WAJIB** menguji kesepakatan silang antara indikator visual terdeteksi di Google Maps dan survei lapangan di SWK menggunakan koefisien **Cohen's Kappa ($\kappa$)** dengan target kesepakatan memadai ($\kappa \ge 0.70$). | WAJIB | Terpenuhi ($\kappa_{\text{QRIS}} = 0.751$, $\kappa_{\text{POS}} = 0.727$) |
| **FR-15** | Pembagian data modeling **WAJIB** dikelompokkan berdasarkan unit usaha (`umkm_id`) menggunakan **GroupKFold ($K=5$)** untuk mencegah kebocoran data (*data leakage*). | WAJIB | Terpenuhi di `model_3_09040624091.ipynb` |
| **FR-16** | Sistem **WAJIB** melatih model klastering K-Means ($k=3$) dan mengevaluasi Silhouette Score ($\ge 0.40$) serta Davies-Bouldin Index. | WAJIB | Terpenuhi (Silhouette: 0.4964, DBI: 0.7916) |
| **FR-17** | Sistem **WAJIB** melatih model klasifikasi kematangan digital berlabel ground truth lapangan (Random Forest). | WAJIB | Terpenuhi (Evaluasi GroupKFold K=5) |

### 1.5 Big Data Analytics (BDA) & Spatial Gap Analysis
| ID | Persyaratan | Prioritas | Status Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-18** | Sistem **WAJIB** mengagregasi penetrasi indikator digital (QRIS, POS, EDC, Ojol) di 6 kecamatan Surabaya Timur. | WAJIB | Terpenuhi di `bda_3_09040624091.ipynb` |
| **FR-19** | Sistem **WAJIB** menghitung **Digital Gap Index** untuk menetapkan prioritas intervensi dinas (Prioritas Tinggi, Sedang, Mandiri). | WAJIB | Terpenuhi |
| **FR-20** | Sistem **WAJIB** menyusun rekomendasi kebijakan konkret untuk akselerasi ekosistem digital UMKM kuliner Surabaya Timur. | WAJIB | Terpenuhi |

---

## 2. Persyaratan Non-Fungsional (Non-Functional Requirements)

| ID | Kategori | Persyaratan |
| :--- | :--- | :--- |
| **NFR-01** | **Bebas Identitas Personal** | Seluruh file notebook (`eda_3_*.ipynb`, `model_3_*.ipynb`, `bda_3_*.ipynb`) **TIDAK BOLEH** memuat identitas personal (tanpa nama mahasiswa, NIM, maupun nama dosen pembimbing) di dalam sel markdown atau komentar kode. |
| **NFR-02** | **Reproduksibilitas Penuh** | Seluruh pipeline dapat dijalankan ulang dari skrip generator [`create_student_notebooks.py`](file:///C:/big-data-analytic/scripts/create_student_notebooks.py) dan dieksekusi secara otomatis oleh [`run_notebook.py`](file:///C:/big-data-analytic/scripts/run_notebook.py). |
| **NFR-03** | **Kepatuhan Privasi (GDPR/UU PDP)** | Seluruh identitas pengulas, nomor telepon, dan wajah pengunjung wajib disanitasi sebelum data disimpan ke format Parquet. |
| **NFR-04** | **Ketahanan Penyimpanan Virtual** | Sinkronisasi data ke Google Drive for Desktop (`G:\My Drive`) menggunakan transfer direktori unbundled untuk mencegah locking file virtual drive. |
| **NFR-05** | **Format Data Kolumnar** | Dataset bersih wajib menggunakan format Apache Parquet (`.parquet`) dengan kompresi Snappy untuk efisiensi I/O analitik. |
