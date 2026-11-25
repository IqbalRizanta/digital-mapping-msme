"""
Konfigurasi terpusat penyimpanan Google Drive terstruktur untuk proyek BDA UMKM Surabaya Timur.

Target Root Google Drive:
- https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing
- Local Mount: G:\\My Drive\\BDA-2026-Iqbal
  (Fallback: G:\\.shortcut-targets-by-id\\12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c\\BDA-2026-Iqbal)

Struktur 7 Zona Utama (Sesuai Bukti Penugasan Presentasi 25 November 2026):
  01_Diagram_DrawIO/            : File draw.io 4 Halaman (Page 1-2 Jurnal, Page 3-4 Modifikasi) + 4 Preview PNG
  02_Journal_Rujukan/           : PDF Jurnal Utama, Jurnal Gambar Indonesia (JuTISI 2025), & Ringkasan Bukti Dataset
  Repeat/                       : Notebook Repeat (.ipynb), Script, Dataset Asli 48 SWK, & Hasil Repeat (Tabel 3 & 4)
  Modifikasi/                   : Notebook Modifikasi (.ipynb), EDA/Model/BDA, Script, Dataset Dokumen (MIDD & CORD-v2), & Hasil
  04_Dataset_Multimodal_Saya/   : 01_raw/, 02_clean/ (>= 200 MB), 03_staging_spark/, 04_feature_store/
  05_Models_Tersimpan/          : Model ML (.pkl), DL (.pt/.pth/.h5), YOLOv8 (.pt), & Pipeline
  07_Reports_Lengkap_Proyek/    : Seluruh Grafik & Laporan Proyek (eda/, model_evaluation/, bda/)
"""

from pathlib import Path

GDRIVE_FOLDER_ID = "12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c"
GDRIVE_FOLDER_NAME = "BDA-2026-Iqbal"

_PRIMARY_ROOT = Path(r"G:\My Drive") / GDRIVE_FOLDER_NAME
_SHORTCUT_ROOT = Path(r"G:\.shortcut-targets-by-id") / GDRIVE_FOLDER_ID / GDRIVE_FOLDER_NAME


def get_gdrive_project_root() -> Path:
    """Mengembalikan path mount Google Drive aktif untuk folder proyek BDA-2026-Iqbal."""
    if _PRIMARY_ROOT.exists():
        return _PRIMARY_ROOT
    if _SHORTCUT_ROOT.exists():
        return _SHORTCUT_ROOT
    return _PRIMARY_ROOT


GDRIVE_PROJECT_ROOT = get_gdrive_project_root()

# Zona 1-4: Diagram Draw.io, Jurnal Rujukan, Folder Coding Repeat/ & Modifikasi/
GDRIVE_DIAGRAM_DIR = GDRIVE_PROJECT_ROOT / "01_Diagram_DrawIO"
GDRIVE_JOURNAL_DIR = GDRIVE_PROJECT_ROOT / "02_Journal_Rujukan"

GDRIVE_REPEAT_DIR = GDRIVE_PROJECT_ROOT / "Repeat"
GDRIVE_REPEAT_DATA_DIR = GDRIVE_REPEAT_DIR / "dataset_jurnal_utama"
GDRIVE_REPEAT_RESULTS_DIR = GDRIVE_REPEAT_DIR / "hasil_repeat"

GDRIVE_MODIFIKASI_DIR = GDRIVE_PROJECT_ROOT / "Modifikasi"
GDRIVE_MODIFIKASI_IMG_DIR = GDRIVE_MODIFIKASI_DIR / "dataset_gambar_indonesia_serupa"
GDRIVE_MODIFIKASI_DOCS_DIR = GDRIVE_MODIFIKASI_DIR / "dataset_dokumen_tambahan"
GDRIVE_MODIFIKASI_RESULTS_DIR = GDRIVE_MODIFIKASI_DIR / "hasil_modifikasi"

# Zona 5-7: Dataset Multimodal Proyek, Model Registry, & Reports
GDRIVE_DATASET_ROOT = GDRIVE_PROJECT_ROOT / "04_Dataset_Multimodal_Saya"
GDRIVE_RAW_DIR = GDRIVE_DATASET_ROOT / "01_raw"
GDRIVE_RAW_IMAGES_DIR = GDRIVE_RAW_DIR / "images"
GDRIVE_RAW_JSON_DIR = GDRIVE_RAW_DIR / "json_google_maps"
GDRIVE_RAW_DOCS_DIR = GDRIVE_RAW_DIR / "documents"
GDRIVE_RAW_META_DIR = GDRIVE_RAW_DIR / "metadata"

GDRIVE_CLEAN_DIR = GDRIVE_DATASET_ROOT / "02_clean"
GDRIVE_CLEAN_IMAGES_DIR = GDRIVE_CLEAN_DIR / "images"
GDRIVE_CLEAN_TABULAR_DIR = GDRIVE_CLEAN_DIR / "tabular_text"

GDRIVE_STAGING_SPARK_DIR = GDRIVE_DATASET_ROOT / "03_staging_spark"
GDRIVE_FEATURE_STORE_DIR = GDRIVE_DATASET_ROOT / "04_feature_store"

GDRIVE_MODELS_DIR = GDRIVE_PROJECT_ROOT / "05_Models_Tersimpan"

GDRIVE_REPORTS_DIR = GDRIVE_PROJECT_ROOT / "07_Reports_Lengkap_Proyek"
GDRIVE_REPORTS_EDA_DIR = GDRIVE_REPORTS_DIR / "eda"
GDRIVE_REPORTS_EVAL_DIR = GDRIVE_REPORTS_DIR / "model_evaluation"
GDRIVE_REPORTS_BDA_DIR = GDRIVE_REPORTS_DIR / "bda"

ALL_STRUCTURED_DIRS = [
    GDRIVE_DIAGRAM_DIR,
    GDRIVE_JOURNAL_DIR,
    GDRIVE_REPEAT_DATA_DIR,
    GDRIVE_REPEAT_RESULTS_DIR,
    GDRIVE_MODIFIKASI_IMG_DIR / "sample_27_images",
    GDRIVE_MODIFIKASI_DOCS_DIR / "MIDD_Invoice_IOB",
    GDRIVE_MODIFIKASI_DOCS_DIR / "CORD_v2_Receipts_Indonesia",
    GDRIVE_MODIFIKASI_RESULTS_DIR,
    GDRIVE_RAW_IMAGES_DIR / "Google_Maps",
    GDRIVE_RAW_IMAGES_DIR / "Survei_Lapangan_SWK",
    GDRIVE_RAW_IMAGES_DIR / "Ditolak_Non_Target",
    GDRIVE_RAW_JSON_DIR,
    GDRIVE_RAW_DOCS_DIR,
    GDRIVE_RAW_META_DIR,
    GDRIVE_CLEAN_IMAGES_DIR / "Google_Maps",
    GDRIVE_CLEAN_IMAGES_DIR / "Survei_Lapangan_SWK",
    GDRIVE_CLEAN_TABULAR_DIR,
    GDRIVE_STAGING_SPARK_DIR,
    GDRIVE_FEATURE_STORE_DIR,
    GDRIVE_MODELS_DIR,
    GDRIVE_REPORTS_EDA_DIR,
    GDRIVE_REPORTS_EVAL_DIR,
    GDRIVE_REPORTS_BDA_DIR,
]


def ensure_gdrive_structure() -> bool:
    """Membuat seluruh hierarki folder terstruktur pada BDA-2026-Iqbal jika Google Drive terpasang."""
    root = get_gdrive_project_root()
    if not root.exists():
        return False
    for d in ALL_STRUCTURED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    return True
