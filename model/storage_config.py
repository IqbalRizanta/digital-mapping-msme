"""
Konfigurasi terpusat penyimpanan Google Drive terstruktur untuk proyek BDA UMKM Surabaya Timur.

Target Root Google Drive:
- https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing
- Local Mount: G:\\My Drive\\BDA-2026-Iqbal
  (Fallback: G:\\.shortcut-targets-by-id\\12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c\\BDA-2026-Iqbal)

Struktur 7 Zona Utama (Search Journal -> Repeat Journal -> Adjust Journal -> Dataset & Output Proyek):
  01_Search_Journal/            : PDF Jurnal Rujukan Resmi & Ringkasan Perbandingan
  02_Repeat_Journal/            : Alur Repeat (PNG+Draw.io), Dataset Asli Jurnal, Code+Notebook, & Hasil Evaluasi
  03_Adjust_Journal/            : Alur Adjust (PNG+Draw.io), Code+Notebook Multimodal, & Hasil Perbandingan
  04_Dataset_Multimodal_Saya/   : 01_raw/, 02_clean/ (>= 200 MB), 03_staging_spark/, 04_feature_store/
  05_Models_Tersimpan/          : Model ML (.pkl), DL (.pt/.pth/.h5), YOLOv8 (.pt), & Pipeline
  06_Notebooks_Lengkap_Proyek/  : Seluruh Notebook Tereksekusi (Repeat, Adjust, EDA, Model, BDA)
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

# Tahap 1–3: Search Journal -> Repeat Journal -> Adjust Journal
GDRIVE_SEARCH_JOURNAL_DIR = GDRIVE_PROJECT_ROOT / "01_Search_Journal"

GDRIVE_REPEAT_JOURNAL_DIR = GDRIVE_PROJECT_ROOT / "02_Repeat_Journal"
GDRIVE_REPEAT_FLOW_DIR = GDRIVE_REPEAT_JOURNAL_DIR / "alur_flowchart"
GDRIVE_REPEAT_DATA_DIR = GDRIVE_REPEAT_JOURNAL_DIR / "dataset_jurnal"
GDRIVE_REPEAT_CODE_DIR = GDRIVE_REPEAT_JOURNAL_DIR / "code_dan_notebook"
GDRIVE_REPEAT_RESULTS_DIR = GDRIVE_REPEAT_JOURNAL_DIR / "hasil_evaluasi"

GDRIVE_ADJUST_JOURNAL_DIR = GDRIVE_PROJECT_ROOT / "03_Adjust_Journal"
GDRIVE_ADJUST_FLOW_DIR = GDRIVE_ADJUST_JOURNAL_DIR / "alur_flowchart"
GDRIVE_ADJUST_CODE_DIR = GDRIVE_ADJUST_JOURNAL_DIR / "code_dan_notebook"
GDRIVE_ADJUST_RESULTS_DIR = GDRIVE_ADJUST_JOURNAL_DIR / "hasil_evaluasi"

# Tahap 4–7: Dataset Multimodal Proyek, Model Registry, Notebooks, & Reports
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
GDRIVE_NOTEBOOKS_DIR = GDRIVE_PROJECT_ROOT / "06_Notebooks_Lengkap_Proyek"

GDRIVE_REPORTS_DIR = GDRIVE_PROJECT_ROOT / "07_Reports_Lengkap_Proyek"
GDRIVE_REPORTS_EDA_DIR = GDRIVE_REPORTS_DIR / "eda"
GDRIVE_REPORTS_EVAL_DIR = GDRIVE_REPORTS_DIR / "model_evaluation"
GDRIVE_REPORTS_BDA_DIR = GDRIVE_REPORTS_DIR / "bda"

ALL_STRUCTURED_DIRS = [
    GDRIVE_SEARCH_JOURNAL_DIR,
    GDRIVE_REPEAT_FLOW_DIR,
    GDRIVE_REPEAT_DATA_DIR,
    GDRIVE_REPEAT_CODE_DIR,
    GDRIVE_REPEAT_RESULTS_DIR,
    GDRIVE_ADJUST_FLOW_DIR,
    GDRIVE_ADJUST_CODE_DIR,
    GDRIVE_ADJUST_RESULTS_DIR,
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
    GDRIVE_NOTEBOOKS_DIR,
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
