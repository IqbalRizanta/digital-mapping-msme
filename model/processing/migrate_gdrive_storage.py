"""
Migrasi & Penataan Terstruktur Google Drive BDA-2026-Iqbal:
- Target Folder: https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing
- Local Mount  : G:\\My Drive\\BDA-2026-Iqbal

Menata seluruh aset secara informatif mengikuti alur tugas dosen:
  00_PANDUAN_ISI_DRIVE.txt
  01_Search_Journal/            (PDF jurnal rujukan + tabel ringkasan perbandingan)
  02_Repeat_Journal/            (alur_flowchart/, dataset_jurnal/, code_dan_notebook/, hasil_evaluasi/)
  03_Adjust_Journal/            (alur_flowchart/, code_dan_notebook/, hasil_evaluasi/)
  04_Dataset_Multimodal_Saya/   (01_raw/, 02_clean/ >= 200 MB, 03_staging_spark/, 04_feature_store/)
  05_Models_Tersimpan/          (ML .pkl, DL .pt/.pth/.h5, YOLO .pt, pipeline)
  06_Notebooks_Lengkap_Proyek/  (seluruh 5 notebook tereksekusi)
  07_Reports_Lengkap_Proyek/    (eda/, model_evaluation/, bda/)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.processing.pyspark_staging import _copy_if_needed, sync_scraping_to_gdrive_gmaps
from model.storage_config import (
    GDRIVE_ADJUST_CODE_DIR,
    GDRIVE_ADJUST_FLOW_DIR,
    GDRIVE_ADJUST_JOURNAL_DIR,
    GDRIVE_ADJUST_RESULTS_DIR,
    GDRIVE_CLEAN_DIR,
    GDRIVE_CLEAN_IMAGES_DIR,
    GDRIVE_DATASET_ROOT,
    GDRIVE_FEATURE_STORE_DIR,
    GDRIVE_FOLDER_ID,
    GDRIVE_MODELS_DIR,
    GDRIVE_NOTEBOOKS_DIR,
    GDRIVE_PROJECT_ROOT,
    GDRIVE_RAW_DIR,
    GDRIVE_REPEAT_CODE_DIR,
    GDRIVE_REPEAT_DATA_DIR,
    GDRIVE_REPEAT_FLOW_DIR,
    GDRIVE_REPEAT_JOURNAL_DIR,
    GDRIVE_REPEAT_RESULTS_DIR,
    GDRIVE_REPORTS_BDA_DIR,
    GDRIVE_REPORTS_DIR,
    GDRIVE_REPORTS_EDA_DIR,
    GDRIVE_REPORTS_EVAL_DIR,
    GDRIVE_SEARCH_JOURNAL_DIR,
    GDRIVE_STAGING_SPARK_DIR,
    ensure_gdrive_structure,
)

CLEAN_DIR = ROOT_DIR / "data" / "clean"
GOLD_DIR = ROOT_DIR / "data" / "gold"
SPARK_DIR = ROOT_DIR / "data" / "staging_spark"
JOURNAL_RAW_DIR = ROOT_DIR / "data" / "raw" / "journal_repeat"
JOURNAL_CODE_DIR = ROOT_DIR / "model" / "journal"
SAVED_DIR = ROOT_DIR / "model" / "saved"
NOTEBOOKS_DIR = ROOT_DIR / "model" / "notebooks"
REPORTS_DIR = ROOT_DIR / "peta-digital-umkm" / "reports"


def _copy_tree_if_needed(src_dir: Path, dst_dir: Path) -> int:
    copied = 0
    if not src_dir.exists():
        return copied
    for item in src_dir.rglob("*"):
        if item.is_file() and not item.name.endswith(".cache"):
            rel = item.relative_to(src_dir)
            _copy_if_needed(item, dst_dir / rel)
            copied += 1
    return copied


def _folder_stats(folder: Path) -> dict:
    if not folder.exists():
        return {"files": 0, "mb": 0.0}
    files = [f for f in folder.rglob("*") if f.is_file() and f.name.lower() != "desktop.ini"]
    total_bytes = sum(f.stat().st_size for f in files)
    return {"files": len(files), "mb": round(total_bytes / (1024 * 1024), 2)}


def _relocate_previous_flat_folders() -> None:
    """Memindahkan secara instan folder flat tahap sebelumnya ke hierarki 7 zona baru tanpa menggandakan kuota."""
    GDRIVE_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    moves = [
        (GDRIVE_PROJECT_ROOT / "01_raw", GDRIVE_RAW_DIR),
        (GDRIVE_PROJECT_ROOT / "02_clean", GDRIVE_CLEAN_DIR),
        (GDRIVE_PROJECT_ROOT / "03_staging_spark", GDRIVE_STAGING_SPARK_DIR),
        (GDRIVE_PROJECT_ROOT / "04_feature_store", GDRIVE_FEATURE_STORE_DIR),
        (GDRIVE_PROJECT_ROOT / "05_models", GDRIVE_MODELS_DIR),
        (GDRIVE_PROJECT_ROOT / "06_notebooks", GDRIVE_NOTEBOOKS_DIR),
        (GDRIVE_PROJECT_ROOT / "07_reports", GDRIVE_REPORTS_DIR),
    ]
    for old_p, new_p in moves:
        if old_p.exists() and not new_p.exists():
            print(f"-> [RELOCATE] {old_p.name} -> {new_p.relative_to(GDRIVE_PROJECT_ROOT)}", flush=True)
            shutil.move(str(old_p), str(new_p))

    old_readme = GDRIVE_PROJECT_ROOT / "README_STRUKTUR_STORAGE.txt"
    if old_readme.exists():
        old_readme.unlink(missing_ok=True)


def migrate_all() -> dict:
    if not GDRIVE_PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Google Drive target tidak ditemukan di: {GDRIVE_PROJECT_ROOT}. "
            "Pastikan Google Drive for Desktop aktif dan shortcut BDA-2026-Iqbal terpasang."
        )

    print(f"=== Memulai Penataan & Migrasi Lengkap ke: {GDRIVE_PROJECT_ROOT} ===", flush=True)
    _relocate_previous_flat_folders()
    ensure_gdrive_structure()

    # ------------------------------------------------------------------
    # 1. TAHAP 1: 01_Search_Journal/
    # ------------------------------------------------------------------
    print("[1/7] Sinkronisasi 01_Search_Journal/ ...", flush=True)
    for src in [
        JOURNAL_RAW_DIR / "bdcc_2024_ghatora_10.3390_bdcc8120199.pdf",
        REPORTS_DIR / "journal_repeat" / "ringkasan_jurnal_dan_perbandingan.csv",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_SEARCH_JOURNAL_DIR / src.name)

    # ------------------------------------------------------------------
    # 2. TAHAP 2: 02_Repeat_Journal/
    # ------------------------------------------------------------------
    print("[2/7] Sinkronisasi 02_Repeat_Journal/ ...", flush=True)
    for src in [
        REPORTS_DIR / "journal_repeat" / "Alur_Repeat_Journal_BDCC2024.png",
        REPORTS_DIR / "journal_repeat" / "Alur_Repeat_Journal_BDCC2024.drawio",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_REPEAT_FLOW_DIR / src.name)

    if (JOURNAL_RAW_DIR / "Dataset-SA.csv").exists():
        _copy_if_needed(JOURNAL_RAW_DIR / "Dataset-SA.csv", GDRIVE_REPEAT_DATA_DIR / "Dataset-SA.csv")

    for src in [
        JOURNAL_CODE_DIR / "repeat_journal_bdcc2024.py",
        NOTEBOOKS_DIR / "01_repeat_journal_bdcc2024_09040624091.ipynb",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_REPEAT_CODE_DIR / src.name)

    for fname in [
        "repeat_vs_paper_metrics.csv",
        "repeat_summary.json",
        "01_repeat_vs_paper_comparison.png",
        "02_confusion_matrix_review.png",
        "03_confusion_matrix_summary.png",
    ]:
        src = REPORTS_DIR / "journal_repeat" / fname
        if src.exists():
            _copy_if_needed(src, GDRIVE_REPEAT_RESULTS_DIR / src.name)

    # ------------------------------------------------------------------
    # 3. TAHAP 3: 03_Adjust_Journal/
    # ------------------------------------------------------------------
    print("[3/7] Sinkronisasi 03_Adjust_Journal/ ...", flush=True)
    for src in [
        REPORTS_DIR / "journal_adjust" / "Alur_Adjust_Journal_Multimodal_UMKM.png",
        REPORTS_DIR / "journal_adjust" / "Alur_Adjust_Journal_Multimodal_UMKM.drawio",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_ADJUST_FLOW_DIR / src.name)

    for src in [
        JOURNAL_CODE_DIR / "adjust_journal_multimodal.py",
        NOTEBOOKS_DIR / "02_adjust_journal_multimodal_umkm_09040624091.ipynb",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_ADJUST_CODE_DIR / src.name)

    for fname in [
        "adjust_model_benchmark.csv",
        "adjust_vs_repeat_comparison.csv",
        "adjust_summary.json",
        "01_repeat_vs_adjust_accuracy_f1.png",
        "02_adjust_multimodal_confusion_matrix.png",
        "03_adjust_spatial_bda_gap.png",
    ]:
        src = REPORTS_DIR / "journal_adjust" / fname
        if src.exists():
            _copy_if_needed(src, GDRIVE_ADJUST_RESULTS_DIR / src.name)

    # ------------------------------------------------------------------
    # 4. TAHAP 4: 04_Dataset_Multimodal_Saya/ (idempoten, melewati file yg sudah dipindah)
    # ------------------------------------------------------------------
    print("[4/7] Verifikasi & Sinkronisasi 04_Dataset_Multimodal_Saya/ ...", flush=True)
    df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
    df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
    df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")
    staged_file = SPARK_DIR / "umkm_spark_staged.parquet"
    pdf_staged = pd.read_parquet(staged_file)

    sync_info = sync_scraping_to_gdrive_gmaps(
        df_umkm=df_umkm,
        df_ulasan=df_ulasan,
        df_foto=df_foto,
        pdf_staged=pdf_staged,
        staged_file=staged_file,
    )

    for src in [
        GOLD_DIR / "feature_store.parquet",
        SAVED_DIR / "embeddings_multimodal.npy",
        SAVED_DIR / "embeddings_multimodal.npz",
        SAVED_DIR / "embeddings_multimodal.parquet",
    ]:
        if src.exists():
            _copy_if_needed(src, GDRIVE_FEATURE_STORE_DIR / src.name)

    # ------------------------------------------------------------------
    # 5. TAHAP 5: 05_Models_Tersimpan/
    # ------------------------------------------------------------------
    print("[5/7] Sinkronisasi 05_Models_Tersimpan/ ...", flush=True)
    model_exts = {".pkl", ".pt", ".pth", ".h5", ".joblib", ".onnx"}
    for mfile in SAVED_DIR.iterdir():
        if mfile.is_file() and mfile.suffix.lower() in model_exts:
            _copy_if_needed(mfile, GDRIVE_MODELS_DIR / mfile.name)

    # ------------------------------------------------------------------
    # 6. TAHAP 6: 06_Notebooks_Lengkap_Proyek/
    # ------------------------------------------------------------------
    print("[6/7] Sinkronisasi 06_Notebooks_Lengkap_Proyek/ ...", flush=True)
    for nb_name in [
        "01_repeat_journal_bdcc2024_09040624091.ipynb",
        "02_adjust_journal_multimodal_umkm_09040624091.ipynb",
        "eda_3_09040624091.ipynb",
        "model_3_09040624091.ipynb",
        "bda_3_09040624091.ipynb",
    ]:
        nb_src = NOTEBOOKS_DIR / nb_name
        if nb_src.exists():
            _copy_if_needed(nb_src, GDRIVE_NOTEBOOKS_DIR / nb_name)

    # ------------------------------------------------------------------
    # 7. TAHAP 7: 07_Reports_Lengkap_Proyek/
    # ------------------------------------------------------------------
    print("[7/7] Sinkronisasi 07_Reports_Lengkap_Proyek/ ...", flush=True)
    _copy_tree_if_needed(REPORTS_DIR / "eda", GDRIVE_REPORTS_EDA_DIR)
    _copy_tree_if_needed(REPORTS_DIR / "modeling", GDRIVE_REPORTS_EVAL_DIR)
    _copy_tree_if_needed(REPORTS_DIR / "yolo", GDRIVE_REPORTS_EVAL_DIR / "yolo_outputs")
    _copy_tree_if_needed(REPORTS_DIR / "bda", GDRIVE_REPORTS_BDA_DIR)

    # Panduan informatif di root Drive agar dosen mudah menelusuri setiap bukti
    guide_text = (
        "================================================================================\n"
        "PANDUAN STRUKTUR GOOGLE DRIVE — BDA-2026-Iqbal (NIM: 09040624091)\n"
        "Alur Tugas: Search Journal -> Repeat Journal -> Adjust Journal (Multimodal UMKM)\n"
        "URL Drive : https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing\n"
        "================================================================================\n\n"
        "01_Search_Journal/\n"
        "  - bdcc_2024_ghatora_10.3390_bdcc8120199.pdf  : Paper Open Access resmi Ghatora et al. (2024),\n"
        "                                                 Big Data and Cognitive Computing (MDPI), Vol. 8(12), Art. 199\n"
        "  - ringkasan_jurnal_dan_perbandingan.csv      : Tabel perbandingan Jurnal Asli vs Repeat vs Adjust\n\n"
        "02_Repeat_Journal/  [REPRODUKSI PERSIS JURNAL]\n"
        "  - alur_flowchart/    : Alur_Repeat_Journal_BDCC2024.png & .drawio (bukti alur persis jurnal)\n"
        "  - dataset_jurnal/    : Dataset-SA.csv (205.052 baris asli Kaggle — Flipkart Product Reviews)\n"
        "  - code_dan_notebook/ : repeat_journal_bdcc2024.py & 01_repeat_journal_bdcc2024_09040624091.ipynb (tereksekusi)\n"
        "  - hasil_evaluasi/    : repeat_vs_paper_metrics.csv, grafik perbandingan & confusion matrix\n"
        "                         (Review RF 0.690 vs Paper 0.680; Summary RF 0.766 vs Paper 0.767)\n\n"
        "03_Adjust_Journal/  [PENYESUAIAN DATASET MULTIMODAL + METODE/LIBRARY/FRAMEWORK BARU]\n"
        "  - alur_flowchart/    : Alur_Adjust_Journal_Multimodal_UMKM.png & .drawio (bukti alur sesudah adjust)\n"
        "  - code_dan_notebook/ : adjust_journal_multimodal.py & 02_adjust_journal_multimodal_umkm_09040624091.ipynb\n"
        "  - hasil_evaluasi/    : adjust_vs_repeat_comparison.csv, adjust_model_benchmark.csv, & 3 grafik evaluasi\n"
        "                         (RF 0.9740, Gradient Boosting 0.9792, PyTorch MLP 1.000, K-Means Sil 0.4737)\n\n"
        "04_Dataset_Multimodal_Saya/  [DATASET TAMBAHAN: GAMBAR, TEKS/NUMERIK, DOKUMEN]\n"
        "  - 01_raw/            : 2.272 foto mentah (635.75 MB), JSON Google Maps, 3 PDF izin, metadata survei\n"
        "  - 02_clean/          : 549 foto bersih (279.99 MB >= 200 MB) + 385 UMKM + 1.164 ulasan asli + 3 PDF\n"
        "  - 03_staging_spark/  : umkm_spark_staged.{parquet,csv} (PySpark SQL multimodal staging)\n"
        "  - 04_feature_store/  : feature_store.parquet & embeddings_multimodal.{npy,npz,parquet}\n\n"
        "05_Models_Tersimpan/         : Seluruh artefak ML (.pkl), DL (.pt/.pth/.h5), YOLOv8 (.pt), & pipeline\n"
        "06_Notebooks_Lengkap_Proyek/ : 5 notebook tereksekusi (Repeat, Adjust, EDA, Model-Evaluation, BDA)\n"
        "07_Reports_Lengkap_Proyek/   : Seluruh grafik & laporan EDA, Model Evaluation (YOLO/SHAP), & Spatial BDA\n"
    )
    (GDRIVE_PROJECT_ROOT / "00_PANDUAN_ISI_DRIVE.txt").write_text(guide_text, encoding="utf-8")

    audit = {
        "gdrive_folder_id": GDRIVE_FOLDER_ID,
        "gdrive_root": str(GDRIVE_PROJECT_ROOT),
        "zones": {
            "01_Search_Journal": _folder_stats(GDRIVE_SEARCH_JOURNAL_DIR),
            "02_Repeat_Journal": _folder_stats(GDRIVE_REPEAT_JOURNAL_DIR),
            "03_Adjust_Journal": _folder_stats(GDRIVE_ADJUST_JOURNAL_DIR),
            "04_Dataset_Multimodal_Saya": _folder_stats(GDRIVE_DATASET_ROOT),
            "04_clean_images_only": _folder_stats(GDRIVE_CLEAN_IMAGES_DIR),
            "05_Models_Tersimpan": _folder_stats(GDRIVE_MODELS_DIR),
            "06_Notebooks_Lengkap_Proyek": _folder_stats(GDRIVE_NOTEBOOKS_DIR),
            "07_Reports_Lengkap_Proyek": _folder_stats(GDRIVE_REPORTS_DIR),
        },
        "sync_info": sync_info,
    }
    print("\n=== Ringkasan Audit Struktur Baru BDA-2026-Iqbal ===", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    migrate_all()
