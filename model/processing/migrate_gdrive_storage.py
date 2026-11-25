"""
Migrasi & Penataan Terstruktur Google Drive BDA-2026-Iqbal:
- Target Folder: https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing
- Local Mount  : G:\\My Drive\\BDA-2026-Iqbal

Menata seluruh aset secara informatif sesuai bukti penugasan:
  00_PANDUAN_ISI_DRIVE.txt
  01_Diagram_DrawIO/            (BDA_Journal_Repeat_dan_Modifikasi_4Pages.drawio + 4 PNG preview)
  02_Journal_Rujukan/           (Journal_Utama.pdf, Journal_Refrensi.pdf, JuTISI 2025, ringkasan CSV)
  Repeat/                       (01_repeat_jurnal_utama_swk_09040624091.ipynb, script, 48 SWK CSV, hasil_repeat/)
  Modifikasi/                   (02_modifikasi_multimodal_umkm + EDA/Model/BDA .ipynb, script, MIDD & CORD-v2, hasil_modifikasi/)
  04_Dataset_Multimodal_Saya/   (01_raw/, 02_clean/ >= 200 MB, 03_staging_spark/, 04_feature_store/)
  05_Models_Tersimpan/          (ML .pkl, DL .pt/.pth/.h5, YOLOv8 .pt, pipeline)
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
    GDRIVE_CLEAN_IMAGES_DIR,
    GDRIVE_DATASET_ROOT,
    GDRIVE_DIAGRAM_DIR,
    GDRIVE_FEATURE_STORE_DIR,
    GDRIVE_FOLDER_ID,
    GDRIVE_JOURNAL_DIR,
    GDRIVE_MODELS_DIR,
    GDRIVE_MODIFIKASI_DIR,
    GDRIVE_MODIFIKASI_DOCS_DIR,
    GDRIVE_MODIFIKASI_IMG_DIR,
    GDRIVE_MODIFIKASI_RESULTS_DIR,
    GDRIVE_PROJECT_ROOT,
    GDRIVE_REPEAT_DATA_DIR,
    GDRIVE_REPEAT_DIR,
    GDRIVE_REPEAT_RESULTS_DIR,
    GDRIVE_REPORTS_BDA_DIR,
    GDRIVE_REPORTS_DIR,
    GDRIVE_REPORTS_EDA_DIR,
    GDRIVE_REPORTS_EVAL_DIR,
    ensure_gdrive_structure,
)

CLEAN_DIR = ROOT_DIR / "data" / "clean"
GOLD_DIR = ROOT_DIR / "data" / "gold"
SPARK_DIR = ROOT_DIR / "data" / "staging_spark"
RAW_DIR = ROOT_DIR / "data" / "raw"
JOURNAL_CODE_DIR = ROOT_DIR / "model" / "journal"
SAVED_DIR = ROOT_DIR / "model" / "saved"
NOTEBOOKS_DIR = ROOT_DIR / "model" / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"
REPORTS_DIR = DOCS_DIR / "reports"


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


def _cleanup_legacy_folders() -> None:
    """Menghapus folder jurnal lama (BDCC 2024 / 06_Notebooks lama) agar struktur Drive bersih."""
    legacy_dirs = [
        "01_Search_Journal",
        "02_Repeat_Journal",
        "03_Adjust_Journal",
        "06_Notebooks_Lengkap_Proyek",
    ]
    for name in legacy_dirs:
        p = GDRIVE_PROJECT_ROOT / name
        if p.exists():
            print(f"-> [CLEANUP] Menghapus folder lama: {name}/", flush=True)
            shutil.rmtree(p, ignore_errors=True)

    # Hapus juga grafik lama Flipkart di 07_Reports_Lengkap_Proyek/eda jika ada
    old_flipkart_png = GDRIVE_REPORTS_EDA_DIR / "00_eda_jurnal_flipkart_distribusi.png"
    if old_flipkart_png.exists():
        old_flipkart_png.unlink(missing_ok=True)


def migrate_all() -> dict:
    if not GDRIVE_PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Google Drive target tidak ditemukan di: {GDRIVE_PROJECT_ROOT}. "
            "Pastikan Google Drive for Desktop aktif dan folder BDA-2026-Iqbal terpasang."
        )

    print(f"=== Memulai Penataan & Migrasi Lengkap ke: {GDRIVE_PROJECT_ROOT} ===", flush=True)
    _cleanup_legacy_folders()
    ensure_gdrive_structure()

    # ------------------------------------------------------------------
    # 1. 01_Diagram_DrawIO/ (4-Page XML + 4 PNG Preview)
    # ------------------------------------------------------------------
    print("[1/7] Sinkronisasi 01_Diagram_DrawIO/ ...", flush=True)
    for fname in [
        "BDA_Journal_Repeat_dan_Modifikasi_4Pages.drawio",
        "Page1_Architecture_Jurnal.png",
        "Page2_Flowchart_Jurnal.png",
        "Page3_Architecture_Modifikasi.png",
        "Page4_Flowchart_Modifikasi.png",
    ]:
        src = DOCS_DIR / "diagrams" / fname
        if src.exists():
            _copy_if_needed(src, GDRIVE_DIAGRAM_DIR / fname)

    # ------------------------------------------------------------------
    # 2. 02_Journal_Rujukan/
    # ------------------------------------------------------------------
    print("[2/7] Sinkronisasi 02_Journal_Rujukan/ ...", flush=True)
    for fname in [
        "Journal_Utama.pdf",
        "Journal_Refrensi.pdf",
        "Journal_Gambar_JuTISI2025_Indonesian_Food.pdf",
        "ringkasan_jurnal_dan_bukti_dataset.csv",
    ]:
        src = DOCS_DIR / "Journal" / fname
        if src.exists():
            _copy_if_needed(src, GDRIVE_JOURNAL_DIR / fname)

    # ------------------------------------------------------------------
    # 3. Repeat/ (Notebook + Script + Dataset Asli 48 SWK + Hasil Repeat)
    # ------------------------------------------------------------------
    print("[3/7] Sinkronisasi Repeat/ ...", flush=True)
    for old_png in ["01_eda_jurnal_utama_swk.png", "02_repeat_ahc_dendrogram_dan_silhouette.png", "03_repeat_vs_jurnal_utama_metrics.png"]:
        (GDRIVE_REPEAT_RESULTS_DIR / old_png).unlink(missing_ok=True)
    for src, dst in [
        (NOTEBOOKS_DIR / "Repeat" / "01_repeat_jurnal_utama_swk_09040624091.ipynb", GDRIVE_REPEAT_DIR / "01_repeat_jurnal_utama_swk_09040624091.ipynb"),
        (JOURNAL_CODE_DIR / "repeat_journal_utama_swk.py", GDRIVE_REPEAT_DIR / "repeat_journal_utama_swk.py"),
        (RAW_DIR / "journal_utama_swk" / "kode11500kode25245kode3135.csv", GDRIVE_REPEAT_DATA_DIR / "kode11500kode25245kode3135.csv"),
        (RAW_DIR / "journal_utama_swk" / "SUMBER_ASLI_WAYBACK_MACHINE.txt", GDRIVE_REPEAT_DATA_DIR / "SUMBER_ASLI_WAYBACK_MACHINE.txt"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
    _copy_tree_if_needed(REPORTS_DIR / "journal_repeat", GDRIVE_REPEAT_RESULTS_DIR)

    # ------------------------------------------------------------------
    # 4. Modifikasi/ (Notebooks + Script + MIDD & CORD-v2 + Hasil Modifikasi)
    # ------------------------------------------------------------------
    print("[4/7] Sinkronisasi Modifikasi/ ...", flush=True)
    for src, dst in [
        (NOTEBOOKS_DIR / "Modifikasi" / "02_modifikasi_multimodal_umkm_09040624091.ipynb", GDRIVE_MODIFIKASI_DIR / "02_modifikasi_multimodal_umkm_09040624091.ipynb"),
        (NOTEBOOKS_DIR / "eda_3_09040624091.ipynb", GDRIVE_MODIFIKASI_DIR / "eda_3_09040624091.ipynb"),
        (NOTEBOOKS_DIR / "model_3_09040624091.ipynb", GDRIVE_MODIFIKASI_DIR / "model_3_09040624091.ipynb"),
        (NOTEBOOKS_DIR / "bda_3_09040624091.ipynb", GDRIVE_MODIFIKASI_DIR / "bda_3_09040624091.ipynb"),
        (JOURNAL_CODE_DIR / "adjust_journal_multimodal.py", GDRIVE_MODIFIKASI_DIR / "adjust_journal_multimodal.py"),
    ]:
        if src.exists():
            _copy_if_needed(src, dst)

    # Dataset Gambar Masakan Indonesia Serupa (fahmindra/padang-cuisine — metadata + fitur + 27 sampel foto)
    padang_src = RAW_DIR / "journal_gambar_padang_id"
    if padang_src.exists():
        for fn in ["padang_cuisine_metadata.csv", "padang_cuisine_features.parquet"]:
            if (padang_src / fn).exists():
                _copy_if_needed(padang_src / fn, GDRIVE_MODIFIKASI_IMG_DIR / fn)
        (GDRIVE_MODIFIKASI_IMG_DIR / "SUMBER_DATASET_PUBLIK.txt").write_text(
            "Dataset Gambar Masakan Indonesia Publik Serupa (Pengganti Mendeley 10.17632/vtjd68bmwt.1 HTTP 403 pada JuTISI 2025):\n"
            "- HuggingFace (ODbL): https://huggingface.co/datasets/fahmindra/padang-cuisine\n"
            "- Mirror Kaggle     : https://www.kaggle.com/datasets/faldoae/padangfood\n"
            "- Total Citra Lokal : 993 foto (115.59 MB), 9 kelas masakan Indonesia seimbang (train/test resmi)\n"
            "- Folder ini memuat metadata lengkap 993 foto, fitur ekstraksi visual (parquet), dan 27 foto sampel (3 per kelas).\n",
            encoding="utf-8",
        )
        train_dir = padang_src / "train"
        if train_dir.exists():
            for cls_dir in sorted(train_dir.iterdir()):
                if cls_dir.is_dir():
                    for img in sorted(cls_dir.glob("*"))[:3]:
                        if img.is_file():
                            _copy_if_needed(img, GDRIVE_MODIFIKASI_IMG_DIR / "sample_27_images" / cls_dir.name / img.name)

    _copy_tree_if_needed(RAW_DIR / "journal_dokumen_midd" / "IOB", GDRIVE_MODIFIKASI_DOCS_DIR / "MIDD_Invoice_IOB")
    _copy_tree_if_needed(RAW_DIR / "journal_dokumen_cord", GDRIVE_MODIFIKASI_DOCS_DIR / "CORD_v2_Receipts_Indonesia")
    _copy_tree_if_needed(REPORTS_DIR / "journal_adjust", GDRIVE_MODIFIKASI_RESULTS_DIR)

    # ------------------------------------------------------------------
    # 5. 04_Dataset_Multimodal_Saya/ (cepat & idempoten — lewati rescan 2.800+ foto jika sudah terverifikasi)
    # ------------------------------------------------------------------
    print("[5/7] Verifikasi & Sinkronisasi 04_Dataset_Multimodal_Saya/ ...", flush=True)
    clean_img_count = sum(
        1 for f in GDRIVE_CLEAN_IMAGES_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ) if GDRIVE_CLEAN_IMAGES_DIR.exists() else 0

    if clean_img_count >= 540:
        print(f"-> [SKIP IMAGE RESCAN] 02_clean/images sudah terisi {clean_img_count} foto di Drive; sinkronisasi tabular & feature store saja.", flush=True)
        for cname in ["umkm.parquet", "ulasan.parquet", "foto.parquet", "dokumen.parquet"]:
            if (CLEAN_DIR / cname).exists():
                _copy_if_needed(CLEAN_DIR / cname, GDRIVE_DATASET_ROOT / "02_clean" / "tabular_text" / cname)
        staged_file = SPARK_DIR / "umkm_spark_staged.parquet"
        if staged_file.exists():
            _copy_if_needed(staged_file, GDRIVE_DATASET_ROOT / "03_staging_spark" / staged_file.name)
        sync_info = {"skipped_image_rescan": True, "clean_images_on_drive": clean_img_count}
    else:
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
    # 6. 05_Models_Tersimpan/
    # ------------------------------------------------------------------
    print("[6/7] Sinkronisasi 05_Models_Tersimpan/ ...", flush=True)
    model_exts = {".pkl", ".pt", ".pth", ".h5", ".joblib", ".onnx"}
    for mfile in SAVED_DIR.iterdir():
        if mfile.is_file() and mfile.suffix.lower() in model_exts:
            _copy_if_needed(mfile, GDRIVE_MODELS_DIR / mfile.name)

    # ------------------------------------------------------------------
    # 7. 07_Reports_Lengkap_Proyek/
    # ------------------------------------------------------------------
    print("[7/7] Sinkronisasi 07_Reports_Lengkap_Proyek/ ...", flush=True)
    _copy_tree_if_needed(REPORTS_DIR / "eda", GDRIVE_REPORTS_EDA_DIR)
    _copy_tree_if_needed(REPORTS_DIR / "modeling", GDRIVE_REPORTS_EVAL_DIR)
    _copy_tree_if_needed(REPORTS_DIR / "model_evaluation", GDRIVE_REPORTS_EVAL_DIR)
    _copy_tree_if_needed(REPORTS_DIR / "yolo", GDRIVE_REPORTS_EVAL_DIR / "yolo_outputs")
    _copy_tree_if_needed(REPORTS_DIR / "bda", GDRIVE_REPORTS_BDA_DIR)

    # Panduan informatif di root Drive
    guide_text = (
        "================================================================================\n"
        "PANDUAN STRUKTUR GOOGLE DRIVE — BDA-2026-Iqbal (NIM: 09040624091)\n"
        "Presentasi 25 November 2026 | Repeat Jurnal Utama -> Modifikasi 3 Modalitas\n"
        "URL Drive : https://drive.google.com/drive/folders/12FkjUbkSLU3T8sycDNsIN4hbn1bDGz5c?usp=sharing\n"
        "================================================================================\n\n"
        "01_Diagram_DrawIO/  [ARCHITECTURE & FLOWCHART 4 HALAMAN]\n"
        "  - BDA_Journal_Repeat_dan_Modifikasi_4Pages.drawio (XML 4 tab siap buka di app.diagrams.net):\n"
        "      * Page-1 Architecture Jurnal     : Module-only Architecture Jurnal Utama (Selayanti et al., 2024)\n"
        "      * Page-2 Flowchart Jurnal        : Flowchart AHC 48 SWK Surabaya (selaras 1-to-1 notebook Repeat/)\n"
        "      * Page-3 Architecture Modifikasi : Module-only Architecture Multimodal (Dataset A, B, C)\n"
        "      * Page-4 Flowchart Modifikasi    : Flowchart bercabang Dataset A, B, C (selaras notebook Modifikasi/)\n"
        "  - Page1..Page4_*.png : Pratinjau cepat masing-masing halaman diagram\n\n"
        "02_Journal_Rujukan/  [JURNAL UTAMA & PENDAMPING INDONESIA + DOKUMEN]\n"
        "  - Journal_Utama.pdf                              : Selayanti, Putri, & Fahrudin (2024, JoMMiT, DOI 10.46961/jommit.v8i2)\n"
        "  - Journal_Gambar_JuTISI2025_Indonesian_Food.pdf  : Wiranata & Lestari (2025, JuTISI, Univ. Ciputra Surabaya, DOI 10.28932/jutisi.v11i1.10284)\n"
        "  - Journal_Refrensi.pdf                           : Fachri & Zahrotun (2026, arsip pembanding K-Means)\n"
        "  - ringkasan_jurnal_dan_bukti_dataset.csv         : Ringkasan DOI/URL valid (Satu Data Surabaya, Mendeley, CORD-v2 HF Bucket, MIDD Zenodo)\n\n"
        "Repeat/  [CODING & HASIL REPEAT PERSIS JURNAL UTAMA — SELARAS PAGE 2]\n"
        "  - 01_repeat_jurnal_utama_swk_09040624091.ipynb   : Notebook Colab tereksekusi mengikuti Page 2\n"
        "  - repeat_journal_utama_swk.py                    : Script Python reproduksi\n"
        "  - dataset_jurnal_utama/kode11500kode25245kode3135.csv : Dataset asli 48 SWK Surabaya\n"
        "  - hasil_repeat/                                  : Tabel 3, Tabel 4 (match 6 desimal), dendrogram, & grafik perbandingan\n\n"
        "Modifikasi/  [CODING & HASIL MODIFIKASI MULTIMODAL BERCABANG A/B/C — SELARAS PAGE 4]\n"
        "  - 02_modifikasi_multimodal_umkm_09040624091.ipynb: Notebook utama bercabang Dataset A, B, C (selaras Page 4)\n"
        "  - eda_3_09040624091.ipynb                        : EDA lengkap Jurnal Utama + 3 Modalitas\n"
        "  - model_3_09040624091.ipynb                      : Modeling & Evaluation lengkap (AHC, K-Means, Autoencoder, RF/SVM/GB/MLP, MIDD, CORD)\n"
        "  - bda_3_09040624091.ipynb                        : Spatial BDA (Digital Gap Index 6 Kecamatan, Peta, Spider Chart, SWK)\n"
        "  - adjust_journal_multimodal.py                   : Script Python terpadu 3 modalitas\n"
        "  - dataset_gambar_indonesia_serupa/               : Padang Cuisine (HF fahmindra/padang-cuisine, 993 foto 9 kelas — pengganti Mendeley 403)\n"
        "  - dataset_dokumen_tambahan/                      : MIDD_Invoice_IOB (708 CSV) & CORD_v2_Receipts_Indonesia (200 struk + 10 PNG)\n"
        "  - hasil_modifikasi/                              : Tabel benchmark & grafik evaluasi multimodal, gambar Padang, & dokumen\n\n"
        "04_Dataset_Multimodal_Saya/  [DATASET TAMBAHAN UMKM & SWK SURABAYA TIMUR]\n"
        "  - 01_raw/            : 2.272 foto mentah, JSON Google Maps, 3 PDF izin penelitian, metadata survei\n"
        "  - 02_clean/          : 549 foto bersih (279.99 MB >= 200 MB) + 385 UMKM + 1.164 ulasan asli (en/id) + 3 PDF\n"
        "  - 03_staging_spark/  : umkm_spark_staged.{parquet,csv} (PySpark SQL multimodal staging)\n"
        "  - 04_feature_store/  : feature_store.parquet & embeddings_multimodal.{npy,npz,parquet}\n\n"
        "05_Models_Tersimpan/       : Seluruh artefak ML (.pkl), DL (.pt/.pth/.h5), YOLOv8 (.pt), & pipeline\n"
        "07_Reports_Lengkap_Proyek/ : Seluruh grafik & laporan EDA, Model Evaluation (YOLO/SHAP), & Spatial BDA\n"
    )
    (GDRIVE_PROJECT_ROOT / "00_PANDUAN_ISI_DRIVE.txt").write_text(guide_text, encoding="utf-8")

    audit = {
        "gdrive_folder_id": GDRIVE_FOLDER_ID,
        "gdrive_root": str(GDRIVE_PROJECT_ROOT),
        "zones": {
            "01_Diagram_DrawIO": _folder_stats(GDRIVE_DIAGRAM_DIR),
            "02_Journal_Rujukan": _folder_stats(GDRIVE_JOURNAL_DIR),
            "Repeat": _folder_stats(GDRIVE_REPEAT_DIR),
            "Modifikasi": _folder_stats(GDRIVE_MODIFIKASI_DIR),
            "04_clean_images_only": _folder_stats(GDRIVE_CLEAN_IMAGES_DIR),
            "05_Models_Tersimpan": _folder_stats(GDRIVE_MODELS_DIR),
            "07_Reports_Lengkap_Proyek": _folder_stats(GDRIVE_REPORTS_DIR),
        },
        "sync_info": sync_info,
    }
    print("\n=== Ringkasan Audit Struktur Baru BDA-2026-Iqbal ===", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    migrate_all()
