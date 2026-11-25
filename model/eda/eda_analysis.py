"""
Exploratory Data Analysis (EDA) Multimodal UMKM Kuliner Surabaya Timur.

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- Pandas Descriptive Statistics: https://pandas.pydata.org/docs/user_guide/basics.html#descriptive-statistics
- Matplotlib Pyplot Tutorial: https://matplotlib.org/stable/tutorials/pyplot.html
- Seaborn Categorical Plots: https://seaborn.pydata.org/tutorial/categorical.html
"""

import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.storage_config import GDRIVE_REPORTS_EDA_DIR

CLEAN_DIR = ROOT_DIR / "data" / "clean"
EDA_REPORT_DIR = ROOT_DIR / "docs" / "reports" / "eda"
GDRIVE_EDA_DIR = GDRIVE_REPORTS_EDA_DIR


def run_eda_analysis() -> dict:
    EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
    df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
    df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")
    df_dok = pd.read_parquet(CLEAN_DIR / "dokumen.parquet")

    saved_plots = []

    # 1. Distribusi UMKM per Kecamatan
    fig, ax = plt.subplots(figsize=(8, 4.5))
    kec_counts = df_umkm["kecamatan"].value_counts()
    sns.barplot(x=kec_counts.values, y=kec_counts.index, hue=kec_counts.index, palette="Blues_r", legend=False, ax=ax)
    ax.set_title("Distribusi UMKM Kuliner per Kecamatan di Surabaya Timur")
    ax.set_xlabel("Jumlah UMKM")
    ax.set_ylabel("Kecamatan")
    plt.tight_layout()
    p1 = EDA_REPORT_DIR / "01_distribusi_umkm_kecamatan.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_plots.append(p1)

    # 2. Distribusi Rating UMKM
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.histplot(df_umkm["rating"], bins=15, kde=True, color="#2b5c8f", ax=ax)
    ax.set_title("Sebaran Rating Google Maps UMKM Surabaya Timur")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frekuensi UMKM")
    plt.tight_layout()
    p2 = EDA_REPORT_DIR / "02_distribusi_rating.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_plots.append(p2)

    # 3. Frekuensi Penyebutan Indikator Digital pada Teks Ulasan
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mention_sums = {
        "QRIS": int(df_ulasan["mention_qris"].sum()),
        "Ojek Online": int(df_ulasan["mention_online"].sum()),
        "Kasir POS": int(df_ulasan["mention_pos"].sum()),
        "Mesin EDC": int(df_ulasan["mention_edc"].sum()),
    }
    sns.barplot(
        x=list(mention_sums.keys()),
        y=list(mention_sums.values()),
        hue=list(mention_sums.keys()),
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Frekuensi Penyebutan Indikator Digital pada Ulasan Pelanggan")
    ax.set_ylabel("Jumlah Ulasan")
    plt.tight_layout()
    p3 = EDA_REPORT_DIR / "03_frekuensi_indikator_ulasan.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_plots.append(p3)

    # 4. Distribusi Panjang Kata Ulasan per Kecamatan
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.boxplot(
        data=df_ulasan,
        x="word_count",
        y="kecamatan",
        hue="kecamatan",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Distribusi Panjang Kata Ulasan per Kecamatan")
    ax.set_xlabel("Jumlah Kata")
    ax.set_ylabel("Kecamatan")
    plt.tight_layout()
    p4 = EDA_REPORT_DIR / "04_panjang_kata_ulasan.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_plots.append(p4)

    # 5. Hasil Deteksi Visual Indikator pada Foto Terkurasi (YOLOv8)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    vis_sums = {
        "QRIS": int(df_foto["has_qris"].sum()),
        "Banner Ojol": int(df_foto["has_merchant_online"].sum()),
        "Kasir POS": int(df_foto["has_pos"].sum()),
        "Mesin EDC": int(df_foto["has_edc"].sum()),
        "Stan/Gerobak": int(df_foto["has_stan"].sum()),
    }
    sns.barplot(
        x=list(vis_sums.keys()),
        y=list(vis_sums.values()),
        hue=list(vis_sums.keys()),
        palette="magma",
        legend=False,
        ax=ax,
    )
    ax.set_title("Deteksi Bukti Visual Indikator pada Foto Terkurasi (n=203)")
    ax.set_ylabel("Jumlah Foto")
    plt.tight_layout()
    p5 = EDA_REPORT_DIR / "05_distribusi_indikator_citra_yolo.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved_plots.append(p5)

    # 6. Proporsi Kelas Ground Truth Survei Lapangan SWK
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    df_swk = df_foto[df_foto["sumber"] == "survei_lapangan_swk"]
    gt_counts = df_swk["label_ground_truth"].value_counts()
    sns.barplot(
        x=gt_counts.index,
        y=gt_counts.values,
        hue=gt_counts.index,
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Distribusi Kelas Ground Truth Survei Lapangan SWK")
    ax.set_xlabel("Kelas Kematangan Digital")
    ax.set_ylabel("Jumlah Sampel Survei")
    plt.tight_layout()
    p6 = EDA_REPORT_DIR / "06_distribusi_ground_truth_swk.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    saved_plots.append(p6)

    if GDRIVE_EDA_DIR.parent.exists():
        GDRIVE_EDA_DIR.mkdir(parents=True, exist_ok=True)
        for p in saved_plots:
            shutil.copy2(p, GDRIVE_EDA_DIR / p.name)

    return {
        "total_umkm": int(len(df_umkm)),
        "total_ulasan": int(len(df_ulasan)),
        "total_foto_clean": int(len(df_foto)),
        "total_faces_blurred": int(df_foto["face_blurred_count"].sum()),
        "total_dokumen": int(len(df_dok)),
        "plots_generated": len(saved_plots),
    }


if __name__ == "__main__":
    print(run_eda_analysis())
