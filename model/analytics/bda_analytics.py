"""
Analitik Big Data Spasial (BDA) dan Visualisasi Kesenjangan Digital per Kecamatan.

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- Pandas GroupBy Aggregation: https://pandas.pydata.org/docs/user_guide/groupby.html
- GeoPandas Spatial Operations: https://geopandas.org/en/stable/docs/user_guide/projections.html
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

from model.storage_config import GDRIVE_REPORTS_BDA_DIR

GOLD_DIR = ROOT_DIR / "data" / "gold"
BDA_DIR = ROOT_DIR / "docs" / "reports" / "bda"
GDRIVE_BDA_DIR = GDRIVE_REPORTS_BDA_DIR


def compute_spatial_bda() -> pd.DataFrame:
    BDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_parquet(GOLD_DIR / "feature_store.parquet")

    agg = (
        df.groupby("kecamatan", as_index=False)
        .agg(
            total_umkm=("place_id", "count"),
            qris_rate=("indikator_qris", lambda s: round(100.0 * s.mean(), 2)),
            ojol_rate=("indikator_ojol", lambda s: round(100.0 * s.mean(), 2)),
            pos_rate=("indikator_pos", lambda s: round(100.0 * s.mean(), 2)),
            edc_rate=("indikator_edc", lambda s: round(100.0 * s.mean(), 2)),
            avg_rating=("rating", lambda s: round(s.mean(), 2)),
            avg_maturity_score=("skor_kematangan_kontinu", lambda s: round(s.mean(), 2)),
        )
    )

    agg["digital_gap_index"] = (100.0 - agg["avg_maturity_score"]).round(2)

    def _assign_priority(gap: float) -> str:
        if gap >= 80.0:
            return "Prioritas Tinggi (Pendampingan Intensif QRIS & Ojol)"
        if gap >= 72.0:
            return "Prioritas Menengah (Akselerasi POS Kasir Digital)"
        return "Prioritas Pemeliharaan (Integrasi Keuangan & Ekspansi)"

    agg["rekomendasi_intervensi"] = agg["digital_gap_index"].apply(_assign_priority)
    agg = agg.sort_values("digital_gap_index", ascending=False).reset_index(drop=True)

    out_csv = BDA_DIR / "kecamatan_digital_gap.csv"
    agg.to_csv(out_csv, index=False)

    # Visualisasi 1: Penetrasi Indikator Digital per Kecamatan
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    melted = agg.melt(
        id_vars="kecamatan",
        value_vars=["qris_rate", "ojol_rate", "pos_rate", "edc_rate"],
        var_name="Indikator",
        value_name="Persentase (%)",
    )
    sns.barplot(data=melted, x="kecamatan", y="Persentase (%)", hue="Indikator", palette="Paired", ax=ax)
    ax.set_title("Penetrasi Indikator Pembayaran & Layanan Digital per Kecamatan (%)")
    ax.set_xlabel("Kecamatan")
    ax.set_ylabel("Adopsi (%)")
    plt.tight_layout()
    p1 = BDA_DIR / "01_penetrasi_digital_kecamatan.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # Visualisasi 2: Skor Kematangan vs Digital Gap Index per Kecamatan
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(
        data=agg,
        x="digital_gap_index",
        y="kecamatan",
        hue="kecamatan",
        palette="YlOrRd_r",
        legend=False,
        ax=ax,
    )
    ax.set_title("Indeks Kesenjangan Digital (Digital Gap Index) per Kecamatan")
    ax.set_xlabel("Digital Gap Index (0-100)")
    ax.set_ylabel("Kecamatan")
    plt.tight_layout()
    p2 = BDA_DIR / "02_indeks_kematangan_kecamatan.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # Visualisasi 3: Peta Sebaran Titik Koordinat Kematangan UMKM Surabaya Timur
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sns.scatterplot(
        data=df,
        x="lon",
        y="lat",
        hue="kelas_kematangan",
        style="kecamatan",
        palette={"Konvensional": "#d95f02", "Berkembang": "#7570b3", "Maju": "#1b9e77"},
        s=75,
        ax=ax,
    )
    ax.set_title("Sebaran Geospasial Kelas Kematangan Digital UMKM Surabaya Timur")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    p3 = BDA_DIR / "03_peta_sebaran_kematangan_umkm.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    if GDRIVE_BDA_DIR.parent.exists():
        GDRIVE_BDA_DIR.mkdir(parents=True, exist_ok=True)
        for p in [out_csv, p1, p2, p3]:
            shutil.copy2(p, GDRIVE_BDA_DIR / p.name)

    return agg


if __name__ == "__main__":
    print(compute_spatial_bda().to_string(index=False))
