"""
REPEAT JOURNAL UTAMA — Selayanti, Putri, & Fahrudin (2024)
Judul  : Analisis Segmentasi Sentra Wisata Kuliner untuk Optimalisasi Omzet UMKM di Surabaya
         Menggunakan Metode Agglomerative Hierarchical Clustering
Jurnal : Jurnal Multi Media dan IT (JoMMiT), Vol. 08 No. 02 (2024), pp. 113-122
DOI    : https://doi.org/10.46961/jommit.v8i2
Dataset: Jumlah Sentra Wisata Kuliner Kota Surabaya (48 SWK, 11 variabel)
         Sumber resmi: Satu Data Surabaya (arsip Wayback Machine 2023-03-22:
         kode11500kode25245kode3135.csv)

Reproduksi Detail 100% Seluruh 48 Data SWK & Visual Grafik Persis Jurnal (Halaman 4–8):
  - Gambar 2 : Grafik Garis Jumlah Pelaku Usaha vs Omzet > 1 Juta (48 Nama Sentra)
  - Gambar 3 : Grafik Batang (Histogram) Jumlah Pelaku Usaha di Setiap Sentra (48 SWK + angka)
  - Gambar 4 : Grafik Batang Horizontal Jumlah Pelaku Usaha vs Kapasitas Sentra (48 SWK alfabetis + angka)
  - Tabel 1  : Dataset 48 SWK Setelah Cleaning & Imputasi Mean (Luas Sentra)
  - Tabel 2  : Dataset 48 SWK Setelah Normalisasi Z-Score (StandardScaler)
  - Gambar 5 : Dendrogram Single Linkage (48 Data Points + garis merah potong k=3 + daftar 48 SWK)
  - Gambar 6 : Dendrogram Average Linkage (48 Data Points + garis merah potong k=3 + daftar 48 SWK)
  - Gambar 7a: Dendrogram Complete Linkage (48 Data Points + garis merah potong k=3 + daftar 48 SWK)
  - Gambar 8 : Dendrogram Ward Linkage (48 Data Points + garis merah potong k=3 + daftar 48 SWK)
  - Tabel 3  : Perbandingan Cophenetic, Silhouette, & Proporsi Cluster 4 Algoritma AHC
  - Gambar 7b: Grafik Silhouette Score Complete Linkage (k = 2..10 + label angka)
  - Tabel 4  : Rata-Rata Z-Score 3 Cluster Terbentuk (Complete Linkage) & Daftar Lengkap 48 SWK
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CSV = ROOT_DIR / "data" / "raw" / "journal_utama_swk" / "kode11500kode25245kode3135.csv"
OUT_DIR = ROOT_DIR / "docs" / "reports" / "journal_repeat"

PAPER_TABLE3 = {
    "single": {"cophenetic": 0.9111, "silhouette": 0.5461, "proporsi": "46 / 1 / 1"},
    "average": {"cophenetic": 0.9317, "silhouette": 0.5461, "proporsi": "46 / 1 / 1"},
    "complete": {"cophenetic": 0.8734, "silhouette": 0.4864, "proporsi": "4 / 43 / 1"},
    "ward": {"cophenetic": 0.6368, "silhouette": 0.3954, "proporsi": "31 / 16 / 1"},
}


def parse_luas_m2(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).replace("m2", "").strip()
    if not s:
        return np.nan
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def load_and_preprocess_swk(csv_path: Path = DATA_CSV) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df_raw = pd.read_csv(csv_path, sep=";")
    df = df_raw.copy()
    df["Luas_Sentra_m2"] = df["Luas sentra"].apply(parse_luas_m2)

    feature_cols = [
        "Luas_Sentra_m2",
        "Kapasitas",
        "Jumlah pelaku usaha",
        "Jumlah pelaku usaha yang omzetnya lebih dari 1 juta rupiah",
    ]
    df_feat = df[feature_cols].copy()
    mean_luas = df_feat["Luas_Sentra_m2"].mean()
    df_feat["Luas_Sentra_m2"] = df_feat["Luas_Sentra_m2"].fillna(mean_luas)

    X_scaled = StandardScaler().fit_transform(df_feat.values)
    return df, df_feat, X_scaled


def _cutoff_height_for_k(Z: np.ndarray, k: int = 3) -> float:
    """Menghitung tinggi garis potong horizontal merah tepat di tengah pemisah k cluster."""
    d_low = float(Z[-(k), 2])
    d_high = float(Z[-(k - 1), 2])
    return 0.5 * (d_low + d_high)


def _save_dendrogram_journal(Z: np.ndarray, title: str, out_png: Path, k: int = 3) -> None:
    cut_y = _cutoff_height_for_k(Z, k=k)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    dendrogram(
        Z,
        labels=[str(i) for i in range(Z.shape[0] + 1)],
        leaf_rotation=90,
        leaf_font_size=7.5,
        color_threshold=cut_y,
        above_threshold_color="#1f77b4",
        ax=ax,
    )
    ax.axhline(y=cut_y, color="red", linestyle="--", linewidth=1.3, label=f"Cut-off k={k} (d={cut_y:.2f})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Data Points", fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
    ax.grid(axis="y", linestyle="-", alpha=0.45)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Bersihkan grafik lama jika ada agar folder hasil_repeat konsisten
    for old_name in ["01_eda_jurnal_utama_swk.png", "02_repeat_ahc_dendrogram_dan_silhouette.png", "03_repeat_vs_jurnal_utama_metrics.png"]:
        old_p = OUT_DIR / old_name
        if old_p.exists():
            old_p.unlink()

    df, df_feat, X_scaled = load_and_preprocess_swk()
    missing_swks = df.loc[df["Luas_Sentra_m2"].isna(), "Nama sentra"].tolist()
    print(f"Dataset Asli SWK Surabaya: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Missing Luas Sentra awal : {len(missing_swks)} baris {missing_swks} -> imputasi mean ({df_feat['Luas_Sentra_m2'].mean():.5f} m2)")

    # ------------------------------------------------------------------
    # GAMBAR 2 JURNAL: Line Plot 48 SWK — Pelaku Usaha vs Omzet > 1 Juta
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    x = np.arange(len(df))
    ax.plot(x, df["Jumlah pelaku usaha"], marker="o", markersize=4.5, lw=1.6, color="#1f77b4", label="Jumlah Pelaku Usaha")
    ax.plot(x, df["Jumlah pelaku usaha yang omzetnya lebih dari 1 juta rupiah"], marker="o", markersize=4.5, lw=1.6, color="#ff7f0e", label="Pelaku Usaha Omzet > 1 juta")
    ax.set_title("Jumlah pelaku usaha vs Jumlah pelaku usaha omzetnya > 1 juta", fontsize=12)
    ax.set_xlabel("Nama Sentra", fontsize=10)
    ax.set_ylabel("Jumlah Pelaku Usaha", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Nama sentra"], rotation=90, fontsize=7.5)
    ax.grid(True, linestyle="-", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_gambar2_pelaku_usaha_vs_omzet_48_swk.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # GAMBAR 3 JURNAL: Vertical Bar Chart 48 SWK + label angka per batang
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16.0, 6.2))
    bars = ax.bar(x, df["Jumlah pelaku usaha"], color="#1f77b4", width=0.72)
    ax.set_title("Jumlah Pelaku Usaha di Setiap Sentra", fontsize=12)
    ax.set_xlabel("Nama Sentra", fontsize=10)
    ax.set_ylabel("Jumlah Pelaku Usaha", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Nama sentra"], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, df["Jumlah pelaku usaha"].max() + 10)
    ax.grid(axis="y", linestyle="-", alpha=0.45)
    for bar in bars:
        h = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, str(h), ha="center", va="bottom", fontsize=7.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_gambar3_jumlah_pelaku_usaha_48_swk.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # GAMBAR 4 JURNAL: Horizontal Grouped Bar Chart 48 SWK Alfabetis
    # ------------------------------------------------------------------
    df_alpha = df.sort_values("Nama sentra", key=lambda s: s.str.lower()).reset_index(drop=True)
    y = np.arange(len(df_alpha))
    bar_h = 0.40
    fig, ax = plt.subplots(figsize=(11.0, 14.5))
    b1 = ax.barh(y - bar_h / 2, df_alpha["Jumlah pelaku usaha"], height=bar_h, color="#4c72b0", label="Jumlah Pelaku Usaha")
    b2 = ax.barh(y + bar_h / 2, df_alpha["Kapasitas"], height=bar_h, color="#f4c430", label="Kapasitas Sentra")
    ax.set_title("Jumlah Pelaku Usaha vs. Kapasitas Sentra di Setiap Sentra", fontsize=12)
    ax.set_xlabel("Jumlah", fontsize=10)
    ax.set_ylabel("Sentra", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(df_alpha["Nama sentra"], fontsize=8.0)
    ax.set_xlim(0, df_alpha["Kapasitas"].max() + 20)
    ax.grid(axis="x", linestyle="-", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.5)
    for bar in b1:
        w = int(bar.get_width())
        ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, str(w), va="center", ha="left", fontsize=6.8)
    for bar in b2:
        w = int(bar.get_width())
        ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, str(w), va="center", ha="left", fontsize=6.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_gambar4_pelaku_usaha_vs_kapasitas_48_swk.png", dpi=180)
    plt.close(fig)

    # Simpan Tabel 1 (Cleaned 48 SWK) & Tabel 2 (Normalized Z-Score 48 SWK)
    df_table1 = pd.DataFrame({
        "No": np.arange(1, len(df) + 1),
        "Nama_Sentra": df["Nama sentra"],
        "Luas_Sentra_m2": df_feat["Luas_Sentra_m2"].round(5),
        "Kapasitas": df_feat["Kapasitas"].astype(int),
        "Jumlah_Pelaku_Usaha": df_feat["Jumlah pelaku usaha"].astype(int),
        "Jumlah_Pelaku_Usaha_Omzet_gt_1_Juta": df_feat["Jumlah pelaku usaha yang omzetnya lebih dari 1 juta rupiah"].astype(int),
    })
    df_table1.to_csv(OUT_DIR / "repeat_table1_cleaned_48_swk.csv", index=False)

    df_table2 = pd.DataFrame(
        X_scaled,
        columns=["Luas Sentra (m2)", "Kapasitas", "Jumlah Pelaku Usaha", "Jumlah Pelaku Usaha Omzetnya > 1 juta"],
    ).round(6)
    df_table2.insert(0, "Nama_Sentra", df["Nama sentra"].values)
    df_table2.insert(0, "No", np.arange(1, len(df) + 1))
    df_table2.to_csv(OUT_DIR / "repeat_table2_normalized_48_swk.csv", index=False)

    # ------------------------------------------------------------------
    # GAMBAR 5, 6, 7a, 8 JURNAL: Dendrogram 4 Linkage + Tabel 3
    # ------------------------------------------------------------------
    D = pdist(X_scaled, metric="euclidean")
    rows = []
    Z_map = {}
    lbl_map = {}
    dendro_files = {
        "single": ("Single Linkage", OUT_DIR / "04_gambar5_dendrogram_single_linkage.png"),
        "average": ("Average Linkage", OUT_DIR / "05_gambar6_dendrogram_average_linkage.png"),
        "complete": ("Complete Linkage", OUT_DIR / "06_gambar7_dendrogram_complete_linkage.png"),
        "ward": ("Ward Linkage", OUT_DIR / "07_gambar8_dendrogram_ward_linkage.png"),
    }

    for method in ["single", "average", "complete", "ward"]:
        Z = linkage(X_scaled, method=method)
        coph, _ = cophenet(Z, D)
        lbl = fcluster(Z, t=3, criterion="maxclust")
        sil = silhouette_score(X_scaled, lbl)
        Z_map[method] = Z
        lbl_map[method] = lbl

        title, out_png = dendro_files[method]
        _save_dendrogram_journal(Z, title, out_png, k=3)

        counts = pd.Series(lbl).value_counts().sort_values(ascending=False).tolist()
        ref = PAPER_TABLE3[method]
        rows.append({
            "Algoritma": f"{method.title()} Linkage",
            "Paper_Cophenetic": ref["cophenetic"],
            "Repeat_Cophenetic": round(float(coph), 4),
            "Paper_Silhouette": ref["silhouette"],
            "Repeat_Silhouette": round(float(sil), 4),
            "Paper_Proporsi_k3": ref["proporsi"],
            "Repeat_Proporsi_k3": " / ".join(str(c) for c in counts),
            "Selisih_Silhouette": round(float(sil) - ref["silhouette"], 4),
        })

    df_table3 = pd.DataFrame(rows)
    print("\n=== TABEL 3: PERBANDINGAN 4 ALGORITMA AHC (k = 3) ===")
    print(df_table3.to_string(index=False))

    # ------------------------------------------------------------------
    # GAMBAR 7b (Halaman 8 Jurnal): Grafik Silhouette Score (k = 2..10)
    # ------------------------------------------------------------------
    Z_complete = Z_map["complete"]
    lbl_complete = lbl_map["complete"]
    sil_k = {k: round(float(silhouette_score(X_scaled, fcluster(Z_complete, t=k, criterion="maxclust"))), 4) for k in range(2, 11)}

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ks = list(sil_k.keys())
    vs = list(sil_k.values())
    ax.plot(ks, vs, marker="o", markersize=6, lw=1.8, color="#1f77b4")
    ax.set_title("Grafik Silhouette Score", fontsize=12)
    ax.set_xlabel("Number of Clusters", fontsize=10)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.set_xticks(ks)
    ax.set_ylim(0.24, 0.80)
    for k, v in zip(ks, vs):
        ax.text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9.0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_gambar7_silhouette_score_k2_10.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # TABEL 4 JURNAL: Rata-Rata Z-Score 3 Cluster (Complete Linkage)
    # ------------------------------------------------------------------
    raw_counts = pd.Series(lbl_complete).value_counts()
    cluster_map = {
        int(raw_counts[raw_counts == 4].index[0]): 1,
        int(raw_counts[raw_counts == 43].index[0]): 2,
        int(raw_counts[raw_counts == 1].index[0]): 3,
    }
    df["Cluster_AHC"] = [cluster_map[int(c)] for c in lbl_complete]

    df_z_only = pd.DataFrame(
        X_scaled,
        columns=["Luas Sentra (m2)", "Kapasitas", "Jumlah Pelaku Usaha", "Jumlah Pelaku Usaha Omzetnya > 1 juta"],
    )
    df_z_only["Cluster_AHC"] = df["Cluster_AHC"]
    df_table4 = df_z_only.groupby("Cluster_AHC").mean().T.round(6)
    df_table4.columns = [f"Cluster {c}" for c in df_table4.columns]
    df_table4.loc["Rata-Rata"] = df_table4.mean(axis=0).round(6)
    print("\n=== TABEL 4: RATA-RATA Z-SCORE PER CLUSTER (COMPLETE LINKAGE) ===")
    print(df_table4.to_string())

    # Simpan artefak CSV & JSON
    df_table3.to_csv(OUT_DIR / "repeat_table3_ahc_comparison.csv", index=False)
    df_table4.to_csv(OUT_DIR / "repeat_table4_cluster_profile.csv")
    df_final = df_table1.copy()
    df_final.insert(2, "Kecamatan", df["Kecamatan"].values)
    df_final["Cluster_AHC"] = df["Cluster_AHC"].values
    df_final.to_csv(OUT_DIR / "swk_surabaya_clustered_48.csv", index=False)

    # Grafik tambahan: Perbandingan Metrik Repeat vs Tabel 3 Jurnal
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    melted = df_table3.melt(
        id_vars="Algoritma",
        value_vars=["Paper_Cophenetic", "Repeat_Cophenetic", "Paper_Silhouette", "Repeat_Silhouette"],
        var_name="Metrik",
        value_name="Skor",
    )
    sns.barplot(data=melted, x="Algoritma", y="Skor", hue="Metrik", palette="Set2", ax=ax)
    ax.set_title("Verifikasi Repeat vs Jurnal Utama (Selayanti et al., 2024) — AHC k = 3")
    ax.set_ylim(0, 1.08)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2, h + 0.015), ha="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "09_verifikasi_repeat_vs_jurnal_metrics.png", dpi=180)
    plt.close(fig)

    summary = {
        "journal": {
            "title": "Analisis Segmentasi Sentra Wisata Kuliner untuk Optimalisasi Omzet UMKM di Surabaya Menggunakan Metode Agglomerative Hierarchical Clustering",
            "authors": "Nabilah Selayanti, Shafira Amanda Putri, Tresna Maulana Fahrudin (UPN Veteran Jawa Timur)",
            "venue": "Jurnal Multi Media dan IT (JoMMiT), Vol. 08 No. 02 (2024), pp. 113-122",
            "doi": "https://doi.org/10.46961/jommit.v8i2",
            "dataset_source": "https://web.archive.org/web/20230322060020/https://opendata.surabaya.go.id/dataset/ee6acc0e-3a6b-4305-b229-124fde04deaf/resource/7a4c926d-0059-4eba-8344-7782e018517d/download/kode11500kode25245kode3135.csv",
        },
        "instances": int(len(df)),
        "silhouette_by_k_complete": sil_k,
        "cluster_1_members": df[df["Cluster_AHC"] == 1]["Nama sentra"].tolist(),
        "cluster_2_members": df[df["Cluster_AHC"] == 2]["Nama sentra"].tolist(),
        "cluster_3_members": df[df["Cluster_AHC"] == 3]["Nama sentra"].tolist(),
        "table3": df_table3.to_dict(orient="records"),
    }
    (OUT_DIR / "repeat_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSeluruh 8 grafik persis jurnal + tabel 48 SWK tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
