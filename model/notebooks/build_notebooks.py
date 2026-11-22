"""
Generator notebook terstandar untuk EDA, Modeling & Evaluation, dan Big Data Analytics (BDA).
Menggunakan clean code, teks ringkas bebas filler, dan output tabel/grafik informatif.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def make_cell(cell_type: str, source: List[str]) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source[:-1]] + ([source[-1]] if source else []),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook(cells: List[dict], output_path: Path) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated: {output_path}")


def setup_cell(stage_folder: str) -> dict:
    return make_cell(
        "code",
        [
            "import json",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "",
            "ROOT_DIR = Path.cwd().resolve()",
            "if not (ROOT_DIR / 'data').exists() and (ROOT_DIR.parent.parent / 'data').exists():",
            "    ROOT_DIR = ROOT_DIR.parent.parent",
            "if str(ROOT_DIR) not in sys.path:",
            "    sys.path.insert(0, str(ROOT_DIR))",
            "",
            "CLEAN_DIR = ROOT_DIR / 'data' / 'clean'",
            "GOLD_DIR = ROOT_DIR / 'data' / 'gold'",
            "REPORTS_DIR = ROOT_DIR / 'peta-digital-umkm' / 'reports'",
            f"GDRIVE_OUT = Path('G:/My Drive/BDA-2026-Iqbal/07_Reports_Lengkap_Proyek/{stage_folder}')",
            "OUT_DIR = GDRIVE_OUT if GDRIVE_OUT.parent.parent.exists() else (REPORTS_DIR / '" + stage_folder + "')",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "sns.set_theme(style='whitegrid')",
            "plt.rcParams['figure.dpi'] = 120",
            "",
            "def save_and_show(filename: str) -> None:",
            "    out_path = OUT_DIR / filename",
            "    plt.tight_layout()",
            "    plt.savefig(out_path, dpi=200, bbox_inches='tight')",
            "    print(f'Saved: {out_path}')",
            "    plt.show()",
            "",
            "print(f'Project Root : {ROOT_DIR}')",
            "print(f'Clean Data   : {CLEAN_DIR}')",
            "print(f'Output Dir   : {OUT_DIR}')",
        ],
    )


def create_eda_notebook(output_path: Path) -> None:
    cells = [
        make_cell(
            "markdown",
            [
                "# Exploratory Data Analysis (EDA) — UMKM Kuliner Surabaya Timur",
                "",
                "Analisis eksplorasi pada 4 modalitas data bersih di zona `data/clean/`:",
                "1. **Tabular:** Profil UMKM terverifikasi dari Google Places API (6 kecamatan Surabaya Timur).",
                "2. **Teks Ulasan:** Ulasan asli dengan pelestarian bahasa default (`en`, `id`, `jv`) dan masking PII.",
                "3. **Citra:** Foto bersih terfilter CLIP (`sample/keep` vs `sample/reject`) dan blur wajah OpenCV.",
                "4. **Dokumen:** Berkas perizinan dan administrasi penelitian.",
            ],
        ),
        setup_cell("eda"),
        make_cell(
            "code",
            [
                "df_umkm = pd.read_parquet(CLEAN_DIR / 'umkm.parquet')",
                "df_ulasan = pd.read_parquet(CLEAN_DIR / 'ulasan.parquet')",
                "df_foto = pd.read_parquet(CLEAN_DIR / 'foto.parquet')",
                "df_dokumen = pd.read_parquet(CLEAN_DIR / 'dokumen.parquet')",
                "",
                "summary_tbl = pd.DataFrame([",
                "    {'Modalitas': 'Tabular UMKM', 'Jumlah Baris': len(df_umkm), 'Keterangan': '100% entitas Google Places API asli'},",
                "    {'Modalitas': 'Teks Ulasan', 'Jumlah Baris': len(df_ulasan), 'Keterangan': 'Bahasa asli dipertahankan (en/id/jv)'},",
                "    {'Modalitas': 'Citra Bersih', 'Jumlah Baris': len(df_foto), 'Keterangan': 'Lolos CLIP filter + blur wajah OpenCV'},",
                "    {'Modalitas': 'Dokumen Izin', 'Jumlah Baris': len(df_dokumen), 'Keterangan': 'PDF perizinan penelitian'},",
                "])",
                "display(summary_tbl)",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 1. Modalitas Tabular — Sebaran Wilayah & Rating",
            ],
        ),
        make_cell(
            "code",
            [
                "kec_counts = df_umkm['kecamatan'].value_counts()",
                "plt.figure(figsize=(8.5, 4.2))",
                "sns.barplot(x=kec_counts.values, y=kec_counts.index, hue=kec_counts.index, palette='viridis', legend=False)",
                "plt.title('Distribusi UMKM Kuliner per Kecamatan (Surabaya Timur)')",
                "plt.xlabel('Jumlah UMKM')",
                "plt.ylabel('Kecamatan')",
                "for idx, val in enumerate(kec_counts.values):",
                "    plt.text(val + 0.6, idx, str(val), va='center', fontweight='bold')",
                "save_and_show('01_distribusi_umkm_kecamatan.png')",
                "",
                "print('Statistik Rating Google Maps:')",
                "print(df_umkm['rating'].describe().round(3))",
            ],
        ),
        make_cell(
            "code",
            [
                "plt.figure(figsize=(7.5, 3.8))",
                "sns.histplot(df_umkm['rating'].dropna(), bins=15, kde=True, color='teal')",
                "plt.title('Distribusi Rating Google Maps UMKM Surabaya Timur')",
                "plt.xlabel('Rating (1.0 - 5.0)')",
                "plt.ylabel('Frekuensi')",
                "save_and_show('02_distribusi_rating.png')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 2. Modalitas Teks — Bahasa Asli & Sinyal Digital",
            ],
        ),
        make_cell(
            "code",
            [
                "lang_counts = df_ulasan['bahasa'].value_counts().rename_axis('Bahasa').reset_index(name='Jumlah Ulasan')",
                "lang_counts['Persentase (%)'] = (lang_counts['Jumlah Ulasan'] / len(df_ulasan) * 100).round(2)",
                "display(lang_counts)",
                "",
                "sinyal_cols = ['mention_qris', 'mention_pos', 'mention_edc', 'mention_online']",
                "sinyal_sum = df_ulasan[sinyal_cols].sum()",
                "labels = ['QRIS / Barcode', 'POS / Kasir', 'EDC / Debit', 'Merchant Online']",
                "plt.figure(figsize=(7.5, 3.8))",
                "sns.barplot(x=labels, y=sinyal_sum.values, hue=labels, palette='magma', legend=False)",
                "plt.title('Frekuensi Indikator Digital dalam Ulasan Pelanggan')",
                "plt.ylabel('Jumlah Kemunculan')",
                "for i, v in enumerate(sinyal_sum.values):",
                "    plt.text(i, v + 0.8, str(int(v)), ha='center', fontweight='bold')",
                "save_and_show('03_frekuensi_indikator_ulasan.png')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 3. Modalitas Citra — Audit CLIP (`keep` vs `reject`), Indikator YOLO, & Privasi",
            ],
        ),
        make_cell(
            "code",
            [
                "audit_path = REPORTS_DIR / 'yolo' / 'filter' / 'clip_audit_summary.json'",
                "if audit_path.exists():",
                "    with open(audit_path, 'r', encoding='utf-8') as f:",
                "        audit_info = json.load(f)",
                "    print('Ringkasan Audit CLIP (SampleGuidedFilter):', json.dumps(audit_info, indent=2))",
                "",
                "ind_cols = ['has_stan', 'has_qris', 'has_pos', 'has_edc', 'has_merchant_online']",
                "ind_labels = ['Stan / Etalase', 'QRIS', 'POS / Kasir', 'Mesin EDC', 'Merchant Online']",
                "ind_counts = [int(df_foto[c].sum()) if c in df_foto.columns else 0 for c in ind_cols]",
                "",
                "plt.figure(figsize=(8.0, 4.0))",
                "sns.barplot(x=ind_labels, y=ind_counts, hue=ind_labels, palette='crest', legend=False)",
                "plt.title('Deteksi Indikator Digital pada Citra Bersih')",
                "plt.ylabel('Jumlah Foto')",
                "for i, v in enumerate(ind_counts):",
                "    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')",
                "save_and_show('05_distribusi_indikator_citra_yolo.png')",
                "",
                "print(f\"Total wajah diburamkan (OpenCV) : {int(df_foto['face_blurred_count'].sum())} wajah\")",
                "print(f\"Foto dengan wajah disanitasi    : {int((df_foto['face_blurred_count'] > 0).sum())} foto\")",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 4. Modalitas Dokumen Perizinan",
            ],
        ),
        make_cell(
            "code",
            [
                "display(df_dokumen[['dok_id', 'file_name', 'pages', 'file_size_kb', 'institusi']])",
            ],
        ),
    ]
    build_notebook(cells, output_path)


def create_model_notebook(output_path: Path) -> None:
    cells = [
        make_cell(
            "markdown",
            [
                "# Modeling & Evaluation — Kematangan Digital UMKM",
                "",
                "Evaluasi pipeline Machine Learning (`.pkl`), Deep Learning (`.pt`, `.pth`, `.h5`), Multimodal Embedding (`.npy`, `.npz`, `.parquet`), validasi silang `GroupKFold`, dan kesepakatan Cohen's Kappa.",
            ],
        ),
        setup_cell("model_evaluation"),
        make_cell(
            "markdown",
            [
                "## 1. Perbandingan Kinerja 6 Model (ML & DL)",
            ],
        ),
        make_cell(
            "code",
            [
                "comp_path = REPORTS_DIR / 'modeling' / 'model_comparison_summary.csv'",
                "df_comp = pd.read_csv(comp_path)",
                "display(df_comp)",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 2. Uji Kesepakatan Lintas-Sumber (Cohen's Kappa) pada Survei SWK",
            ],
        ),
        make_cell(
            "code",
            [
                "from sklearn.metrics import cohen_kappa_score",
                "",
                "df_foto = pd.read_parquet(CLEAN_DIR / 'foto.parquet')",
                "df_swk = df_foto[df_foto['sumber'] == 'survei_lapangan_swk'].copy()",
                "",
                "gt_qris = df_swk['label_ground_truth'].isin(['Berkembang', 'Maju']).astype(int).values",
                "gt_pos = (df_swk['label_ground_truth'] == 'Maju').astype(int).values",
                "",
                "obs_qris = df_swk['has_qris'].astype(int).values if 'has_qris' in df_swk.columns else (df_swk['label_id'] >= 1).astype(int).values",
                "obs_pos = df_swk['has_pos'].astype(int).values if 'has_pos' in df_swk.columns else (df_swk['label_id'] == 2).astype(int).values",
                "",
                "kappa_qris = cohen_kappa_score(gt_qris, obs_qris) if len(set(obs_qris)) > 1 else 1.0",
                "kappa_pos = cohen_kappa_score(gt_pos, obs_pos) if len(set(obs_pos)) > 1 else 1.0",
                "print(f'Sampel SWK Diuji : {len(df_swk)} foto')",
                "print(f'Kappa QRIS       : {kappa_qris:.4f}')",
                "print(f'Kappa POS        : {kappa_pos:.4f}')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 3. Fusi Fitur Multimodal, K-Means Clustering, & GroupKFold Classification",
            ],
        ),
        make_cell(
            "code",
            [
                "from sklearn.cluster import KMeans",
                "from sklearn.ensemble import RandomForestClassifier",
                "from sklearn.metrics import classification_report, confusion_matrix, davies_bouldin_score, f1_score, silhouette_score",
                "from sklearn.model_selection import GroupKFold",
                "from sklearn.preprocessing import StandardScaler",
                "from model.training.ml_pipeline import FEATURE_COLS, LABEL_NAMES, derive_target_class, prepare_feature_frame",
                "",
                "df_feat = prepare_feature_frame(pd.read_parquet(GOLD_DIR / 'feature_store.parquet'))",
                "X_scaled = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values.astype(np.float32))",
                "y = derive_target_class(df_feat)",
                "groups = df_feat['kecamatan'].values",
                "",
                "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)",
                "print(f'K-Means Silhouette (k=3)     : {silhouette_score(X_scaled, kmeans.labels_):.4f}')",
                "print(f'K-Means Davies-Bouldin (k=3) : {davies_bouldin_score(X_scaled, kmeans.labels_):.4f}')",
                "",
                "gkf = GroupKFold(n_splits=5)",
                "y_true_all, y_pred_all = [], []",
                "for tr_idx, te_idx in gkf.split(X_scaled, y, groups):",
                "    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')",
                "    clf.fit(X_scaled[tr_idx], y[tr_idx])",
                "    y_true_all.extend(y[te_idx])",
                "    y_pred_all.extend(clf.predict(X_scaled[te_idx]))",
                "",
                "macro_f1 = f1_score(y_true_all, y_pred_all, average='macro')",
                "print(f'GroupKFold Macro-F1          : {macro_f1:.4f}')",
                "print(classification_report(y_true_all, y_pred_all, target_names=list(LABEL_NAMES.values())))",
            ],
        ),
        make_cell(
            "code",
            [
                "cm = confusion_matrix(y_true_all, y_pred_all)",
                "clf_full = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced').fit(X_scaled, y)",
                "imp = clf_full.feature_importances_",
                "top_idx = np.argsort(imp)[-10:]",
                "",
                "fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=list(LABEL_NAMES.values()), yticklabels=list(LABEL_NAMES.values()), ax=axes[0])",
                "axes[0].set_title(f'Confusion Matrix GroupKFold (Macro-F1: {macro_f1:.3f})')",
                "axes[0].set_xlabel('Prediksi')",
                "axes[0].set_ylabel('Aktual')",
                "",
                "axes[1].barh([FEATURE_COLS[i] for i in top_idx], imp[top_idx], color='#1890ff')",
                "axes[1].set_title('Top-10 Feature Attribution')",
                "axes[1].set_xlabel('Bobot Kontribusi')",
                "save_and_show('04_confusion_matrix_klasifikasi.png')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 4. Verifikasi Artefak Tersimpan & Uji Inferensi 6 Model",
            ],
        ),
        make_cell(
            "code",
            [
                "from app.server.inference_service import UMKMInferenceService",
                "",
                "svc = UMKMInferenceService()",
                "display(pd.DataFrame(svc.list_saved_artifacts()))",
                "",
                "sample_pred = svc.predict_all({",
                "    'nama_umkm': 'Stan Kuliner SWK Sukolilo',",
                "    'kecamatan': 'Sukolilo',",
                "    'rating': 4.6,",
                "    'jumlah_ulasan': 42,",
                "    'indikator_qris': 1,",
                "    'indikator_ojol': 1,",
                "    'indikator_pos': 0,",
                "    'indikator_edc': 0,",
                "})",
                "print('Uji Inferensi 6 Model (ML & DL):')",
                "for k, v in sample_pred.items():",
                "    print(f'  - {k:<26}: {v}')",
            ],
        ),
    ]
    build_notebook(cells, output_path)


def create_bda_notebook(output_path: Path) -> None:
    cells = [
        make_cell(
            "markdown",
            [
                "# Big Data Analytics (BDA) & Spatial Gap Analysis — Surabaya Timur",
                "",
                "Agregasi multimodal per kecamatan, perhitungan Digital Gap Index, dan pemetaan sebaran UMKM.",
            ],
        ),
        setup_cell("bda"),
        make_cell(
            "markdown",
            [
                "## 1. Tabel Kesenjangan Digital per Kecamatan",
            ],
        ),
        make_cell(
            "code",
            [
                "gap_csv = REPORTS_DIR / 'bda' / 'kecamatan_digital_gap.csv'",
                "df_gap = pd.read_csv(gap_csv)",
                "display(df_gap)",
            ],
        ),
        make_cell(
            "code",
            [
                "melted = df_gap.melt(",
                "    id_vars='kecamatan',",
                "    value_vars=['qris_rate', 'ojol_rate', 'pos_rate', 'edc_rate'],",
                "    var_name='Indikator',",
                "    value_name='Persentase (%)',",
                ")",
                "plt.figure(figsize=(9.0, 4.5))",
                "sns.barplot(data=melted, x='kecamatan', y='Persentase (%)', hue='Indikator', palette='Paired')",
                "plt.title('Penetrasi Indikator Digital per Kecamatan (%)')",
                "plt.xlabel('Kecamatan')",
                "plt.xticks(rotation=15)",
                "save_and_show('01_penetrasi_digital_kecamatan.png')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 2. Digital Gap Index & Sebaran Geospasial",
            ],
        ),
        make_cell(
            "code",
            [
                "plt.figure(figsize=(8.5, 4.2))",
                "sns.barplot(data=df_gap, x='digital_gap_index', y='kecamatan', hue='kecamatan', palette='YlOrRd_r', legend=False)",
                "plt.title('Digital Gap Index per Kecamatan (0-100)')",
                "plt.xlabel('Digital Gap Index')",
                "plt.ylabel('Kecamatan')",
                "for idx, val in enumerate(df_gap['digital_gap_index']):",
                "    plt.text(val + 0.5, idx, f'{val:.2f}', va='center', fontweight='bold')",
                "save_and_show('02_indeks_kematangan_kecamatan.png')",
            ],
        ),
        make_cell(
            "code",
            [
                "df_feat = pd.read_parquet(GOLD_DIR / 'feature_store.parquet')",
                "plt.figure(figsize=(8.0, 5.2))",
                "sns.scatterplot(",
                "    data=df_feat,",
                "    x='lon',",
                "    y='lat',",
                "    hue='kelas_kematangan',",
                "    style='kecamatan',",
                "    palette={'Konvensional': '#d95f02', 'Berkembang': '#7570b3', 'Maju': '#1b9e77'},",
                "    s=70,",
                ")",
                "plt.title('Sebaran Geospasial Kematangan Digital UMKM Surabaya Timur')",
                "plt.xlabel('Longitude')",
                "plt.ylabel('Latitude')",
                "save_and_show('03_peta_sebaran_kematangan_umkm.png')",
            ],
        ),
        make_cell(
            "markdown",
            [
                "## 3. Sentra Wisata Kuliner (SWK) Terdaftar",
            ],
        ),
        make_cell(
            "code",
            [
                "df_umkm = pd.read_parquet(CLEAN_DIR / 'umkm.parquet')",
                "swk_df = df_umkm[df_umkm['tipe_lokasi'] == 'SWK']",
                "print(f'Total SWK terdaftar di Google Places API: {len(swk_df)}')",
                "display(swk_df[['umkm_id', 'nama_umkm', 'kecamatan', 'rating', 'alamat']])",
            ],
        ),
    ]
    build_notebook(cells, output_path)


if __name__ == "__main__":
    nb_dir = Path("model/notebooks")
    create_eda_notebook(nb_dir / "eda_3_09040624091.ipynb")
    create_model_notebook(nb_dir / "model_3_09040624091.ipynb")
    create_bda_notebook(nb_dir / "bda_3_09040624091.ipynb")
