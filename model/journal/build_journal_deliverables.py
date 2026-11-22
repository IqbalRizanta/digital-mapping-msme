"""
Generator Bukti Alur (Flowchart PNG + Draw.io XML), Tabel Ringkasan Search Journal,
serta 2 Notebook Sederhana bergaya AMFLearning (Google Colab Ready):
  1. 01_repeat_journal_bdcc2024_09040624091.ipynb  (Repeat Persis Jurnal Ghatora et al. 2024)
  2. 02_adjust_journal_multimodal_umkm_09040624091.ipynb (Adjust Multimodal UMKM Surabaya Timur)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT_DIR / "peta-digital-umkm" / "reports"
REPEAT_DIR = REPORTS_DIR / "journal_repeat"
ADJUST_DIR = REPORTS_DIR / "journal_adjust"
NB_DIR = ROOT_DIR / "model" / "notebooks"


def _draw_box(ax, x, y, w, h, text, facecolor="#E8F1FA", edgecolor="#1F4E79", fontsize=9.5, bold=False):
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#111111",
    )


def _draw_arrow(ax, x1, y1, x2, y2, color="#222222"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
    )


def build_repeat_flowchart(out_png: Path, out_drawio: Path) -> None:
    """Membuat flowchart PNG & Draw.io untuk alur Repeat Journal Ghatora et al. (2024)."""
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "ALUR REPEAT JOURNAL — Ghatora et al. (2024)\nBig Data and Cognitive Computing (MDPI), Vol. 8(12), Art. 199 | DOI: 10.3390/bdcc8120199",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#0B2E59",
    )

    steps = [
        ("1. Data Acquisition (Kaggle Public Dataset)\nDataset-SA.csv — 205.052 baris x 6 kolom (Product, Price, Rate, Review, Summary, Sentiment)", "#DCEBF7", "#1F4E79"),
        ("2. Data Cleaning — Hapus Missing Values (Section 3.1.4)\ndropna() menghapus 24.664 null Review & 11 null Summary -> 180.379 baris (Neutral = 8.807)", "#DCEBF7", "#1F4E79"),
        ("3. Class Balancing untuk Menghilangkan Bias (Figure 5)\nEksperimen 1 ('Review'): 8.807 x 3 kelas = 26.421  |  Eksperimen 2 ('Summary'): 3.000 x 3 kelas = 9.000", "#E8F5E9", "#2E7D32"),
        ("4. Text Pre-Processing (Section 3.2a — NLTK & Regex)\nLowercase -> Hapus URL/Hashtag/Simbol -> Tokenisasi -> Stop-words Removal -> PorterStemmer", "#FFF8E1", "#B26A00"),
        ("5. Ekstraksi Fitur Bag of Words (Section 3.2b — Scikit-Learn)\nCountVectorizer(max_features=1500) mengubah teks menjadi matriks frekuensi kata", "#FFF8E1", "#B26A00"),
        ("6. Train/Test Split (80:20 Stratified) & Pelatihan 3 Model ML (Section 3.2c)\nRandom Forest (n=100)   |   Naive Bayes (GaussianNB)   |   Support Vector Machine (LinearSVC)", "#F3E5F5", "#6A1B9A"),
        ("7. Evaluasi & Verifikasi Hasil vs Tabel 2 & 3 Jurnal\nReview: RF 0.690 (Paper 0.680), SVM 0.690 (0.682), NB 0.457 (0.449)\nSummary: RF 0.766 (Paper 0.767), SVM 0.754 (0.768), NB 0.452 (0.426)", "#E0F2F1", "#00695C"),
    ]

    y_top = 0.81
    h = 0.088
    gap = 0.032
    for i, (label, fc, ec) in enumerate(steps):
        y = y_top - i * (h + gap)
        _draw_box(ax, 0.07, y, 0.86, h, label, facecolor=fc, edgecolor=ec, fontsize=9.2, bold=(i in [0, 6]))
        if i < len(steps) - 1:
            _draw_arrow(ax, 0.5, y, 0.5, y - gap)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Versi Draw.io XML
    cells = [
        '<mxCell id="0" />',
        '<mxCell id="1" parent="0" />',
    ]
    y_px = 40
    for i, (label, fc, ec) in enumerate(steps):
        safe = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "&lt;br&gt;")
        cells.append(
            f'<mxCell id="r{i}" value="{safe}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={ec};fontSize=12;fontStyle=1;" vertex="1" parent="1">'
            f'<mxGeometry x="80" y="{y_px}" width="760" height="72" as="geometry" /></mxCell>'
        )
        if i > 0:
            cells.append(
                f'<mxCell id="re{i}" edge="1" parent="1" source="r{i-1}" target="r{i}" style="edgeStyle=orthogonalEdgeStyle;endArrow=classic;strokeWidth=2;">'
                '<mxGeometry relative="1" as="geometry" /></mxCell>'
            )
        y_px += 105

    xml = '<mxGraphModel><root>\n  ' + '\n  '.join(cells) + '\n</root></mxGraphModel>'
    out_drawio.write_text(xml, encoding="utf-8")


def build_adjust_flowchart(out_png: Path, out_drawio: Path) -> None:
    """Membuat flowchart PNG & Draw.io untuk alur Adjust Journal (Multimodal UMKM Surabaya Timur)."""
    fig, ax = plt.subplots(figsize=(12.5, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "ALUR ADJUST JOURNAL — Ekspansi Multimodal Big Data & AI UMKM Kuliner Surabaya Timur\n(Penambahan Dataset Gambar, Teks/Numerik, Dokumen + Metode, Library, & Framework Baru)",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color="#0B2E59",
    )

    # Baris 1: 3 Modalitas Input (Tambahan Dataset)
    _draw_box(
        ax, 0.03, 0.78, 0.29, 0.11,
        "A. DATASET GAMBAR (BARU)\n549 Foto Bersih (279.99 MB >= 200 MB)\n505 Google Maps + 44 Survei SWK\n+ 2.272 Foto Mentah (01_raw)",
        facecolor="#E3F2FD", edgecolor="#1565C0", fontsize=8.5, bold=True,
    )
    _draw_box(
        ax, 0.355, 0.78, 0.29, 0.11,
        "B. DATASET TEKS & NUMERIK (BARU)\n385 UMKM Surabaya Timur (6 Kec.)\n1.164 Ulasan Asli (en/id, tanpa translasi)\n+ Label Survei SWK & Flipkart Baseline",
        facecolor="#E8F5E9", edgecolor="#2E7D32", fontsize=8.5, bold=True,
    )
    _draw_box(
        ax, 0.68, 0.78, 0.29, 0.11,
        "C. DATASET DOKUMEN (BARU)\n3 Dokumen PDF Perizinan Penelitian\n(7 halaman diekstrak dengan pypdf)\n-> dokumen.parquet",
        facecolor="#FFF3E0", edgecolor="#EF6C00", fontsize=8.5, bold=True,
    )

    # Baris 2: Preprocessing & Big Data Framework Baru
    _draw_box(
        ax, 0.03, 0.61, 0.94, 0.10,
        "2. PREPROCESSING MULTIMODAL & BIG DATA STAGING (FRAMEWORK & LIBRARY BARU)\n"
        "Gambar: OpenCLIP (Filter Keep vs Reject) + OpenCV (Blur 824 Wajah) + Ultralytics YOLOv8 (Deteksi QRIS/POS/EDC/Ojol/Stan)\n"
        "Teks/Dokumen/Staging: Regex PII Masking + pypdf + PySpark SQL Staging (umkm_spark_staged.parquet) -> Google Drive BDA-2026-Iqbal",
        facecolor="#FFF8E1", edgecolor="#B26A00", fontsize=8.6,
    )
    _draw_arrow(ax, 0.175, 0.78, 0.175, 0.71)
    _draw_arrow(ax, 0.500, 0.78, 0.500, 0.71)
    _draw_arrow(ax, 0.825, 0.78, 0.825, 0.71)

    # Baris 3: Feature Engineering Multimodal
    _draw_box(
        ax, 0.08, 0.46, 0.84, 0.09,
        "3. EKSTRAKSI & FUSI FITUR MULTIMODAL (PENGGANTI BAG-OF-WORDS TUNGGAL)\n"
        "TF-IDF Ulasan + Indikator Visual YOLO + Fitur Numerik/Rating + Metadata Dokumen -> StandardScaler + PCA\n"
        "Feature Store: feature_store.parquet, embeddings_multimodal.npy / .npz / .parquet",
        facecolor="#EDE7F6", edgecolor="#4527A0", fontsize=8.8,
    )
    _draw_arrow(ax, 0.50, 0.61, 0.50, 0.55)

    # Baris 4: 3 Cabang Pemodelan (ML + DL)
    _draw_box(
        ax, 0.03, 0.27, 0.29, 0.12,
        "4A. UNSUPERVISED CLUSTERING (BARU)\nK-Means (k=3): Silhouette 0.4737\nPyTorch Autoencoder:\nLatent Silhouette 0.4834",
        facecolor="#F3E5F5", edgecolor="#6A1B9A", fontsize=8.4,
    )
    _draw_box(
        ax, 0.355, 0.27, 0.29, 0.12,
        "4B. CLASSIFICATION (GROUPKFOLD)\nNB: 0.8857 | SVM: 0.9766 | RF: 0.9740\nGradient Boosting (Baru): Acc 0.9792, F1 0.9533\nPyTorch MLP (DL Baru): F1 1.0000",
        facecolor="#E0F2F1", edgecolor="#00695C", fontsize=8.2, bold=True,
    )
    _draw_box(
        ax, 0.68, 0.27, 0.29, 0.12,
        "4C. REGRESSION & XAI (BARU)\nRF Regressor: R2 0.9768, RMSE 2.93\nPyTorch MLP Regressor: R2 0.9993\nSHAP Attribution & Cohen's Kappa",
        facecolor="#FCE4EC", edgecolor="#AD1457", fontsize=8.4,
    )
    _draw_arrow(ax, 0.25, 0.46, 0.175, 0.39)
    _draw_arrow(ax, 0.50, 0.46, 0.500, 0.39)
    _draw_arrow(ax, 0.75, 0.46, 0.825, 0.39)

    # Baris 5: Spatial BDA & AI Application
    _draw_box(
        ax, 0.08, 0.08, 0.84, 0.11,
        "5. BIG DATA ANALYTICS (SPASIAL) & AI APPLICATION (OUTPUT AKHIR)\n"
        "Agregasi 6 Kecamatan Surabaya Timur -> Digital Gap Index (Tertinggi: Gubeng 88.12, Terendah: Mulyorejo 76.03)\n"
        "Visualisasi Peta & Grafik (Matplotlib/Seaborn) -> Inference REST API (FastAPI) & Web Dashboard (Streamlit)",
        facecolor="#E8F5E9", edgecolor="#1B5E20", fontsize=8.8, bold=True,
    )
    _draw_arrow(ax, 0.175, 0.27, 0.30, 0.19)
    _draw_arrow(ax, 0.500, 0.27, 0.50, 0.19)
    _draw_arrow(ax, 0.825, 0.27, 0.70, 0.19)

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Versi Draw.io XML
    nodes = [
        ("a1", "A. Dataset Gambar (Baru): 549 foto bersih (279.99 MB >= 200 MB) + 2.272 foto mentah", 40, 40, 260, 80, "#E3F2FD", "#1565C0"),
        ("a2", "B. Dataset Teks & Numerik (Baru): 385 UMKM Surabaya Timur + 1.164 ulasan asli (en/id)", 330, 40, 260, 80, "#E8F5E9", "#2E7D32"),
        ("a3", "C. Dataset Dokumen (Baru): 3 PDF perizinan penelitian (pypdf)", 620, 40, 260, 80, "#FFF3E0", "#EF6C00"),
        ("a4", "2. Preprocessing Multimodal & PySpark Staging (OpenCLIP, OpenCV Blur, YOLOv8, Regex, pypdf, PySpark)", 40, 160, 840, 76, "#FFF8E1", "#B26A00"),
        ("a5", "3. Ekstraksi & Fusi Fitur Multimodal (TF-IDF + Visual YOLO + Tabular + StandardScaler & PCA -> Feature Store)", 80, 275, 760, 72, "#EDE7F6", "#4527A0"),
        ("a6", "4A. Unsupervised Clustering: K-Means (Sil 0.4737) & PyTorch Autoencoder (Sil 0.4834)", 40, 390, 260, 82, "#F3E5F5", "#6A1B9A"),
        ("a7", "4B. Classification GroupKFold: RF (0.9740), SVM (0.9766), Gradient Boosting (0.9792), PyTorch MLP (1.000)", 330, 390, 260, 82, "#E0F2F1", "#00695C"),
        ("a8", "4C. Regression & XAI: RF Regressor (R2 0.9768), PyTorch MLP Regressor (R2 0.9993), SHAP & Cohen Kappa", 620, 390, 260, 82, "#FCE4EC", "#AD1457"),
        ("a9", "5. Spatial BDA (Digital Gap Index 6 Kecamatan Surabaya Timur) -> FastAPI & Streamlit Dashboard", 80, 515, 760, 76, "#E8F5E9", "#1B5E20"),
    ]
    edges = [
        ("a1", "a4"), ("a2", "a4"), ("a3", "a4"),
        ("a4", "a5"),
        ("a5", "a6"), ("a5", "a7"), ("a5", "a8"),
        ("a6", "a9"), ("a7", "a9"), ("a8", "a9"),
    ]
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    for nid, txt, x, y, w, h, fc, ec in nodes:
        safe = txt.replace("&", "&amp;")
        cells.append(
            f'<mxCell id="{nid}" value="{safe}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={ec};fontSize=11;fontStyle=1;" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'
        )
    for idx, (s, t) in enumerate(edges):
        cells.append(
            f'<mxCell id="ae{idx}" edge="1" parent="1" source="{s}" target="{t}" style="edgeStyle=orthogonalEdgeStyle;endArrow=classic;strokeWidth=2;">'
            '<mxGeometry relative="1" as="geometry" /></mxCell>'
        )
    out_drawio.write_text('<mxGraphModel><root>\n  ' + '\n  '.join(cells) + '\n</root></mxGraphModel>', encoding="utf-8")


def build_search_summary_csv(out_csv: Path) -> None:
    """Membuat tabel ringkasan perbandingan: Jurnal Asli vs Repeat Journal vs Adjust Journal."""
    rows = [
        {
            "Aspek": "Judul / Fokus Penelitian",
            "Jurnal_Asli_BDCC_2024": "Sentiment Analysis of Product Reviews Using Machine Learning and Pre-Trained LLM (Ghatora et al., 2024)",
            "Tahap_Repeat_Journal": "Reproduksi persis eksperimen ML (Review & Summary) dari Ghatora et al. (2024)",
            "Tahap_Adjust_Journal": "Pemetaan & Klasifikasi Kematangan Digital UMKM Kuliner Surabaya Timur (Multimodal Big Data & AI)",
        },
        {
            "Aspek": "Jurnal & DOI",
            "Jurnal_Asli_BDCC_2024": "Big Data and Cognitive Computing (MDPI), Vol. 8(12), Art. 199 | DOI: 10.3390/bdcc8120199",
            "Tahap_Repeat_Journal": "Sama persis (PDF Open Access tersimpan di 01_Search_Journal/)",
            "Tahap_Adjust_Journal": "Ekspansi atas Limitation & Future Work Section 6 jurnal (multi-modal & real-world business case)",
        },
        {
            "Aspek": "Dataset & Modalitas",
            "Jurnal_Asli_BDCC_2024": "Teks & Numerik tunggal: Flipkart Dataset-SA.csv (205.052 baris, 6 kolom)",
            "Tahap_Repeat_Journal": "Persis Dataset-SA.csv (205.052 -> 180.379 bersih -> Review 8.807x3 & Summary 3.000x3)",
            "Tahap_Adjust_Journal": "3 Modalitas Baru: Gambar (549 foto bersih = 279.99 MB + 2.272 mentah), Teks/Numerik (385 UMKM + 1.164 ulasan en/id), Dokumen (3 PDF)",
        },
        {
            "Aspek": "Preprocessing & Ekstraksi Fitur",
            "Jurnal_Asli_BDCC_2024": "Drop null, balancing kelas, tokenisasi, normalisasi, stopwords, PorterStemmer, Bag of Words",
            "Tahap_Repeat_Journal": "Sama persis: Regex, NLTK PorterStemmer, CountVectorizer (Bag of Words)",
            "Tahap_Adjust_Journal": "PySpark Staging, OpenCLIP filter, OpenCV Face Blur, Ultralytics YOLOv8, pypdf, TF-IDF + PCA -> Feature Store",
        },
        {
            "Aspek": "Metode ML / DL / BDA",
            "Jurnal_Asli_BDCC_2024": "Random Forest, Naive Bayes, SVM (+ GPT-4 zero-shot)",
            "Tahap_Repeat_Journal": "Random Forest, Naive Bayes (GaussianNB), SVM (LinearSVC) — Split 80:20",
            "Tahap_Adjust_Journal": "Clustering (K-Means, PyTorch Autoencoder) + Classification (RF, SVM, NB, Gradient Boosting, PyTorch MLP) + Regression + Spatial BDA",
        },
        {
            "Aspek": "Library & Framework",
            "Jurnal_Asli_BDCC_2024": "Python, Pandas, NLTK, Scikit-Learn",
            "Tahap_Repeat_Journal": "Python, Pandas, NLTK, Scikit-Learn, Matplotlib, Seaborn",
            "Tahap_Adjust_Journal": "PySpark, PyTorch, OpenCLIP, OpenCV, Ultralytics YOLOv8, pypdf, Scikit-Learn, SHAP, FastAPI, Streamlit",
        },
        {
            "Aspek": "Hasil Utama (Metrik)",
            "Jurnal_Asli_BDCC_2024": "Review: RF 0.680, SVM 0.682, NB 0.449 | Summary: RF 0.767, SVM 0.768, NB 0.426",
            "Tahap_Repeat_Journal": "Review: RF 0.690, SVM 0.690, NB 0.457 | Summary: RF 0.766, SVM 0.754, NB 0.452",
            "Tahap_Adjust_Journal": "GroupKFold: RF Acc 0.9740 (F1 0.9402), GB Acc 0.9792 (F1 0.9533), PyTorch MLP 1.000; K-Means Sil 0.4737; RF Reg R2 0.9768; Gap Gubeng 88.12",
        },
    ]
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _cell(cell_type: str, lines: List[str]) -> dict:
    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else []),
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def _save_nb(cells: List[dict], out_path: Path) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated notebook: {out_path}")


def build_repeat_notebook(out_path: Path) -> None:
    """Notebook Repeat Journal sederhana bergaya AMFLearning (Google Colab Ready)."""
    cells = [
        _cell("markdown", [
            "# 01 — Repeat Journal: Ghatora et al. (2024)",
            "",
            "- **Judul Jurnal:** *Sentiment Analysis of Product Reviews Using Machine Learning and Pre-Trained LLM*",
            "- **Jurnal:** *Big Data and Cognitive Computing* (MDPI), Vol. 8, Issue 12, Art. 199 (2024)",
            "- **DOI:** https://doi.org/10.3390/bdcc8120199",
            "- **Dataset Asli:** Flipkart Product Reviews with Sentiment Dataset (`Dataset-SA.csv`, 205.052 baris)",
            "- **NIM:** `09040624091` — Muhammad Iqbal Rizanta",
            "",
            "Notebook ini dibuat sederhana (gaya tutorial Colab) dan mengikuti persis langkah Section 3.1–4.2 pada jurnal.",
        ]),
        _cell("code", [
            "# 1. Import Library Utama",
            "import re",
            "from pathlib import Path",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from nltk.stem import PorterStemmer",
            "from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.ensemble import RandomForestClassifier",
            "from sklearn.naive_bayes import GaussianNB",
            "from sklearn.svm import LinearSVC",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix",
            "",
            "sns.set_theme(style='whitegrid')",
            "ROOT = Path.cwd().resolve()",
            "if not (ROOT / 'data').exists() and (ROOT.parent.parent / 'data').exists():",
            "    ROOT = ROOT.parent.parent",
            "DATA_PATH = ROOT / 'data' / 'raw' / 'journal_repeat' / 'Dataset-SA.csv'",
            "print('Dataset Path:', DATA_PATH)",
        ]),
        _cell("code", [
            "# 2. Load Dataset Asli & Cek Distribusi Awal (Tabel 1 Jurnal)",
            "df = pd.read_csv(DATA_PATH)",
            "print('Ukuran Dataset Asli:', df.shape)",
            "display(df.head(5))",
            "print('\\nJumlah Missing Value per Kolom:')",
            "print(df.isnull().sum())",
            "print('\\nDistribusi Sentimen Awal (Tabel 1):')",
            "print(df['Sentiment'].value_counts())",
        ]),
        _cell("code", [
            "# 3. Hapus Baris Null sesuai Section 3.1.4 Jurnal",
            "df_clean = df.dropna(subset=['Review', 'Summary', 'Sentiment']).reset_index(drop=True)",
            "print('Ukuran Setelah Hapus Null:', df_clean.shape)",
            "counts = df_clean['Sentiment'].value_counts()",
            "print(counts)",
            "",
            "plt.figure(figsize=(6.5, 3.6))",
            "sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette='Set2', legend=False)",
            "plt.title('Distribusi Kelas Sentimen Setelah Hapus Null (Neutral = 8.807)')",
            "plt.ylabel('Jumlah Baris')",
            "for i, v in enumerate(counts.values):",
            "    plt.text(i, v + 2000, f'{v:,}', ha='center', fontweight='bold')",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        _cell("code", [
            "# 4. Fungsi Cleaning Teks (Section 3.2a) & Fungsi Eksperimen (Section 3.2b-3.2c)",
            "stemmer = PorterStemmer()",
            "stop_words = set(ENGLISH_STOP_WORDS)",
            "",
            "def clean_text(text):",
            "    text = re.sub(r'http\\S+|www\\S+|#\\w+|<.*?>', ' ', str(text).lower())",
            "    text = re.sub(r'[^a-z\\s]', ' ', text)",
            "    return ' '.join(stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 1)",
            "",
            "def repeat_experiment(df_in, col_name, n_per_class):",
            "    # Class balancing sesuai jurnal",
            "    df_bal = pd.concat([g.sample(n=n_per_class, random_state=42) for _, g in df_in.groupby('Sentiment')], ignore_index=True)",
            "    # Bag of Words (CountVectorizer)",
            "    X = CountVectorizer(max_features=1500).fit_transform(df_bal[col_name].apply(clean_text)).toarray()",
            "    y = df_bal['Sentiment'].to_numpy(dtype=str)",
            "    # Train/Test Split 80:20",
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)",
            "",
            "    models = {",
            "        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),",
            "        'Naive Bayes': GaussianNB(),",
            "        'SVM': LinearSVC(random_state=42, dual='auto', max_iter=2000),",
            "    }",
            "    rows, cms = [], {}",
            "    labels = ['negative', 'neutral', 'positive']",
            "    for name, clf in models.items():",
            "        clf.fit(X_train, y_train)",
            "        y_pred = clf.predict(X_test)",
            "        rows.append({",
            "            'Experiment': col_name,",
            "            'Model': name,",
            "            'Repeat_Accuracy': round(accuracy_score(y_test, y_pred), 3),",
            "            'Repeat_Precision': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 3),",
            "            'Repeat_Recall': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 3),",
            "            'Repeat_F1': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 3),",
            "        })",
            "        cms[name] = confusion_matrix(y_test, y_pred, labels=labels)",
            "    return pd.DataFrame(rows), cms",
        ]),
        _cell("code", [
            "# 5. Jalankan Eksperimen 1 (Review: 8.807 x 3) & Eksperimen 2 (Summary: 3.000 x 3)",
            "res_review, cms_review = repeat_experiment(df_clean, 'Review', 8807)",
            "res_summary, cms_summary = repeat_experiment(df_clean, 'Summary', 3000)",
            "df_repeat = pd.concat([res_review, res_summary], ignore_index=True)",
            "",
            "# Bandingkan langsung dengan angka Tabel 2 & Tabel 3 Jurnal Ghatora et al. (2024)",
            "df_paper = pd.DataFrame([",
            "    {'Experiment': 'Review',  'Model': 'Random Forest', 'Paper_Accuracy': 0.680, 'Paper_F1': 0.672},",
            "    {'Experiment': 'Review',  'Model': 'Naive Bayes',   'Paper_Accuracy': 0.449, 'Paper_F1': 0.347},",
            "    {'Experiment': 'Review',  'Model': 'SVM',           'Paper_Accuracy': 0.682, 'Paper_F1': 0.673},",
            "    {'Experiment': 'Summary', 'Model': 'Random Forest', 'Paper_Accuracy': 0.767, 'Paper_F1': 0.766},",
            "    {'Experiment': 'Summary', 'Model': 'Naive Bayes',   'Paper_Accuracy': 0.426, 'Paper_F1': 0.380},",
            "    {'Experiment': 'Summary', 'Model': 'SVM',           'Paper_Accuracy': 0.768, 'Paper_F1': 0.768},",
            "])",
            "df_compare = df_paper.merge(df_repeat, on=['Experiment', 'Model'])",
            "df_compare['Selisih_Acc'] = (df_compare['Repeat_Accuracy'] - df_compare['Paper_Accuracy']).round(3)",
            "display(df_compare)",
        ]),
        _cell("code", [
            "# 6. Visualisasi Perbandingan Hasil Repeat vs Hasil Jurnal",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))",
            "for ax, exp in zip(axes, ['Review', 'Summary']):",
            "    sub = df_compare[df_compare['Experiment'] == exp]",
            "    melted = sub.melt(id_vars='Model', value_vars=['Paper_Accuracy', 'Repeat_Accuracy', 'Paper_F1', 'Repeat_F1'], var_name='Metrik', value_name='Skor')",
            "    sns.barplot(data=melted, x='Model', y='Skor', hue='Metrik', palette='Set2', ax=ax)",
            "    ax.set_title(f'Repeat vs Jurnal — Fitur {exp}')",
            "    ax.set_ylim(0, 1.0)",
            "plt.tight_layout()",
            "plt.show()",
        ]),
    ]
    _save_nb(cells, out_path)


def build_adjust_notebook(out_path: Path) -> None:
    """Notebook Adjust Journal sederhana bergaya AMFLearning (Google Colab Ready)."""
    cells = [
        _cell("markdown", [
            "# 02 — Adjust Journal: Ekspansi Multimodal UMKM Kuliner Surabaya Timur",
            "",
            "Melanjutkan **Repeat Journal** (Ghatora et al., 2024 — teks tunggal Flipkart dengan Bag-of-Words & ML dasar), tahap **Adjust Journal** ini menambahkan:",
            "1. **3 Modalitas Dataset Asli Surabaya Timur:**",
            "   - **Gambar (`279.99 MB >= 200 MB`):** 549 foto bersih (`505` Google Maps + `44` Survei Lapangan SWK) + 2.272 foto mentah",
            "   - **Teks & Numerik:** 385 UMKM Surabaya Timur (6 kecamatan) + 1.164 ulasan bahasa asli (`en`/`id`)",
            "   - **Dokumen:** 3 dokumen PDF perizinan penelitian (`pypdf`)",
            "2. **Metode, Library, & Framework Baru:** `PySpark`, `OpenCLIP`, `OpenCV`, `Ultralytics YOLOv8`, `TF-IDF + PCA`, `K-Means`, `Gradient Boosting`, `PyTorch Autoencoder & MLP`, `GroupKFold`, `Cohen's Kappa`, & **Spatial BDA (Digital Gap Index)**.",
        ]),
        _cell("code", [
            "# 1. Import Library Utama",
            "import json",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.cluster import KMeans",
            "from sklearn.model_selection import GroupKFold",
            "from sklearn.naive_bayes import GaussianNB",
            "from sklearn.svm import LinearSVC",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor",
            "from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, silhouette_score, r2_score, mean_squared_error, confusion_matrix",
            "",
            "sns.set_theme(style='whitegrid')",
            "ROOT = Path.cwd().resolve()",
            "if not (ROOT / 'data').exists() and (ROOT.parent.parent / 'data').exists():",
            "    ROOT = ROOT.parent.parent",
            "if str(ROOT) not in sys.path:",
            "    sys.path.insert(0, str(ROOT))",
            "from model.training.ml_pipeline import FEATURE_COLS, LABEL_NAMES, derive_target_class, prepare_feature_frame",
        ]),
        _cell("code", [
            "# 2. Muat 3 Modalitas Dataset Tambahan (Gambar, Teks/Numerik, Dokumen)",
            "df_umkm = pd.read_parquet(ROOT / 'data' / 'clean' / 'umkm.parquet')",
            "df_ulasan = pd.read_parquet(ROOT / 'data' / 'clean' / 'ulasan.parquet')",
            "df_foto = pd.read_parquet(ROOT / 'data' / 'clean' / 'foto.parquet')",
            "df_dok = pd.read_parquet(ROOT / 'data' / 'clean' / 'dokumen.parquet')",
            "photo_mb = round(sum(f.stat().st_size for f in (ROOT / 'data' / 'clean' / 'photos').rglob('*') if f.is_file()) / (1024**2), 2)",
            "",
            "ringkasan_modalitas = pd.DataFrame([",
            "    {'Modalitas': '1. Gambar (Citra Bersih)', 'Jumlah': f'{len(df_foto)} foto ({photo_mb} MB)', 'Library/Proses': 'OpenCLIP filter + OpenCV Blur + YOLOv8'},",
            "    {'Modalitas': '2. Teks & Numerik UMKM', 'Jumlah': f'{len(df_umkm)} UMKM, {len(df_ulasan)} ulasan', 'Library/Proses': 'Google Places API + PySpark + TF-IDF'},",
            "    {'Modalitas': '3. Dokumen PDF', 'Jumlah': f'{len(df_dok)} PDF ({int(df_dok[\"pages\"].sum())} hal.)', 'Library/Proses': 'pypdf text & metadata extraction'},",
            "])",
            "display(ringkasan_modalitas)",
        ]),
        _cell("code", [
            "# 3. Penambahan Metode Unsupervised Clustering (K-Means) & Supervised Classification (GroupKFold)",
            "df_feat = prepare_feature_frame(pd.read_parquet(ROOT / 'data' / 'gold' / 'feature_store.parquet'))",
            "X = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values.astype(np.float32))",
            "y = derive_target_class(df_feat)",
            "y_reg = df_feat['skor_kematangan_kontinu'].values.astype(np.float32)",
            "groups = df_feat['kecamatan'].to_numpy(dtype=str)",
            "",
            "# Unsupervised Clustering (Metode Tambahan)",
            "km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)",
            "print(f'K-Means Clustering (k=3) Silhouette Score : {silhouette_score(X, km.labels_):.4f}')",
            "",
            "# Supervised Classification dengan GroupKFold (5 Kecamatan)",
            "gkf = GroupKFold(n_splits=5)",
            "models = {",
            "    'Naive Bayes (Baseline Jurnal)': lambda: GaussianNB(),",
            "    'SVM (Baseline Jurnal)': lambda: LinearSVC(random_state=42, dual='auto', max_iter=3000),",
            "    'Random Forest (Adjusted)': lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),",
            "    'Gradient Boosting (Metode Baru)': lambda: GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),",
            "}",
            "rows, best_cm = [], None",
            "for name, make_clf in models.items():",
            "    yt, yp = [], []",
            "    for tr, te in gkf.split(X, y, groups):",
            "        clf = make_clf().fit(X[tr], y[tr])",
            "        yt.extend(y[te]); yp.extend(clf.predict(X[te]))",
            "    rows.append({",
            "        'Model': name,",
            "        'Accuracy': round(accuracy_score(yt, yp), 4),",
            "        'Macro_F1': round(f1_score(yt, yp, average='macro'), 4),",
            "        'Cohens_Kappa': round(cohen_kappa_score(yt, yp), 4),",
            "    })",
            "    if 'Gradient Boosting' in name:",
            "        best_cm = confusion_matrix(yt, yp)",
            "",
            "df_adj = pd.DataFrame(rows)",
            "display(df_adj)",
        ]),
        _cell("code", [
            "# 4. Penambahan Metode Regression & Perbandingan Repeat Journal vs Adjust Journal",
            "yr_t, yr_p = [], []",
            "for tr, te in gkf.split(X, y_reg, groups):",
            "    reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(X[tr], y_reg[tr])",
            "    yr_t.extend(y_reg[te]); yr_p.extend(reg.predict(X[te]))",
            "print(f'Random Forest Regressor (GroupKFold) -> R2: {r2_score(yr_t, yr_p):.4f} | RMSE: {np.sqrt(mean_squared_error(yr_t, yr_p)):.4f}')",
            "",
            "df_comp = pd.read_csv(ROOT / 'peta-digital-umkm' / 'reports' / 'journal_adjust' / 'adjust_vs_repeat_comparison.csv')",
            "display(df_comp)",
            "",
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
            "melted = df_comp.melt(id_vars='Tahap', value_vars=['Accuracy', 'Macro_F1'], var_name='Metrik', value_name='Skor')",
            "sns.barplot(data=melted, x='Tahap', y='Skor', hue='Metrik', palette='viridis', ax=axes[0])",
            "axes[0].set_title('Perbandingan Repeat Journal vs Adjust Journal')",
            "axes[0].set_ylim(0, 1.1)",
            "axes[0].tick_params(axis='x', rotation=15)",
            "",
            "sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens', xticklabels=list(LABEL_NAMES.values()), yticklabels=list(LABEL_NAMES.values()), ax=axes[1])",
            "axes[1].set_title('Confusion Matrix — Gradient Boosting (Adjust Journal)')",
            "axes[1].set_xlabel('Prediksi'); axes[1].set_ylabel('Aktual')",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        _cell("code", [
            "# 5. Penambahan Spatial Big Data Analytics (Digital Gap Index 6 Kecamatan Surabaya Timur)",
            "df_gap = pd.read_csv(ROOT / 'peta-digital-umkm' / 'reports' / 'bda' / 'kecamatan_digital_gap.csv')",
            "display(df_gap[['kecamatan', 'total_umkm', 'qris_rate', 'ojol_rate', 'avg_maturity_score', 'digital_gap_index', 'rekomendasi_intervensi']])",
            "",
            "plt.figure(figsize=(8.5, 4.2))",
            "sns.barplot(data=df_gap, x='digital_gap_index', y='kecamatan', hue='kecamatan', palette='YlOrRd_r', legend=False)",
            "plt.title('Digital Gap Index per Kecamatan — Hasil Spatial BDA (Adjust Journal)')",
            "plt.xlabel('Digital Gap Index (0 - 100)')",
            "for i, val in enumerate(df_gap['digital_gap_index']):",
            "    plt.text(val + 0.5, i, f'{val:.2f}', va='center', fontweight='bold')",
            "plt.tight_layout()",
            "plt.show()",
        ]),
    ]
    _save_nb(cells, out_path)


def main() -> None:
    REPEAT_DIR.mkdir(parents=True, exist_ok=True)
    ADJUST_DIR.mkdir(parents=True, exist_ok=True)

    build_repeat_flowchart(
        REPEAT_DIR / "Alur_Repeat_Journal_BDCC2024.png",
        REPEAT_DIR / "Alur_Repeat_Journal_BDCC2024.drawio",
    )
    build_adjust_flowchart(
        ADJUST_DIR / "Alur_Adjust_Journal_Multimodal_UMKM.png",
        ADJUST_DIR / "Alur_Adjust_Journal_Multimodal_UMKM.drawio",
    )
    build_search_summary_csv(REPEAT_DIR / "ringkasan_jurnal_dan_perbandingan.csv")
    build_repeat_notebook(NB_DIR / "01_repeat_journal_bdcc2024_09040624091.ipynb")
    build_adjust_notebook(NB_DIR / "02_adjust_journal_multimodal_umkm_09040624091.ipynb")
    print("Seluruh flowchart (PNG + Draw.io), tabel ringkasan, dan 2 notebook berhasil dibuat.")


if __name__ == "__main__":
    main()
