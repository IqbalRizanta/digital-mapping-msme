"""
REPEAT JOURNAL — Ghatora et al. (2024)
Judul  : Sentiment Analysis of Product Reviews Using Machine Learning and Pre-Trained LLM
Jurnal : Big Data and Cognitive Computing (MDPI), Vol. 8, Issue 12, Art. 199 (2024)
DOI    : https://doi.org/10.3390/bdcc8120199
Dataset: Flipkart Product Reviews with Sentiment Dataset (Dataset-SA.csv, 205.052 baris)

Alur Repeat Persis Sesuai Jurnal (Section 3.1 - 4.2):
1. Load Dataset (205.052 baris, 6 kolom)
2. Hapus Missing Values (dropna -> 8.807 kelas neutral)
3. Balancing Kelas (Review: 8.807 x 3 = 26.421; Summary: 3.000 x 3 = 9.000)
4. Text Cleaning (normalisasi, tokenisasi, hapus stopwords, PorterStemmer)
5. Ekstraksi Fitur Bag of Words (CountVectorizer)
6. Split Train/Test 80:20 & Training 3 Model ML (Random Forest, Naive Bayes, SVM)
7. Evaluasi (Accuracy, Macro Precision, Macro Recall, Macro F1) & Perbandingan vs Tabel 2 & 3 Jurnal
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_CSV = ROOT_DIR / "data" / "raw" / "journal_repeat" / "Dataset-SA.csv"
OUT_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "journal_repeat"

# Hasil rujukan langsung dari Tabel 2 & Tabel 3 Jurnal Ghatora et al. (2024)
PAPER_BENCHMARK = {
    "Review": {
        "Random Forest": {"Accuracy": 0.680, "Precision": 0.677, "Recall": 0.680, "F1": 0.672},
        "Naive Bayes": {"Accuracy": 0.449, "Precision": 0.471, "Recall": 0.446, "F1": 0.347},
        "SVM": {"Accuracy": 0.682, "Precision": 0.678, "Recall": 0.682, "F1": 0.673},
    },
    "Summary": {
        "Random Forest": {"Accuracy": 0.767, "Precision": 0.766, "Recall": 0.768, "F1": 0.766},
        "Naive Bayes": {"Accuracy": 0.426, "Precision": 0.508, "Recall": 0.421, "F1": 0.380},
        "SVM": {"Accuracy": 0.768, "Precision": 0.768, "Recall": 0.769, "F1": 0.768},
    },
}

stemmer = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Cleaning sesuai Section 3.2(a): normalisasi, tokenisasi, hapus stopwords, stemming."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|#\w+|<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def run_experiment(df_clean: pd.DataFrame, feature_col: str, n_per_class: int) -> tuple[pd.DataFrame, dict]:
    """Menjalankan 1 eksperimen penuh (Review atau Summary) mengikuti alur jurnal."""
    # Balancing kelas agar tidak bias
    df_bal = pd.concat(
        [g.sample(n=n_per_class, random_state=42) for _, g in df_clean.groupby("Sentiment")],
        ignore_index=True,
    )

    # Cleaning teks dan vektorisasi Bag of Words
    corpus = df_bal[feature_col].apply(clean_text)
    vectorizer = CountVectorizer(max_features=1500)
    X = vectorizer.fit_transform(corpus).toarray()
    y = df_bal["Sentiment"].to_numpy(dtype=str)

    # Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "SVM": LinearSVC(random_state=42, dual="auto", max_iter=2000),
    }

    rows = []
    cms = {}
    labels = ["negative", "neutral", "positive"]

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = round(float(accuracy_score(y_test, y_pred)), 3)
        prec = round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 3)
        rec = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 3)
        f1 = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 3)

        ref = PAPER_BENCHMARK[feature_col][name]
        rows.append(
            {
                "Experiment": feature_col,
                "Samples_Per_Class": n_per_class,
                "Model": name,
                "Paper_Accuracy": ref["Accuracy"],
                "Repeat_Accuracy": acc,
                "Paper_Precision": ref["Precision"],
                "Repeat_Precision": prec,
                "Paper_Recall": ref["Recall"],
                "Repeat_Recall": rec,
                "Paper_F1": ref["F1"],
                "Repeat_F1": f1,
                "Diff_Accuracy": round(acc - ref["Accuracy"], 3),
                "Diff_F1": round(f1 - ref["F1"], 3),
            }
        )
        cms[name] = confusion_matrix(y_test, y_pred, labels=labels)

    return pd.DataFrame(rows), cms


def plot_comparison(df_res: pd.DataFrame, out_path: Path) -> None:
    """Grafik perbandingan Accuracy & Macro F1: Hasil Jurnal vs Hasil Repeat."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ax, exp_name in zip(axes, ["Review", "Summary"]):
        sub = df_res[df_res["Experiment"] == exp_name]
        melted = sub.melt(
            id_vars="Model",
            value_vars=["Paper_Accuracy", "Repeat_Accuracy", "Paper_F1", "Repeat_F1"],
            var_name="Metrik",
            value_name="Skor",
        )
        sns.barplot(data=melted, x="Model", y="Skor", hue="Metrik", palette="Set2", ax=ax)
        ax.set_title(f"Repeat vs Jurnal — Fitur '{exp_name}'")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Skor (0 - 1)")
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2, h + 0.015), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrices(cms: dict, title_prefix: str, out_path: Path) -> None:
    """Visualisasi Confusion Matrix untuk Random Forest, Naive Bayes, dan SVM."""
    labels = ["negative", "neutral", "positive"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{title_prefix} — {name}")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset asli jurnal
    df_raw = pd.read_csv(DATA_CSV)
    print(f"Dataset asli: {df_raw.shape[0]:,} baris x {df_raw.shape[1]} kolom")
    print("Distribusi sentimen awal:", df_raw["Sentiment"].value_counts().to_dict())

    # 2. Hapus baris kosong (null) sesuai Section 3.1.4
    df_clean = df_raw.dropna(subset=["Review", "Summary", "Sentiment"]).reset_index(drop=True)
    print(f"Setelah hapus null: {df_clean.shape[0]:,} baris")
    print("Distribusi setelah hapus null:", df_clean["Sentiment"].value_counts().to_dict())

    # 3. Jalankan Eksperimen 1 (Review: 8.807/kelas) & Eksperimen 2 (Summary: 3.000/kelas)
    print("\nMenjalankan Eksperimen 1 (Fitur Review, 8.807 per kelas)...")
    res_review, cms_review = run_experiment(df_clean, "Review", n_per_class=8807)

    print("Menjalankan Eksperimen 2 (Fitur Summary, 3.000 per kelas)...")
    res_summary, cms_summary = run_experiment(df_clean, "Summary", n_per_class=3000)

    df_all = pd.concat([res_review, res_summary], ignore_index=True)
    print("\n=== HASIL PERBANDINGAN REPEAT VS JURNAL GHATORA ET AL. (2024) ===")
    print(df_all.to_string(index=False))

    # Simpan hasil CSV, JSON, dan grafik
    df_all.to_csv(OUT_DIR / "repeat_vs_paper_metrics.csv", index=False)
    summary_json = {
        "journal": {
            "title": "Sentiment Analysis of Product Reviews Using Machine Learning and Pre-Trained LLM",
            "authors": "Ghatora, P.S.; Hosseini, S.E.; Pervez, S.; Iqbal, M.J.; Shaukat, N.",
            "venue": "Big Data and Cognitive Computing (MDPI), 2024, 8(12), 199",
            "doi": "https://doi.org/10.3390/bdcc8120199",
            "dataset": "Flipkart Product Reviews with Sentiment Dataset (Dataset-SA.csv, 205,052 rows)",
        },
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_clean)),
        "results": df_all.to_dict(orient="records"),
    }
    (OUT_DIR / "repeat_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    plot_comparison(df_all, OUT_DIR / "01_repeat_vs_paper_comparison.png")
    plot_confusion_matrices(cms_review, "Review", OUT_DIR / "02_confusion_matrix_review.png")
    plot_confusion_matrices(cms_summary, "Summary", OUT_DIR / "03_confusion_matrix_summary.png")
    print(f"\nSeluruh hasil Repeat Journal tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
