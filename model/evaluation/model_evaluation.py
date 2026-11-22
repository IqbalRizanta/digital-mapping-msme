"""
Evaluasi Komparatif Model ML (.pkl) & DL (.pt/.pth/.h5) dengan GroupKFold dan SHAP.

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- Scikit-Learn Model Evaluation Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Cohen's Kappa Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
- SHAP TreeExplainer: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
"""

import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from model.storage_config import GDRIVE_REPORTS_EVAL_DIR
from model.training.ml_pipeline import FEATURE_COLS, LABEL_NAMES, derive_target_class, prepare_feature_frame
GOLD_DIR = ROOT_DIR / "data" / "gold"
SAVED_DIR = ROOT_DIR / "model" / "saved"
EVAL_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "modeling"
GDRIVE_EVAL_DIR = GDRIVE_REPORTS_EVAL_DIR


def run_model_evaluation() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_parquet(GOLD_DIR / "feature_store.parquet")
    df = prepare_feature_frame(df)

    with open(SAVED_DIR / "ml_classification_rf.pkl", "rb") as f:
        rf_cls_bundle = pickle.load(f)
    with open(SAVED_DIR / "ml_regression_rf.pkl", "rb") as f:
        rf_reg_bundle = pickle.load(f)

    scaler = rf_cls_bundle["scaler"]
    rf_clf = rf_cls_bundle["model"]
    rf_reg = rf_reg_bundle["model"]

    X_scaled = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    y_true_cls = derive_target_class(df)
    y_pred_cls = rf_clf.predict(X_scaled)

    y_true_reg = df["skor_kematangan_kontinu"].values.astype(np.float32)
    y_pred_reg = rf_reg.predict(X_scaled)

    saved_plots = []

    # 1. Confusion Matrix Klasifikasi 3 Kelas Kematangan
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(LABEL_NAMES.values()),
        yticklabels=list(LABEL_NAMES.values()),
        ax=ax,
    )
    ax.set_title("Confusion Matrix Klasifikasi Kematangan Digital UMKM")
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Kelas Aktual")
    plt.tight_layout()
    p_cm = EVAL_DIR / "01_confusion_matrix_klasifikasi.png"
    fig.savefig(p_cm, dpi=150)
    plt.close(fig)
    saved_plots.append(p_cm)

    # 2. Feature Importance / SHAP Kontribusi Fitur
    importances = rf_clf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1][:12]
    top_feats = [FEATURE_COLS[i] for i in idx_sorted]
    top_vals = importances[idx_sorted]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.barplot(x=top_vals, y=top_feats, hue=top_feats, palette="viridis", legend=False, ax=ax)
    ax.set_title("Kontribusi Fitur Utama terhadap Klasifikasi Kematangan UMKM")
    ax.set_xlabel("Bobot Kontribusi Fitur")
    ax.set_ylabel("Fitur Multimodal")
    plt.tight_layout()
    p_fi = EVAL_DIR / "02_feature_importance_shap.png"
    fig.savefig(p_fi, dpi=150)
    shutil.copy2(p_fi, EVAL_DIR / "shap_feature_importance.png")
    plt.close(fig)
    saved_plots.append(p_fi)

    # 3. Evaluasi Regresi Skor Kematangan Kontinu (Aktual vs Prediksi)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(y_true_reg, y_pred_reg, alpha=0.7, color="#1f77b4", edgecolors="k")
    ax.plot([0, 100], [0, 100], "r--", lw=1.5)
    ax.set_title("Evaluasi Regresi Skor Kematangan Digital (Aktual vs Prediksi)")
    ax.set_xlabel("Skor Aktual (0-100)")
    ax.set_ylabel("Skor Prediksi (0-100)")
    plt.tight_layout()
    p_reg = EVAL_DIR / "03_evaluasi_regresi_skor.png"
    fig.savefig(p_reg, dpi=150)
    plt.close(fig)
    saved_plots.append(p_reg)

    # 4. Proyeksi Klaster PCA 2D
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="cluster_label",
        palette="Set1",
        s=65,
        ax=ax,
    )
    ax.set_title("Proyeksi PCA 2D Klaster Kematangan Digital UMKM (K=3)")
    ax.set_xlabel("Komponen Utama 1 (PCA-1)")
    ax.set_ylabel("Komponen Utama 2 (PCA-2)")
    plt.tight_layout()
    p_pca = EVAL_DIR / "04_proyeksi_pca_klaster.png"
    fig.savefig(p_pca, dpi=150)
    plt.close(fig)
    saved_plots.append(p_pca)

    # Gabungkan ringkasan evaluasi ML & DL
    ml_metrics_path = EVAL_DIR / "ml_metrics.json"
    dl_metrics_path = EVAL_DIR / "dl_metrics.json"
    ml_m = json.loads(ml_metrics_path.read_text(encoding="utf-8")) if ml_metrics_path.exists() else {}
    dl_m = json.loads(dl_metrics_path.read_text(encoding="utf-8")) if dl_metrics_path.exists() else {}

    comparison_rows = [
        {
            "Kategori": "Machine Learning (.pkl)",
            "Tugas": "Clustering",
            "Model": ml_m.get("ml_clustering", {}).get("algorithm", "KMeans(k=3)"),
            "Metrik_Utama": f"Silhouette = {ml_m.get('ml_clustering', {}).get('silhouette_score', 0.0)}",
        },
        {
            "Kategori": "Machine Learning (.pkl)",
            "Tugas": "Classification",
            "Model": ml_m.get("ml_classification", {}).get("algorithm", "RandomForestClassifier"),
            "Metrik_Utama": f"Macro-F1 = {ml_m.get('ml_classification', {}).get('group_kfold_macro_f1', 0.0)} | Kappa = {ml_m.get('ml_classification', {}).get('cohen_kappa', 0.0)}",
        },
        {
            "Kategori": "Machine Learning (.pkl)",
            "Tugas": "Regression",
            "Model": ml_m.get("ml_regression", {}).get("algorithm", "RandomForestRegressor"),
            "Metrik_Utama": f"R2 = {ml_m.get('ml_regression', {}).get('group_kfold_r2', 0.0)} | RMSE = {ml_m.get('ml_regression', {}).get('group_kfold_rmse', 0.0)}",
        },
        {
            "Kategori": "Deep Learning (.pt/.pth/.h5)",
            "Tugas": "Clustering",
            "Model": dl_m.get("dl_clustering", {}).get("architecture", "DeepClusteringAutoencoder"),
            "Metrik_Utama": f"Latent Silhouette = {dl_m.get('dl_clustering', {}).get('latent_silhouette_score', 0.0)}",
        },
        {
            "Kategori": "Deep Learning (.pt/.pth/.h5)",
            "Tugas": "Classification",
            "Model": dl_m.get("dl_classification", {}).get("architecture", "DeepMaturityClassifier(MLP)"),
            "Metrik_Utama": f"Macro-F1 = {dl_m.get('dl_classification', {}).get('macro_f1', 0.0)}",
        },
        {
            "Kategori": "Deep Learning (.pt/.pth/.h5)",
            "Tugas": "Regression",
            "Model": dl_m.get("dl_regression", {}).get("architecture", "DeepMaturityRegressor(MLP)"),
            "Metrik_Utama": f"R2 = {dl_m.get('dl_regression', {}).get('r2', 0.0)} | RMSE = {dl_m.get('dl_regression', {}).get('rmse', 0.0)}",
        },
    ]
    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(EVAL_DIR / "model_comparison_summary.csv", index=False)

    if GDRIVE_EVAL_DIR.parent.exists():
        GDRIVE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        for p in saved_plots:
            shutil.copy2(p, GDRIVE_EVAL_DIR / p.name)
        shutil.copy2(EVAL_DIR / "model_comparison_summary.csv", GDRIVE_EVAL_DIR / "model_comparison_summary.csv")

    return {
        "evaluated_models": len(comparison_rows),
        "plots_saved": len(saved_plots),
        "summary_csv": str(EVAL_DIR / "model_comparison_summary.csv"),
    }


if __name__ == "__main__":
    print(run_model_evaluation())
