"""
Pemodelan Machine Learning: Clustering, Classification, dan Regression (.pkl).

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- Scikit-Learn KMeans Clustering: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Scikit-Learn Random Forest (Classifier & Regressor): https://scikit-learn.org/stable/modules/ensemble.html#forest
- Scikit-Learn GroupKFold Cross-Validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- Python Pickle Serialization: https://docs.python.org/3/library/pickle.html
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
GOLD_DIR = ROOT_DIR / "data" / "gold"
SAVED_DIR = ROOT_DIR / "model" / "saved"
REPORTS_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "modeling"

FEATURE_COLS = [
    "rating",
    "jumlah_ulasan",
    "indikator_qris",
    "indikator_pos",
    "indikator_edc",
    "indikator_ojol",
    "review_count",
    "avg_word_count",
    "text_qris",
    "text_ojol_alias",
    "text_pos",
    "text_edc",
    "photo_count",
    "vis_qris",
    "vis_online",
    "vis_pos",
    "vis_edc",
]

LABEL_NAMES = {0: "Konvensional", 1: "Berkembang", 2: "Maju"}


def derive_target_class(df: pd.DataFrame) -> np.ndarray:
    n_ind = (
        df["indikator_qris"].astype(int)
        + df["indikator_ojol"].astype(int)
        + df["indikator_pos"].astype(int)
        + df["indikator_edc"].astype(int)
    )
    y = np.zeros(len(df), dtype=int)
    y[n_ind == 1] = 1
    y[n_ind >= 2] = 2
    return y


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "text_ojol_alias" not in df.columns:
        df["text_ojol_alias"] = df.get("text_online", 0.0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def train_and_save_ml_models() -> dict:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(GOLD_DIR / "feature_store.parquet")
    df = prepare_feature_frame(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_cls = derive_target_class(df)
    y_reg = df["skor_kematangan_kontinu"].values.astype(np.float32)
    groups = df["kecamatan"].astype(str).values

    # 1. ML CLUSTERING (K-Means, K=3)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    raw_clusters = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, raw_clusters))

    cluster_means = {
        c: float(y_reg[raw_clusters == c].mean()) for c in np.unique(raw_clusters)
    }
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    cluster_map = {old_c: new_c for new_c, old_c in enumerate(sorted_clusters)}
    aligned_clusters = np.array([cluster_map[c] for c in raw_clusters], dtype=int)

    df["cluster_kmeans"] = aligned_clusters
    df["cluster_label"] = [LABEL_NAMES[i] for i in aligned_clusters]
    df["kelas_kematangan"] = [LABEL_NAMES[i] for i in y_cls]
    df.to_parquet(GOLD_DIR / "feature_store.parquet", index=False)

    with open(SAVED_DIR / "ml_clustering_kmeans.pkl", "wb") as f:
        pickle.dump(
            {
                "model": kmeans,
                "scaler": scaler,
                "cluster_map": cluster_map,
                "feature_cols": FEATURE_COLS,
            },
            f,
        )

    # 2. ML CLASSIFICATION (Random Forest & XGBoost dengan GroupKFold)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(df), dtype=int)
    for train_idx, val_idx in gkf.split(X_scaled, y_cls, groups=groups):
        clf_cv = RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42
        )
        clf_cv.fit(X_scaled[train_idx], y_cls[train_idx])
        oof_preds[val_idx] = clf_cv.predict(X_scaled[val_idx])

    cls_acc = float(accuracy_score(y_cls, oof_preds))
    cls_f1 = float(f1_score(y_cls, oof_preds, average="macro"))
    cls_kappa = float(cohen_kappa_score(y_cls, oof_preds))

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf_clf.fit(X_scaled, y_cls)

    with open(SAVED_DIR / "ml_classification_rf.pkl", "wb") as f:
        pickle.dump(
            {
                "model": rf_clf,
                "scaler": scaler,
                "feature_cols": FEATURE_COLS,
                "label_names": LABEL_NAMES,
            },
            f,
        )

    if HAS_XGB:
        xgb_clf = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
            eval_metric="mlogloss",
        )
        xgb_clf.fit(X_scaled, y_cls)
        with open(SAVED_DIR / "ml_classification_xgb.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": xgb_clf,
                    "scaler": scaler,
                    "feature_cols": FEATURE_COLS,
                    "label_names": LABEL_NAMES,
                },
                f,
            )

    # 3. ML REGRESSION (Random Forest Regressor & Ridge Regression)
    oof_reg = np.zeros(len(df), dtype=np.float32)
    for train_idx, val_idx in gkf.split(X_scaled, y_reg, groups=groups):
        reg_cv = RandomForestRegressor(
            n_estimators=150, max_depth=8, random_state=42
        )
        reg_cv.fit(X_scaled[train_idx], y_reg[train_idx])
        oof_reg[val_idx] = reg_cv.predict(X_scaled[val_idx])

    reg_rmse = float(np.sqrt(mean_squared_error(y_reg, oof_reg)))
    reg_mae = float(mean_absolute_error(y_reg, oof_reg))
    reg_r2 = float(r2_score(y_reg, oof_reg))

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_reg.fit(X_scaled, y_reg)

    ridge_reg = Ridge(alpha=1.0, random_state=42)
    ridge_reg.fit(X_scaled, y_reg)

    with open(SAVED_DIR / "ml_regression_rf.pkl", "wb") as f:
        pickle.dump(
            {"model": rf_reg, "scaler": scaler, "feature_cols": FEATURE_COLS}, f
        )

    with open(SAVED_DIR / "ml_regression_ridge.pkl", "wb") as f:
        pickle.dump(
            {"model": ridge_reg, "scaler": scaler, "feature_cols": FEATURE_COLS}, f
        )

    metrics = {
        "ml_clustering": {
            "algorithm": "KMeans(k=3)",
            "silhouette_score": round(sil_score, 4),
        },
        "ml_classification": {
            "algorithm": "RandomForestClassifier",
            "group_kfold_accuracy": round(cls_acc, 4),
            "group_kfold_macro_f1": round(cls_f1, 4),
            "cohen_kappa": round(cls_kappa, 4),
        },
        "ml_regression": {
            "algorithm": "RandomForestRegressor",
            "group_kfold_rmse": round(reg_rmse, 4),
            "group_kfold_mae": round(reg_mae, 4),
            "group_kfold_r2": round(reg_r2, 4),
        },
    }

    with open(REPORTS_DIR / "ml_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print(train_and_save_ml_models())
