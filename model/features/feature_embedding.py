"""
Ekstraksi fitur multimodal dan penyimpanan vektor embedding (.npy, .npz, .parquet).

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- Scikit-Learn Preprocessing & PCA: https://scikit-learn.org/stable/modules/preprocessing.html
- Scikit-Learn TF-IDF Vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- NumPy Binary I/O (.npy / .npz): https://numpy.org/doc/stable/reference/routines.io.html
"""

import pickle
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.storage_config import GDRIVE_FEATURE_STORE_DIR, GDRIVE_PROJECT_ROOT

CLEAN_DIR = ROOT_DIR / "data" / "clean"
GOLD_DIR = ROOT_DIR / "data" / "gold"
SAVED_DIR = ROOT_DIR / "model" / "saved"
GDRIVE_ROOT = GDRIVE_PROJECT_ROOT


def build_and_save_embeddings() -> dict:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
    df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
    df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")

    ulasan_per_umkm = (
        df_ulasan.groupby("umkm_id")
        .agg(
            combined_text=("teks_bersih", lambda x: " ".join(str(v) for v in x)),
            text_qris=("mention_qris", "mean"),
            text_pos=("mention_pos", "mean"),
            text_edc=("mention_edc", "mean"),
            text_online=("mention_online", "mean"),
            avg_word_count=("word_count", "mean"),
            review_count=("ulasan_id", "count"),
            avg_review_rating=("rating_pengulas", "mean"),
        )
        .reset_index()
    )

    foto_per_umkm = (
        df_foto.groupby("umkm_id")
        .agg(
            vis_qris=("has_qris", lambda x: int(x.any())),
            vis_pos=("has_pos", lambda x: int(x.any())),
            vis_edc=("has_edc", lambda x: int(x.any())),
            vis_online=("has_merchant_online", lambda x: int(x.any())),
            vis_stan=("has_stan", lambda x: int(x.any())),
            vis_num_indicators=("num_indicators", "mean"),
            vis_max_conf=("max_conf", "max"),
            photo_count=("foto_id", "count"),
            label_ground_truth=("label_ground_truth", lambda x: x.iloc[0]),
            label_id=("label_id", "max"),
        )
        .reset_index()
    )

    df = (
        df_umkm.merge(ulasan_per_umkm, on="umkm_id", how="left")
        .merge(foto_per_umkm, on="umkm_id", how="left")
    )
    df["combined_text"] = df["combined_text"].fillna(df["nama_umkm"].fillna(""))

    num_cols = [
        "rating",
        "user_ratings_total_clean",
        "text_qris",
        "text_pos",
        "text_edc",
        "text_online",
        "avg_word_count",
        "review_count",
        "vis_qris",
        "vis_pos",
        "vis_edc",
        "vis_online",
        "vis_stan",
        "vis_num_indicators",
        "vis_max_conf",
        "photo_count",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["avg_review_rating"] = df["avg_review_rating"].fillna(df["rating"])
    df["label_ground_truth"] = df["label_ground_truth"].fillna("Unlabeled")
    df["label_id"] = df["label_id"].fillna(-1).astype(int)

    df["indikator_qris"] = ((df["vis_qris"] > 0) | (df["text_qris"] > 0)).astype(int)
    df["indikator_pos"] = ((df["vis_pos"] > 0) | (df["text_pos"] > 0)).astype(int)
    df["indikator_edc"] = ((df["vis_edc"] > 0) | (df["text_edc"] > 0)).astype(int)
    df["indikator_ojol"] = ((df["vis_online"] > 0) | (df["text_online"] > 0)).astype(int)
    df["jumlah_ulasan"] = df["user_ratings_total_clean"].astype(int)

    df["skor_kematangan_kontinu"] = np.clip(
        28.0 * df["indikator_qris"]
        + 24.0 * df["indikator_ojol"]
        + 22.0 * df["indikator_pos"]
        + 16.0 * df["indikator_edc"]
        + 5.0 * np.minimum(df["rating"] / 5.0, 1.0)
        + 5.0 * np.minimum(np.log1p(df["jumlah_ulasan"]) / 7.0, 1.0),
        0.0,
        100.0,
    ).round(2)

    tfidf = TfidfVectorizer(max_features=32, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(df["combined_text"]).toarray().astype(np.float32)

    core_features = [
        "rating",
        "user_ratings_total_clean",
        "text_qris",
        "text_pos",
        "text_edc",
        "text_online",
        "avg_word_count",
        "vis_qris",
        "vis_pos",
        "vis_edc",
        "vis_online",
        "vis_stan",
        "vis_num_indicators",
    ]
    X_core = df[core_features].values.astype(np.float32)
    X_combined = np.hstack([X_core, tfidf_mat])

    fusion_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=6, random_state=42)),
        ]
    )
    X_pca = fusion_pipe.fit_transform(X_combined).astype(np.float32)
    for i in range(X_pca.shape[1]):
        df[f"pca_{i+1}"] = X_pca[:, i]

    with open(SAVED_DIR / "feature_fusion_pipeline.pkl", "wb") as f:
        pickle.dump(
            {
                "pipeline": fusion_pipe,
                "tfidf": tfidf,
                "core_features": core_features,
            },
            f,
        )

    npy_path = SAVED_DIR / "embeddings_multimodal.npy"
    npz_path = SAVED_DIR / "embeddings_multimodal.npz"
    pq_path = SAVED_DIR / "embeddings_multimodal.parquet"

    np.save(npy_path, X_pca)
    np.savez_compressed(
        npz_path,
        X_core=X_core,
        tfidf_mat=tfidf_mat,
        X_pca=X_pca,
        umkm_id=df["umkm_id"].astype(str).values,
        place_id=df["place_id"].astype(str).values,
        skor_kontinu=df["skor_kematangan_kontinu"].values,
    )

    pca_cols = [f"pca_{i+1}" for i in range(X_pca.shape[1])]
    tfidf_cols = [f"tfidf_{i+1:02d}" for i in range(tfidf_mat.shape[1])]
    df_tfidf = pd.DataFrame(tfidf_mat, columns=tfidf_cols, index=df.index)
    emb_table = pd.concat(
        [
            df[["umkm_id", "place_id", "nama_umkm", "kecamatan", "skor_kematangan_kontinu"] + pca_cols],
            df_tfidf,
        ],
        axis=1,
    )
    emb_table.to_parquet(pq_path, index=False)

    df_out = df.drop(columns=["combined_text"])
    df_out.to_parquet(GOLD_DIR / "feature_store.parquet", index=False)
    shutil.copy2(npy_path, GOLD_DIR / npy_path.name)
    shutil.copy2(npz_path, GOLD_DIR / "multimodal_embeddings.npz")
    shutil.copy2(pq_path, GOLD_DIR / pq_path.name)

    if GDRIVE_ROOT.exists():
        GDRIVE_FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
        for p in [npy_path, npz_path, pq_path, GOLD_DIR / "feature_store.parquet"]:
            shutil.copy2(p, GDRIVE_FEATURE_STORE_DIR / p.name)

    return {
        "n_samples": int(X_pca.shape[0]),
        "pca_dim": int(X_pca.shape[1]),
        "tfidf_dim": int(tfidf_mat.shape[1]),
        "npy_path": str(npy_path),
        "npz_path": str(npz_path),
        "parquet_path": str(pq_path),
    }


if __name__ == "__main__":
    print(build_and_save_embeddings())
