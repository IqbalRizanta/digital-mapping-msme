"""
Layanan inferensi terpadu yang memuat model ML (.pkl) dan DL (.pt/.pth/.h5).

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- PyTorch Inference Mode: https://pytorch.org/docs/stable/generated/torch.no_grad.html
- Scikit-Learn Predict API: https://scikit-learn.org/stable/glossary.html#term-predict
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from model.training.dl_pipeline import (
    DeepClusteringAutoencoder,
    DeepMaturityClassifier,
    DeepMaturityRegressor,
)
from model.training.ml_pipeline import FEATURE_COLS, LABEL_NAMES

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAVED_DIR = ROOT_DIR / "model" / "saved"
GOLD_DIR = ROOT_DIR / "data" / "gold"
BDA_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "bda"


class UMKMInferenceService:
    def __init__(self):
        self.ml_cluster = self._load_pickle("ml_clustering_kmeans.pkl")
        self.ml_clf = self._load_pickle("ml_classification_rf.pkl")
        self.ml_reg = self._load_pickle("ml_regression_rf.pkl")

        self.dl_cluster_ckpt = torch.load(
            SAVED_DIR / "dl_clustering_autoencoder.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.dl_clf_ckpt = torch.load(
            SAVED_DIR / "dl_classification_mlp.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.dl_reg_ckpt = torch.load(
            SAVED_DIR / "dl_regression_mlp.pt",
            map_location="cpu",
            weights_only=False,
        )

        input_dim = int(self.dl_clf_ckpt["input_dim"])
        self.ae_net = DeepClusteringAutoencoder(
            input_dim=input_dim, latent_dim=int(self.dl_cluster_ckpt["latent_dim"])
        )
        self.ae_net.load_state_dict(self.dl_cluster_ckpt["state_dict"])
        self.ae_net.eval()

        self.clf_net = DeepMaturityClassifier(input_dim=input_dim, num_classes=3)
        self.clf_net.load_state_dict(self.dl_clf_ckpt["state_dict"])
        self.clf_net.eval()

        self.reg_net = DeepMaturityRegressor(input_dim=input_dim)
        self.reg_net.load_state_dict(self.dl_reg_ckpt["state_dict"])
        self.reg_net.eval()

    @staticmethod
    def _load_pickle(filename: str) -> dict:
        with open(SAVED_DIR / filename, "rb") as f:
            return pickle.load(f)

    def build_feature_vector(self, payload: dict) -> np.ndarray:
        qris = float(payload.get("indikator_qris", 0))
        ojol = float(payload.get("indikator_ojol", 0))
        pos = float(payload.get("indikator_pos", 0))
        edc = float(payload.get("indikator_edc", 0))

        row = {
            "rating": float(payload.get("rating", 4.3)),
            "jumlah_ulasan": float(payload.get("jumlah_ulasan", 20)),
            "indikator_qris": qris,
            "indikator_pos": pos,
            "indikator_edc": edc,
            "indikator_ojol": ojol,
            "review_count": float(payload.get("rev_count", 4)),
            "avg_word_count": float(payload.get("rev_word_mean", 12.0)),
            "text_qris": 0.4 * qris,
            "text_ojol_alias": 0.4 * ojol,
            "text_pos": 0.3 * pos,
            "text_edc": 0.3 * edc,
            "photo_count": float(payload.get("img_count", 1)),
            "vis_qris": qris,
            "vis_online": ojol,
            "vis_pos": pos,
            "vis_edc": edc,
        }
        return np.array([[row[c] for c in FEATURE_COLS]], dtype=np.float32)

    def predict_all(self, payload: dict) -> dict:
        x_raw = self.build_feature_vector(payload)
        scaler = self.ml_clf["scaler"]
        x_scaled = scaler.transform(x_raw).astype(np.float32)

        raw_c = int(self.ml_cluster["model"].predict(x_scaled)[0])
        ml_c = int(self.ml_cluster["cluster_map"].get(raw_c, raw_c))
        ml_cls_id = int(self.ml_clf["model"].predict(x_scaled)[0])
        ml_cls_label = LABEL_NAMES.get(ml_cls_id, "Konvensional")
        ml_reg_score = float(
            np.clip(self.ml_reg["model"].predict(x_scaled)[0], 0.0, 100.0)
        )

        x_tensor = torch.from_numpy(x_scaled)
        with torch.no_grad():
            z_lat, _ = self.ae_net(x_tensor)
            centers = self.dl_cluster_ckpt["kmeans_centers"]
            dists = np.linalg.norm(centers - z_lat.numpy()[0], axis=1)
            dl_c = int(np.argmin(dists))

            dl_cls_id = int(self.clf_net(x_tensor).argmax(dim=1).item())
            dl_cls_label = LABEL_NAMES.get(dl_cls_id, "Konvensional")
            dl_reg_score = float(np.clip(self.reg_net(x_tensor).item(), 0.0, 100.0))

        if ml_cls_label == "Konvensional":
            rekom = "Fasilitasi pendaftaran QRIS statis dan kemitraan pesan-antar (GoFood/GrabFood/ShopeeFood)."
        elif ml_cls_label == "Berkembang":
            rekom = "Integrasi aplikasi kasir POS pintar (Moka/Pawoon/Majoo) dan pencatatan transaksi digital."
        else:
            rekom = "Perluasan akses pembiayaan mikro berbasis rekam jejak transaksi digital dan promosi lintas kanal."

        return {
            "nama_umkm": str(payload.get("nama_umkm", "UMKM Surabaya Timur")),
            "kecamatan": str(payload.get("kecamatan", "Sukolilo")),
            "ml_clustering_cluster": ml_c,
            "ml_classification_label": ml_cls_label,
            "ml_regression_score": round(ml_reg_score, 2),
            "dl_clustering_cluster": dl_c,
            "dl_classification_label": dl_cls_label,
            "dl_regression_score": round(dl_reg_score, 2),
            "rekomendasi": rekom,
        }

    @staticmethod
    def list_saved_artifacts() -> list:
        if not SAVED_DIR.exists():
            return []
        items = []
        for f in sorted(SAVED_DIR.iterdir()):
            if f.is_file():
                items.append(
                    {
                        "filename": f.name,
                        "format": f.suffix,
                        "size_kb": round(f.stat().st_size / 1024.0, 2),
                    }
                )
        return items

    @staticmethod
    def get_bda_summary() -> list:
        csv_path = BDA_DIR / "kecamatan_digital_gap.csv"
        if not csv_path.exists():
            return []
        return pd.read_csv(csv_path).to_dict(orient="records")

    @staticmethod
    def get_umkm_table(kecamatan: str | None = None) -> list:
        pq_path = GOLD_DIR / "feature_store.parquet"
        if not pq_path.exists():
            return []
        df = pd.read_parquet(pq_path)
        if kecamatan and not hasattr(kecamatan, "default") and str(kecamatan) != "Semua":
            df = df[df["kecamatan"].astype(str) == str(kecamatan)]
        cols = [
            "umkm_id",
            "place_id",
            "nama_umkm",
            "kecamatan",
            "rating",
            "jumlah_ulasan",
            "indikator_qris",
            "indikator_ojol",
            "indikator_pos",
            "indikator_edc",
            "skor_kematangan_kontinu",
            "kelas_kematangan",
        ]
        avail_cols = [c for c in cols if c in df.columns]
        return df[avail_cols].to_dict(orient="records")
