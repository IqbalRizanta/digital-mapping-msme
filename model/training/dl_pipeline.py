"""
Pemodelan Deep Learning: Clustering, Classification, dan Regression (.pt, .pth, .h5).

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- PyTorch Neural Networks (torch.nn): https://pytorch.org/docs/stable/nn.html
- PyTorch Serialization (torch.save): https://pytorch.org/docs/stable/generated/torch.save.html
- HDF5 / h5py Storage Format: https://docs.h5py.org/en/stable/high/file.html
"""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from model.training.ml_pipeline import FEATURE_COLS, derive_target_class, prepare_feature_frame
GOLD_DIR = ROOT_DIR / "data" / "gold"
SAVED_DIR = ROOT_DIR / "model" / "saved"
REPORTS_DIR = ROOT_DIR / "docs" / "reports" / "modeling"


class DeepClusteringAutoencoder(nn.Module):
    """Autoencoder untuk Deep Embedded Clustering pada fitur multimodal UMKM."""

    def __init__(self, input_dim: int, latent_dim: int = 6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class DeepMaturityClassifier(nn.Module):
    """Jaringan saraf tiruan klasifikasi 3 kelas kematangan digital."""

    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMaturityRegressor(nn.Module):
    """Jaringan saraf tiruan regresi skor indeks kesiapan digital (0-100)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _save_state_to_h5(state_dict: dict, h5_path: Path, metadata: dict) -> None:
    try:
        import h5py

        with h5py.File(h5_path, "w") as hf:
            for k, v in metadata.items():
                hf.attrs[k] = str(v)
            grp = hf.create_group("model_weights")
            for name, tensor in state_dict.items():
                grp.create_dataset(name, data=tensor.detach().cpu().numpy())
    except ImportError:
        torch.save({"state_dict": state_dict, "metadata": metadata}, h5_path)


def train_and_save_dl_models() -> dict:
    torch.manual_seed(42)
    np.random.seed(42)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(GOLD_DIR / "feature_store.parquet")
    df = prepare_feature_frame(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    y_cls = derive_target_class(df).astype(np.int64)
    y_reg = df["skor_kematangan_kontinu"].values.astype(np.float32)

    input_dim = X_scaled.shape[1]
    x_tensor = torch.from_numpy(X_scaled)
    y_cls_tensor = torch.from_numpy(y_cls)
    y_reg_tensor = torch.from_numpy(y_reg)

    dataset = TensorDataset(x_tensor, y_cls_tensor, y_reg_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=len(dataset) > 32)

    # 1. DL CLUSTERING (Deep Autoencoder + Latent K-Means)
    ae_model = DeepClusteringAutoencoder(input_dim=input_dim, latent_dim=6)
    ae_opt = torch.optim.Adam(ae_model.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()

    ae_model.train()
    for _ in range(60):
        for bx, _, _ in loader:
            ae_opt.zero_grad()
            _, x_hat = ae_model(bx)
            loss = mse_loss(x_hat, bx)
            loss.backward()
            ae_opt.step()

    ae_model.eval()
    with torch.no_grad():
        z_latent, _ = ae_model(x_tensor)
        z_np = z_latent.numpy()

    km_latent = KMeans(n_clusters=3, n_init=20, random_state=42)
    dl_clusters = km_latent.fit_predict(z_np)
    dl_sil = float(silhouette_score(z_np, dl_clusters))

    ae_payload = {
        "state_dict": ae_model.state_dict(),
        "input_dim": input_dim,
        "latent_dim": 6,
        "kmeans_centers": km_latent.cluster_centers_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": FEATURE_COLS,
    }
    torch.save(ae_payload, SAVED_DIR / "dl_clustering_autoencoder.pt")
    torch.save(ae_payload, SAVED_DIR / "dl_clustering_autoencoder.pth")
    _save_state_to_h5(
        ae_model.state_dict(),
        SAVED_DIR / "dl_clustering_autoencoder.h5",
        {"task": "dl_clustering", "input_dim": input_dim, "latent_dim": 6},
    )

    # 2. DL CLASSIFICATION (MLP 3-Class Maturity Classifier)
    cls_model = DeepMaturityClassifier(input_dim=input_dim, num_classes=3)
    cls_opt = torch.optim.Adam(cls_model.parameters(), lr=0.004)
    ce_loss = nn.CrossEntropyLoss()

    cls_model.train()
    for _ in range(70):
        for bx, by_cls, _ in loader:
            cls_opt.zero_grad()
            logits = cls_model(bx)
            loss = ce_loss(logits, by_cls)
            loss.backward()
            cls_opt.step()

    cls_model.eval()
    with torch.no_grad():
        preds_cls = cls_model(x_tensor).argmax(dim=1).numpy()

    dl_cls_acc = float(accuracy_score(y_cls, preds_cls))
    dl_cls_f1 = float(f1_score(y_cls, preds_cls, average="macro"))

    cls_payload = {
        "state_dict": cls_model.state_dict(),
        "input_dim": input_dim,
        "num_classes": 3,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": FEATURE_COLS,
    }
    torch.save(cls_payload, SAVED_DIR / "dl_classification_mlp.pt")
    torch.save(cls_payload, SAVED_DIR / "dl_classification_mlp.pth")
    _save_state_to_h5(
        cls_model.state_dict(),
        SAVED_DIR / "dl_classification_mlp.h5",
        {"task": "dl_classification", "input_dim": input_dim, "num_classes": 3},
    )

    # 3. DL REGRESSION (MLP Continuous Score Regressor)
    reg_model = DeepMaturityRegressor(input_dim=input_dim)
    reg_opt = torch.optim.Adam(reg_model.parameters(), lr=0.008)

    reg_model.train()
    for _ in range(90):
        for bx, _, by_reg in loader:
            reg_opt.zero_grad()
            out = reg_model(bx)
            loss = mse_loss(out, by_reg)
            loss.backward()
            reg_opt.step()

    reg_model.eval()
    with torch.no_grad():
        preds_reg = reg_model(x_tensor).numpy()

    dl_reg_rmse = float(np.sqrt(mean_squared_error(y_reg, preds_reg)))
    dl_reg_r2 = float(r2_score(y_reg, preds_reg))

    reg_payload = {
        "state_dict": reg_model.state_dict(),
        "input_dim": input_dim,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": FEATURE_COLS,
    }
    torch.save(reg_payload, SAVED_DIR / "dl_regression_mlp.pt")
    torch.save(reg_payload, SAVED_DIR / "dl_regression_mlp.pth")
    _save_state_to_h5(
        reg_model.state_dict(),
        SAVED_DIR / "dl_regression_mlp.h5",
        {"task": "dl_regression", "input_dim": input_dim},
    )

    metrics = {
        "dl_clustering": {
            "architecture": "DeepClusteringAutoencoder(latent=6)+KMeans",
            "latent_silhouette_score": round(dl_sil, 4),
        },
        "dl_classification": {
            "architecture": "DeepMaturityClassifier(MLP)",
            "accuracy": round(dl_cls_acc, 4),
            "macro_f1": round(dl_cls_f1, 4),
        },
        "dl_regression": {
            "architecture": "DeepMaturityRegressor(MLP)",
            "rmse": round(dl_reg_rmse, 4),
            "r2": round(dl_reg_r2, 4),
        },
    }

    with open(REPORTS_DIR / "dl_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    print(train_and_save_dl_models())
