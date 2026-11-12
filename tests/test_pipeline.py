"""
Suite pengujian unit dan integrasi untuk pipeline Big Data & AI UMKM Surabaya Timur.

Menjalankan validasi dataset bersih, penyimpanan model ML (.pkl) & DL (.pt/.pth/.h5),
analitik spasial BDA, serta endpoint backend FastAPI.
"""

import unittest
from pathlib import Path
import pandas as pd

from app.server.inference_service import UMKMInferenceService
from app.server.main import health_check, bda_summary, umkm_list, predict_multimodal
from app.server.schemas import UMKMPredictRequest

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT_DIR / "data" / "clean"
SAVED_DIR = ROOT_DIR / "model" / "saved"
BDA_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "bda"


class TestUMKMPipeline(unittest.TestCase):
    def setUp(self):
        self.svc = UMKMInferenceService()

    def test_clean_datasets_exist_and_valid(self):
        df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
        df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
        df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")
        df_dok = pd.read_parquet(CLEAN_DIR / "dokumen.parquet")

        self.assertEqual(len(df_umkm), 256)
        self.assertEqual(len(df_ulasan), 1017)
        self.assertEqual(len(df_foto), 203)
        self.assertEqual(len(df_dok), 3)

    def test_saved_models_and_embeddings_exist(self):
        expected_files = [
            "ml_clustering_kmeans.pkl",
            "ml_classification_rf.pkl",
            "ml_regression_rf.pkl",
            "dl_clustering_autoencoder.pt",
            "dl_clustering_autoencoder.pth",
            "dl_clustering_autoencoder.h5",
            "dl_classification_mlp.pt",
            "dl_classification_mlp.pth",
            "dl_classification_mlp.h5",
            "dl_regression_mlp.pt",
            "dl_regression_mlp.pth",
            "dl_regression_mlp.h5",
            "embeddings_multimodal.npy",
            "embeddings_multimodal.npz",
            "embeddings_multimodal.parquet",
        ]
        for fname in expected_files:
            fpath = SAVED_DIR / fname
            self.assertTrue(fpath.exists(), f"File model {fname} tidak ditemukan di model/saved/")

    def test_inference_service_predictions(self):
        # 1. Uji Konvensional
        pred_konv = self.svc.predict_all({
            "nama_umkm": "Warung Konvensional",
            "kecamatan": "Sukolilo",
            "rating": 4.1,
            "jumlah_ulasan": 5,
            "indikator_qris": 0,
            "indikator_ojol": 0,
            "indikator_pos": 0,
            "indikator_edc": 0,
        })
        self.assertEqual(pred_konv["ml_classification_label"], "Konvensional")

        # 2. Uji Berkembang
        pred_berk = self.svc.predict_all({
            "nama_umkm": "Warung Berkembang",
            "kecamatan": "Gubeng",
            "rating": 4.5,
            "jumlah_ulasan": 35,
            "indikator_qris": 1,
            "indikator_ojol": 0,
            "indikator_pos": 0,
            "indikator_edc": 0,
        })
        self.assertEqual(pred_berk["ml_classification_label"], "Berkembang")

        # 3. Uji Maju
        pred_maju = self.svc.predict_all({
            "nama_umkm": "Resto Maju",
            "kecamatan": "Rungkut",
            "rating": 4.8,
            "jumlah_ulasan": 95,
            "indikator_qris": 1,
            "indikator_ojol": 1,
            "indikator_pos": 1,
            "indikator_edc": 0,
        })
        self.assertEqual(pred_maju["ml_classification_label"], "Maju")

    def test_fastapi_endpoints(self):
        # Test /health
        h = health_check()
        self.assertEqual(h["status"], "ok")
        self.assertGreaterEqual(h["saved_models_count"], 15)

        # Test /bda/summary
        bda = bda_summary()
        self.assertEqual(len(bda), 6)

        # Test /umkm/list
        umkms = umkm_list()
        self.assertEqual(len(umkms), 256)

        # Test /predict/multimodal
        req = UMKMPredictRequest(
            nama_umkm="Warung Test API",
            kecamatan="Sukolilo",
            rating=4.7,
            jumlah_ulasan=50,
            indikator_qris=1,
            indikator_ojol=1,
            indikator_pos=0,
            indikator_edc=0,
        )
        res = predict_multimodal(req)
        self.assertIn(res["ml_classification_label"], ["Berkembang", "Maju"])

    def test_bda_spatial_gap_table(self):
        csv_path = BDA_DIR / "kecamatan_digital_gap.csv"
        self.assertTrue(csv_path.exists())
        df_gap = pd.read_csv(csv_path)
        self.assertEqual(len(df_gap), 6)
        expected_kecamatan = {"Sukolilo", "Gubeng", "Rungkut", "Mulyorejo", "Tenggilis Mejoyo", "Gunung Anyar"}
        self.assertEqual(set(df_gap["kecamatan"]), expected_kecamatan)


if __name__ == "__main__":
    unittest.main()
