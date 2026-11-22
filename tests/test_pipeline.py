"""
Suite pengujian unit dan integrasi untuk pipeline Big Data & AI UMKM Surabaya Timur.

Menjalankan validasi dataset bersih:
1. Pelestarian bahasa asli ulasan (Indonesian/English/Javanese).
2. Kebenaran data ingestion (100% Google Places API asli tanpa entitas fiktif).
3. Volume penyimpanan citra Google Maps (Minimal >= 200 MB).
4. Penyimpanan model ML (.pkl) & DL (.pt/.pth/.h5).
5. Analitik spasial BDA & endpoint backend FastAPI.
"""

import unittest
from pathlib import Path
import pandas as pd

from app.server.inference_service import UMKMInferenceService
from app.server.main import health_check, bda_summary, umkm_list, predict_multimodal
from app.server.schemas import UMKMPredictRequest
from model.storage_config import (
    GDRIVE_CLEAN_IMAGES_DIR,
    GDRIVE_PROJECT_ROOT,
    GDRIVE_RAW_IMAGES_DIR,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT_DIR / "data" / "clean"
SAVED_DIR = ROOT_DIR / "model" / "saved"
BDA_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "bda"
GDRIVE_IMAGES_GMAPS = GDRIVE_PROJECT_ROOT


class TestUMKMPipeline(unittest.TestCase):
    def setUp(self):
        self.svc = UMKMInferenceService()

    def test_clean_datasets_exist_and_valid(self):
        df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
        df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
        df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")
        df_dok = pd.read_parquet(CLEAN_DIR / "dokumen.parquet")

        # 1. Validasi entitas UMKM 100% dari Google Places API (bebas entitas palsu)
        self.assertGreaterEqual(len(df_umkm), 246)
        self.assertFalse(df_umkm["place_id"].str.contains("SURVEI").any(), "Ditemukan entitas buatan/palsu di umkm.parquet!")
        self.assertTrue((df_umkm["sumber"] == "google_places_api").all())

        # 2. Validasi ulasan melestarikan bahasa asli
        self.assertGreaterEqual(len(df_ulasan), 1000)
        self.assertIn("teks_asli", df_ulasan.columns)
        self.assertIn("bahasa", df_ulasan.columns)
        self.assertIn("en", set(df_ulasan["bahasa"]))
        self.assertIn("id", set(df_ulasan["bahasa"]))

        # 3. Validasi foto bersih terfilter CLIP (penamaan rapi per kecamatan & nama tempat)
        self.assertGreaterEqual(len(df_foto), 250)
        self.assertIn("nama_tempat", df_foto.columns)
        self.assertFalse((df_foto["kecamatan"] == "Google Maps Hires").any())
        self.assertFalse(df_foto["clean_path"].str.contains("_hires_").any())
        self.assertEqual(len(df_dok), 3)

    def test_image_storage_volume_ge_200mb(self):
        """Validasi struktur 7 zona BDA-2026-Iqbal (Search->Repeat->Adjust->Dataset) & clean images >= 200 MB."""
        if GDRIVE_PROJECT_ROOT.exists():
            for top_dir in [
                "01_Search_Journal",
                "02_Repeat_Journal",
                "03_Adjust_Journal",
                "04_Dataset_Multimodal_Saya",
                "05_Models_Tersimpan",
                "06_Notebooks_Lengkap_Proyek",
                "07_Reports_Lengkap_Proyek",
            ]:
                self.assertTrue(
                    (GDRIVE_PROJECT_ROOT / top_dir).exists(),
                    f"Folder terstruktur {top_dir}/ belum ada di {GDRIVE_PROJECT_ROOT}!",
                )

            self.assertTrue(GDRIVE_RAW_IMAGES_DIR.exists(), "Folder 04_Dataset_Multimodal_Saya/01_raw/images belum ada!")
            self.assertTrue(GDRIVE_CLEAN_IMAGES_DIR.exists(), "Folder 04_Dataset_Multimodal_Saya/02_clean/images belum ada!")
            self.assertTrue((GDRIVE_CLEAN_IMAGES_DIR / "Google_Maps").exists(), "Folder 02_clean/images/Google_Maps belum ada!")
            self.assertTrue((GDRIVE_CLEAN_IMAGES_DIR / "Survei_Lapangan_SWK").exists(), "Folder 02_clean/images/Survei_Lapangan_SWK belum ada!")

            clean_files = [
                f for f in GDRIVE_CLEAN_IMAGES_DIR.rglob("*")
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
            ]
            total_bytes = sum(f.stat().st_size for f in clean_files)
            total_mb = total_bytes / (1024 * 1024)
            self.assertGreaterEqual(
                total_mb,
                200.0,
                f"Ukuran gambar bersih pada 04_Dataset_Multimodal_Saya/02_clean/images ({total_mb:.2f} MB) belum memenuhi target minimal 200 MB!",
            )

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
        self.assertGreaterEqual(len(umkms), 246)

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
