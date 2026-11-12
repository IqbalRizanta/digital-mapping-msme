"""
Main Ingestion Execution Script.
Runs Google Places API ingestion for Surabaya Timur subdistricts and scans offline survey images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

from src.umkm.ingestion.places_client import GooglePlacesIngestion
from src.umkm.ingestion.survey_scanner import SurveyImageScanner


def main():
    parser = argparse.ArgumentParser(description="Run Ingestion for Surabaya Timur UMKM.")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test API connection and survey folder without running full ingestion.",
    )
    parser.add_argument(
        "--kecamatan",
        type=str,
        default=None,
        help="Specific subdistrict to ingest (e.g. 'Gubeng'). If omitted, runs for configured subdistricts.",
    )
    parser.add_argument(
        "--max-places",
        type=int,
        default=10,
        help="Maximum places to fetch per query (default: 10 to conserve API quota).",
    )
    parser.add_argument(
        "--download-photos",
        action="store_true",
        default=True,
        help="Whether to download photos from Google Places.",
    )
    args = parser.parse_args()

    print("=======================================================")
    print("  FASE 1: INGESTION (Google Places API & Survei Offline)")
    print("=======================================================\n")

    # 1. Test Google Places API Connection
    print("[1/3] Memeriksa Kredensial & Koneksi Google Places API...")
    try:
        places_client = GooglePlacesIngestion()
        conn_test = places_client.test_connection()
        print(f"Hasil Tes Koneksi: {conn_test['status']} - {conn_test['message']}")
        if conn_test["status"] != "SUCCESS":
            print("[Error] Mohon periksa kembali kuota atau aktivasi Places API pada Google Cloud Console.")
            if args.test_only:
                sys.exit(1)
    except Exception as e:
        print(f"[Fatal Error] Gagal inisialisasi Places Client: {e}")
        sys.exit(1)

    # 2. Scan Offline Survey Images
    print("\n[2/3] Memindai Penyimpanan Data Gambar Offline (data/raw/survei_offline/)...")
    scanner = SurveyImageScanner()
    survey_summary = scanner.scan_images()
    print(f"Total Gambar Offline Ditemukan : {survey_summary['total_images_found']}")
    print(f"Total Ukuran Gambar Offline    : {survey_summary['total_size_mb']} MB")
    print(f"Status Target Volume (>=200MB) : {'TERPENUHI' if survey_summary['gate_200mb_passed'] else 'BELUM TERPENUHI (Silakan tambahkan data gambar)'}")

    if args.test_only:
        print("\n[Mode --test-only Selesai]")
        return

    # 3. Run Ingestion for Kecamatan
    print("\n[3/3] Menjalankan Ingestion Data Google Places API...")
    config_path = Path("configs/kecamatan.yaml")
    if not config_path.exists():
        print(f"[Error] Config {config_path} tidak ditemukan.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    kec_list = [k["nama"] for k in cfg.get("kecamatan", [])]
    if args.kecamatan:
        kec_list = [args.kecamatan]

    total_places_all = 0
    total_reviews_all = 0
    total_photos_all = 0

    for kec in kec_list:
        summary = places_client.ingest_kecamatan(
            kecamatan=kec,
            max_places_per_query=args.max_places,
            download_photos=args.download_photos,
            max_photos_per_place=2,
        )
        total_places_all += summary["total_places"]
        total_reviews_all += summary["total_reviews"]
        total_photos_all += summary["downloaded_photos"]

    print("\n=======================================================")
    print("               RINGKASAN AKHIR INGESTION              ")
    print("=======================================================")
    print(f"Total Kecamatan Diproses   : {len(kec_list)}")
    print(f"Total UMKM Kuliner Didapat : {total_places_all}")
    print(f"Total Baris Ulasan Didapat : {total_reviews_all}")
    print(f"Total Foto Online Didownload: {total_photos_all}")
    print(f"Total Gambar Offline Tersedia: {survey_summary['total_images_found']} ({survey_summary['total_size_mb']} MB)")
    print("Data mentah tersimpan aman di: data/raw/")
    print("=======================================================")


if __name__ == "__main__":
    main()
