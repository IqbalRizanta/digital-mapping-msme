"""
Phase 2: Build Clean Dataset.
Processes, validates, anonymizes, and saves multimodal data into clean Parquet tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pypdf

from src.umkm.processing.image_processor import ImageProcessor
from src.umkm.processing.text_processor import TextProcessor


def build_clean_datasets():
    print("=======================================================")
    print("  FASE 2: PEMROSESAN, ANONIMISASI & ZONA CLEAN (PARQUET)")
    print("=======================================================\n")

    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_photos_dir = clean_dir / "photos"
    clean_photos_dir.mkdir(parents=True, exist_ok=True)

    img_processor = ImageProcessor(clean_photo_dir=str(clean_photos_dir))

    # ---------------------------------------------------------
    # 1. PROSES TABEL UMKM (DARI GOOGLE PLACES & SWK SURVEI)
    # ---------------------------------------------------------
    print("[1/4] Memproses data tabular UMKM...")
    raw_maps_dir = Path("data/raw/google_maps")
    umkm_records: List[Dict[str, Any]] = []
    place_id_to_umkm_id: Dict[str, str] = {}

    counter = 1
    for meta_file in raw_maps_dir.glob("*/places_metadata.json"):
        with open(meta_file, "r", encoding="utf-8") as f:
            places = json.load(f)

        for p in places:
            pid = p.get("place_id")
            if not pid or pid in place_id_to_umkm_id:
                continue

            umkm_id = f"UMKM_SBY_{counter:04d}"
            counter += 1
            place_id_to_umkm_id[pid] = umkm_id

            name = p.get("name", "")
            is_swk = "sentra wisata kuliner" in name.lower() or "wisata kuliner" in name.lower() or "swk" in name.lower()
            swk_name = None
            if "convention" in name.lower():
                swk_name = "SWK Convention Hall"
            elif "deles" in name.lower():
                swk_name = "SWK Deles"
            elif "semolowaru" in name.lower():
                swk_name = "SWK Semolowaru"

            umkm_records.append({
                "umkm_id": umkm_id,
                "place_id": pid,
                "nama_umkm": name,
                "kecamatan": p.get("kecamatan"),
                "alamat": p.get("formatted_address"),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "rating": p.get("rating"),
                "user_ratings_total": p.get("user_ratings_total"),
                "tipe_lokasi": "SWK" if is_swk else "Reguler",
                "swk_nama": swk_name,
                "sumber": "google_places_api",
            })

    # Tambahkan entitas UMKM stan dari survei SWK Convention Hall & Deles
    survey_inv_path = Path("data/raw/survei_offline/survey_images_inventory.csv")
    if survey_inv_path.exists():
        df_survey_inv = pd.read_csv(survey_inv_path)
        for idx, row in df_survey_inv.iterrows():
            umkm_id = f"UMKM_SURVEI_{idx+1:03d}"
            umkm_records.append({
                "umkm_id": umkm_id,
                "place_id": f"SURVEI_STALL_{idx+1:03d}",
                "nama_umkm": f"Stan Kuliner SWK #{idx+1} ({row['label_ground_truth']})",
                "kecamatan": "Sukolilo",
                "alamat": "Sentra Wisata Kuliner Surabaya Timur (Convention Hall / Deles)",
                "lat": -7.2885,  # Titik sentral SWK Sukolilo
                "lon": 112.7845,
                "rating": None,
                "user_ratings_total": None,
                "tipe_lokasi": "SWK_Stan_Survei",
                "swk_nama": "SWK Sukolilo",
                "sumber": "survei_lapangan_primer",
            })

    df_umkm = pd.DataFrame(umkm_records)
    umkm_parquet = clean_dir / "umkm.parquet"
    df_umkm.to_parquet(umkm_parquet, index=False)
    print(f"-> Berhasil menyimpan {len(df_umkm)} entitas UMKM ke {umkm_parquet}")

    # ---------------------------------------------------------
    # 2. PROSES & ANONIMISASI TEKS ULASAN
    # ---------------------------------------------------------
    print("\n[2/4] Menganonimkan & memproses teks ulasan pengunjung...")
    review_records: List[Dict[str, Any]] = []
    rev_counter = 1

    for rev_file in raw_maps_dir.glob("*/reviews_metadata.json"):
        with open(rev_file, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        for r in reviews:
            pid = r.get("place_id")
            umkm_id = place_id_to_umkm_id.get(pid, "UNKNOWN")
            raw_text = r.get("text", "")
            if not raw_text.strip():
                continue

            cleaned_info = TextProcessor.clean_and_normalize(raw_text)
            rev_id = f"REV_{rev_counter:05d}"
            rev_counter += 1

            review_records.append({
                "ulasan_id": rev_id,
                "umkm_id": umkm_id,
                "place_id": pid,
                "kecamatan": r.get("kecamatan"),
                "rating_pengulas": r.get("rating"),
                "teks_anonim": cleaned_info["teks_anonim"],
                "teks_bersih": cleaned_info["teks_bersih"],
                "word_count": cleaned_info["word_count"],
                "mention_qris": cleaned_info["mention_qris"],
                "mention_pos": cleaned_info["mention_pos"],
                "mention_edc": cleaned_info["mention_edc"],
                "mention_online": cleaned_info["mention_online"],
                "waktu": r.get("relative_publish_time"),
            })

    df_reviews = pd.DataFrame(review_records)
    reviews_parquet = clean_dir / "ulasan.parquet"
    df_reviews.to_parquet(reviews_parquet, index=False)
    print(f"-> Berhasil memproses & menganonimkan {len(df_reviews)} baris ulasan ke {reviews_parquet}")

    # ---------------------------------------------------------
    # 3. PROSES FOTO & BLUR WAJAH (PRIVASI)
    # ---------------------------------------------------------
    print("\n[3/4] Menganonimkan foto (blur wajah OpenCV) & standardisasi...")
    photo_records: List[Dict[str, Any]] = []
    photo_counter = 1
    total_faces_blurred = 0

    # 3.1 Foto Online Google Maps
    print("-> Memproses foto Google Maps...")
    for kec_folder in raw_maps_dir.glob("*"):
        if not kec_folder.is_dir():
            continue
        kec_name = kec_folder.name
        photos_folder = kec_folder / "photos"
        for pfile in photos_folder.glob("*.jpg"):
            pid = pfile.name.split("_photo_")[0]
            umkm_id = place_id_to_umkm_id.get(pid, "UNKNOWN")

            proc_res = img_processor.process_and_save(
                raw_path=pfile,
                subfolder=f"online/{kec_name}",
            )
            if proc_res:
                total_faces_blurred += proc_res["face_blurred_count"]
                photo_records.append({
                    "foto_id": f"FOTO_ONL_{photo_counter:04d}",
                    "umkm_id": umkm_id,
                    "sumber": "google_places_api",
                    "kecamatan": kec_name.title().replace("_", " "),
                    "label_ground_truth": "Unlabeled",
                    "label_id": -1,
                    "width": proc_res["width"],
                    "height": proc_res["height"],
                    "face_blurred_count": proc_res["face_blurred_count"],
                    "clean_path": proc_res["clean_path"],
                    "sha256": proc_res["clean_sha256"],
                })
                photo_counter += 1

    # 3.2 Foto Offline Survei SWK
    print("-> Memproses foto survei lapangan SWK Convention Hall & Deles...")
    if survey_inv_path.exists():
        for idx, row in df_survey_inv.iterrows():
            raw_path = Path(row["relative_path"])
            proc_res = img_processor.process_and_save(
                raw_path=raw_path,
                subfolder=f"offline/{row['folder_class']}",
            )
            if proc_res:
                total_faces_blurred += proc_res["face_blurred_count"]
                photo_records.append({
                    "foto_id": f"FOTO_OFF_{idx+1:03d}",
                    "umkm_id": f"UMKM_SURVEI_{idx+1:03d}",
                    "sumber": "survei_lapangan_swk",
                    "kecamatan": "Sukolilo",
                    "label_ground_truth": row["label_ground_truth"],
                    "label_id": int(row["label_id"]),
                    "width": proc_res["width"],
                    "height": proc_res["height"],
                    "face_blurred_count": proc_res["face_blurred_count"],
                    "clean_path": proc_res["clean_path"],
                    "sha256": proc_res["clean_sha256"],
                })

    df_photos = pd.DataFrame(photo_records)
    photos_parquet = clean_dir / "foto.parquet"
    df_photos.to_parquet(photos_parquet, index=False)
    print(f"-> Berhasil membersihkan {len(df_photos)} foto (Total wajah diburamkan: {total_faces_blurred}) ke {photos_parquet}")

    # ---------------------------------------------------------
    # 4. PROSES DOKUMEN REGULASI / PERIZINAN
    # ---------------------------------------------------------
    print("\n[4/4] Memproses dokumen perizinan & proposal penelitian...")
    doc_records: List[Dict[str, Any]] = []
    pdf_dir = Path("data/raw/survei_offline/Surat izin penelitian")
    if pdf_dir.exists():
        for d_idx, pdf_path in enumerate(pdf_dir.glob("*.pdf")):
            try:
                reader = pypdf.PdfReader(pdf_path)
                full_text = " ".join([page.extract_text() or "" for page in reader.pages])
                clean_full = TextProcessor.clean_and_normalize(full_text)["teks_bersih"]
                doc_records.append({
                    "dok_id": f"DOK_{d_idx+1:02d}",
                    "file_name": pdf_path.name,
                    "pages": len(reader.pages),
                    "file_size_kb": round(pdf_path.stat().st_size / 1024, 2),
                    "institusi": "Dinas Koperasi UKM dan Perdagangan Kota Surabaya & UIN Sunan Ampel",
                    "ringkasan_teks": clean_full[:500],
                    "total_karakter": len(clean_full),
                })
            except Exception as e:
                print(f"[Warning] Gagal membaca PDF {pdf_path.name}: {e}")

    df_docs = pd.DataFrame(doc_records)
    docs_parquet = clean_dir / "dokumen.parquet"
    df_docs.to_parquet(docs_parquet, index=False)
    print(f"-> Berhasil mengekstrak {len(df_docs)} dokumen ke {docs_parquet}")

    print("\n=======================================================")
    print("           RINGKASAN ZONA CLEAN SELESAI               ")
    print("=======================================================")
    print(f"1. data/clean/umkm.parquet    : {len(df_umkm)} baris")
    print(f"2. data/clean/ulasan.parquet  : {len(df_reviews)} baris")
    print(f"3. data/clean/foto.parquet    : {len(df_photos)} baris (Wajah diburamkan: {total_faces_blurred})")
    print(f"4. data/clean/dokumen.parquet : {len(df_docs)} baris")
    print("=======================================================")


if __name__ == "__main__":
    build_clean_datasets()
