"""
Phase 2: Build Clean Dataset.
Processes, validates, anonymizes, and saves multimodal data into clean Parquet tables.
Strictly ensures:
1. Customer reviews preserve their default original language (no cross-language alterations).
2. 100% genuine, verifiable data (zero synthetic/fabricated UMKM entities or dummy coordinates).
3. Comprehensive multimodal data linking real Google Places entities and primary SWK survey observations.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List
import pandas as pd
import pypdf

from model.ingestion.curate_clean_images import (
    CATALOG_PATH,
    extract_place_id,
    resolve_east_surabaya_subdistrict,
    sanitize_folder_name,
)
from model.processing.image_processor import ImageProcessor
from model.processing.text_processor import TextProcessor


def build_clean_datasets():
    print("=======================================================")
    print("  FASE 2: PEMROSESAN, ANONIMISASI & ZONA CLEAN (PARQUET)")
    print("=======================================================\n")

    clean_dir = Path("data/clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_photos_dir = clean_dir / "photos"
    for old_sub in ["online", "offline", "Google_Maps", "Survei_Lapangan_SWK"]:
        old_path = clean_photos_dir / old_sub
        if old_path.exists():
            shutil.rmtree(old_path, ignore_errors=True)
    clean_photos_dir.mkdir(parents=True, exist_ok=True)

    img_processor = ImageProcessor(clean_photo_dir=str(clean_photos_dir))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else {}

    # ---------------------------------------------------------
    # 1. PROSES TABEL UMKM (100% GOOGLE PLACES API SURABAYA TIMUR)
    # ---------------------------------------------------------
    print("[1/4] Memproses data tabular UMKM terverifikasi...")
    raw_maps_dirs = [
        Path("data/raw/google_maps"),
        Path("data/raw/google_maps_ext"),
        Path("data/raw/google_maps_hires"),
    ]
    umkm_records: List[Dict[str, Any]] = []
    place_id_to_umkm_id: Dict[str, str] = {}
    place_id_to_info: Dict[str, Dict[str, str]] = {}

    counter = 1
    for raw_dir in raw_maps_dirs:
        if not raw_dir.exists():
            continue
        for meta_file in sorted(raw_dir.glob("*/places_metadata.json")):
            with open(meta_file, "r", encoding="utf-8") as f:
                places = json.load(f)

            for p in places:
                pid = p.get("place_id")
                if not pid or pid in place_id_to_umkm_id:
                    continue

                addr = p.get("formatted_address", "")
                verified_kec = resolve_east_surabaya_subdistrict(addr) or catalog.get(pid, {}).get("kecamatan_verified")
                if not verified_kec:
                    continue

                name = catalog.get(pid, {}).get("name") or p.get("name", "")
                umkm_id = f"UMKM_SBY_{counter:04d}"
                counter += 1
                place_id_to_umkm_id[pid] = umkm_id
                place_id_to_info[pid] = {"name": name, "kecamatan": verified_kec}

                name_lower = name.lower()
                is_swk = (
                    "sentra wisata kuliner" in name_lower
                    or "wisata kuliner" in name_lower
                    or "swk" in name_lower
                )
                swk_name = None
                if "convention" in name_lower:
                    swk_name = "SWK Convention Hall"
                elif "deles" in name_lower:
                    swk_name = "SWK Deles"
                elif "semolowaru" in name_lower:
                    swk_name = "SWK Semolowaru"
                elif "bratang" in name_lower:
                    swk_name = "SWK Bratang Binangun"
                elif "dharmahusada" in name_lower:
                    swk_name = "SWK Dharmahusada"
                elif "rungkut" in name_lower:
                    swk_name = "SWK Rungkut Kidul"

                umkm_records.append({
                    "umkm_id": umkm_id,
                    "place_id": pid,
                    "nama_umkm": name,
                    "kecamatan": verified_kec,
                    "alamat": addr or catalog.get(pid, {}).get("formatted_address", ""),
                    "lat": p.get("lat"),
                    "lon": p.get("lon"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("user_ratings_total"),
                    "tipe_lokasi": "SWK" if is_swk else "Reguler",
                    "swk_nama": swk_name,
                    "sumber": "google_places_api",
                })

    # Tambahkan tempat dari katalog verified yang fotonya sudah dikurasi bila belum tercatat di places_metadata
    for pid, info in catalog.items():
        if pid in place_id_to_umkm_id or not info.get("kecamatan_verified"):
            continue
        umkm_id = f"UMKM_SBY_{counter:04d}"
        counter += 1
        place_id_to_umkm_id[pid] = umkm_id
        place_id_to_info[pid] = {"name": info.get("name", pid), "kecamatan": info["kecamatan_verified"]}
        umkm_records.append({
            "umkm_id": umkm_id,
            "place_id": pid,
            "nama_umkm": info.get("name", pid),
            "kecamatan": info["kecamatan_verified"],
            "alamat": info.get("formatted_address", ""),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "rating": info.get("rating"),
            "user_ratings_total": info.get("user_ratings_total"),
            "tipe_lokasi": "SWK" if "swk" in info.get("name", "").lower() else "Reguler",
            "swk_nama": None,
            "sumber": "google_places_api",
        })

    df_umkm = pd.DataFrame(umkm_records)
    umkm_parquet = clean_dir / "umkm.parquet"
    df_umkm.to_parquet(umkm_parquet, index=False)
    print(f"-> Berhasil memvalidasi & menyimpan {len(df_umkm)} entitas UMKM asli ke {umkm_parquet}")

    # Identifikasi entitas SWK Sukolilo untuk relasi foto survei primer
    swk_sukolilo_id = None
    for _, row in df_umkm.iterrows():
        if row["kecamatan"] == "Sukolilo" and row["tipe_lokasi"] == "SWK":
            swk_sukolilo_id = row["umkm_id"]
            break
    if not swk_sukolilo_id and len(df_umkm) > 0:
        swk_sukolilo_id = df_umkm.iloc[0]["umkm_id"]

    # ---------------------------------------------------------
    # 2. PROSES & ANONIMISASI TEKS ULASAN (BAHASA ASLI DIPERTAHANKAN)
    # ---------------------------------------------------------
    print("\n[2/4] Menganonimkan teks ulasan dengan mempertahankan bahasa asli...")
    review_records: List[Dict[str, Any]] = []
    rev_counter = 1
    seen_review_keys = set()

    for raw_dir in raw_maps_dirs:
        if not raw_dir.exists():
            continue
        for rev_file in sorted(raw_dir.glob("*/reviews_metadata.json")):
            with open(rev_file, "r", encoding="utf-8") as f:
                reviews = json.load(f)

            for r in reviews:
                pid = r.get("place_id")
                if pid not in place_id_to_umkm_id:
                    continue
                umkm_id = place_id_to_umkm_id[pid]
                kec_verified = place_id_to_info.get(pid, {}).get("kecamatan", r.get("kecamatan"))
                raw_text = r.get("text", "")
                if not raw_text.strip():
                    continue

                # Deduplicate reviews if collected across multiple queries
                rev_key = (pid, r.get("author_name"), raw_text.strip()[:60])
                if rev_key in seen_review_keys:
                    continue
                seen_review_keys.add(rev_key)

                cleaned_info = TextProcessor.clean_and_normalize(raw_text)
                rev_id = f"REV_{rev_counter:05d}"
                rev_counter += 1

                review_records.append({
                    "ulasan_id": rev_id,
                    "umkm_id": umkm_id,
                    "place_id": pid,
                    "kecamatan": kec_verified,
                    "rating_pengulas": r.get("rating"),
                    "bahasa": cleaned_info["bahasa"],
                    "teks_asli": cleaned_info["teks_asli"],
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
    print(f"-> Berhasil menyimpan {len(df_reviews)} ulasan (Bahasa asli dipertahankan) ke {reviews_parquet}")

    # ---------------------------------------------------------
    # 3. PROSES FOTO & BLUR WAJAH (PRIVASI) DARI SEMUA SUMBER ASLI
    # ---------------------------------------------------------
    print("\n[3/4] Menganonimkan foto (blur wajah OpenCV) & standardisasi...")
    photo_records: List[Dict[str, Any]] = []
    photo_counter = 1
    total_faces_blurred = 0

    # 3.1 Foto Online Google Maps (terstruktur per Kecamatan & Nama Tempat UMKM)
    place_photo_seq: Dict[str, int] = {}
    for p_dir in raw_maps_dirs:
        if not p_dir.exists():
            continue
        print(f"-> Memproses foto dari {p_dir}...")
        for pfile in sorted(p_dir.rglob("*.jpg")):
            pid = extract_place_id(pfile.name)
            pinfo = place_id_to_info.get(pid)
            if not pinfo:
                continue

            umkm_id = place_id_to_umkm_id[pid]
            kec_display = pinfo["kecamatan"]
            place_display = pinfo["name"] or pid
            kec_folder = sanitize_folder_name(kec_display, 30)
            place_folder = sanitize_folder_name(place_display, 50)

            seq = place_photo_seq.get(pid, 0) + 1
            place_photo_seq[pid] = seq
            clean_fname = f"{place_folder}_{seq:02d}.jpg"

            proc_res = img_processor.process_and_save(
                raw_path=pfile,
                subfolder=f"Google_Maps/{kec_folder}/{place_folder}",
                output_filename=clean_fname,
            )
            if proc_res:
                total_faces_blurred += proc_res["face_blurred_count"]
                photo_records.append({
                    "foto_id": f"FOTO_ONL_{photo_counter:04d}",
                    "umkm_id": umkm_id,
                    "place_id": pid,
                    "nama_tempat": place_display,
                    "sumber": "google_places_api",
                    "kecamatan": kec_display,
                    "label_ground_truth": "Unlabeled",
                    "label_id": -1,
                    "width": proc_res["width"],
                    "height": proc_res["height"],
                    "face_blurred_count": proc_res["face_blurred_count"],
                    "raw_path": proc_res["raw_path"],
                    "clean_path": proc_res["clean_path"],
                    "sha256": proc_res["clean_sha256"],
                })
                photo_counter += 1

    # 3.2 Foto Offline Survei SWK Lapangan Primer
    print("-> Memproses foto survei lapangan SWK Convention Hall & Deles...")
    survey_inv_path = Path("data/raw/survei_offline/survey_images_inventory.csv")
    if survey_inv_path.exists():
        df_survey_inv = pd.read_csv(survey_inv_path)
        class_seq: Dict[str, int] = {}
        for idx, row in df_survey_inv.iterrows():
            raw_path = Path(row["relative_path"])
            if not raw_path.exists():
                continue
            cls_label = sanitize_folder_name(str(row["label_ground_truth"]), 30)
            seq = class_seq.get(cls_label, 0) + 1
            class_seq[cls_label] = seq
            swk_fname = f"Stan_{cls_label}_{seq:02d}.jpg"

            proc_res = img_processor.process_and_save(
                raw_path=raw_path,
                subfolder=f"Survei_Lapangan_SWK/Sukolilo/Stan_{cls_label}",
                output_filename=swk_fname,
            )
            if proc_res:
                total_faces_blurred += proc_res["face_blurred_count"]
                photo_records.append({
                    "foto_id": f"FOTO_OFF_{idx+1:03d}",
                    "umkm_id": swk_sukolilo_id,
                    "place_id": "SWK_SURVEI_SUKOLILO",
                    "nama_tempat": f"SWK Sukolilo ({row['label_ground_truth']})",
                    "sumber": "survei_lapangan_swk",
                    "kecamatan": "Sukolilo",
                    "label_ground_truth": row["label_ground_truth"],
                    "label_id": int(row["label_id"]),
                    "width": proc_res["width"],
                    "height": proc_res["height"],
                    "face_blurred_count": proc_res["face_blurred_count"],
                    "raw_path": proc_res["raw_path"],
                    "clean_path": proc_res["clean_path"],
                    "sha256": proc_res["clean_sha256"],
                })

    df_photos = pd.DataFrame(photo_records)

    # Enrich with YOLO detections if available
    classes = ["qris", "edc", "pos", "merchant_online", "stan"]
    for c in classes:
        df_photos[f"det_{c}"] = 0
        df_photos[f"has_{c}"] = False
    df_photos["detected_classes"] = ""
    df_photos["num_indicators"] = 0
    df_photos["max_conf"] = 0.0

    from model.storage_config import GDRIVE_REPORTS_EVAL_DIR
    gdrive_det_csv = GDRIVE_REPORTS_EVAL_DIR / "yolo_outputs" / "detect" / "predictions_detect.csv"
    local_det_csv = Path("docs/reports/yolo/detect/predictions_detect.csv")
    det_csv = gdrive_det_csv if gdrive_det_csv.exists() else local_det_csv
    if det_csv.exists():
        df_det = pd.read_csv(det_csv)
        det_map = df_det.set_index("foto_id")
        for idx, row in df_photos.iterrows():
            fid = row["foto_id"]
            if fid in det_map.index:
                det_row = det_map.loc[fid]
                if isinstance(det_row, pd.DataFrame):
                    det_row = det_row.iloc[0]
                for c in classes:
                    val = int(det_row.get(f"det_{c}", 0) or 0)
                    df_photos.at[idx, f"det_{c}"] = val
                    df_photos.at[idx, f"has_{c}"] = val > 0
                df_photos.at[idx, "detected_classes"] = str(det_row.get("detected_classes", ""))
                df_photos.at[idx, "max_conf"] = float(det_row.get("max_conf", 0.0) or 0.0)
                df_photos.at[idx, "num_indicators"] = sum(df_photos.at[idx, f"has_{c}"] for c in classes)

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
        for d_idx, pdf_path in enumerate(sorted(pdf_dir.glob("*.pdf"))):
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
    print(f"1. data/clean/umkm.parquet    : {len(df_umkm)} baris (100% entitas Google Places nyata)")
    print(f"2. data/clean/ulasan.parquet  : {len(df_reviews)} baris (Bahasa asli dipertahankan)")
    print(f"3. data/clean/foto.parquet    : {len(df_photos)} baris (Wajah diburamkan: {total_faces_blurred})")
    print(f"4. data/clean/dokumen.parquet : {len(df_docs)} baris")
    print("=======================================================")


if __name__ == "__main__":
    build_clean_datasets()
