"""
Tahap 1 & 2 Mandiri:
1. Identifikasi seluruh gambar di sample/keep sebagai Positive Visual Anchors (CLIP ViT-B/32).
2. Memilah foto yang sudah ada maupun foto baru saat Ingestion agar hemat kuota API:
   - Hanya mempertahankan foto yang selaras dengan sample/keep (stan, etalase, kasir, QRIS, EDC,
     printer POS/struk, layar aplikasi merchant GoFood/GrabFood/ShopeeFood, laci kas/catatan stan).
   - Foto makanan, minuman, foto random tidak jelas/blur/meja kosong, dan foto AI/ilustrasi
     langsung di-reject dan dimasukkan ke folder sample/reject/.
3. Menambah kuota pengambilan data Google Places API secara efisien hingga melampaui
   Gerbang Volume (>= 200 MB foto & >= 2.000 baris multimodal).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFile

from src.umkm.ingestion.places_client import GooglePlacesIngestion
from src.umkm.ingestion.expand_umkm_photos import CHAIN_KEYWORDS, is_big_business

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = Path(__file__).resolve().parents[1]
SAMPLE_KEEP_DIR = BASE / "sample" / "keep"
SAMPLE_REJECT_DIR = BASE / "sample" / "reject"
RAW_MAPS = BASE / "data" / "raw" / "google_maps"
RAW_MAPS_EXT = BASE / "data" / "raw" / "google_maps_ext"
SURVEY_DIR = BASE / "data" / "raw" / "survei_offline"
OUT_FILTER = BASE / "reports" / "yolo" / "filter"

# Kueri diperluas untuk menambah kuota UMKM di 6 kecamatan Surabaya Timur
EXPANDED_UMKM_QUERIES = [
    "warung makan {k} Surabaya",
    "depot kuliner {k} Surabaya",
    "kedai makanan {k} Surabaya",
    "angkringan {k} Surabaya",
    "pedagang kaki lima kuliner {k} Surabaya",
    "gerobak jajanan {k} Surabaya",
    "warkop kopi {k} Surabaya",
    "penyetan bebek ayam {k} Surabaya",
    "sentra wisata kuliner {k} Surabaya",
    "pujasera food court {k} Surabaya",
    "warung bakso mie ayam {k} Surabaya",
    "warung soto rawon pecel {k} Surabaya",
    "nasi goreng nasi campur {k} Surabaya",
    "jajanan pasar camilan {k} Surabaya",
]

KEEP_TEXT_PROMPTS = [
    "a photo of a QRIS QR code payment sticker or acrylic standee at a shop",
    "a photo of a street food stall or booth with signboard and menu banner",
    "a photo of a food stall glass display counter or etalase with banners",
    "a photo of a cashier counter with thermal receipt printer or POS machine",
    "a close-up photo of a smartphone screen showing food delivery merchant app orders",
    "a photo of printed thermal receipts for online food delivery orders",
    "a photo of a GoFood GrabFood ShopeeFood delivery banner or sticker on a stall",
    "a photo of a cash drawer with banknotes or manual ledger book at a food stall",
    "a photo of a customer paying at a food court stall counter",
]

REJECT_TEXT_PROMPTS = {
    "makanan": [
        "a close-up photo of a plate or bowl of cooked food",
        "a photo of rice, meat, noodles, soup, or snacks served on a dish",
        "a photo of food on a dining table ready to eat",
    ],
    "minuman": [
        "a close-up photo of a glass, cup, or bottle of drink, coffee, tea, or juice",
        "a photo of iced drinks or beverages on a table",
    ],
    "foto_ai_grafis": [
        "an AI generated digital illustration or 3D render of food",
        "a digital promotional poster graphic, flyer, or social media banner design",
        "a cartoon or vector logo graphic",
    ],
    "random_gajelas": [
        "a blurry, dark, or unclear random photo",
        "a selfie or group portrait of people sitting at a table",
        "an empty dining room with chairs and tables or parking lot street view",
        "a close-up of a printed paper menu list only without stall or QR code",
    ],
}


class SampleGuidedFilter:
    """Filter berbasis embedding visual sample/keep + zero-shot CLIP."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.keep_paths = sorted(
            [p for p in SAMPLE_KEEP_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not self.keep_paths:
            raise RuntimeError(f"Tidak ada file gambar di {SAMPLE_KEEP_DIR}!")

        self.keep_embs, self.keep_names = self._encode_keep_anchors()
        self.txt_embs, self.txt_labels, self.txt_is_keep = self._encode_text_prompts()
        print(f"[SampleGuidedFilter] Berhasil mengidentifikasi {len(self.keep_paths)} foto acuan di sample/keep.")

    def _encode_keep_anchors(self) -> Tuple[torch.Tensor, List[str]]:
        embs = []
        names = []
        with torch.no_grad():
            for p in self.keep_paths:
                pil = Image.open(p).convert("RGB")
                tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
                feat = self.model.encode_image(tensor).float()
                feat /= feat.norm(dim=-1, keepdim=True)
                embs.append(feat)
                names.append(p.name)
        return torch.cat(embs, dim=0), names

    def _encode_text_prompts(self) -> Tuple[torch.Tensor, List[str], np.ndarray]:
        prompts = []
        labels = []
        is_keep = []
        for p in KEEP_TEXT_PROMPTS:
            prompts.append(p)
            labels.append("keep_target")
            is_keep.append(True)
        for cat, ps in REJECT_TEXT_PROMPTS.items():
            for p in ps:
                prompts.append(p)
                labels.append(cat)
                is_keep.append(False)
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            txt = self.model.encode_text(tokens).float()
            txt /= txt.norm(dim=-1, keepdim=True)
        return txt, labels, np.array(is_keep, dtype=bool)

    def evaluate_image(self, img_path: Path) -> Dict:
        """Evaluasi apakah sebuah foto mirip dengan sample/keep atau harus di-reject."""
        try:
            pil = Image.open(img_path).convert("RGB")
            arr = np.asarray(pil)
            if arr.std() < 10.0 or min(pil.size) < 120:
                return {
                    "accepted": False,
                    "category": "random_gajelas",
                    "reason": "reject:random_gajelas (gambar kosong/terlalu kecil)",
                    "max_keep_sim": 0.0,
                    "mean_top3_keep_sim": 0.0,
                    "p_keep_prompt": 0.0,
                    "nearest_keep_sample": "",
                }
            # Cek keburaman ekstrem (Laplacian variance)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_var < 18.0:
                return {
                    "accepted": False,
                    "category": "random_gajelas",
                    "reason": f"reject:random_gajelas (blur ekstrem var={blur_var:.1f})",
                    "max_keep_sim": 0.0,
                    "mean_top3_keep_sim": 0.0,
                    "p_keep_prompt": 0.0,
                    "nearest_keep_sample": "",
                }

            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        except Exception as e:
            return {
                "accepted": False,
                "category": "random_gajelas",
                "reason": f"reject:random_gajelas (unreadable: {e})",
                "max_keep_sim": 0.0,
                "mean_top3_keep_sim": 0.0,
                "p_keep_prompt": 0.0,
                "nearest_keep_sample": "",
            }

        with torch.no_grad():
            feat = self.model.encode_image(tensor).float()
            feat /= feat.norm(dim=-1, keepdim=True)

            # 1. Kemiripan kosinus terhadap 43 gambar sample/keep
            sims = (feat @ self.keep_embs.T).cpu().numpy()[0]
            best_idx = int(sims.argmax())
            max_sim = float(sims[best_idx])
            top3_sim = float(np.sort(sims)[-3:].mean())
            nearest_sample = self.keep_names[best_idx]

            # 2. Probabilitas prompt CLIP (Keep vs Makanan/Minuman/AI/Random)
            logits = (100.0 * feat @ self.txt_embs.T).softmax(dim=-1).cpu().numpy()[0]
            p_keep = float(logits[self.txt_is_keep].sum())

            # Cari kategori reject tertinggi jika ditolak
            reject_indices = np.where(~self.txt_is_keep)[0]
            best_rej_idx = reject_indices[int(logits[reject_indices].argmax())]
            best_rej_cat = self.txt_labels[best_rej_idx]

        # Aturan Keputusan Ketat Berbasis sample/keep:
        # Diterima HANYA jika:
        # - Probabilitas prompt keep >= 0.58 DAN kemiripan ke sample/keep memadai (max_sim >= 0.62 atau top3 >= 0.59)
        # - ATAU sangat identik dengan sample/keep (max_sim >= 0.74 dan p_keep >= 0.45)
        accepted = bool(
            (p_keep >= 0.58 and (max_sim >= 0.62 or top3_sim >= 0.59))
            or (max_sim >= 0.74 and p_keep >= 0.45)
        )

        if accepted:
            category = "keep_target"
            reason = f"keep:mirip_{nearest_sample} (sim={max_sim:.2f}, p_keep={p_keep:.2f})"
        else:
            category = best_rej_cat
            reason = f"reject:{best_rej_cat} (sim={max_sim:.2f}, p_keep={p_keep:.2f})"

        return {
            "accepted": accepted,
            "category": category,
            "reason": reason,
            "max_keep_sim": round(max_sim, 4),
            "mean_top3_keep_sim": round(top3_sim, 4),
            "p_keep_prompt": round(p_keep, 4),
            "nearest_keep_sample": nearest_sample,
        }


def sync_sample_keep_to_survey(filter_engine: SampleGuidedFilter):
    """
    Pastikan foto di sample/keep juga diinventarisasi dan dikenali kategorinya.
    Simpan ringkasan identifikasi sample/keep ke reports/yolo/filter/sample_keep_profile.csv.
    """
    OUT_FILTER.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in filter_engine.keep_paths:
        pil = Image.open(p).convert("RGB")
        w, h = pil.size
        kb = round(p.stat().st_size / 1024, 1)
        with torch.no_grad():
            t = filter_engine.preprocess(pil).unsqueeze(0)
            f = filter_engine.model.encode_image(t).float()
            f /= f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ filter_engine.txt_embs[: len(KEEP_TEXT_PROMPTS)].T).softmax(dim=-1).cpu().numpy()[0]
            top_i = int(probs.argmax())
        rows.append({
            "filename": p.name,
            "width": w,
            "height": h,
            "size_kb": kb,
            "identified_target_type": KEEP_TEXT_PROMPTS[top_i],
            "confidence": round(float(probs[top_i]), 4),
        })
    df_prof = pd.DataFrame(rows)
    df_prof.to_csv(OUT_FILTER / "sample_keep_profile.csv", index=False)
    print("\n=== Hasil Identifikasi 43 Foto di sample/keep ===")
    print(df_prof["identified_target_type"].value_counts().to_string())
    return df_prof


def get_total_raw_volume_mb() -> float:
    files = (
        list(RAW_MAPS.glob("*/photos/*.jpg"))
        + list(RAW_MAPS_EXT.glob("*/photos/*.jpg"))
        + list(SURVEY_DIR.rglob("*.jpg"))
        + list(SURVEY_DIR.rglob("*.jpeg"))
        + list(SAMPLE_KEEP_DIR.glob("*.*"))
    )
    # Hindari hitung ganda jika ada file identik berdasarkan path
    unique_paths = {p.resolve() for p in files if p.is_file()}
    return sum(p.stat().st_size for p in unique_paths) / (1024 * 1024)


def run_quota_efficient_ingestion(filter_engine: SampleGuidedFilter, target_mb: float = 215.0):
    """
    Menambahkan kuota pengambilan data Google Maps secara efisien:
    - Memilah metadata foto sebelum unduh (ukuran resolusi wajar >= 400px, bukan rasio banner sempit).
    - Mengunduh dengan resolusi max_size=1200 agar detail QRIS/stiker tajam.
    - Langsung mengevaluasi foto dengan SampleGuidedFilter saat ingestion:
      * Jika sebuah UMKM sudah mendapatkan >= 2 foto target KEEP, hentikan unduhan foto berikutnya
        untuk tempat tersebut agar hemat kuota API!
      * Jika 2 foto pertama suatu tempat semuanya REJECT (makanan/minuman/AI), hentikan unduhan
        foto sisa dari tempat tersebut agar tidak membuang kuota!
    """
    current_mb = get_total_raw_volume_mb()
    print(f"\n=== Cek Volume Raw Data Saat Ini: {current_mb:.2f} MB (Target Gate >= {target_mb:.0f} MB) ===")
    if current_mb >= target_mb:
        print("-> Volume Raw Data sudah melampaui target >= 200 MB!")
        return

    cfg = yaml.safe_load((BASE / "configs" / "kecamatan.yaml").read_text(encoding="utf-8"))
    kec_list = [k["nama"] for k in cfg["kecamatan"]]
    bbox = {k["nama"]: k["bounding_box"] for k in cfg["kecamatan"]}

    client = GooglePlacesIngestion(output_base_dir=str(RAW_MAPS_EXT))
    conn = client.test_connection()
    print(f"-> Status API: {conn['status']} ({conn.get('message', '')})")
    if conn["status"] != "SUCCESS":
        print("[Warning] API tidak dapat dihubungi, melewati ekspansi online.")
        return

    seen_ids = set()
    for d in (RAW_MAPS, RAW_MAPS_EXT):
        for f in d.glob("*/places_metadata.json"):
            seen_ids.update(p["place_id"] for p in json.loads(f.read_text(encoding="utf-8")))

    added_places = 0
    added_reviews = 0
    added_photos = 0
    skipped_quota_saved = 0

    for kec in kec_list:
        if get_total_raw_volume_mb() >= target_mb and added_places >= 60:
            break

        kdir = RAW_MAPS_EXT / kec.lower().replace(" ", "_")
        pdir = kdir / "photos"
        pdir.mkdir(parents=True, exist_ok=True)
        meta_f = kdir / "places_metadata.json"
        rev_f = kdir / "reviews_metadata.json"
        places = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else []
        reviews = json.loads(rev_f.read_text(encoding="utf-8")) if rev_f.exists() else []
        bb = bbox[kec]

        print(f"\n[Smart Ingestion] Kecamatan: {kec} (Volume saat ini: {get_total_raw_volume_mb():.2f} MB)")
        for q_tmpl in EXPANDED_UMKM_QUERIES:
            if get_total_raw_volume_mb() >= target_mb and added_places >= 60:
                break
            q = q_tmpl.format(k=kec)
            res = client.search_places(q, max_results=20)
            for p in res:
                pid = p.get("id")
                loc = p.get("location", {}) or {}
                lat, lon = loc.get("latitude"), loc.get("longitude")
                if not pid or pid in seen_ids or lat is None:
                    continue
                if not (
                    bb["min_lat"] - 0.008 <= lat <= bb["max_lat"] + 0.008
                    and bb["min_lon"] - 0.008 <= lon <= bb["max_lon"] + 0.008
                ):
                    continue
                types = p.get("types", [])
                if not any(
                    t in types
                    for t in (
                        "restaurant",
                        "food",
                        "cafe",
                        "meal_takeaway",
                        "bakery",
                        "coffee_shop",
                        "food_store",
                        "fast_food_restaurant",
                    )
                ):
                    continue

                seen_ids.add(pid)
                big = is_big_business(p)
                places.append({
                    "place_id": pid,
                    "kecamatan": kec,
                    "name": (p.get("displayName") or {}).get("text", ""),
                    "formatted_address": p.get("formattedAddress"),
                    "lat": lat,
                    "lon": lon,
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("userRatingCount"),
                    "types": types,
                    "price_level": p.get("priceLevel"),
                    "is_big_business": big,
                    "query": q,
                    "source": "google_places_api_smart_ext",
                })
                added_places += 1

                for rv in p.get("reviews", []) or []:
                    reviews.append({
                        "place_id": pid,
                        "kecamatan": kec,
                        "rating": rv.get("rating"),
                        "text": (rv.get("text") or {}).get("text", ""),
                        "publish_time": rv.get("publishTime"),
                    })
                    added_reviews += 1

                # Pemilahan foto cerdas saat ingestion untuk menghemat kuota API
                raw_photo_list = p.get("photos") or []
                # Filter metadata foto: pilih yang resolusinya memadai dan rasio wajar
                candidate_photos = []
                for ph in raw_photo_list:
                    w_px = ph.get("widthPx", 0) or 0
                    h_px = ph.get("heightPx", 0) or 0
                    if w_px < 400 or h_px < 300:
                        skipped_quota_saved += 1
                        continue
                    aspect = max(w_px, h_px) / max(1, min(w_px, h_px))
                    if aspect > 2.4:  # lewati banner/panorama aneh
                        skipped_quota_saved += 1
                        continue
                    candidate_photos.append(ph)

                keep_found_for_place = 0
                reject_streak = 0
                for i, ph in enumerate(candidate_photos[:5]):
                    # Hemat kuota: jika sudah dapat 2 foto KEEP untuk UMKM ini, stop download!
                    if keep_found_for_place >= 2:
                        skipped_quota_saved += len(candidate_photos[:5]) - i
                        break
                    # Hemat kuota: jika 2 foto awal berturut-turut adalah makanan/minuman, stop!
                    if reject_streak >= 2 and keep_found_for_place == 0:
                        skipped_quota_saved += len(candidate_photos[:5]) - i
                        break

                    dest = pdir / f"{pid}_photo_{i}.jpg"
                    if not dest.exists():
                        if client.download_photo(ph["name"], dest, max_size=1200):
                            added_photos += 1
                            eval_res = filter_engine.evaluate_image(dest)
                            if eval_res["accepted"]:
                                keep_found_for_place += 1
                                reject_streak = 0
                            else:
                                reject_streak += 1
                            time.sleep(0.1)

        meta_f.write_text(json.dumps(places, ensure_ascii=False, indent=2), encoding="utf-8")
        rev_f.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")

    final_mb = get_total_raw_volume_mb()
    print("\n=== RINGKASAN EKSPANSI INGESTION HEMAT KUOTA ===")
    print(f"-> Tambahan Tempat UMKM Baru : +{added_places}")
    print(f"-> Tambahan Ulasan Baru      : +{added_reviews}")
    print(f"-> Tambahan Foto Diunduh     : +{added_photos}")
    print(f"-> Panggilan Foto Dihemat    : {skipped_quota_saved} request foto dihemat")
    print(f"-> Total Volume Raw Data Kini: {final_mb:.2f} MB")


def filter_all_and_populate_sample_reject(filter_engine: SampleGuidedFilter) -> pd.DataFrame:
    """
    Saring seluruh foto mentah menggunakan SampleGuidedFilter:
    - Foto yang diterima (sesuai sample/keep) masuk ke daftar accepted untuk data/clean/photos.
    - Foto yang ditolak (makanan, minuman, random foto gajelas, foto AI) disalin ke sample/reject/
      dengan penamaan kategori yang jelas: <kategori>__<nama_file>.jpg
    """
    SAMPLE_REJECT_DIR.mkdir(parents=True, exist_ok=True)
    # Bersihkan isi lama sample/reject jika ada
    for old_f in SAMPLE_REJECT_DIR.glob("*.*"):
        old_f.unlink()

    all_maps_photos = sorted(RAW_MAPS.glob("*/photos/*.jpg")) + sorted(RAW_MAPS_EXT.glob("*/photos/*.jpg"))
    print(f"\n=== Menyaring {len(all_maps_photos)} Foto Mentah Berdasarkan Acuan sample/keep ===")

    records = []
    seen_hashes = set()
    copied_reject_count = 0

    for idx, p in enumerate(all_maps_photos, 1):
        # Cek duplikat byte/hash terlebih dahulu
        try:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            continue
        if sha in seen_hashes:
            continue
        seen_hashes.add(sha)

        res = filter_engine.evaluate_image(p)
        pid = p.name.split("_photo_")[0]
        kec = p.parent.parent.name
        src_dir = p.parents[2].name

        records.append({
            "raw_path": str(p),
            "place_id": pid,
            "kecamatan": kec,
            "source_dir": src_dir,
            "p_target": res["p_keep_prompt"],
            "p_place_gate": res["max_keep_sim"],
            "mean_top3_keep_sim": res["mean_top3_keep_sim"],
            "top_group": "stan_or_indicator" if res["accepted"] else res["category"],
            "top_prompt": res["nearest_keep_sample"] if res["accepted"] else res["category"],
            "top_prob": res["max_keep_sim"],
            "accepted": res["accepted"],
            "reason": res["reason"],
        })

        # Jika ditolak, masukkan ke folder sample/reject/
        if not res["accepted"]:
            cat = res["category"]
            rej_dest = SAMPLE_REJECT_DIR / f"{cat}__{p.name}"
            try:
                shutil.copy2(p, rej_dest)
                copied_reject_count += 1
            except Exception:
                pass

        if idx % 100 == 0:
            print(f"   Diproses: {idx}/{len(all_maps_photos)} foto...")

    df_scores = pd.DataFrame(records)
    OUT_FILTER.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(OUT_FILTER / "clip_filter_scores.csv", index=False)

    n_acc = int(df_scores["accepted"].sum())
    n_rej = len(df_scores) - n_acc
    print("\n=== HASIL PEMILAHAN KEEP vs REJECT ===")
    print(f"-> Total Foto Unik Diperiksa : {len(df_scores)}")
    print(f"-> KEEP (Sesuai sample/keep) : {n_acc} foto")
    print(f"-> REJECT (Ke sample/reject) : {n_rej} foto (Disalin ke sample/reject/: {copied_reject_count} file)")
    print("\nRincian Kategori Foto di sample/reject/:")
    print(df_scores[~df_scores["accepted"]]["top_group"].value_counts().to_string())
    return df_scores


if __name__ == "__main__":
    engine = SampleGuidedFilter(device="cpu")
    sync_sample_keep_to_survey(engine)
    run_quota_efficient_ingestion(engine, target_mb=215.0)
    filter_all_and_populate_sample_reject(engine)
