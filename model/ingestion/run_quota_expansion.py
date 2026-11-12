"""
Tahap 1 & 2 Mandiri (Full Execution):
1. Identifikasi seluruh 43 gambar di sample/keep sebagai Positive Visual Anchors (CLIP ViT-B/32).
2. Ekspansi Kuota Ingestion Google Places API secara cerdas & hemat kuota:
   - Menjalankan kueri baru di 6 kecamatan Surabaya Timur dengan kuota diperbesar.
   - Memilah foto SAAT INGESTION:
     * Cek dimensi & rasio foto dari metadata API sebelum download (hemat kuota dari thumbnail/banner rusak).
     * Begitu foto diunduh, langsung diuji terhadap acuan sample/keep:
       - Bila cocok dengan sample/keep -> disimpan sebagai data mentah target.
       - Bila TIDAK cocok (makanan, minuman, random gajelas, foto AI) -> langsung dipindah ke sample/reject/
         dan hentikan unduhan foto berikutnya pada tempat tersebut bila 2x berturut-turut berisi makanan/non-target!
3. Penyaringan ketat menuju Tahap Clean Data (B2):
   - Hanya foto yang benar-benar selaras dengan sample/keep yang masuk ke tahap Clean Data.
   - Seluruh foto non-target dari pool lama maupun baru dikumpulkan di sample/reject/
     (makanan, minuman, random_gajelas, foto_ai_grafis).
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

from model.ingestion.places_client import GooglePlacesIngestion
from model.ingestion.expand_umkm_photos import CHAIN_KEYWORDS, is_big_business

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = Path(__file__).resolve().parents[2]
SAMPLE_KEEP_DIR = BASE / "data" / "sample" / "keep"
SAMPLE_REJECT_DIR = BASE / "data" / "sample" / "reject"
RAW_MAPS = BASE / "data" / "raw" / "google_maps"
RAW_MAPS_EXT = BASE / "data" / "raw" / "google_maps_ext"
SURVEY_DIR = BASE / "data" / "raw" / "survei_offline"
OUT_FILTER = BASE / "peta-digital-umkm" / "reports" / "yolo" / "filter"

# Kueri tambahan untuk memperbesar kuota data UMKM di 6 kecamatan Surabaya Timur
NEW_QUOTA_QUERIES = [
    "warung bakso mie ayam {k} Surabaya",
    "warung soto rawon pecel {k} Surabaya",
    "nasi goreng nasi campur {k} Surabaya",
    "jajanan pasar camilan {k} Surabaya",
    "kuliner malam kaki lima {k} Surabaya",
    "ayam bakar bebek goreng {k} Surabaya",
    "es teh kopi susu kedai {k} Surabaya",
    "kantin pujasera {k} Surabaya",
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
        "a photo of a buffet tray or pan full of food",
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
        "a random outdoor street, tree, car, motorcycle, or building roof",
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
        """Evaluasi ketat apakah sebuah foto mirip dengan sample/keep atau harus di-reject."""
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
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_var < 20.0:
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

            reject_indices = np.where(~self.txt_is_keep)[0]
            best_rej_idx = reject_indices[int(logits[reject_indices].argmax())]
            best_rej_cat = self.txt_labels[best_rej_idx]

        # Kriteria ketat berbasis sample/keep:
        # Harus memiliki kemiripan kuat dengan foto di sample/keep DAN prompt keep dominan
        accepted = bool(
            (p_keep >= 0.62 and max_sim >= 0.64 and top3_sim >= 0.61)
            or (max_sim >= 0.75 and p_keep >= 0.50)
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


def run_quota_expansion_with_live_filtering(filter_engine: SampleGuidedFilter, max_new_places_per_kec: int = 35):
    """
    Menambah kuota pengambilan data UMKM & ulasan di 6 kecamatan Surabaya Timur,
    serta memilah gambar secara langsung saat ingestion:
    - Foto yang lolos kriteria sample/keep disimpan ke data/raw/google_maps_ext/<kec>/photos/
    - Foto yang gagal (makanan/minuman/random/AI) langsung dimasukkan ke sample/reject/
      dan sistem menghentikan unduhan foto sisa pada tempat tersebut jika 2 foto berturut-turut ditolak
      (menghemat kuota Photo API).
    """
    SAMPLE_REJECT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((BASE / "configs" / "kecamatan.yaml").read_text(encoding="utf-8"))
    kec_list = [k["nama"] for k in cfg["kecamatan"]]
    bbox = {k["nama"]: k["bounding_box"] for k in cfg["kecamatan"]}

    client = GooglePlacesIngestion(output_base_dir=str(RAW_MAPS_EXT))
    conn = client.test_connection()
    print(f"\n=== EKSPANSI KUOTA INGESTION CERDAS (Status API: {conn['status']}) ===")
    if conn["status"] != "SUCCESS":
        print("[Warning] API tidak aktif, melewati ekspansi jaringan.")
        return

    seen_ids = set()
    for d in (RAW_MAPS, RAW_MAPS_EXT):
        for f in d.glob("*/places_metadata.json"):
            seen_ids.update(p["place_id"] for p in json.loads(f.read_text(encoding="utf-8")))
    print(f"-> UMKM eksisting sebelum ekspansi kuota: {len(seen_ids)} tempat")

    added_places = 0
    added_reviews = 0
    kept_new_photos = 0
    rejected_new_photos = 0
    saved_photo_calls = 0

    for kec in kec_list:
        kdir = RAW_MAPS_EXT / kec.lower().replace(" ", "_")
        pdir = kdir / "photos"
        pdir.mkdir(parents=True, exist_ok=True)
        meta_f = kdir / "places_metadata.json"
        rev_f = kdir / "reviews_metadata.json"
        places = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else []
        reviews = json.loads(rev_f.read_text(encoding="utf-8")) if rev_f.exists() else []
        bb = bbox[kec]

        kec_new_places = 0
        print(f"\n-> Mengambil kuota tambahan untuk Kecamatan: {kec}...")
        for q_tmpl in NEW_QUOTA_QUERIES:
            if kec_new_places >= max_new_places_per_kec:
                break
            q = q_tmpl.format(k=kec)
            res = client.search_places(q, max_results=20)
            for p in res:
                if kec_new_places >= max_new_places_per_kec:
                    break
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
                kec_new_places += 1

                for rv in p.get("reviews", []) or []:
                    reviews.append({
                        "place_id": pid,
                        "kecamatan": kec,
                        "rating": rv.get("rating"),
                        "text": (rv.get("text") or {}).get("text", ""),
                        "publish_time": rv.get("publishTime"),
                    })
                    added_reviews += 1

                # Seleksi metadata foto sebelum download (hemat kuota)
                raw_photos = p.get("photos") or []
                candidates = []
                for ph in raw_photos:
                    w_px = ph.get("widthPx", 0) or 0
                    h_px = ph.get("heightPx", 0) or 0
                    if w_px < 400 or h_px < 300:
                        saved_photo_calls += 1
                        continue
                    aspect = max(w_px, h_px) / max(1, min(w_px, h_px))
                    if aspect > 2.3:
                        saved_photo_calls += 1
                        continue
                    candidates.append(ph)

                keep_for_this_place = 0
                reject_streak = 0
                for i, ph in enumerate(candidates[:4]):
                    # Jika sudah dapat 2 foto KEEP untuk UMKM ini, hemat kuota!
                    if keep_for_this_place >= 2:
                        saved_photo_calls += len(candidates[:4]) - i
                        break
                    # Jika 2 foto pertama berturut-turut REJECT (makanan/minuman/AI), hemat kuota!
                    if reject_streak >= 2 and keep_for_this_place == 0:
                        saved_photo_calls += len(candidates[:4]) - i
                        break

                    dest = pdir / f"{pid}_photo_{i}.jpg"
                    if not dest.exists() and client.download_photo(ph["name"], dest, max_size=1000):
                        eval_res = filter_engine.evaluate_image(dest)
                        if eval_res["accepted"]:
                            kept_new_photos += 1
                            keep_for_this_place += 1
                            reject_streak = 0
                        else:
                            # Foto baru yang ditolak langsung dipindah ke sample/reject/
                            # agar raw folder baru hanya menyimpan foto yang layak!
                            rejected_new_photos += 1
                            reject_streak += 1
                            cat = eval_res["category"]
                            rej_path = SAMPLE_REJECT_DIR / f"{cat}__{dest.name}"
                            shutil.move(str(dest), str(rej_path))
                        time.sleep(0.1)

        meta_f.write_text(json.dumps(places, ensure_ascii=False, indent=2), encoding="utf-8")
        rev_f.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   +{kec_new_places} UMKM baru di {kec}")

    print("\n=== RINGKASAN TAMBAHAN KUOTA & INGESTION HEMAT KUOTA ===")
    print(f"-> UMKM Baru Ditambahkan         : +{added_places} UMKM (Total kini: {len(seen_ids)} UMKM)")
    print(f"-> Ulasan Teks Baru Ditambahkan  : +{added_reviews} ulasan")
    print(f"-> Foto Baru Lolos (KEEP di Raw) : +{kept_new_photos} foto")
    print(f"-> Foto Baru Ditolak -> Reject   : +{rejected_new_photos} foto (dipindah ke sample/reject/)")
    print(f"-> Kuota API Foto Berhasil Dihemat: {saved_photo_calls} panggilan")


if __name__ == "__main__":
    from model.ingestion.sample_keep_filter import (
        sync_sample_keep_to_survey,
        filter_all_and_populate_sample_reject,
    )
    engine = SampleGuidedFilter(device="cpu")
    sync_sample_keep_to_survey(engine)
    # 1. Saring foto lama terlebih dahulu & isi sample/reject/
    filter_all_and_populate_sample_reject(engine)
    # 2. Tambah kuota baru dari Google Places API dengan pemilahan langsung saat ingestion
    run_quota_expansion_with_live_filtering(engine, max_new_places_per_kec=25)
    # 3. Perbarui skor manifest gabungan tanpa menghapus foto reject baru
    all_maps_photos = sorted(RAW_MAPS.glob("*/photos/*.jpg")) + sorted(RAW_MAPS_EXT.glob("*/photos/*.jpg"))
    rows = []
    seen_sha = set()
    for p in all_maps_photos:
        try:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            continue
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        res = engine.evaluate_image(p)
        rows.append({
            "raw_path": str(p),
            "place_id": p.name.split("_photo_")[0],
            "kecamatan": p.parent.parent.name,
            "source_dir": p.parents[2].name,
            "p_target": res["p_keep_prompt"],
            "p_place_gate": res["max_keep_sim"],
            "mean_top3_keep_sim": res["mean_top3_keep_sim"],
            "top_group": "stan_or_indicator" if res["accepted"] else res["category"],
            "top_prompt": res["nearest_keep_sample"] if res["accepted"] else res["category"],
            "top_prob": res["max_keep_sim"],
            "accepted": res["accepted"],
            "reason": res["reason"],
        })
    df_final = pd.DataFrame(rows)
    df_final.to_csv(OUT_FILTER / "clip_filter_scores.csv", index=False)
    print(f"\n=== FINAL POOL SETELAH TAMBAH KUOTA ===")
    print(f"-> Total Foto Raw Unik : {len(df_final)} | KEEP: {df_final['accepted'].sum()} | REJECT di sample/reject: {len(list(SAMPLE_REJECT_DIR.glob('*.*')))}")

