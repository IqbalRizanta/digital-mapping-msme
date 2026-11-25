"""
Kurasi, Validasi Alamat Surabaya Timur, Penyaringan Visual Ketat (CLIP ViT-B/32),
dan Penataan Direktori Terstruktur per Kecamatan & Nama Tempat UMKM.

Memastikan:
1. Kecamatan setiap UMKM diverifikasi langsung dari detail alamat (formatted_address).
   Entitas di luar 6 kecamatan Surabaya Timur dikeluarkan agar analisis spasial akurat.
2. Gerbang CLIP ketat menolak makanan, minuman, daftar menu/grafis AI, dan foto acak.
3. Ekspansi foto beresolusi tinggi (1600px) dari Google Places API disaring langsung
   hingga volume data bersih >= 200 MB.
4. Mendukung pengayaan/verifikasi halaman publik menggunakan Scrapling bila diperlukan.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import clip
import cv2
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from PIL import Image, ImageFile

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.ingestion.expand_umkm_photos import is_big_business
from model.ingestion.run_quota_expansion import KEEP_TEXT_PROMPTS

ImageFile.LOAD_TRUNCATED_IMAGES = True
load_dotenv(ROOT_DIR / ".env")

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_SOURCE_DIRS = [
    RAW_DIR / "google_maps",
    RAW_DIR / "google_maps_ext",
    RAW_DIR / "google_maps_hires",
]
SURVEY_DIR = RAW_DIR / "survei_offline"
SAMPLE_KEEP_DIR = ROOT_DIR / "data" / "sample" / "keep"
REJECT_DIR = ROOT_DIR / "data" / "sample" / "reject"
CATALOG_PATH = RAW_DIR / "verified_places_catalog.json"
FILTER_REPORT_PATH = ROOT_DIR / "docs" / "reports" / "yolo" / "filter" / "clip_audit_summary.json"

SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places"

EAST_SURABAYA_SUBDISTRICTS: Dict[str, Tuple[str, ...]] = {
    "Gubeng": ("kec. gubeng", "gubeng"),
    "Gunung Anyar": ("kec. gn. anyar", "kec. gunung anyar", "gn. anyar", "gunung anyar"),
    "Mulyorejo": ("kec. mulyorejo", "mulyorejo"),
    "Rungkut": ("kec. rungkut", "rungkut"),
    "Sukolilo": ("kec. sukolilo", "sukolilo"),
    "Tenggilis Mejoyo": ("kec. tenggilis mejoyo", "tenggilis mejoyo", "tenggilis"),
}

STRICT_REJECT_PROMPTS = {
    "Makanan": [
        "a close-up photo of a plate or bowl of cooked food",
        "a photo of rice, meat, noodles, soup, satay, or snacks served on a dish",
        "a photo of food on a dining table ready to eat",
        "a photo of a buffet tray or pan full of cooked dishes",
        "a photo of fried chicken, duck, fish, or grilled food closeup",
    ],
    "Minuman": [
        "a close-up photo of a glass, cup, or bottle of drink, coffee, tea, or juice",
        "a photo of iced beverages or drinks on a dining table",
    ],
    "Grafis_AI_Menu": [
        "an AI generated digital illustration or 3D render of food",
        "a digital promotional poster graphic, flyer, or social media banner design",
        "a close-up of a printed paper menu list, price board, or poster graphic",
        "a cartoon, logo, or vector graphic artwork",
    ],
    "Tidak_Relevan": [
        "a blurry, dark, or unclear random photo",
        "a selfie or portrait of people sitting at a dining table",
        "an empty dining room with chairs, ceiling, wall, or floor only",
        "a random outdoor street, sky, tree, car, motorcycle, or roof",
    ],
}

EXPANSION_QUERIES = [
    "warung makan {k} Surabaya",
    "kedai kopi warkop {k} Surabaya",
    "depot nasi pecel soto {k} Surabaya",
    "sentra wisata kuliner pujasera {k} Surabaya",
    "bakso mie ayam {k} Surabaya",
    "ayam penyet bebek goreng {k} Surabaya",
    "martabak terang bulan kaki lima {k} Surabaya",
    "angkringan warung tenda {k} Surabaya",
    "es teh kopi susu kedai {k} Surabaya",
    "nasi goreng mie goreng kaki lima {k} Surabaya",
    "penyetan sambelan lamongan {k} Surabaya",
    "rujak cingur gado gado {k} Surabaya",
    "kantin warung kopi {k} Surabaya",
    "kuliner tenda gerobak {k} Surabaya",
]


def sanitize_folder_name(name: str, max_len: int = 55) -> str:
    """Konversi nama tempat atau kecamatan menjadi nama folder/file yang rapi dan aman di Windows."""
    cleaned = re.sub(r'[<>:"/\\|?*\'`]+', "", name.strip())
    cleaned = re.sub(r"[\s\-&,().]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return (cleaned[:max_len].strip("_")) or "Tanpa_Nama"


def resolve_east_surabaya_subdistrict(address: str) -> Optional[str]:
    """
    Verifikasi kecamatan asli berdasarkan ringkasan detail alamat (formatted_address).
    Mengembalikan nama kecamatan Surabaya Timur baku atau None bila di luar wilayah target.
    """
    if not address:
        return None
    addr_low = address.lower()

    # Prioritaskan kecocokan eksplisit 'Kec. <Nama>' terlebih dahulu agar nama jalan tidak menipu
    kec_match = re.search(r"kec\.\s*([^,]+)", addr_low)
    if kec_match:
        kec_text = kec_match.group(1).strip()
        for std_name, aliases in EAST_SURABAYA_SUBDISTRICTS.items():
            if any(alias in kec_text for alias in aliases):
                return std_name
        return None

    for std_name, aliases in EAST_SURABAYA_SUBDISTRICTS.items():
        if any(alias in addr_low for alias in aliases):
            return std_name
    return None


def extract_place_id(filename: str) -> str:
    """Ekstrak Google place_id dari pola penamaan file lama maupun baru."""
    stem = Path(filename).stem
    if stem.startswith("clean_"):
        stem = stem[6:]
    for sep in ("_photo_", "_hires_", "_exp_"):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


class StrictVisualFilter:
    """Gerbang ganda CLIP ViT-B/32 ketat untuk menyaring citra stan/indikator vs makanan/AI/random."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.keep_paths = sorted(
            [p for p in SAMPLE_KEEP_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not self.keep_paths:
            raise RuntimeError(f"Tidak ada file acuan di {SAMPLE_KEEP_DIR}")

        self.keep_embs = self._encode_anchors()
        self.txt_embs, self.txt_labels, self.txt_is_keep = self._encode_prompts()
        print(f"[StrictVisualFilter] Memuat {len(self.keep_paths)} foto acuan target.")

    def _encode_anchors(self) -> torch.Tensor:
        embs = []
        with torch.no_grad():
            for p in self.keep_paths:
                pil = Image.open(p).convert("RGB")
                tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
                feat = self.model.encode_image(tensor).float()
                feat /= feat.norm(dim=-1, keepdim=True)
                embs.append(feat)
        return torch.cat(embs, dim=0)

    def _encode_prompts(self) -> Tuple[torch.Tensor, List[str], np.ndarray]:
        prompts, labels, is_keep = [], [], []
        for p in KEEP_TEXT_PROMPTS:
            prompts.append(p)
            labels.append("Keep_Target")
            is_keep.append(True)
        for cat, ps in STRICT_REJECT_PROMPTS.items():
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
        """Evaluasi ketat: tolak makanan, minuman, poster/AI, dan foto buram/tidak relevan."""
        try:
            pil = Image.open(img_path).convert("RGB")
            arr = np.asarray(pil)
            if arr.std() < 10.0 or min(pil.size) < 120:
                return {"accepted": False, "category": "Tidak_Relevan", "p_keep": 0.0, "max_sim": 0.0}
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 22.0:
                return {"accepted": False, "category": "Tidak_Relevan", "p_keep": 0.0, "max_sim": 0.0}
            tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        except Exception:
            return {"accepted": False, "category": "Tidak_Relevan", "p_keep": 0.0, "max_sim": 0.0}

        with torch.no_grad():
            feat = self.model.encode_image(tensor).float()
            feat /= feat.norm(dim=-1, keepdim=True)
            sims = (feat @ self.keep_embs.T).cpu().numpy()[0]
            max_sim = float(sims.max())
            top3_sim = float(np.sort(sims)[-3:].mean())

            logits = (100.0 * feat @ self.txt_embs.T).softmax(dim=-1).cpu().numpy()[0]
            p_keep = float(logits[self.txt_is_keep].sum())

            rej_idx = np.where(~self.txt_is_keep)[0]
            best_rej_idx = rej_idx[int(logits[rej_idx].argmax())]
            best_rej_cat = self.txt_labels[best_rej_idx]

        # Aturan ketat: wajib p_keep >= 0.78 agar foto makanan/minuman/menu tidak pernah lolos
        accepted = bool(p_keep >= 0.78 and max_sim >= 0.65 and top3_sim >= 0.62)
        return {
            "accepted": accepted,
            "category": "Keep_Target" if accepted else best_rej_cat,
            "p_keep": round(p_keep, 4),
            "max_sim": round(max_sim, 4),
        }


def fetch_place_details(place_id: str, api_key: str) -> Optional[Dict]:
    """Ambil detail resmi (nama & alamat lengkap) dari Google Places API v1."""
    if not api_key or not place_id:
        return None
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "id,displayName,formattedAddress,location,rating,userRatingCount",
    }
    try:
        resp = requests.get(f"{PLACE_DETAILS_URL}/{place_id}", headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        d = resp.json()
        loc = d.get("location", {})
        return {
            "place_id": d.get("id", place_id),
            "name": d.get("displayName", {}).get("text", ""),
            "formatted_address": d.get("formattedAddress", ""),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
            "rating": d.get("rating"),
            "user_ratings_total": d.get("userRatingCount"),
        }
    except Exception:
        return None


def build_verified_place_catalog(api_key: str) -> Dict[str, Dict]:
    """Bangun katalog place_id -> info tempat lengkap dengan verifikasi kecamatan dari alamat."""
    catalog: Dict[str, Dict] = {}
    if CATALOG_PATH.exists():
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    for raw_dir in RAW_SOURCE_DIRS:
        if not raw_dir.exists():
            continue
        for meta_file in sorted(raw_dir.rglob("places_metadata.json")):
            places = json.loads(meta_file.read_text(encoding="utf-8"))
            for p in places:
                pid = p.get("place_id") or p.get("id")
                if not pid:
                    continue
                addr = p.get("formatted_address", "")
                verified_kec = resolve_east_surabaya_subdistrict(addr)
                catalog[pid] = {
                    "place_id": pid,
                    "name": p.get("name", ""),
                    "formatted_address": addr,
                    "kecamatan_verified": verified_kec,
                    "in_east_surabaya": verified_kec is not None,
                    "lat": p.get("lat"),
                    "lon": p.get("lon"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("user_ratings_total"),
                }

    # Lengkapi metadata untuk place_id yang belum punya catatan alamat/nama
    missing_pids: Set[str] = set()
    for raw_dir in RAW_SOURCE_DIRS:
        if raw_dir.exists():
            for img in raw_dir.rglob("*.jpg"):
                pid = extract_place_id(img.name)
                if pid not in catalog or not catalog[pid].get("formatted_address"):
                    missing_pids.add(pid)

    if missing_pids:
        print(f"[Katalog] Melengkapi detail alamat Google Maps untuk {len(missing_pids)} tempat...")
        for pid in sorted(missing_pids):
            details = fetch_place_details(pid, api_key)
            if details:
                addr = details.get("formatted_address", "")
                verified_kec = resolve_east_surabaya_subdistrict(addr)
                catalog[pid] = {
                    **details,
                    "kecamatan_verified": verified_kec,
                    "in_east_surabaya": verified_kec is not None,
                }
                time.sleep(0.1)

    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def move_rejected_photo(img_path: Path, category: str, kec_label: str, place_label: str) -> None:
    """Pindahkan foto yang ditolak ke data/sample/reject/<Kategori>/<Kecamatan>/<Nama_Tempat>/."""
    dst_dir = REJECT_DIR / category / sanitize_folder_name(kec_label, 30) / sanitize_folder_name(place_label, 50)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / img_path.name
    if dst_file.exists():
        img_path.unlink(missing_ok=True)
    else:
        shutil.move(str(img_path), str(dst_file))


def audit_existing_photos(filter_engine: StrictVisualFilter, catalog: Dict[str, Dict]) -> Dict[str, int]:
    """Audit seluruh foto lokal: tolak wilayah luar Surabaya Timur dan tolak makanan/minuman/AI/random."""
    stats = {"kept": 0, "rejected_visual": 0, "rejected_region": 0, "kept_bytes": 0}
    for raw_dir in RAW_SOURCE_DIRS:
        if not raw_dir.exists():
            continue
        files = sorted(raw_dir.rglob("*.jpg"))
        print(f"\n[Audit Ketat] Mengevaluasi {len(files)} foto di {raw_dir.relative_to(ROOT_DIR)}...")
        for img_path in files:
            pid = extract_place_id(img_path.name)
            info = catalog.get(pid, {})
            place_name = info.get("name") or pid
            kec_verified = info.get("kecamatan_verified")

            if not kec_verified:
                stats["rejected_region"] += 1
                move_rejected_photo(img_path, "Luar_Surabaya_Timur", "Luar_Wilayah", place_name)
                continue

            res = filter_engine.evaluate_image(img_path)
            if res["accepted"]:
                stats["kept"] += 1
                stats["kept_bytes"] += img_path.stat().st_size
            else:
                stats["rejected_visual"] += 1
                move_rejected_photo(img_path, res["category"], kec_verified, place_name)
    return stats


def compute_survey_bytes() -> int:
    inv_csv = SURVEY_DIR / "survey_images_inventory.csv"
    if not inv_csv.exists():
        return 0
    df = pd.read_csv(inv_csv)
    return sum((ROOT_DIR / str(r["relative_path"])).stat().st_size for _, r in df.iterrows() if (ROOT_DIR / str(r["relative_path"])).exists())


def expand_strict_east_surabaya_photos(
    filter_engine: StrictVisualFilter,
    catalog: Dict[str, Dict],
    current_clean_bytes: int,
    api_key: str,
    target_mb: float = 212.0,
) -> Dict[str, int]:
    """
    Unduh foto 1600px dari Google Places API khusus kecamatan Surabaya Timur terverifikasi
    dan hanya terima foto yang lolos gerbang ketat (p_keep >= 0.78).
    """
    target_bytes = int(target_mb * 1024 * 1024)
    exp_stats = {"downloaded_kept": 0, "downloaded_rejected": 0, "added_bytes": 0}
    if current_clean_bytes >= target_bytes or not api_key:
        return exp_stats

    target_dir = RAW_DIR / "google_maps_hires"
    seen_pids: Set[str] = {extract_place_id(p.name) for d in RAW_SOURCE_DIRS if d.exists() for p in d.rglob("*.jpg")}
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.photos",
    }

    for kec_name in EAST_SURABAYA_SUBDISTRICTS:
        if current_clean_bytes >= target_bytes:
            break
        kec_slug = sanitize_folder_name(kec_name).lower()
        kec_dir = target_dir / kec_slug
        kec_dir.mkdir(parents=True, exist_ok=True)
        meta_path = kec_dir / "places_metadata.json"
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        existing_pids = {m.get("place_id") for m in existing_meta if m.get("place_id")}

        for q_tmpl in EXPANSION_QUERIES:
            if current_clean_bytes >= target_bytes:
                break
            query = q_tmpl.format(k=kec_name)
            print(f"\n[Ekspansi 1600px] '{query}' (Clean: {current_clean_bytes/(1024*1024):.2f} MB)")
            try:
                resp = requests.post(SEARCH_TEXT_URL, json={"textQuery": query, "pageSize": 15}, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue
                places = resp.json().get("places", [])
            except Exception:
                continue

            for p in places:
                if current_clean_bytes >= target_bytes:
                    break
                pid = p.get("id")
                pname = p.get("displayName", {}).get("text", "")
                addr = p.get("formattedAddress", "")
                verified_kec = resolve_east_surabaya_subdistrict(addr)

                # Wajib berada di dalam 6 kecamatan Surabaya Timur berdasarkan detail alamat
                if not pid or is_big_business(p) or not verified_kec or pid in seen_pids:
                    continue
                seen_pids.add(pid)

                ver_slug = sanitize_folder_name(verified_kec).lower()
                ver_dir = target_dir / ver_slug
                ver_dir.mkdir(parents=True, exist_ok=True)

                loc = p.get("location", {})
                catalog[pid] = {
                    "place_id": pid,
                    "name": pname,
                    "formatted_address": addr,
                    "kecamatan_verified": verified_kec,
                    "in_east_surabaya": True,
                    "lat": loc.get("latitude"),
                    "lon": loc.get("longitude"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("userRatingCount"),
                }
                if pid not in existing_pids:
                    existing_meta.append({
                        "place_id": pid,
                        "name": pname,
                        "kecamatan": verified_kec,
                        "formatted_address": addr,
                        "lat": loc.get("latitude"),
                        "lon": loc.get("longitude"),
                        "rating": p.get("rating"),
                        "user_ratings_total": p.get("userRatingCount"),
                    })
                    existing_pids.add(pid)

                consecutive_rej = 0
                for idx, ph in enumerate(p.get("photos", [])[:8]):
                    if current_clean_bytes >= target_bytes:
                        break
                    ref = ph.get("name")
                    if not ref:
                        continue
                    out_path = ver_dir / f"{pid}_hires_{idx}.jpg"
                    if out_path.exists():
                        continue
                    try:
                        p_resp = requests.get(f"https://places.googleapis.com/v1/{ref}/media?maxWidthPx=1600&key={api_key}", stream=True, timeout=15)
                        if p_resp.status_code != 200:
                            continue
                        with open(out_path, "wb") as f:
                            for chunk in p_resp.iter_content(chunk_size=16384):
                                f.write(chunk)

                        res = filter_engine.evaluate_image(out_path)
                        if res["accepted"]:
                            sz = out_path.stat().st_size
                            current_clean_bytes += sz
                            exp_stats["downloaded_kept"] += 1
                            exp_stats["added_bytes"] += sz
                            consecutive_rej = 0
                            print(f"  + KEEP [{verified_kec}] [{pname[:22]}] #{idx} ({sz/1024:.0f} KB, p={res['p_keep']:.2f})")
                        else:
                            exp_stats["downloaded_rejected"] += 1
                            consecutive_rej += 1
                            move_rejected_photo(out_path, res["category"], verified_kec, pname)
                            if consecutive_rej >= 3:
                                break
                        time.sleep(0.15)
                    except Exception:
                        continue

        meta_path.write_text(json.dumps(existing_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return exp_stats


def main() -> None:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    catalog = build_verified_place_catalog(api_key)
    filter_engine = StrictVisualFilter(device="cpu")

    audit_stats = audit_existing_photos(filter_engine, catalog)
    survey_bytes = compute_survey_bytes()
    clean_bytes = audit_stats["kept_bytes"] + survey_bytes

    print(f"\n[Pasca Audit Ketat] Lolos: {audit_stats['kept']} foto | Tolak Visual: {audit_stats['rejected_visual']} | Tolak Luar Surabaya Timur: {audit_stats['rejected_region']} | Bersih: {clean_bytes/(1024*1024):.2f} MB")

    exp_stats = expand_strict_east_surabaya_photos(filter_engine, catalog, clean_bytes, api_key, target_mb=212.0)
    final_clean_mb = (clean_bytes + exp_stats["added_bytes"]) / (1024 * 1024)

    summary = {
        "audit_strict": audit_stats,
        "survey_bytes": survey_bytes,
        "expansion": exp_stats,
        "final_clean_mb": round(final_clean_mb, 2),
    }
    FILTER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FILTER_REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== HASIL KURASI & VALIDASI ALAMAT SURABAYA TIMUR ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
