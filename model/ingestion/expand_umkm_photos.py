"""
Perluasan data foto Google Maps berfokus pada UMKM kuliner Surabaya Timur.

- Kueri diarahkan ke usaha mikro/kecil (warung, depot, kedai, angkringan, PKL, gerobak, SWK, dll).
- Mengambil hingga --max-photos foto per tempat (Places API memberi maks 10 referensi foto).
- Data disimpan terpisah di data/raw/google_maps_ext/<kecamatan>/ agar data lama tidak tertimpa.
- Tempat yang sudah ada di data/raw/google_maps dilewati (deduplikasi place_id).
- Penandaan usaha besar (is_big_business) dilakukan di sini; aturan "usaha besar wajib punya foto
  kasir/QRIS/EDC" diterapkan pada tahap clean di yolo_full_pipeline.py setelah verifikasi CLIP.

Pemakaian:
  python -m src.umkm.ingestion.expand_umkm_photos --max-places 20 --max-photos 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.umkm.ingestion.places_client import GooglePlacesIngestion

BASE = Path(__file__).resolve().parents[3]
OLD_DIR = BASE / "data" / "raw" / "google_maps"
EXT_DIR = BASE / "data" / "raw" / "google_maps_ext"

UMKM_QUERIES = [
    "warung makan {k} Surabaya",
    "depot {k} Surabaya",
    "kedai {k} Surabaya",
    "angkringan {k} Surabaya",
    "pedagang kaki lima {k} Surabaya",
    "gerobak makanan {k} Surabaya",
    "warkop {k} Surabaya",
    "penyetan {k} Surabaya",
    "sentra wisata kuliner {k} Surabaya",
    "pujasera food court {k} Surabaya",
]

CHAIN_KEYWORDS = [
    "mcdonald", "kfc", "starbucks", "solaria", "richeese", "burger king", "pizza hut", "hokben",
    "chatime", "janji jiwa", "kopi kenangan", "mixue", "wingstop", "a&w", "j.co", "jco", "dunkin",
    "yoshinoya", "marugame", "sushi tei", "gacoan", "point coffee", "fore coffee", "tomoro",
    "es teh indonesia", "haus!", "geprek bensu", "bakso boedjangan", "d'cost", "domino",
]


def is_big_business(place: dict) -> bool:
    name = (place.get("displayName", {}) or {}).get("text", "").lower()
    if any(k in name for k in CHAIN_KEYWORDS):
        return True
    if (place.get("userRatingCount") or 0) >= 1000:
        return True
    return place.get("priceLevel") in ("PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE")


def existing_place_ids() -> set[str]:
    ids = set()
    for d in (OLD_DIR, EXT_DIR):
        for f in d.glob("*/places_metadata.json"):
            ids.update(p["place_id"] for p in json.loads(f.read_text(encoding="utf-8")))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-places", type=int, default=20, help="maks tempat per kueri")
    ap.add_argument("--max-photos", type=int, default=6, help="maks foto per tempat")
    ap.add_argument("--kecamatan", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load((BASE / "configs" / "kecamatan.yaml").read_text(encoding="utf-8"))
    kec_list = [k["nama"] for k in cfg["kecamatan"]]
    bbox = {k["nama"]: k["bounding_box"] for k in cfg["kecamatan"]}
    if a.kecamatan:
        kec_list = [a.kecamatan]

    client = GooglePlacesIngestion(output_base_dir=str(EXT_DIR))
    print(client.test_connection())
    seen = existing_place_ids()
    print(f"-> place_id yang sudah ada (dilewati): {len(seen)}")

    grand = {"places": 0, "photos": 0, "big": 0}
    for kec in kec_list:
        kdir = EXT_DIR / kec.lower().replace(" ", "_")
        pdir = kdir / "photos"
        pdir.mkdir(parents=True, exist_ok=True)
        meta_f = kdir / "places_metadata.json"
        rev_f = kdir / "reviews_metadata.json"
        places = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else []
        reviews = json.loads(rev_f.read_text(encoding="utf-8")) if rev_f.exists() else []
        bb = bbox[kec]
        n_new = n_ph = 0
        print(f"\n=== {kec} ===")
        for q in UMKM_QUERIES:
            q = q.format(k=kec)
            res = client.search_places(q, max_results=a.max_places)
            print(f"-> '{q}': {len(res)} hasil")
            for p in res:
                pid = p.get("id")
                loc = p.get("location", {}) or {}
                lat, lon = loc.get("latitude"), loc.get("longitude")
                if not pid or pid in seen or lat is None:
                    continue
                # hanya yang berada di dalam bounding box kecamatan (toleransi kecil)
                if not (bb["min_lat"] - 0.005 <= lat <= bb["max_lat"] + 0.005 and
                        bb["min_lon"] - 0.005 <= lon <= bb["max_lon"] + 0.005):
                    continue
                types = p.get("types", [])
                if not any(t in types for t in ("restaurant", "food", "cafe", "meal_takeaway", "bakery",
                                                 "coffee_shop", "food_store", "fast_food_restaurant")):
                    continue
                seen.add(pid)
                big = is_big_business(p)
                places.append({
                    "place_id": pid, "kecamatan": kec,
                    "name": (p.get("displayName") or {}).get("text", ""),
                    "formatted_address": p.get("formattedAddress"), "lat": lat, "lon": lon,
                    "rating": p.get("rating"), "user_ratings_total": p.get("userRatingCount"),
                    "types": types, "price_level": p.get("priceLevel"),
                    "is_big_business": big, "query": q, "source": "google_places_api_ext",
                })
                for rv in p.get("reviews", []) or []:
                    reviews.append({"place_id": pid, "kecamatan": kec, "rating": rv.get("rating"),
                                    "text": (rv.get("text") or {}).get("text", ""),
                                    "publish_time": rv.get("publishTime")})
                for i, ph in enumerate((p.get("photos") or [])[: a.max_photos]):
                    dest = pdir / f"{pid}_photo_{i}.jpg"
                    if not dest.exists() and client.download_photo(ph["name"], dest):
                        n_ph += 1
                        time.sleep(0.15)
                n_new += 1
                grand["big"] += int(big)
        meta_f.write_text(json.dumps(places, ensure_ascii=False, indent=2), encoding="utf-8")
        rev_f.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {kec}: +{n_new} tempat baru, +{n_ph} foto")
        grand["places"] += n_new
        grand["photos"] += n_ph
    print(f"\n=== TOTAL: +{grand['places']} tempat ({grand['big']} usaha besar), +{grand['photos']} foto ===")


if __name__ == "__main__":
    main()
