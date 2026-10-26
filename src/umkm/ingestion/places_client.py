"""
Google Places API (New) Ingestion Client for UMKM Kuliner Surabaya Timur.
Fetches place metadata, reviews, and photos compliant with project requirements.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GooglePlacesIngestion:
    """Client for querying Google Places API (New) for culinary MSMEs in Surabaya Timur."""

    SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

    def __init__(self, api_key: Optional[str] = None, output_base_dir: str = "data/raw/google_maps"):
        self.api_key = api_key or os.getenv("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_PLACES_API_KEY tidak ditemukan di environment atau .env file! "
                "Harap set GOOGLE_PLACES_API_KEY terlebih dahulu."
            )
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def test_connection(self) -> Dict[str, Any]:
        """Test API key validity with a lightweight test query."""
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.id,places.displayName",
        }
        payload = {"textQuery": "kuliner Gubeng Surabaya", "pageSize": 1}
        try:
            resp = requests.post(self.SEARCH_TEXT_URL, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                found = len(data.get("places", []))
                return {
                    "status": "SUCCESS",
                    "message": f"Koneksi Google Places API (New) Berhasil! Ditemukan tempat uji: {found}",
                }
            else:
                err_text = resp.text[:300]
                return {
                    "status": "ERROR",
                    "code": resp.status_code,
                    "message": f"HTTP {resp.status_code}: {err_text}",
                }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def search_places(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search places using Places API (New) with pagination support."""
        results: List[Dict[str, Any]] = []
        page_token: Optional[str] = None

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,places.location,"
                "places.rating,places.userRatingCount,places.reviews,places.photos,places.types,places.priceLevel,"
                "nextPageToken"
            ),
        }

        while len(results) < max_results:
            page_size = min(20, max_results - len(results))
            payload: Dict[str, Any] = {"textQuery": query, "pageSize": page_size}
            if page_token:
                payload["pageToken"] = page_token
                time.sleep(1.5)

            try:
                resp = requests.post(self.SEARCH_TEXT_URL, json=payload, headers=headers, timeout=15)
                if resp.status_code != 200:
                    print(f"[Warning] Places search returned HTTP {resp.status_code}: {resp.text[:200]}")
                    break
                data = resp.json()
            except Exception as e:
                print(f"[Error] Request error: {e}")
                break

            batch = data.get("places", [])
            if not batch:
                break
            results.extend(batch)

            page_token = data.get("nextPageToken")
            if not page_token or len(results) >= max_results:
                break

        return results[:max_results]

    def download_photo(self, photo_name: str, save_path: Path, max_size: int = 800) -> bool:
        """Download photo from Google Places Photo Media API."""
        url = f"https://places.googleapis.com/v1/{photo_name}/media?maxHeightPx={max_size}&maxWidthPx={max_size}&key={self.api_key}"
        try:
            resp = requests.get(url, stream=True, timeout=15)
            if resp.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"[Warning] Photo fetch returned status: {resp.status_code}")
        except Exception as e:
            print(f"[Error] Failed to download photo {photo_name[:20]}...: {e}")
        return False

    def ingest_kecamatan(
        self,
        kecamatan: str,
        queries: Optional[List[str]] = None,
        max_places_per_query: int = 15,
        download_photos: bool = True,
        max_photos_per_place: int = 2,
    ) -> Dict[str, Any]:
        """Ingest culinary places for a specific subdistrict."""
        if queries is None:
            queries = [
                f"kuliner {kecamatan} Surabaya",
                f"warung makan {kecamatan} Surabaya",
                f"kafe {kecamatan} Surabaya",
            ]

        kec_dir = self.output_base_dir / kecamatan.lower().replace(" ", "_")
        photos_dir = kec_dir / "photos"
        kec_dir.mkdir(parents=True, exist_ok=True)
        photos_dir.mkdir(parents=True, exist_ok=True)

        seen_place_ids = set()
        collected_places: List[Dict[str, Any]] = []
        collected_reviews: List[Dict[str, Any]] = []
        downloaded_photos_count = 0

        print(f"\n=== Memulai Ingestion untuk Kecamatan: {kecamatan} ===")
        for q in queries:
            print(f"-> Kueri: '{q}'...")
            raw_places = self.search_places(q, max_results=max_places_per_query)

            for p in raw_places:
                pid = p.get("id")
                if not pid or pid in seen_place_ids:
                    continue
                seen_place_ids.add(pid)

                display_name = p.get("displayName", {}).get("text", "")
                loc = p.get("location", {})
                place_record = {
                    "place_id": pid,
                    "kecamatan": kecamatan,
                    "name": display_name,
                    "formatted_address": p.get("formattedAddress"),
                    "lat": loc.get("latitude"),
                    "lon": loc.get("longitude"),
                    "rating": p.get("rating"),
                    "user_ratings_total": p.get("userRatingCount"),
                    "types": p.get("types", []),
                    "price_level": p.get("priceLevel"),
                    "source": "google_places_api",
                }
                collected_places.append(place_record)

                # Extract reviews
                for rev in p.get("reviews", []):
                    author = rev.get("authorAttribution", {}).get("displayName", "Pengulas Anonim")
                    rev_text = rev.get("text", {}).get("text", "")
                    rev_record = {
                        "place_id": pid,
                        "kecamatan": kecamatan,
                        "author_name": author,
                        "rating": rev.get("rating"),
                        "text": rev_text,
                        "publish_time": rev.get("publishTime"),
                        "relative_publish_time": rev.get("relativePublishTimeDescription"),
                    }
                    collected_reviews.append(rev_record)

                # Download photos if requested
                if download_photos and p.get("photos"):
                    photos_to_get = p["photos"][:max_photos_per_place]
                    for idx, photo_item in enumerate(photos_to_get):
                        pname = photo_item.get("name")
                        if pname:
                            photo_filename = f"{pid}_photo_{idx}.jpg"
                            photo_path = photos_dir / photo_filename
                            if not photo_path.exists():
                                if self.download_photo(pname, photo_path):
                                    downloaded_photos_count += 1
                                    time.sleep(0.2)

        # Save metadata to disk
        meta_file = kec_dir / "places_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(collected_places, f, ensure_ascii=False, indent=2)

        reviews_file = kec_dir / "reviews_metadata.json"
        with open(reviews_file, "w", encoding="utf-8") as f:
            json.dump(collected_reviews, f, ensure_ascii=False, indent=2)

        summary = {
            "kecamatan": kecamatan,
            "total_places": len(collected_places),
            "total_reviews": len(collected_reviews),
            "downloaded_photos": downloaded_photos_count,
            "output_dir": str(kec_dir),
        }
        print(f"Hasil {kecamatan}: {len(collected_places)} tempat, {len(collected_reviews)} ulasan, {downloaded_photos_count} foto.")
        return summary
