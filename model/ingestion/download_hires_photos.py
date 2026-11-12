import os, sys
sys.path.insert(0, r"C:\big-data-analytic")
import json
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(r"C:\big-data-analytic\.env")
import requests

api_key = os.getenv("GOOGLE_PLACES_API_KEY")
if not api_key:
    print("No API KEY")
    sys.exit(1)

SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": api_key,
    "X-Goog-FieldMask": "places.id,places.displayName,places.photos",
}

target_dir = Path("data/raw/google_maps_hires")
target_dir.mkdir(parents=True, exist_ok=True)

# List of real popular culinary places in Surabaya Timur to enrich real photo dataset
queries = [
    "kuliner Rungkut Surabaya",
    "kuliner Sukolilo Surabaya",
    "kuliner Gubeng Surabaya",
    "kuliner Mulyorejo Surabaya",
    "kuliner Tenggilis Mejoyo Surabaya",
    "kuliner Gunung Anyar Surabaya",
]

total_downloaded = 0
total_bytes = 0

for q in queries:
    print(f"Searching: {q}")
    payload = {"textQuery": q, "pageSize": 10}
    try:
        resp = requests.post(SEARCH_TEXT_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"Error {resp.status_code}")
            continue
        data = resp.json()
        places = data.get("places", [])
        for p in places:
            pid = p.get("id")
            pname = p.get("displayName", {}).get("text", "Unknown")
            photos = p.get("photos", [])
            for idx, ph in enumerate(photos[:4]): # top 4 photos per place
                ref = ph.get("name")
                if not ref:
                    continue
                out_path = target_dir / f"{pid}_hires_{idx}.jpg"
                if out_path.exists():
                    continue
                # Fetch at max resolution (1600px)
                photo_url = f"https://places.googleapis.com/v1/{ref}/media?maxWidthPx=1600&key={api_key}"
                try:
                    p_resp = requests.get(photo_url, stream=True, timeout=15)
                    if p_resp.status_code == 200:
                        with open(out_path, "wb") as f:
                            for chunk in p_resp.iter_content(chunk_size=16384):
                                f.write(chunk)
                        sz = out_path.stat().st_size
                        total_bytes += sz
                        total_downloaded += 1
                        print(f"  + Downloaded [{pname[:20]}] photo {idx} ({sz/1024:.1f} KB)")
                        time.sleep(0.3)
                except Exception as e:
                    print(f"  - Failed photo: {e}")
    except Exception as e:
        print(f"Search failed: {e}")

print(f"\nDone! Downloaded {total_downloaded} high-res photos ({total_bytes / (1024*1024):.2f} MB).")
