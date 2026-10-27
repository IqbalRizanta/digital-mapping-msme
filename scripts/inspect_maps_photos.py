"""
Inspect Google Maps photos to classify them into relevant (stan, cashier, facade)
vs irrelevant (food dish, drink, menu, macro shots).
"""

from pathlib import Path
from PIL import Image
import numpy as np

def inspect_maps_photos():
    raw_maps_dir = Path("data/raw/google_maps")
    all_photos = list(raw_maps_dir.glob("*/photos/*.jpg"))
    print(f"Total Google Maps photos: {len(all_photos)}")

    relevant_candidates = []
    food_or_menu = []

    for p in all_photos:
        try:
            with Image.open(p) as img:
                w, h = img.size
                aspect = w / h
                # Basic check: very square or high resolution
                # Check color and edge distribution
                arr = np.array(img.convert('RGB'))
                # Mean saturation or color warmth
                # Store sample info
                relevant_candidates.append((p, w, h, aspect))
        except Exception as e:
            print(f"Error reading {p}: {e}")

    print(f"Analyzed {len(relevant_candidates)} photos.")

if __name__ == '__main__':
    inspect_maps_photos()
