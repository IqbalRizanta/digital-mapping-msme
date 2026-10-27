"""
Use pretrained YOLOv8n to inspect what objects are in Google Maps photos.
Identifies food/drink objects vs storefront/tables/appliances.
"""

from pathlib import Path
from ultralytics import YOLO
import pandas as pd

def run_coco_screening():
    model = YOLO("yolov8n.pt")
    raw_maps_dir = Path("data/raw/google_maps")
    photos = list(raw_maps_dir.glob("*/photos/*.jpg"))
    print(f"Total Google Maps photos to screen: {len(photos)}")

    food_classes = {'bowl', 'sandwich', 'pizza', 'hot dog', 'donut', 'cake', 'banana', 'apple', 'orange', 'broccoli', 'carrot'}
    establishment_classes = {'tv', 'laptop', 'cell phone', 'refrigerator', 'chair', 'couch', 'potted plant', 'dining table'}

    records = []
    # Test on a batch of 50 photos first
    results = model(photos[:60], verbose=False)

    for p, res in zip(photos[:60], results):
        detected_names = [res.names[int(cls)] for cls in res.boxes.cls.tolist()]
        has_food = any(c in food_classes for c in detected_names)
        has_est = any(c in establishment_classes for c in detected_names)
        records.append({
            "path": str(p),
            "detected": detected_names,
            "has_food": has_food,
            "has_est": has_est
        })

    df = pd.DataFrame(records)
    print("Screening Results on first 60:")
    print("Pure food photos count:", len(df[df['has_food'] & ~df['has_est']]))
    print("Establishment / venue photos count:", len(df[df['has_est']]))
    print("Sample detected:", df[['detected', 'has_food', 'has_est']].head(10))

if __name__ == '__main__':
    run_coco_screening()
