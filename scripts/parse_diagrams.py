import json
import xml.etree.ElementTree as ET
from pathlib import Path
import html
import re

def parse_file(path_str):
    path = Path(path_str)
    print(f"==================================================")
    print(f"FILE: {path.name}")
    print(f"==================================================")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pages = data.get("pages", [])
    print(f"Total Pages: {len(pages)}")
    for p_idx, page in enumerate(pages):
        print(f"\n--- PAGE {p_idx+1} keys: {list(page.keys())} ---")
        for k, v in page.items():
            if isinstance(v, (str, bytes)):
                print(f"  key {k}: type {type(v)}, len {len(v)}")
                if len(v) > 0 and len(v) < 500:
                    print(f"    val: {v}")
                elif len(v) >= 500:
                    print(f"    val snippet: {v[:200]}")
            elif isinstance(v, (list, dict)):
                print(f"  key {k}: type {type(v)}, len {len(v)}")
            else:
                print(f"  key {k}: {v}")


if __name__ == "__main__":
    parse_file("diagram-architecture/flowchart-mandiri.json")
    parse_file("diagram-architecture/H7D.5-3_diagram-architecture-2026 (1).json")
