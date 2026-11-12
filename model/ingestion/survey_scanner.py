"""
Survey Image Scanner and Intake Utility.
Scans, catalogs, and labels user-provided offline survey images from SWK Convention Hall & Deles.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


class SurveyImageScanner:
    """Scans and catalogs user-provided offline survey images from Sentra Wisata Kuliner (SWK)."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Mapping of folder names to standardized digital maturity labels
    CLASS_MAPPING = {
        "kelas_1_konvensional": {
            "label_id": 0,
            "label_name": "Konvensional",
            "description": "Transaksi tunai / manual, belum adopsi QRIS/EDC/Online",
        },
        "kelas_2_transisi_digital": {
            "label_id": 1,
            "label_name": "Berkembang",
            "description": "Transisi digital, telah adopsi QRIS / EDC mandiri",
        },
        "kelas_3_go_digital_ojol": {
            "label_id": 2,
            "label_name": "Maju",
            "description": "Ekosistem digital terintegrasi (QRIS/EDC + Merchant Online / POS)",
        },
    }

    def __init__(self, raw_survey_dir: str = "data/raw/survei_offline"):
        self.raw_survey_dir = Path(raw_survey_dir)
        self.raw_survey_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Calculate SHA256 hash for deduplication check."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def scan_images(self) -> Dict[str, Any]:
        """Scan directory for all valid image files and generate inventory with class labels."""
        image_records: List[Dict[str, Any]] = []
        total_bytes = 0
        duplicate_hashes = set()
        seen_hashes = {}
        class_distribution = {"Konvensional": 0, "Berkembang": 0, "Maju": 0, "Unclassified": 0}

        all_files = [
            f for f in self.raw_survey_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
        ]

        for file_path in all_files:
            file_size = file_path.stat().st_size
            total_bytes += file_size
            file_hash = self.compute_sha256(file_path)

            is_duplicate = False
            if file_hash in seen_hashes:
                is_duplicate = True
                duplicate_hashes.add(file_hash)
            else:
                seen_hashes[file_hash] = file_path

            # Infer class label from parent folder name
            parent_folder = file_path.parent.name.lower()
            mapped = self.CLASS_MAPPING.get(parent_folder)
            if mapped:
                label_id = mapped["label_id"]
                label_name = mapped["label_name"]
            else:
                label_id = -1
                label_name = "Unclassified"

            class_distribution[label_name] = class_distribution.get(label_name, 0) + 1

            image_records.append({
                "file_name": file_path.name,
                "relative_path": str(file_path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/"),
                "folder_class": file_path.parent.name,
                "label_id": label_id,
                "label_ground_truth": label_name,
                "target_focus": "SWK (Sentra Wisata Kuliner)",
                "kecamatan_focus": "Sukolilo",
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "sha256": file_hash,
                "is_duplicate": is_duplicate,
                "extension": file_path.suffix.lower(),
            })

        df_inventory = pd.DataFrame(image_records) if image_records else pd.DataFrame()
        inventory_csv = self.raw_survey_dir / "survey_images_inventory.csv"
        if not df_inventory.empty:
            df_inventory.to_csv(inventory_csv, index=False)

        total_mb = round(total_bytes / (1024 * 1024), 2)
        summary = {
            "total_images_found": len(image_records),
            "unique_images": len(seen_hashes),
            "duplicates_count": len(duplicate_hashes),
            "total_size_mb": total_mb,
            "class_distribution": class_distribution,
            "inventory_csv": str(inventory_csv) if not df_inventory.empty else None,
        }
        return summary
