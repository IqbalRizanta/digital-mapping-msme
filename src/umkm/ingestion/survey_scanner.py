"""
Survey Image Scanner and Intake Utility.
Scans and registers user-provided offline survey images from data/raw/survei_offline/.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


class SurveyImageScanner:
    """Scans and catalogs user-provided offline survey images."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
        """Scan directory for all valid image files and generate inventory."""
        image_records: List[Dict[str, Any]] = []
        total_bytes = 0
        duplicate_hashes = set()
        seen_hashes = {}

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

            # Attempt to infer subdistrict / merchant name from directory structure
            rel_parts = file_path.relative_to(self.raw_survey_dir).parts
            inferred_kecamatan = rel_parts[0] if len(rel_parts) > 1 else "Unspecified"
            inferred_merchant = rel_parts[1] if len(rel_parts) > 2 else "Unspecified"

            image_records.append({
                "file_name": file_path.name,
                "relative_path": str(file_path.relative_to(Path.cwd())),
                "file_size_bytes": file_size,
                "file_size_kb": round(file_size / 1024, 2),
                "sha256": file_hash,
                "is_duplicate": is_duplicate,
                "inferred_kecamatan": inferred_kecamatan,
                "inferred_merchant": inferred_merchant,
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
            "gate_200mb_passed": total_mb >= 200.0,
            "inventory_csv": str(inventory_csv) if not df_inventory.empty else None,
        }
        return summary
