"""
Image Processor for UMKM Multimodal Pipeline.
Performs face blurring using OpenCV and standardizes images for clean storage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class ImageProcessor:
    """Processes images with face blurring and standardization."""

    def __init__(self, clean_photo_dir: str = "data/clean/photos"):
        self.clean_photo_dir = Path(clean_photo_dir)
        self.clean_photo_dir.mkdir(parents=True, exist_ok=True)

        # Load OpenCV Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Calculate SHA-256 hash."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def blur_faces(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, int]:
        """Detect faces and apply strong Gaussian blur to preserve privacy."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        blurred_count = len(faces)
        for (x, y, w, h) in faces:
            roi = img_bgr[y : y + h, x : x + w]
            # Kernel size must be odd and proportional to face size
            ksize = max(15, (w // 2) * 2 + 1)
            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 30)
            img_bgr[y : y + h, x : x + w] = blurred_roi

        return img_bgr, blurred_count

    def process_and_save(
        self,
        raw_path: Path,
        subfolder: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load raw image, blur faces, resize if requested, and save to clean storage."""
        if not raw_path.exists():
            return None

        # Read image
        img = cv2.imread(str(raw_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # Privacy step: blur faces
        sanitized_img, face_count = self.blur_faces(img)

        # Output destination
        out_dir = self.clean_photo_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_filename = f"clean_{raw_path.name}"
        clean_path = out_dir / clean_filename

        # Write clean sanitized image
        cv2.imwrite(str(clean_path), sanitized_img)

        # Compute hash of clean image
        clean_hash = self.compute_sha256(clean_path)

        return {
            "raw_path": str(raw_path.resolve()).replace("\\", "/"),
            "clean_path": str(clean_path.resolve()).replace("\\", "/"),
            "width": w,
            "height": h,
            "face_blurred_count": face_count,
            "clean_sha256": clean_hash,
            "size_kb": round(clean_path.stat().st_size / 1024, 2),
        }
