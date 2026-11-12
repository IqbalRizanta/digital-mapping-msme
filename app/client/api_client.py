"""
HTTP Client untuk komunikasi antara aplikasi Client (Streamlit) dan Server (FastAPI).

Bagian: AI-Assisted (< 40% HTTP Client Wrapper)
Referensi Dokumentasi Resmi:
- Requests Quickstart: https://requests.readthedocs.io/en/latest/user/quickstart/
"""

import os
import requests
from app.server.inference_service import UMKMInferenceService

SERVER_URL = os.getenv("UMKM_SERVER_URL", "http://127.0.0.1:8000")


class BackendClient:
    def __init__(self, base_url: str = SERVER_URL):
        self.base_url = base_url.rstrip("/")
        self._local_fallback: UMKMInferenceService | None = None

    def _fallback(self) -> UMKMInferenceService:
        if self._local_fallback is None:
            self._local_fallback = UMKMInferenceService()
        return self._local_fallback

    def fetch_health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        artifacts = UMKMInferenceService.list_saved_artifacts()
        return {
            "status": "local_direct_mode",
            "saved_models_count": len(artifacts),
            "artifacts": artifacts,
        }

    def fetch_bda_summary(self) -> list:
        try:
            r = requests.get(f"{self.base_url}/bda/summary", timeout=2.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return UMKMInferenceService.get_bda_summary()

    def fetch_umkm_list(self, kecamatan: str | None = None) -> list:
        try:
            params = {"kecamatan": kecamatan} if kecamatan else {}
            r = requests.get(f"{self.base_url}/umkm/list", params=params, timeout=2.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return UMKMInferenceService.get_umkm_table(kecamatan=kecamatan)

    def predict_umkm(self, payload: dict) -> dict:
        try:
            r = requests.post(
                f"{self.base_url}/predict/multimodal", json=payload, timeout=3.0
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return self._fallback().predict_all(payload)
