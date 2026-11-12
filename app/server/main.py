"""
REST API Server (FastAPI) untuk menyajikan model Big Data & AI UMKM Surabaya Timur.

Bagian: Kelompok / Individu & AI-Assisted (< 40% Routing Boilerplate)
Referensi Dokumentasi Resmi:
- FastAPI Tutorial & Routing: https://fastapi.tiangolo.com/tutorial/first-steps/
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.server.inference_service import UMKMInferenceService
from app.server.schemas import ModelPredictionResponse, UMKMPredictRequest

app = FastAPI(
    title="Peta Digital UMKM Surabaya Timur - Inference Server",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: UMKMInferenceService | None = None


def get_service() -> UMKMInferenceService:
    global _service
    if _service is None:
        _service = UMKMInferenceService()
    return _service


@app.get("/health")
def health_check() -> dict:
    artifacts = UMKMInferenceService.list_saved_artifacts()
    return {
        "status": "ok",
        "saved_models_count": len(artifacts),
        "artifacts": artifacts,
    }


@app.get("/bda/summary")
def bda_summary() -> list:
    return UMKMInferenceService.get_bda_summary()


@app.get("/umkm/list")
def umkm_list(kecamatan: str | None = Query(default=None)) -> list:
    return UMKMInferenceService.get_umkm_table(kecamatan=kecamatan)


@app.post("/predict/multimodal", response_model=ModelPredictionResponse)
def predict_multimodal(req: UMKMPredictRequest) -> dict:
    svc = get_service()
    return svc.predict_all(req.model_dump())
