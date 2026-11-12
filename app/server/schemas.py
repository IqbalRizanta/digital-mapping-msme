"""
Skema validasi request dan response REST API (Pydantic).

Bagian: AI-Assisted (< 40% Boilerplate Schema)
Referensi Dokumentasi Resmi:
- Pydantic Models: https://docs.pydantic.dev/latest/concepts/models/
"""

from pydantic import BaseModel, Field


class UMKMPredictRequest(BaseModel):
    nama_umkm: str = Field(default="Warung Contoh Surabaya Timur")
    kecamatan: str = Field(default="Sukolilo")
    rating: float = Field(default=4.5, ge=0.0, le=5.0)
    jumlah_ulasan: int = Field(default=35, ge=0)
    indikator_qris: int = Field(default=1, ge=0, le=1)
    indikator_pos: int = Field(default=0, ge=0, le=1)
    indikator_edc: int = Field(default=0, ge=0, le=1)
    indikator_ojol: int = Field(default=1, ge=0, le=1)
    rev_count: int = Field(default=5, ge=0)
    rev_word_mean: float = Field(default=14.0, ge=0.0)
    img_count: int = Field(default=2, ge=0)


class ModelPredictionResponse(BaseModel):
    nama_umkm: str
    kecamatan: str
    ml_clustering_cluster: int
    ml_classification_label: str
    ml_regression_score: float
    dl_clustering_cluster: int
    dl_classification_label: str
    dl_regression_score: float
    rekomendasi: str
