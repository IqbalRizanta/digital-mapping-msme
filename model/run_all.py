"""
Orkestrasi eksekusi pipeline Big Data Analytic (Staging, EDA, Embedding, ML, DL, Evaluasi, & BDA).

Bagian: Kelompok / Individu (Non-AI)
"""

import json
from pathlib import Path
from model.processing.pyspark_staging import run_pyspark_staging
from model.eda.eda_analysis import run_eda_analysis
from model.features.feature_embedding import build_and_save_embeddings
from model.training.ml_pipeline import train_and_save_ml_models
from model.training.dl_pipeline import train_and_save_dl_models
from model.evaluation.model_evaluation import run_model_evaluation
from model.analytics.bda_analytics import compute_spatial_bda

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "peta-digital-umkm" / "reports" / "modeling"


def main() -> dict:
    spark_res = run_pyspark_staging()
    eda_res = run_eda_analysis()
    emb_res = build_and_save_embeddings()
    ml_res = train_and_save_ml_models()
    dl_res = train_and_save_dl_models()
    eval_res = run_model_evaluation()
    bda_df = compute_spatial_bda()

    summary = {
        "staging_pyspark": spark_res,
        "eda": eda_res,
        "embeddings": emb_res,
        "machine_learning": ml_res,
        "deep_learning": dl_res,
        "model_evaluation": eval_res,
        "bda_kecamatan_count": int(len(bda_df)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "pipeline_execution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
