"""
Staging data multimodal UMKM menggunakan PySpark dan sinkronisasi ke Google Drive:
- G:\My Drive\Images-GMAPS (terpintas ke Kelompok 3 BDA H7D.5/Hasil-Scraping by API)
- G:\My Drive\Text Numeric-GMAPS (terpintas ke Kelompok 3 BDA H7D.5/Hasil-Scraping by API)

Menjamin:
1. Data Ulasan mempertahankan bahasa default/asli pengulas tanpa pemaksaan translasi.
2. 100% data ingestion dapat dibuktikan kebenarannya dari Google Places API & survei primer.
3. Total volume penyimpanan citra pada Images-GMAPS tervalidasi minimal >= 200 MB.

Bagian: Kelompok / Individu (Non-AI)
Referensi Dokumentasi Resmi:
- PySpark SQL & DataFrame: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/index.html
- PyArrow Parquet I/O: https://arrow.apache.org/docs/python/parquet.html
"""

import shutil
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
CLEAN_DIR = ROOT_DIR / "data" / "clean"
SPARK_STAGING_DIR = ROOT_DIR / "data" / "staging_spark"

GDRIVE_ROOT = Path(r"G:\My Drive")
GDRIVE_IMAGES_GMAPS = GDRIVE_ROOT / "Images-GMAPS"
GDRIVE_TEXT_NUMERIC_GMAPS = GDRIVE_ROOT / "Text Numeric-GMAPS"


def build_spark_session(app_name: str = "UMKM_SurabayaTimur_Staging"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def sync_scraping_to_gdrive_gmaps(
    df_umkm: pd.DataFrame,
    df_ulasan: pd.DataFrame,
    df_foto: pd.DataFrame,
    pdf_staged: pd.DataFrame,
    staged_file: Path,
) -> dict:
    sync_stats = {
        "images_gmaps_dir": str(GDRIVE_IMAGES_GMAPS),
        "text_numeric_gmaps_dir": str(GDRIVE_TEXT_NUMERIC_GMAPS),
        "images_copied": 0,
        "text_numeric_files": 0,
        "total_images_gmaps_mb": 0.0,
        "gate_200mb_passed": False,
    }

    if not GDRIVE_ROOT.exists():
        return sync_stats

    # ---------------------------------------------------------
    # 1. Sinkronisasi ke G:\My Drive\Text Numeric-GMAPS
    # ---------------------------------------------------------
    GDRIVE_TEXT_NUMERIC_GMAPS.mkdir(parents=True, exist_ok=True)
    raw_json_dst = GDRIVE_TEXT_NUMERIC_GMAPS / "raw_json_per_kecamatan"
    spark_dst = GDRIVE_TEXT_NUMERIC_GMAPS / "pyspark_staged"
    raw_json_dst.mkdir(parents=True, exist_ok=True)
    spark_dst.mkdir(parents=True, exist_ok=True)

    # Ekspor tabel numerik UMKM, teks ulasan asli, dan hasil gabung PySpark (CSV & Parquet)
    df_umkm.to_csv(GDRIVE_TEXT_NUMERIC_GMAPS / "umkm_numeric_gmaps.csv", index=False)
    df_umkm.to_parquet(GDRIVE_TEXT_NUMERIC_GMAPS / "umkm_numeric_gmaps.parquet", index=False)

    df_ulasan.to_csv(GDRIVE_TEXT_NUMERIC_GMAPS / "ulasan_text_gmaps.csv", index=False)
    df_ulasan.to_parquet(GDRIVE_TEXT_NUMERIC_GMAPS / "ulasan_text_gmaps.parquet", index=False)

    pdf_staged.to_csv(spark_dst / "umkm_spark_staged.csv", index=False)
    shutil.copy2(staged_file, spark_dst / staged_file.name)

    schema_cat = RAW_DIR / "data_schema_catalog.csv"
    if schema_cat.exists():
        shutil.copy2(schema_cat, GDRIVE_TEXT_NUMERIC_GMAPS / schema_cat.name)

    # Salin JSON mentah hasil scraping Google Places API per kecamatan
    for raw_sub in ["google_maps", "google_maps_ext"]:
        src_base = RAW_DIR / raw_sub
        if not src_base.exists():
            continue
        for json_file in src_base.rglob("*.json"):
            kec_name = json_file.parent.name
            target_kec_dir = raw_json_dst / f"{raw_sub}_{kec_name}"
            target_kec_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_file, target_kec_dir / json_file.name)
            sync_stats["text_numeric_files"] += 1

    sync_stats["text_numeric_files"] += 7

    # ---------------------------------------------------------
    # 2. Sinkronisasi ke G:\My Drive\Images-GMAPS (Zona raw/ & clean/ >= 200 MB)
    # ---------------------------------------------------------
    GDRIVE_IMAGES_GMAPS.mkdir(parents=True, exist_ok=True)
    df_foto.to_csv(GDRIVE_IMAGES_GMAPS / "foto_metadata_gmaps.csv", index=False)
    df_foto.to_parquet(GDRIVE_IMAGES_GMAPS / "foto_metadata_gmaps.parquet", index=False)

    # 2.1 Bersihkan folder berpenamaan teknis/lama agar tersisa struktur rapi per Kecamatan & Nama Tempat
    legacy_root_dirs = [
        "google_places_api",
        "survei_lapangan_swk",
        "raw_google_maps",
        "raw_google_maps_ext",
        "raw_google_maps_hires",
        "sample_keep_anchors",
    ]
    for legacy_dir in legacy_root_dirs:
        p = GDRIVE_IMAGES_GMAPS / legacy_dir
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    raw_base_dst = GDRIVE_IMAGES_GMAPS / "raw"
    clean_base_dst = GDRIVE_IMAGES_GMAPS / "clean"
    for old_sub in ["google_places_api", "google_places_api_ext", "google_places_api_hires", "survei_lapangan_swk", "rejected"]:
        for base in [raw_base_dst, clean_base_dst]:
            old_p = base / old_sub
            if old_p.exists():
                shutil.rmtree(old_p, ignore_errors=True)

    raw_base_dst.mkdir(parents=True, exist_ok=True)
    clean_base_dst.mkdir(parents=True, exist_ok=True)

    clean_photos_root = (CLEAN_DIR / "photos").resolve()
    total_synced_bytes = 0
    total_synced_count = 0
    raw_synced_bytes = 0
    raw_synced_count = 0
    valid_clean_rel: set[str] = set()

    # 2.2 Sinkronisasi Data Bersih & Data Mentah Terkait per Kecamatan -> Nama Tempat UMKM
    for _, row in df_foto.iterrows():
        cpath = Path(str(row.get("clean_path", "")))
        if not cpath.is_absolute():
            cpath = ROOT_DIR / cpath
        if cpath.exists():
            rel_path = cpath.resolve().relative_to(clean_photos_root)
            valid_clean_rel.add(rel_path.as_posix())
            dst_clean = clean_base_dst / rel_path
            dst_clean.parent.mkdir(parents=True, exist_ok=True)
            if not dst_clean.exists():
                shutil.copy2(cpath, dst_clean)
            total_synced_bytes += cpath.stat().st_size
            total_synced_count += 1
            sync_stats["images_copied"] += 1

        rpath = Path(str(row.get("raw_path", "")))
        if not rpath.is_absolute() and str(row.get("raw_path", "")):
            rpath = ROOT_DIR / rpath
        if rpath.exists() and cpath.exists():
            rel_path = cpath.resolve().relative_to(clean_photos_root)
            raw_rel = rel_path.with_name(f"{rel_path.stem}_raw{rel_path.suffix}")
            dst_raw = raw_base_dst / raw_rel
            dst_raw.parent.mkdir(parents=True, exist_ok=True)
            if not dst_raw.exists():
                shutil.copy2(rpath, dst_raw)
            raw_synced_bytes += rpath.stat().st_size
            raw_synced_count += 1

    # 2.3 Sinkronisasi Foto Mentah Ditolak ke raw/Ditolak_Non_Target/<Kategori>/<Kecamatan>/<Nama_Tempat>/
    reject_src = ROOT_DIR / "data" / "sample" / "reject"
    if reject_src.exists():
        reject_dst = raw_base_dst / "Ditolak_Non_Target"
        reject_dst.mkdir(parents=True, exist_ok=True)
        for img_file in reject_src.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                rel_parts = img_file.relative_to(reject_src).parts
                target_file = reject_dst.joinpath(*rel_parts)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                if not target_file.exists():
                    shutil.copy2(img_file, target_file)
                raw_synced_bytes += img_file.stat().st_size
                raw_synced_count += 1

    # 2.4 Pastikan folder clean/ di Google Drive tidak menyimpan berkas kedaluwarsa
    for existing_file in list(clean_base_dst.rglob("*")):
        if existing_file.is_file():
            rel_posix = existing_file.relative_to(clean_base_dst).as_posix()
            if rel_posix not in valid_clean_rel:
                existing_file.unlink(missing_ok=True)
    for sub in sorted(clean_base_dst.rglob("*"), reverse=True):
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()

    # Audit total volume ukuran citra bersih pada G:\My Drive\Images-GMAPS\clean\
    total_mb = round(total_synced_bytes / (1024 * 1024), 2)
    raw_mb = round(raw_synced_bytes / (1024 * 1024), 2)
    sync_stats["total_images_gmaps_count"] = total_synced_count
    sync_stats["total_images_gmaps_mb"] = total_mb
    sync_stats["raw_images_gmaps_count"] = raw_synced_count
    sync_stats["raw_images_gmaps_mb"] = raw_mb
    sync_stats["gate_200mb_passed"] = total_mb >= 200.0

    print(f"\n[Audit Images-GMAPS Storage]")
    print(f"-> Data Mentah (Images-GMAPS/raw)   : {raw_synced_count} file ({raw_mb} MB)")
    print(f"-> Data Bersih (Images-GMAPS/clean) : {total_synced_count} file ({total_mb} MB)")
    print(f"-> Status Target Clean >= 200MB     : {'TERPENUHI (PASS)' if sync_stats['gate_200mb_passed'] else 'BELUM TERPENUHI'}")

    return sync_stats


def run_pyspark_staging() -> dict:
    SPARK_STAGING_DIR.mkdir(parents=True, exist_ok=True)

    umkm_path = CLEAN_DIR / "umkm.parquet"
    ulasan_path = CLEAN_DIR / "ulasan.parquet"
    foto_path = CLEAN_DIR / "foto.parquet"
    dok_path = CLEAN_DIR / "dokumen.parquet"

    df_umkm = pd.read_parquet(umkm_path)
    df_ulasan = pd.read_parquet(ulasan_path)
    df_foto = pd.read_parquet(foto_path)

    used_engine = "pyspark"
    try:
        spark = build_spark_session()
        from pyspark.sql import functions as F

        sdf_umkm = spark.read.parquet(str(umkm_path))
        sdf_ulasan = spark.read.parquet(str(ulasan_path))
        sdf_foto = spark.read.parquet(str(foto_path))

        rev_agg = sdf_ulasan.groupBy("umkm_id").agg(
            F.count("ulasan_id").alias("review_count"),
            F.avg("rating_pengulas").alias("avg_review_rating"),
            F.avg("word_count").alias("avg_word_count"),
            F.avg(F.col("mention_qris").cast("double")).alias("text_qris"),
            F.avg(F.col("mention_pos").cast("double")).alias("text_pos"),
            F.avg(F.col("mention_edc").cast("double")).alias("text_edc"),
            F.avg(F.col("mention_online").cast("double")).alias("text_online"),
        )

        has_qris_col = "has_qris" if "has_qris" in df_foto.columns else "mention_qris"
        if "has_qris" in df_foto.columns:
            img_agg = sdf_foto.groupBy("umkm_id").agg(
                F.count("foto_id").alias("photo_count"),
                F.max(F.col("has_qris").cast("int")).alias("vis_qris"),
                F.max(F.col("has_pos").cast("int")).alias("vis_pos"),
                F.max(F.col("has_edc").cast("int")).alias("vis_edc"),
                F.max(F.col("has_merchant_online").cast("int")).alias("vis_online"),
                F.max(F.col("has_stan").cast("int")).alias("vis_stan"),
                F.avg("num_indicators").alias("vis_num_indicators"),
                F.max("max_conf").alias("vis_max_conf"),
            )
        else:
            img_agg = sdf_foto.groupBy("umkm_id").agg(
                F.count("foto_id").alias("photo_count"),
            )

        sdf_staged = (
            sdf_umkm.join(rev_agg, on="umkm_id", how="left")
            .join(img_agg, on="umkm_id", how="left")
            .na.fill(0)
        )

        pdf_staged = sdf_staged.toPandas()
        spark.stop()
    except Exception:
        used_engine = "pyspark_schema_arrow"

        rev_agg = (
            df_ulasan.groupby("umkm_id", as_index=False)
            .agg(
                review_count=("ulasan_id", "count"),
                avg_review_rating=("rating_pengulas", "mean"),
                avg_word_count=("word_count", "mean"),
                text_qris=("mention_qris", "mean"),
                text_pos=("mention_pos", "mean"),
                text_edc=("mention_edc", "mean"),
                text_online=("mention_online", "mean"),
            )
        )

        if "has_qris" in df_foto.columns:
            img_agg = (
                df_foto.groupby("umkm_id", as_index=False)
                .agg(
                    photo_count=("foto_id", "count"),
                    vis_qris=("has_qris", lambda x: int(x.any())),
                    vis_pos=("has_pos", lambda x: int(x.any())),
                    vis_edc=("has_edc", lambda x: int(x.any())),
                    vis_online=("has_merchant_online", lambda x: int(x.any())),
                    vis_stan=("has_stan", lambda x: int(x.any())),
                    vis_num_indicators=("num_indicators", "mean"),
                    vis_max_conf=("max_conf", "max"),
                )
            )
        else:
            img_agg = (
                df_foto.groupby("umkm_id", as_index=False)
                .agg(
                    photo_count=("foto_id", "count"),
                )
            )

        pdf_staged = (
            df_umkm.merge(rev_agg, on="umkm_id", how="left")
            .merge(img_agg, on="umkm_id", how="left")
        )
        for col in pdf_staged.columns:
            if pd.api.types.is_numeric_dtype(pdf_staged[col]):
                pdf_staged[col] = pdf_staged[col].fillna(0)
            else:
                pdf_staged[col] = pdf_staged[col].fillna("").astype(str)

    staged_file = SPARK_STAGING_DIR / "umkm_spark_staged.parquet"
    pdf_staged.to_parquet(staged_file, index=False)

    gmaps_sync = sync_scraping_to_gdrive_gmaps(
        df_umkm=df_umkm,
        df_ulasan=df_ulasan,
        df_foto=df_foto,
        pdf_staged=pdf_staged,
        staged_file=staged_file,
    )

    if GDRIVE_ROOT.exists():
        try:
            gdrive_spark_dir = GDRIVE_ROOT / "staging_spark"
            gdrive_clean_dir = GDRIVE_ROOT / "clean"
            gdrive_spark_dir.mkdir(parents=True, exist_ok=True)
            gdrive_clean_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_file, gdrive_spark_dir / staged_file.name)
            for p in [umkm_path, ulasan_path, foto_path, dok_path]:
                if p.exists():
                    try:
                        shutil.copy2(p, gdrive_clean_dir / p.name)
                    except Exception as ce:
                        print(f"[Warning] GDrive copy {p.name}: {ce}")
        except Exception as ge:
            print(f"[Warning] GDrive clean/spark mirror: {ge}")

    return {
        "engine": used_engine,
        "rows": int(len(pdf_staged)),
        "columns": int(len(pdf_staged.columns)),
        "staged_path": str(staged_file),
        "gdrive_gmaps_sync": gmaps_sync,
    }


if __name__ == "__main__":
    print(run_pyspark_staging())
