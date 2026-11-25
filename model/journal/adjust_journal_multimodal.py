"""
ADJUST JOURNAL — Ekspansi 3 Modalitas Dataset (Tabular SWK + Gambar Indonesia + Dokumen CORD-v2 & MIDD)
Menggabungkan:
  1. Jurnal Utama (Tabular/Numerik): Selayanti et al. (2024, JoMMiT) — 48 SWK Surabaya (AHC)
  2. Jurnal Pendamping Gambar (Indonesia): Wiranata & Lestari (2025, JuTISI, Univ. Ciputra Surabaya,
     DOI: 10.28932/jutisi.v11i1.10284). Karena dataset asli Mendeley 10.17632/vtjd68bmwt.1 menolak
     unduhan otomatis (HTTP 403), digunakan dataset publik serupa (similar) terunduh langsung:
     - Padang Cuisine Indonesian Food Image Dataset (fahmindra/padang-cuisine, ODbL / Kaggle faldoae/padangfood)
       -> 993 foto masakan Indonesia pada 9 kelas seimbang (train/test resmi)
     - Dipadukan dengan 549 Foto Bersih SWK & UMKM Surabaya Timur (279.99 MB >= 200 MB) + 385 UMKM & 1.164 Ulasan
  3. Jurnal Pendamping Dokumen:
     - CORD (Park et al., 2019, NeurIPS DI Workshop) -> 200 Struk Restoran Indonesia (HF cord-v2-bucket)
     - MIDD (Baviskar et al., 2021, MDPI Data, DOI: 10.3390/data6070078, Zenodo 10.5281/zenodo.5113009)
       -> 708 Invoice Multi-Layout IOB CSV (Layout1..Layout4)
     - 3 Dokumen PDF Perizinan Penelitian Lokal (pypdf)

Menjalankan EDA & Modeling-Evaluation terpadu untuk ketiga modalitas:
  A. Multimodal UMKM/SWK Surabaya Timur (Clustering K-Means/Autoencoder, Classification GroupKFold, Regression)
  B. Indonesian Food Image Classification (Padang Cuisine — 993 Foto, 9 Kelas: HSV Spatial Pyramid + Sobel Texture)
  C1. Document Analytics & Classification MIDD (708 Invoice -> Klasifikasi 4 Layout + Audit Entitas NER IOB)
  C2. Receipt Document Analytics CORD-v2 (200 Struk Restoran Indonesia -> Prediksi Metode Bayar & Analisis Omzet)
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.training.ml_pipeline import FEATURE_COLS, LABEL_NAMES, derive_target_class, prepare_feature_frame

CLEAN_DIR = ROOT_DIR / "data" / "clean"
GOLD_DIR = ROOT_DIR / "data" / "gold"
RAW_DIR = ROOT_DIR / "data" / "raw"
REPORTS_DIR = ROOT_DIR / "docs" / "reports"
ADJUST_DIR = REPORTS_DIR / "journal_adjust"


def _extract_visual_descriptor(img_bgr: np.ndarray) -> np.ndarray:
    """Ekstraksi fitur warna HSV + Lab + tekstur gradien Sobel (grid global + tengah) untuk citra makanan."""
    im = cv2.resize(img_bgr, (128, 128))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

    # Histogram HSV global (8 x 4 x 4 = 128 fitur)
    h_glob = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    h_glob = h_glob / (h_glob.sum() + 1e-6)

    # Histogram HSV area tengah (fokus piring/makanan, 6 x 3 x 3 = 54 fitur)
    hsv_mid = hsv[24:104, 24:104]
    h_mid = cv2.calcHist([hsv_mid], [0, 1, 2], None, [6, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
    h_mid = h_mid / (h_mid.sum() + 1e-6)

    # Histogram orientasi gradien Sobel (16 fitur tekstur)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    eh, _ = np.histogram(ang, bins=16, range=(0, 2 * np.pi), weights=mag)
    eh = eh / (eh.sum() + 1e-6)

    # Statistik momen warna RGB, HSV, & Lab (18 fitur)
    stats = np.concatenate([
        im.mean(axis=(0, 1)), im.std(axis=(0, 1)),
        hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1)),
        lab.mean(axis=(0, 1)), lab.std(axis=(0, 1)),
    ])
    return np.concatenate([h_glob, h_mid, eh, stats]).astype(np.float32)


def analyze_padang_food_images() -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """EDA & Klasifikasi 9 Kelas pada 993 Citra Masakan Indonesia (fahmindra/padang-cuisine, ODbL)."""
    padang_root = RAW_DIR / "journal_gambar_padang_id"
    feat_parquet = padang_root / "padang_cuisine_features.parquet"
    meta_csv = padang_root / "padang_cuisine_metadata.csv"

    if feat_parquet.exists() and meta_csv.exists():
        df_feat_img = pd.read_parquet(feat_parquet)
        df_meta = pd.read_csv(meta_csv)
    else:
        imgs = sorted(
            f for f in padang_root.rglob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        meta_rows, feats = [], []
        for f in imgs:
            im = cv2.imread(str(f))
            if im is None:
                continue
            h, w = im.shape[:2]
            split_name = "test" if "test" in f.parts else "train"
            meta_rows.append(
                {
                    "file_name": f.name,
                    "rel_path": f.relative_to(padang_root).as_posix(),
                    "split": split_name,
                    "kelas": f.parent.name,
                    "width": int(w),
                    "height": int(h),
                    "size_kb": round(f.stat().st_size / 1024.0, 2),
                }
            )
            feats.append(_extract_visual_descriptor(im))

        df_meta = pd.DataFrame(meta_rows)
        feat_mat = np.vstack(feats)
        feat_cols = [f"v_{i:03d}" for i in range(feat_mat.shape[1])]
        df_feat_img = pd.concat([df_meta[["rel_path", "split", "kelas"]], pd.DataFrame(feat_mat, columns=feat_cols)], axis=1)
        df_meta.to_csv(meta_csv, index=False)
        df_feat_img.to_parquet(feat_parquet, index=False)

    feat_cols = [c for c in df_feat_img.columns if c.startswith("v_")]
    X_all = df_feat_img[feat_cols].values.astype(np.float32)
    y_all = df_feat_img["kelas"].to_numpy(dtype=str)
    tr_mask = (df_feat_img["split"] == "train").values
    te_mask = (df_feat_img["split"] == "test").values
    classes = sorted(df_feat_img["kelas"].unique().tolist())

    scaler = StandardScaler().fit(X_all[tr_mask])
    X_tr, X_te = scaler.transform(X_all[tr_mask]), scaler.transform(X_all[te_mask])
    y_tr, y_te = y_all[tr_mask], y_all[te_mask]

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        "LinearSVC": LinearSVC(C=0.1, random_state=42, dual="auto", max_iter=3000),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=300, random_state=42),
    }
    bench, best_cm, best_name, best_acc = [], None, "", -1.0
    for mname, clf in models.items():
        y_pred = clf.fit(X_tr, y_tr).predict(X_te)
        acc = round(float(accuracy_score(y_te, y_pred)), 4)
        f1 = round(float(f1_score(y_te, y_pred, average="macro")), 4)
        bench.append({"Model": mname, "Test_Accuracy": acc, "Test_Macro_F1": f1})
        if acc > best_acc:
            best_acc = acc
            best_name = mname
            best_cm = confusion_matrix(y_te, y_pred, labels=classes)

    metrics = {
        "source": "HuggingFace fahmindra/padang-cuisine (ODbL, mirror Kaggle faldoae/padangfood)",
        "rationale": "Dataset gambar masakan Indonesia publik serupa untuk JuTISI 2025 (Mendeley 10.17632/vtjd68bmwt.1 HTTP 403)",
        "total_images": int(len(df_meta)),
        "train_images": int(tr_mask.sum()),
        "test_images": int(te_mask.sum()),
        "num_classes": len(classes),
        "feature_dims": len(feat_cols),
        "class_counts": df_meta["kelas"].value_counts().sort_index().to_dict(),
        "avg_width": round(float(df_meta["width"].mean()), 1),
        "avg_height": round(float(df_meta["height"].mean()), 1),
        "best_model": best_name,
        "best_test_acc": best_acc,
        "benchmark": bench,
    }
    return df_meta, best_cm, classes, metrics


def analyze_midd_documents() -> tuple[pd.DataFrame, dict]:
    """EDA & Klasifikasi 4 Layout pada 708 Dokumen Invoice MIDD (IOB CSV)."""
    midd_root = RAW_DIR / "journal_dokumen_midd" / "IOB"
    doc_rows = []
    tag_counts: dict[str, int] = {}

    for layout_dir in sorted(midd_root.iterdir()):
        if not layout_dir.is_dir():
            continue
        for csv_f in sorted(layout_dir.glob("*.csv")):
            try:
                d = pd.read_csv(csv_f, on_bad_lines="skip")
                if "Text" not in d.columns or "Tag" not in d.columns:
                    continue
                tokens = d["Text"].fillna("").astype(str).tolist()
                tags = d["Tag"].fillna("O").astype(str).str.strip().tolist()
                for t in tags:
                    if t != "O":
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                doc_rows.append(
                    {
                        "file": csv_f.name,
                        "layout": layout_dir.name,
                        "n_tokens": len(tokens),
                        "n_entity_tokens": sum(1 for t in tags if t != "O"),
                        "text_joined": " ".join(tokens[:250]),
                    }
                )
            except Exception:
                continue

    df_midd = pd.DataFrame(doc_rows)
    X_tfidf = TfidfVectorizer(max_features=1200, stop_words="english").fit_transform(df_midd["text_joined"])
    y_layout = df_midd["layout"].to_numpy(dtype=str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_svm = cross_val_predict(LinearSVC(random_state=42, dual="auto"), X_tfidf, y_layout, cv=skf)
    y_pred_rf = cross_val_predict(RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1), X_tfidf, y_layout, cv=skf)

    midd_metrics = {
        "total_invoices": int(len(df_midd)),
        "layout_counts": df_midd["layout"].value_counts().to_dict(),
        "avg_tokens_per_invoice": round(float(df_midd["n_tokens"].mean()), 1),
        "avg_entity_tokens": round(float(df_midd["n_entity_tokens"].mean()), 1),
        "top_iob_tags": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "layout_clf_svm_acc": round(float(accuracy_score(y_layout, y_pred_svm)), 4),
        "layout_clf_svm_f1": round(float(f1_score(y_layout, y_pred_svm, average="macro")), 4),
        "layout_clf_rf_acc": round(float(accuracy_score(y_layout, y_pred_rf)), 4),
        "layout_clf_rf_f1": round(float(f1_score(y_layout, y_pred_rf, average="macro")), 4),
    }
    return df_midd, midd_metrics


def analyze_cord_receipts() -> tuple[pd.DataFrame, dict]:
    """EDA & Klasifikasi Metode Bayar pada 200 Struk Restoran Indonesia CORD-v2."""
    df_cord = pd.read_parquet(RAW_DIR / "journal_dokumen_cord" / "cord_v2_receipts_structured.parquet")
    df_cord["has_digital_payment"] = (df_cord["payment_method"] == "Non-Tunai (Card/E-Money)").astype(int)

    valid_idr = df_cord["total_price_idr"].dropna()
    valid_idr = valid_idr[(valid_idr >= 1000) & (valid_idr <= 10_000_000)]

    feat_cols = ["num_menu_items", "ocr_line_count", "ocr_word_count", "has_tax", "has_service_charge", "has_discount"]
    X_c = df_cord[feat_cols].astype(float).fillna(0).values
    y_c = df_cord["has_digital_payment"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_c = cross_val_predict(
        RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight="balanced"),
        X_c,
        y_c,
        cv=skf,
    )

    cord_metrics = {
        "total_receipts": int(len(df_cord)),
        "payment_distribution": df_cord["payment_method"].value_counts().to_dict(),
        "avg_menu_items": round(float(df_cord["num_menu_items"].mean()), 2),
        "avg_ocr_words": round(float(df_cord["ocr_word_count"].mean()), 2),
        "median_total_idr": int(valid_idr.median()) if len(valid_idr) else 0,
        "mean_total_idr": int(valid_idr.mean()) if len(valid_idr) else 0,
        "digital_payment_rf_acc": round(float(accuracy_score(y_c, y_pred_c)), 4),
        "digital_payment_rf_f1": round(float(f1_score(y_c, y_pred_c, average="macro")), 4),
    }
    return df_cord, cord_metrics


def main() -> None:
    ADJUST_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Modalitas UMKM/SWK Surabaya Timur (Tabular + Teks + Gambar + PDF)
    df_umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
    df_ulasan = pd.read_parquet(CLEAN_DIR / "ulasan.parquet")
    df_foto = pd.read_parquet(CLEAN_DIR / "foto.parquet")
    df_dok = pd.read_parquet(CLEAN_DIR / "dokumen.parquet")
    df_feat = prepare_feature_frame(pd.read_parquet(GOLD_DIR / "feature_store.parquet"))
    photo_mb = round(sum(f.stat().st_size for f in (CLEAN_DIR / "photos").rglob("*") if f.is_file()) / (1024**2), 2)

    X_sc = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values.astype(np.float32))
    y_cls = derive_target_class(df_feat)
    y_reg = df_feat["skor_kematangan_kontinu"].values.astype(np.float32)
    groups = df_feat["kecamatan"].to_numpy(dtype=str)

    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_sc)
    km_sil = round(float(silhouette_score(X_sc, km.labels_)), 4)
    km_dbi = round(float(davies_bouldin_score(X_sc, km.labels_)), 4)

    gkf = GroupKFold(n_splits=5)
    classifiers = {
        "Naive Bayes": lambda: GaussianNB(),
        "SVM (LinearSVC)": lambda: LinearSVC(random_state=42, dual="auto", max_iter=3000),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42),
    }
    bench_rows, best_cm, best_f1 = [], None, -1.0
    for name, fn in classifiers.items():
        yt, yp = [], []
        for tr, te in gkf.split(X_sc, y_cls, groups):
            c = fn().fit(X_sc[tr], y_cls[tr])
            yt.extend(y_cls[te])
            yp.extend(c.predict(X_sc[te]))
        f1 = round(float(f1_score(yt, yp, average="macro")), 4)
        bench_rows.append(
            {
                "Model": name,
                "Accuracy": round(float(accuracy_score(yt, yp)), 4),
                "Macro_F1": f1,
                "Cohens_Kappa": round(float(cohen_kappa_score(yt, yp)), 4),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_cm = confusion_matrix(yt, yp)

    yr_t, yr_p = [], []
    for tr, te in gkf.split(X_sc, y_reg, groups):
        r = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42).fit(X_sc[tr], y_reg[tr])
        yr_t.extend(y_reg[te])
        yr_p.extend(r.predict(X_sc[te]))
    reg_rmse = round(float(np.sqrt(mean_squared_error(yr_t, yr_p))), 4)
    reg_r2 = round(float(r2_score(yr_t, yr_p)), 4)

    dl_path = REPORTS_DIR / "modeling" / "dl_metrics.json"
    dl_metrics = json.loads(dl_path.read_text(encoding="utf-8")) if dl_path.exists() else {}

    # 2. Analisis & Modeling Gambar Masakan Indonesia Publik Serupa (Padang Cuisine — 993 Foto, 9 Kelas)
    print("Mengekstrak & mengevaluasi 993 citra masakan Indonesia (fahmindra/padang-cuisine)...", flush=True)
    df_padang_meta, padang_cm, padang_classes, padang_metrics = analyze_padang_food_images()

    # 3. Analisis & Modeling Dokumen MIDD & CORD-v2
    df_midd, midd_metrics = analyze_midd_documents()
    df_cord, cord_metrics = analyze_cord_receipts()

    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv(ADJUST_DIR / "adjust_multimodal_umkm_benchmark.csv", index=False)

    best_padang_row = max(padang_metrics["benchmark"], key=lambda x: x["Test_Accuracy"])
    df_summary_all = pd.DataFrame([
        {
            "Modalitas_Tahap": "1. Repeat Jurnal Utama (Tabular 48 SWK)",
            "Sumber_Dataset": "Satu Data Surabaya (Selayanti et al., 2024)",
            "Metode_Utama": "AHC Complete Linkage (k=3)",
            "Metrik_Utama": "Cophenetic = 0.8566 | Silhouette = 0.4893 (Paper: 0.4864)",
        },
        {
            "Modalitas_Tahap": "2A. Modifikasi Gambar + Teks/Numerik UMKM",
            "Sumber_Dataset": f"549 Foto ({photo_mb} MB) + 385 UMKM + 1.164 Ulasan Surabaya Timur",
            "Metode_Utama": "K-Means, Autoencoder, RF, Gradient Boosting, PyTorch MLP",
            "Metrik_Utama": f"GB Acc = {df_bench.iloc[-1]['Accuracy']:.4f}, F1 = {df_bench.iloc[-1]['Macro_F1']:.4f} | K-Means Sil = {km_sil:.4f} | Reg R2 = {reg_r2:.4f}",
        },
        {
            "Modalitas_Tahap": "2B. Gambar Masakan Indonesia Serupa (Padang Cuisine)",
            "Sumber_Dataset": f"HF fahmindra/padang-cuisine (ODbL, {padang_metrics['total_images']} foto, 9 kelas — pengganti Mendeley 403 JuTISI 2025)",
            "Metode_Utama": f"HSV/Lab Spatial Pyramid + Sobel Texture + {best_padang_row['Model']}",
            "Metrik_Utama": f"Test Acc = {best_padang_row['Test_Accuracy']:.4f}, Macro-F1 = {best_padang_row['Test_Macro_F1']:.4f} (9 Kelas Seimbang)",
        },
        {
            "Modalitas_Tahap": "3A. Dokumen Invoice (MIDD - 708 CSV IOB)",
            "Sumber_Dataset": "MIDD Zenodo 10.5281/zenodo.5113009 (Baviskar et al., 2021)",
            "Metode_Utama": "TF-IDF + LinearSVC & Random Forest (4-Layout Classification)",
            "Metrik_Utama": f"SVM Acc = {midd_metrics['layout_clf_svm_acc']:.4f}, F1 = {midd_metrics['layout_clf_svm_f1']:.4f} | RF Acc = {midd_metrics['layout_clf_rf_acc']:.4f}",
        },
        {
            "Modalitas_Tahap": "3B. Dokumen Struk Indonesia (CORD-v2)",
            "Sumber_Dataset": "CORD-v2 (Park et al., 2019; HF iqbal-rizanta/cord-v2-bucket)",
            "Metode_Utama": "Receipt Metadata Feature Extraction + RF Payment Classifier",
            "Metrik_Utama": f"Acc = {cord_metrics['digital_payment_rf_acc']:.4f}, F1 = {cord_metrics['digital_payment_rf_f1']:.4f} | Median Struk Rp {cord_metrics['median_total_idr']:,}",
        },
    ])
    df_summary_all.to_csv(ADJUST_DIR / "adjust_vs_repeat_multimodal_summary.csv", index=False)

    print("=== RINGKASAN MODIFIKASI (3 MODALITAS TERPADU) ===", flush=True)
    print(df_summary_all.to_string(index=False), flush=True)
    print("\nPadang Food Image Metrics:", json.dumps(padang_metrics, indent=2), flush=True)

    # Grafik 1: Performa Model Multimodal UMKM + Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    melted = df_bench.melt(id_vars="Model", value_vars=["Accuracy", "Macro_F1", "Cohens_Kappa"], var_name="Metrik", value_name="Skor")
    sns.barplot(data=melted, x="Model", y="Skor", hue="Metrik", palette="viridis", ax=axes[0])
    axes[0].set_title("Benchmark Klasifikasi Multimodal UMKM (GroupKFold 5 Kecamatan)")
    axes[0].set_ylim(0, 1.12)
    axes[0].tick_params(axis="x", rotation=12)

    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Greens", xticklabels=list(LABEL_NAMES.values()), yticklabels=list(LABEL_NAMES.values()), ax=axes[1])
    axes[1].set_title(f"Confusion Matrix Terbaik — Gradient Boosting (F1: {best_f1:.4f})")
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")
    plt.tight_layout()
    fig.savefig(ADJUST_DIR / "01_adjust_multimodal_umkm_evaluation.png", dpi=160)
    plt.close(fig)

    # Grafik 2: EDA & Evaluasi Dokumen MIDD & CORD-v2
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
    l_ser = pd.Series(midd_metrics["layout_counts"]).sort_index()
    sns.barplot(x=l_ser.index, y=l_ser.values, hue=l_ser.index, palette="Blues_d", legend=False, ax=axes[0])
    axes[0].set_title(f"MIDD Invoice Dataset (699 Valid CSV) — SVM Acc: {midd_metrics['layout_clf_svm_acc']:.4f}")
    axes[0].set_ylabel("Jumlah Dokumen Invoice (.csv)")
    for i, v in enumerate(l_ser.values):
        axes[0].text(i, v + 8, str(v), ha="center", fontweight="bold")

    p_ser = pd.Series(cord_metrics["payment_distribution"])
    sns.barplot(x=p_ser.index, y=p_ser.values, hue=p_ser.index, palette="magma", legend=False, ax=axes[1])
    axes[1].set_title(f"CORD-v2 Struk Restoran Indonesia (200 Struk) — RF Acc: {cord_metrics['digital_payment_rf_acc']:.4f}")
    axes[1].set_ylabel("Jumlah Struk")
    axes[1].tick_params(axis="x", rotation=12)
    for i, v in enumerate(p_ser.values):
        axes[1].text(i, v + 2, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(ADJUST_DIR / "02_adjust_dokumen_midd_dan_cord.png", dpi=160)
    plt.close(fig)

    # Grafik 3: EDA & Evaluasi Gambar Masakan Indonesia Serupa (993 Foto, 9 Kelas)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    cls_ser = pd.Series(padang_metrics["class_counts"])
    sns.barplot(x=cls_ser.values, y=cls_ser.index, hue=cls_ser.index, palette="crest", legend=False, ax=axes[0])
    axes[0].set_title(f"Dataset Gambar Serupa: Padang Cuisine ({padang_metrics['total_images']} Foto, 9 Kelas)")
    axes[0].set_xlabel("Jumlah Foto (Train + Test)")
    for i, v in enumerate(cls_ser.values):
        axes[0].text(v + 1.5, i, str(v), va="center", fontweight="bold")

    short_lbls = [c.replace("_", " ").title() for c in padang_classes]
    sns.heatmap(padang_cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=short_lbls, yticklabels=short_lbls, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix — {padang_metrics['best_model']} (Test Acc: {padang_metrics['best_test_acc']:.4f})")
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(ADJUST_DIR / "03_adjust_gambar_padang_indonesia.png", dpi=160)
    plt.close(fig)

    summary = {
        "multimodal_umkm": {
            "umkm_rows": int(len(df_umkm)),
            "ulasan_rows": int(len(df_ulasan)),
            "foto_clean_count": int(len(df_foto)),
            "foto_clean_mb": photo_mb,
            "dokumen_pdf_count": int(len(df_dok)),
            "kmeans_silhouette": km_sil,
            "kmeans_davies_bouldin": km_dbi,
            "rf_reg_rmse": reg_rmse,
            "rf_reg_r2": reg_r2,
            "dl_metrics": dl_metrics,
            "classification": bench_rows,
        },
        "padang_food_images": padang_metrics,
        "midd_documents": midd_metrics,
        "cord_receipts": cord_metrics,
    }
    (ADJUST_DIR / "adjust_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSeluruh hasil Modifikasi Multimodal tersimpan di: {ADJUST_DIR}", flush=True)


if __name__ == "__main__":
    main()
