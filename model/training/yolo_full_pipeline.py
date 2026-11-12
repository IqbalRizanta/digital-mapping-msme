"""
YOLOv8 full vision pipeline (menggantikan yolo_indicator_pipeline.py).

Tahapan (jalankan berurutan, atau --stage all):
  filter      : CLIP zero-shot memisahkan foto TARGET (stan/etalase, kasir/POS, QRIS, EDC,
                stiker/banner merchant online) dari NON-TARGET (makanan, minuman, menu,
                ilustrasi/hasil generate AI, interior makan, selfie). Hasil -> CSV skor + contact sheet.
  clean       : bangun ulang data/clean/photos (HANYA foto target) + face blur OpenCV.
  autolabel   : YOLO-World (open-vocabulary) membuat pseudo-label bounding box nyata
                untuk 5 kelas: qris, edc, pos, merchant_online, stan.
  detect      : latih YOLOv8n-detect di atas pseudo-label -> results.csv, confusion matrix,
                PR/F1 curve, lalu prediksi -> gambar ber-bounding box berwarna + label + confidence.
  classify    : latih YOLOv8n-cls (Konvensional/Berkembang/Maju) dari foto survei SWK,
                prediksi -> gambar berlabel kelas + probabilitas.
  segment     : YOLOv8n-seg (pretrained COCO) -> gambar dengan mask segmentasi berwarna.
  pose        : YOLOv8n-pose (pretrained COCO) -> gambar dengan keypoint/skeleton.
  parquet     : tulis data/clean/foto.parquet dari deteksi NYATA (bukan dari label ground truth).
  sync        : salin seluruh output ke Google Drive.

Semua gambar beranotasi dibuat dari foto clean (wajah sudah di-blur).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE = Path(__file__).resolve().parents[3]
RAW_MAPS = BASE / "data" / "raw" / "google_maps"
SURVEY_INV = BASE / "data" / "raw" / "survei_offline" / "survey_images_inventory.csv"
CLEAN_DIR = BASE / "data" / "clean"
CLEAN_PHOTOS = CLEAN_DIR / "photos"
OUT = BASE / "reports" / "yolo"
MODELS = BASE / "models"
GDRIVE_OUT = Path("G:/My Drive/Iqbal-Model-Evaluation/yolo_outputs")
GDRIVE_CLEAN = Path("G:/My Drive/clean")

CLASS_NAMES = ["qris", "edc", "pos", "merchant_online", "stan"]
IMGSZ = 416
SEED = 42

# ----------------------------------------------------------------------------- 
# 1. FILTER (CLIP zero-shot)
# -----------------------------------------------------------------------------
TARGET_PROMPTS = {
    "stan": [
        "a photo of a street food stall",
        "a photo of a small food shop storefront with a signboard",
        "a photo of a food cart or kiosk on the street",
        "a photo of a food court booth with a name banner",
    ],
    "pos": [
        "a photo of a cashier counter with a computer or tablet",
        "a photo of a cash register at a shop counter",
    ],
    "qris": [
        "a photo of a QR code payment sticker or standee",
        "a photo of a printed QR code sign at a shop",
    ],
    "edc": ["a photo of a card payment terminal machine"],
    "merchant_online": [
        "a photo of a GoFood GrabFood or ShopeeFood sticker or banner on a shop",
        "a photo of a shop window with food delivery app logos",
    ],
}
# Gerbang biner ke-2: foto harus didominasi tempat usaha, bukan makanan/minuman/grafik
GATE_PLACE = [
    "a photo of a shop building or storefront",
    "a photo of a street food stall or food cart",
    "a photo of a shop counter or cashier desk",
    "a photo of a shop signboard or banner",
    "a photo of a QR code payment sign",
]
GATE_FOOD = [
    "a photo of food on a plate",
    "a photo of many dishes of food on a table",
    "a photo of trays of food in a buffet display",
    "a photo of a drink in a glass or cup",
    "a photo of a person holding a plate of food",
    "a digital graphic, poster or illustration of food or drinks",
    "an AI generated image of food",
    "a photo of a dining room with empty tables and chairs",
]
NONTARGET_PROMPTS = {
    "makanan": [
        "a close-up photo of a plate of food",
        "a photo of a bowl of noodles, soup or meatballs",
        "a photo of fried chicken and rice on a plate",
        "a photo of snacks or cakes",
    ],
    "minuman": [
        "a photo of a drink in a glass or plastic cup",
        "a photo of a cup of coffee or tea",
    ],
    "menu": ["a photo of a printed restaurant menu with prices"],
    "ai_generated": [
        "an AI generated digital illustration of food",
        "a 3D render or digital art of a dish",
        "a promotional poster graphic with text",
    ],
    "interior": ["a photo of a dining room with tables and chairs"],
    "orang": ["a selfie photo of people", "a group photo of people eating"],
    "lainnya": ["a screenshot of a phone app", "a photo of a street or parking lot"],
}


RAW_MAPS_EXT = BASE / "data" / "raw" / "google_maps_ext"


def _list_maps_photos() -> list[Path]:
    return sorted(RAW_MAPS.glob("*/photos/*.jpg")) + sorted(RAW_MAPS_EXT.glob("*/photos/*.jpg"))


def _load_place_meta() -> pd.DataFrame:
    """Gabungkan metadata tempat (lama + ext) beserta flag usaha besar."""
    from src.umkm.ingestion.expand_umkm_photos import CHAIN_KEYWORDS

    rows = []
    for d in (RAW_MAPS, RAW_MAPS_EXT):
        for f in d.glob("*/places_metadata.json"):
            rows += json.loads(f.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows).drop_duplicates("place_id")
    if "is_big_business" not in df.columns:
        df["is_big_business"] = np.nan
    name = df["name"].fillna("").str.lower()
    big = (name.apply(lambda n: any(k in n for k in CHAIN_KEYWORDS))
           | (df["user_ratings_total"].fillna(0) >= 1000)
           | df["price_level"].isin(["PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE"]))
    df["is_big_business"] = df["is_big_business"].fillna(big).astype(bool) | big
    return df


def stage_filter(threshold: float = 0.55) -> pd.DataFrame:
    import clip
    import torch

    out_dir = OUT / "filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    prompts, groups, is_target = [], [], []
    for g, ps in TARGET_PROMPTS.items():
        for p in ps:
            prompts.append(p); groups.append(g); is_target.append(True)
    for g, ps in NONTARGET_PROMPTS.items():
        for p in ps:
            prompts.append(p); groups.append(g); is_target.append(False)
    is_target = np.array(is_target)

    with torch.no_grad():
        txt = model.encode_text(clip.tokenize(prompts).to(device)).float()
        txt /= txt.norm(dim=-1, keepdim=True)
        gate_txt = model.encode_text(clip.tokenize(GATE_PLACE + GATE_FOOD).to(device)).float()
        gate_txt /= gate_txt.norm(dim=-1, keepdim=True)
    n_place = len(GATE_PLACE)

    rows = []
    photos = _list_maps_photos()
    print(f"-> CLIP zero-shot pada {len(photos)} foto Google Maps")
    for i, p in enumerate(photos, 1):
        try:
            pil = Image.open(p).convert("RGB")
            if np.asarray(pil).std() < 8:  # gambar kosong/rusak
                raise ValueError("blank image")
            img = preprocess(pil).unsqueeze(0).to(device)
        except Exception as e:  # file rusak
            rows.append({"raw_path": str(p), "place_id": p.name.split("_photo_")[0],
                         "kecamatan": p.parent.parent.name, "accepted": False,
                         "top_group": "rusak", "reason": f"unreadable: {e}"})
            continue
        with torch.no_grad():
            f = model.encode_image(img).float()
            f /= f.norm(dim=-1, keepdim=True)
            probs = (100.0 * f @ txt.T).softmax(dim=-1).cpu().numpy()[0]
            gate = (100.0 * f @ gate_txt.T).softmax(dim=-1).cpu().numpy()[0]
        top = int(probs.argmax())
        p_target = float(probs[is_target].sum())
        p_place = float(gate[:n_place].sum())
        accepted = bool(is_target[top] and p_target >= threshold and p_place >= 0.6)
        rows.append({
            "raw_path": str(p),
            "place_id": p.name.split("_photo_")[0],
            "kecamatan": p.parent.parent.name,
            "source_dir": p.parents[2].name,
            "p_target": round(p_target, 4),
            "p_place_gate": round(p_place, 4),
            "top_group": groups[top],
            "top_prompt": prompts[top],
            "top_prob": round(float(probs[top]), 4),
            "accepted": accepted,
            "reason": (f"target:{groups[top]}" if accepted else
                       (f"non-target:{groups[top]}" if not is_target[top] or p_target < threshold
                        else "gate:dominan makanan/minuman/grafik")),
        })
        if i % 50 == 0:
            print(f"   {i}/{len(photos)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "clip_filter_scores.csv", index=False)
    acc = df[df.accepted]
    print(f"-> Diterima (target): {len(acc)} | Ditolak: {len(df) - len(acc)}")
    print(df.groupby(["accepted", "top_group"]).size().to_string())
    _contact_sheet(acc.raw_path.tolist(), out_dir / "accepted_contact_sheet.jpg", "DITERIMA (target)")
    _contact_sheet(df[~df.accepted].raw_path.tolist(), out_dir / "rejected_contact_sheet.jpg", "DITOLAK (non-target)")
    return df


def _contact_sheet(paths: list[str], dest: Path, title: str, cell: int = 160, cols: int = 12):
    paths = [p for p in paths if Path(p).exists()][:240]
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell + 40, cols * cell, 3), 255, np.uint8)
    cv2.putText(sheet, f"{title} - {len(paths)} foto", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (cell, cell))
        r, c = divmod(i, cols)
        sheet[40 + r * cell: 40 + (r + 1) * cell, c * cell:(c + 1) * cell] = im
    cv2.imwrite(str(dest), sheet)


# -----------------------------------------------------------------------------
# 2. CLEAN ZONE (hanya foto target) + face blur
# -----------------------------------------------------------------------------
def stage_clean() -> pd.DataFrame:
    from src.umkm.processing.image_processor import ImageProcessor

    scores = pd.read_csv(OUT / "filter" / "clip_filter_scores.csv")
    if CLEAN_PHOTOS.exists():
        shutil.rmtree(CLEAN_PHOTOS)
    proc = ImageProcessor(clean_photo_dir=str(CLEAN_PHOTOS))
    recs = []

    survey = pd.read_csv(SURVEY_INV)
    for idx, r in survey.iterrows():
        raw = BASE / r["relative_path"]
        res = proc.process_and_save(raw, subfolder=f"offline/{r['folder_class']}")
        if res:
            recs.append({"foto_id": f"FOTO_OFF_{idx + 1:03d}", "umkm_id": f"UMKM_SURVEI_{idx + 1:03d}",
                         "sumber": "survei_lapangan_swk", "kecamatan": "Sukolilo",
                         "label_ground_truth": r["label_ground_truth"], "label_id": int(r["label_id"]),
                         "clip_p_target": np.nan, "clip_group": "survei", **_meta(res)})

    umkm = pd.read_parquet(CLEAN_DIR / "umkm.parquet")
    pid2umkm = dict(zip(umkm["place_id"], umkm["umkm_id"]))
    meta = _load_place_meta()
    for i, pid in enumerate(sorted(set(meta.place_id) - set(pid2umkm)), 1):
        pid2umkm[pid] = f"UMKM_EXT_{i:04d}"

    # Aturan usaha besar: hanya dipertahankan jika punya >=1 foto kasir/QRIS/EDC yang lolos CLIP
    big_ids = set(meta.loc[meta.is_big_business, "place_id"])
    acc = scores[scores.accepted].copy()
    pay_ok = set(acc.loc[acc.top_group.isin(["pos", "qris", "edc"]), "place_id"])
    drop_big = big_ids - pay_ok
    acc = acc[~acc.place_id.isin(drop_big)]
    print(f"-> Usaha besar: {len(big_ids & set(scores.place_id))} | dibuang (tanpa foto kasir/pembayaran): "
          f"{len(drop_big & set(scores.place_id))}")
    meta.to_csv(OUT / "filter" / "places_meta_with_bigflag.csv", index=False)
    n = 1
    for _, r in acc.iterrows():
        raw = Path(r["raw_path"])
        res = proc.process_and_save(raw, subfolder=f"online/{r['kecamatan']}")
        if res:
            recs.append({"foto_id": f"FOTO_ONL_{n:04d}", "umkm_id": pid2umkm.get(r["place_id"], "UNKNOWN"),
                         "sumber": "google_places_api", "kecamatan": str(r["kecamatan"]).replace("_", " ").title(),
                         "label_ground_truth": "Unlabeled", "label_id": -1,
                         "clip_p_target": r["p_target"], "clip_group": r["top_group"], **_meta(res)})
            n += 1
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "filter" / "clean_manifest.csv", index=False)
    print(f"-> Clean zone: {len(df)} foto ({(df.sumber == 'survei_lapangan_swk').sum()} survei, "
          f"{(df.sumber == 'google_places_api').sum()} Maps), wajah di-blur: {df.face_blurred_count.sum()}")
    return df


def _meta(res: dict) -> dict:
    return {"width": res["width"], "height": res["height"], "face_blurred_count": res["face_blurred_count"],
            "clean_path": res["clean_path"], "sha256": res["clean_sha256"]}


# -----------------------------------------------------------------------------
# 3. AUTOLABEL (YOLO-World open-vocabulary) -> dataset YOLO
# -----------------------------------------------------------------------------
WORLD_PROMPTS = [  # (prompt, class_id, conf_min)
    ("qr code", 0, 0.10), ("qr code sticker", 0, 0.10),
    ("card payment terminal", 1, 0.10), ("credit card machine", 1, 0.10),
    ("cash register", 2, 0.10), ("tablet computer", 2, 0.15), ("computer monitor", 2, 0.15),
    ("delivery app sticker", 3, 0.10), ("green delivery jacket", 3, 0.15), ("food delivery banner", 3, 0.10),
    ("food stall", 4, 0.20), ("food cart", 4, 0.20), ("storefront", 4, 0.20), ("kiosk", 4, 0.20),
]


def stage_autolabel() -> Path:
    from ultralytics import YOLOWorld

    manifest = pd.read_csv(OUT / "filter" / "clean_manifest.csv")
    ds = BASE / "data" / "yolo"
    if ds.exists():
        shutil.rmtree(ds)
    for s in ("train", "val"):
        (ds / "images" / s).mkdir(parents=True)
        (ds / "labels" / s).mkdir(parents=True)
    prev = OUT / "autolabel" / "preview"
    if prev.parent.exists():
        shutil.rmtree(prev.parent)
    prev.mkdir(parents=True)

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes([p for p, _, _ in WORLD_PROMPTS])

    # split per UMKM (group) agar tidak bocor
    groups = sorted(manifest.umkm_id.unique())
    random.Random(SEED).shuffle(groups)
    val_groups = set(groups[: max(1, int(len(groups) * 0.2))])

    stats = []
    for _, r in manifest.iterrows():
        img_p = Path(r["clean_path"])
        res = model.predict(str(img_p), imgsz=640, conf=0.08, iou=0.5, verbose=False)[0]
        lines, kept_cls = [], []
        for b in res.boxes:
            pi = int(b.cls); conf = float(b.conf)
            _, cid, cmin = WORLD_PROMPTS[pi]
            if conf < cmin:
                continue
            x, y, w, h = b.xywhn[0].tolist()
            lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            kept_cls.append(cid)
        split = "val" if r["umkm_id"] in val_groups else "train"
        dst = ds / "images" / split / f"{r['foto_id']}.jpg"
        shutil.copy2(img_p, dst)
        (ds / "labels" / split / f"{r['foto_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats.append({"foto_id": r["foto_id"], "split": split, "n_boxes": len(lines),
                      **{f"n_{c}": kept_cls.count(i) for i, c in enumerate(CLASS_NAMES)}})
        if len(stats) <= 60:
            cv2.imwrite(str(prev / f"{r['foto_id']}.jpg"), res.plot())

    (ds / "dataset.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "autolabel" / "pseudo_label_stats.csv", index=False)
    print("-> Pseudo-label per kelas:", {c: int(st[f'n_{c}'].sum()) for c in CLASS_NAMES})
    print(f"-> Train: {(st.split == 'train').sum()} | Val: {(st.split == 'val').sum()}")
    return ds / "dataset.yaml"


# -----------------------------------------------------------------------------
# 4. DETECT: train + predict
# -----------------------------------------------------------------------------
def stage_detect(epochs: int):
    from ultralytics import YOLO

    proj = OUT / "detect"
    model = YOLO("yolov8n.pt")
    model.train(data=str(BASE / "data" / "yolo" / "dataset.yaml"), epochs=epochs, imgsz=IMGSZ, batch=8,
                device="cpu", workers=0, project=str(proj), name="train", exist_ok=True, seed=SEED,
                plots=True, verbose=True)
    best = proj / "train" / "weights" / "best.pt"
    shutil.copy2(best, MODELS / "yolov8n_indicator_detect.pt")
    best_model = YOLO(str(best))
    m = best_model.val(data=str(BASE / "data" / "yolo" / "dataset.yaml"), imgsz=IMGSZ, device="cpu",
                       project=str(proj), name="val", exist_ok=True, plots=True)
    metrics = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr),
               "per_class_mAP50": {CLASS_NAMES[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)},
               "epochs": epochs, "imgsz": IMGSZ, "label_source": "YOLO-World pseudo-label (belum diverifikasi manual)"}
    (proj / "metrics_detect.json").write_text(json.dumps(metrics, indent=2))
    print("-> METRIK DETEKSI (nyata):", json.dumps(metrics, indent=2))

    # prediksi seluruh foto clean -> gambar ber-bounding box berwarna + label + conf
    manifest = pd.read_csv(OUT / "filter" / "clean_manifest.csv")
    rows = []
    for _, r in manifest.iterrows():
        res = best_model.predict(r["clean_path"], imgsz=IMGSZ, conf=0.25, device="cpu", save=True,
                                 save_txt=True, save_conf=True, project=str(proj), name="predict",
                                 exist_ok=True, verbose=False)[0]
        cls = [CLASS_NAMES[int(c)] for c in res.boxes.cls.tolist()]
        rows.append({"foto_id": r["foto_id"], "clean_path": r["clean_path"],
                     "detected_classes": ", ".join(sorted(set(cls))),
                     **{f"det_{c}": cls.count(c) for c in CLASS_NAMES},
                     "max_conf": float(res.boxes.conf.max()) if len(res.boxes) else 0.0})
    pd.DataFrame(rows).to_csv(proj / "predictions_detect.csv", index=False)


# -----------------------------------------------------------------------------
# 5. CLASSIFY (kematangan digital) train + predict
# -----------------------------------------------------------------------------
def stage_classify(epochs: int):
    from ultralytics import YOLO

    survey = pd.read_csv(SURVEY_INV)
    manifest = pd.read_csv(OUT / "filter" / "clean_manifest.csv")
    off = manifest[manifest.sumber == "survei_lapangan_swk"].copy()
    ds = BASE / "data" / "yolo_cls"
    if ds.exists():
        shutil.rmtree(ds)
    rng = random.Random(SEED)
    for lbl, grp in off.groupby("label_ground_truth"):
        idx = list(grp.index); rng.shuffle(idx)
        n_val = max(1, round(len(idx) * 0.25))
        for k, i in enumerate(idx):
            split = "val" if k < n_val else "train"
            d = ds / split / lbl
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(off.loc[i, "clean_path"], d / f"{off.loc[i, 'foto_id']}.jpg")

    proj = OUT / "classify"
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(ds), epochs=epochs, imgsz=224, batch=8, device="cpu", workers=0,
                project=str(proj), name="train", exist_ok=True, seed=SEED, plots=True)
    best = proj / "train" / "weights" / "best.pt"
    shutil.copy2(best, MODELS / "yolov8n_maturity_cls.pt")
    bm = YOLO(str(best))
    m = bm.val(data=str(ds), imgsz=224, device="cpu", project=str(proj), name="val", exist_ok=True, plots=True)
    metrics = {"top1_acc": float(m.top1), "top5_acc": float(m.top5), "epochs": epochs,
               "n_train": len(list((ds / 'train').rglob('*.jpg'))), "n_val": len(list((ds / 'val').rglob('*.jpg')))}
    (proj / "metrics_classify.json").write_text(json.dumps(metrics, indent=2))
    print("-> METRIK KLASIFIKASI (nyata):", metrics)

    rows = []
    for _, r in manifest.iterrows():
        res = bm.predict(r["clean_path"], imgsz=224, device="cpu", save=True, project=str(proj),
                         name="predict", exist_ok=True, verbose=False)[0]
        rows.append({"foto_id": r["foto_id"], "pred_class": res.names[int(res.probs.top1)],
                     "pred_conf": float(res.probs.top1conf), "label_ground_truth": r["label_ground_truth"]})
    pd.DataFrame(rows).to_csv(proj / "predictions_classify.csv", index=False)


# -----------------------------------------------------------------------------
# 6/7. SEGMENT & POSE (pretrained COCO)
# -----------------------------------------------------------------------------
def _run_pretrained(task: str, weights: str):
    from ultralytics import YOLO

    manifest = pd.read_csv(OUT / "filter" / "clean_manifest.csv")
    proj = OUT / task
    model = YOLO(weights)
    rows, t0 = [], time.time()
    for _, r in manifest.iterrows():
        res = model.predict(r["clean_path"], imgsz=640, conf=0.3, device="cpu", save=True,
                            project=str(proj), name="predict", exist_ok=True, verbose=False)[0]
        names = [res.names[int(c)] for c in res.boxes.cls.tolist()]
        row = {"foto_id": r["foto_id"], "n_objects": len(names),
               "objects": ", ".join(f"{n}:{names.count(n)}" for n in sorted(set(names))),
               **{k: round(v, 1) for k, v in res.speed.items()}}
        if task == "pose":
            row["n_person_keypoints"] = 0 if res.keypoints is None else int(res.keypoints.shape[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(proj / f"predictions_{task}.csv", index=False)
    summary = {"weights": weights, "n_images": len(df), "total_objects": int(df.n_objects.sum()),
               "images_with_objects": int((df.n_objects > 0).sum()),
               "mean_inference_ms": float(df["inference"].mean()), "wall_time_s": round(time.time() - t0, 1)}
    (proj / f"summary_{task}.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {task.upper()}:", summary)


# -----------------------------------------------------------------------------
# 8. foto.parquet dari deteksi nyata
# -----------------------------------------------------------------------------
def stage_parquet():
    man = pd.read_csv(OUT / "filter" / "clean_manifest.csv")
    det = pd.read_csv(OUT / "detect" / "predictions_detect.csv").drop(columns=["clean_path"])
    df = man.merge(det, on="foto_id", how="left")
    cls_p = OUT / "classify" / "predictions_classify.csv"
    if cls_p.exists():
        df = df.merge(pd.read_csv(cls_p)[["foto_id", "pred_class", "pred_conf"]], on="foto_id", how="left")
    for c in CLASS_NAMES:
        df[f"det_{c}"] = df[f"det_{c}"].fillna(0).astype(int)
        df[f"has_{c}"] = df[f"det_{c}"] > 0
    df["detected_classes"] = df["detected_classes"].fillna("")
    df["num_indicators"] = df[[f"has_{c}" for c in CLASS_NAMES]].sum(axis=1)
    df.to_parquet(CLEAN_DIR / "foto.parquet", index=False)
    print(f"-> foto.parquet: {len(df)} baris")
    print(df[[f"has_{c}" for c in CLASS_NAMES]].sum().to_string())


# -----------------------------------------------------------------------------
# 9. SYNC ke Google Drive
# -----------------------------------------------------------------------------
def stage_sync():
    for src, dst in [(OUT, GDRIVE_OUT), (CLEAN_PHOTOS, GDRIVE_CLEAN / "photos")]:
        if not dst.parent.exists():
            print(f"-> Lewati sync (tidak ada): {dst.parent}")
            continue
        try:
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.cache"), dirs_exist_ok=True)
            print(f"-> [GDRIVE SYNC] {src} -> {dst}")
        except Exception as e:
            print(f"-> [GDRIVE SYNC WARNING] {src} -> {dst}: {e}")
    if GDRIVE_CLEAN.exists():
        try:
            shutil.copy2(CLEAN_DIR / "foto.parquet", GDRIVE_CLEAN / "foto.parquet")
            print("-> [GDRIVE SYNC] foto.parquet tersinkron")
        except Exception as e:
            print(f"-> [GDRIVE SYNC WARNING] foto.parquet: {e}")
    for w in MODELS.glob("yolov8n_*.pt"):
        if GDRIVE_OUT.parent.exists():
            try:
                shutil.copy2(w, GDRIVE_OUT.parent / w.name)
                print(f"-> [GDRIVE SYNC] model {w.name} tersinkron")
            except Exception as e:
                print(f"-> [GDRIVE SYNC WARNING] {w.name}: {e}")


STAGES = ["filter", "clean", "autolabel", "detect", "classify", "segment", "pose", "parquet", "sync"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help=",".join(STAGES + ["all"]))
    ap.add_argument("--det-epochs", type=int, default=5)
    ap.add_argument("--cls-epochs", type=int, default=5)
    a = ap.parse_args()
    todo = STAGES if a.stage == "all" else a.stage.split(",")
    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in todo:
        print(f"\n===== STAGE: {s} =====", flush=True)
        if s == "filter": stage_filter()
        elif s == "clean": stage_clean()
        elif s == "autolabel": stage_autolabel()
        elif s == "detect": stage_detect(a.det_epochs)
        elif s == "classify": stage_classify(a.cls_epochs)
        elif s == "segment": _run_pretrained("segment", "yolov8n-seg.pt")
        elif s == "pose": _run_pretrained("pose", "yolov8n-pose.pt")
        elif s == "parquet": stage_parquet()
        elif s == "sync": stage_sync()
