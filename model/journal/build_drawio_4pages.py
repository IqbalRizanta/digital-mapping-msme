"""
Generator Diagram Draw.io 4 Halaman (format XML <mxfile>) + Preview PNG:
  - Page 1: Architecture Jurnal Utama (Selayanti et al., 2024 — 48 SWK Surabaya, AHC) — Module Only
  - Page 2: Flowchart Jurnal Utama (Selayanti et al., 2024) — Selaras Notebook Repeat/
  - Page 3: Architecture Modifikasi Mahasiswa (Multimodal Dataset A, B, C) — Module Only
  - Page 4: Flowchart Modifikasi Mahasiswa (Bercabang Dataset A, B, C pola Figure (d)) — Selaras Notebook Modifikasi/
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT_DIR / "docs" / "diagrams"
OUT_DRAWIO = DIAG_DIR / "BDA_Journal_Repeat_dan_Modifikasi_4Pages.drawio"


def _mx_box(cid: str, label: str, x: int, y: int, w: int, h: int, style: str, parent: str = "1") -> str:
    val = escape(label).replace("\n", "&lt;br&gt;")
    return (
        f'        <mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="{parent}">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
        f'        </mxCell>'
    )


def _mx_edge(
    eid: str,
    src: str,
    tgt: str,
    label: str = "",
    dashed: bool = False,
    extra: str = "",
    parent: str = "1",
    waypoints: list[tuple[int, int]] | None = None,
) -> str:
    dash_str = "dashed=1;dashPattern=6 4;" if dashed else ""
    val = escape(label)
    style = (
        f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
        f"strokeWidth=1.8;endArrow=block;endFill=1;strokeColor=#1A1A1A;fontColor=#1A1A1A;fontSize=11;fontStyle=1;"
        f"labelBackgroundColor=#FFFFFF;{dash_str}{extra}"
    )
    if waypoints:
        pts = "\n".join(f'              <mxPoint x="{px}" y="{py}" />' for px, py in waypoints)
        geom = (
            '          <mxGeometry relative="1" as="geometry">\n'
            '            <Array as="points">\n'
            f'{pts}\n'
            '            </Array>\n'
            '          </mxGeometry>'
        )
    else:
        geom = '          <mxGeometry relative="1" as="geometry" />'
    return (
        f'        <mxCell id="{eid}" value="{val}" style="{style}" edge="1" parent="{parent}" source="{src}" target="{tgt}">\n'
        f'{geom}\n'
        f'        </mxCell>'
    )


def _wrap_diagram(diag_id: str, name: str, cells: list[str], pw: int = 1654, ph: int = 1169) -> str:
    body = "\n".join(cells)
    return (
        f'  <diagram id="{diag_id}" name="{escape(name)}">\n'
        f'    <mxGraphModel dx="1420" dy="820" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
        f'fold="1" page="1" pageScale="1" pageWidth="{pw}" pageHeight="{ph}" math="0" shadow="0">\n'
        f'      <root>\n'
        f'        <mxCell id="0" />\n'
        f'        <mxCell id="1" parent="0" />\n'
        f'{body}\n'
        f'      </root>\n'
        f'    </mxGraphModel>\n'
        f'  </diagram>'
    )


# ---------------------------------------------------------------------------
# PAGE 1: ARCHITECTURE JURNAL UTAMA (MODULE ONLY)
# ---------------------------------------------------------------------------
def build_page1_arch_jurnal() -> str:
    grp = "rounded=1;arcSize=4;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=12;spacingTop=6;strokeWidth=2;fillColor=#F8FAFC;strokeColor=#1F4E79;fontColor=#0B2E59;"
    mod = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fontSize=11;fillColor=#FFFFFF;strokeColor=#2C5282;fontColor=#111111;"
    cyl = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=12;fontSize=11;fontStyle=1;fillColor=#EDF2F7;strokeColor=#2C5282;fontColor=#111111;"

    c = [
        _mx_box("p1_title", "Page 1 — Architecture Jurnal Utama: Segmentasi SWK Surabaya (Selayanti et al., 2024)", 40, 20, 980, 34, "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#0B2E59;"),
        _mx_box("p1_note", "Catatan: Setiap node hanya menampilkan Module proses sesuai ketentuan (tanpa daftar library).", 40, 56, 800, 22, "text;html=1;align=left;verticalAlign=middle;fontSize=11;fontColor=#4A5568;"),

        # Containers
        _mx_box("g_ds", "DATA SOURCE", 40, 110, 210, 200, grp),
        _mx_box("m_ds1", "Satu Data Surabaya\n(48 SWK, 11 Variabel)", 60, 165, 170, 95, mod),

        _mx_box("g_st", "DATA STORAGE", 290, 110, 210, 320, grp),
        _mx_box("m_st1", "Raw Data\n(Dataset SWK 48 Baris)", 315, 155, 160, 70, cyl),
        _mx_box("m_st2", "Processed Data\n(Dataset Z-Score)", 315, 245, 160, 70, cyl),
        _mx_box("m_st3", "Output Data\n(Label Cluster & Grafik)", 315, 335, 160, 70, cyl),

        _mx_box("g_pre", "PRE-PROCESSING", 550, 110, 250, 280, grp),
        _mx_box("m_pre1", "Seleksi Variabel\n(4 Fitur Utama SWK)", 575, 155, 200, 55, mod),
        _mx_box("m_pre2", "Missing Value\n(Imputasi Mean)", 575, 225, 200, 55, mod),
        _mx_box("m_pre3", "Normalisasi Data\n(Z-Score)", 575, 295, 200, 55, mod),

        _mx_box("g_eda", "EXPLORATORY DATA ANALYSIS (EDA)", 550, 420, 250, 250, grp),
        _mx_box("m_eda1", "Statistika Deskriptif", 575, 465, 200, 48, mod),
        _mx_box("m_eda2", "Perbandingan Pelaku Usaha\n& Omzet > 1 Juta", 575, 525, 200, 52, mod),
        _mx_box("m_eda3", "Histogram & Kapasitas Sentra", 575, 590, 200, 48, mod),

        _mx_box("g_mod", "MODELING (HIERARCHICAL CLUSTERING)", 850, 110, 250, 300, grp),
        _mx_box("m_mod1", "Single Linkage", 875, 155, 200, 48, mod),
        _mx_box("m_mod2", "Average Linkage", 875, 215, 200, 48, mod),
        _mx_box("m_mod3", "Complete Linkage (Terpilih)", 875, 275, 200, 48, mod),
        _mx_box("m_mod4", "Ward Linkage", 875, 335, 200, 48, mod),

        _mx_box("g_ev", "MODEL EVALUATION", 850, 440, 250, 230, grp),
        _mx_box("m_ev1", "Korelasi Cophenetic\n(Validasi Linkage Terbaik)", 875, 490, 200, 62, mod),
        _mx_box("m_ev2", "Silhouette Coefficient\n(Validasi Jumlah Cluster k)", 875, 570, 200, 62, mod),

        _mx_box("g_vis", "ANALYSIS & VISUALIZATION", 1150, 110, 260, 300, grp),
        _mx_box("m_vis1", "Dendrogram Cluster", 1175, 160, 210, 52, mod),
        _mx_box("m_vis2", "Interpretasi Profil Cluster\n(Rata-Rata 3 Cluster)", 1175, 230, 210, 58, mod),
        _mx_box("m_vis3", "Strategi Optimalisasi Omzet\nSWK Surabaya", 1175, 305, 210, 58, mod),

        # Edges
        _mx_edge("e1", "g_ds", "g_st"),
        _mx_edge("e2", "g_st", "g_pre", extra="startArrow=block;startFill=1;"),
        _mx_edge("e3", "g_st", "g_eda"),
        _mx_edge("e4", "g_pre", "g_mod"),
        _mx_edge("e5", "g_mod", "g_ev"),
        _mx_edge("e6", "g_ev", "g_vis"),
        _mx_edge("e7", "g_vis", "g_st", dashed=True, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=1;"),
    ]
    return _wrap_diagram("page1_arch_jurnal", "Page-1 Architecture Jurnal", c, 1500, 760)


# ---------------------------------------------------------------------------
# PAGE 2: FLOWCHART JURNAL UTAMA (SELARAS NOTEBOOK REPEAT/)
# ---------------------------------------------------------------------------
def build_page2_flow_jurnal() -> str:
    start_st = "ellipse;whiteSpace=wrap;html=1;fontSize=12;fontStyle=1;fillColor=#E6FFFA;strokeColor=#00695C;strokeWidth=2;"
    proc_st = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=11;fontStyle=1;fillColor=#FFFFFF;strokeColor=#1A1A1A;strokeWidth=1.8;"
    cyl_st = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=12;fontSize=11;fontStyle=1;fillColor=#EDF2F7;strokeColor=#1A1A1A;strokeWidth=1.8;"
    dec_st = "rhombus;whiteSpace=wrap;html=1;fontSize=11;fontStyle=1;fillColor=#FFFBEB;strokeColor=#B7791F;strokeWidth=1.8;"

    c = [
        _mx_box("p2_title", "Page 2 — Flowchart Jurnal Utama: AHC 48 SWK Surabaya (Selaras Notebook Repeat/)", 40, 20, 960, 34, "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#0B2E59;"),

        # Row 1 (Left -> Right)
        _mx_box("f_start", "Start", 50, 100, 120, 56, start_st),
        _mx_box("f_ds", "Dataset SWK\n(48 Baris, 11 Var)", 220, 92, 145, 72, cyl_st),
        _mx_box("f_eda", "EDA Statistika Deskriptif\n(Pelaku Usaha, Kapasitas,\nOmzet > 1 Juta)", 420, 94, 210, 68, proc_st),
        _mx_box("f_sel", "Pre-processing:\nSeleksi 4 Variabel Utama", 685, 94, 195, 68, proc_st),
        _mx_box("f_miss", "Ada Missing\nValue?", 940, 86, 145, 84, dec_st),
        _mx_box("f_imp", "Imputasi Mean\n(Luas Sentra)", 1145, 94, 170, 68, proc_st),

        # Row 2 (Right -> Left)
        _mx_box("f_norm", "Normalisasi Data\nZ-Score (StandardScaler)", 1125, 250, 210, 68, proc_st),
        _mx_box("f_ahc", "Clustering AHC:\nSingle, Average,\nComplete, Ward Linkage", 840, 250, 220, 68, proc_st),
        _mx_box("f_coph", "Evaluasi Korelasi\nCophenetic (4 Linkage)", 565, 250, 210, 68, proc_st),
        _mx_box("f_sil", "Evaluasi Silhouette\nCoefficient (k = 2..6)", 295, 250, 205, 68, proc_st),
        _mx_box("f_opt", "Linkage & Cluster\nOptimal?", 60, 242, 165, 84, dec_st),

        # Row 3 (Decision loop + Final)
        _mx_box("f_tune", "Bandingkan Linkage &\nProporsi Cluster (k)", 60, 390, 190, 62, proc_st),
        _mx_box("f_pick", "Pilih Complete Linkage\n(k = 3; Cophenetic 0.8566,\nSilhouette 0.4893)", 340, 390, 230, 72, proc_st),
        _mx_box("f_prof", "Dendrogram & Profil\nRata-Rata 3 Cluster\n(Tabel 3 & 4)", 640, 390, 215, 72, proc_st),
        _mx_box("f_vis", "Visualisasi & Strategi\nOptimalisasi Omzet SWK", 925, 390, 215, 72, proc_st),
        _mx_box("f_end", "End", 1210, 398, 120, 56, start_st),

        # Edges
        _mx_edge("fe1", "f_start", "f_ds"),
        _mx_edge("fe2", "f_ds", "f_eda"),
        _mx_edge("fe3", "f_eda", "f_sel"),
        _mx_edge("fe4", "f_sel", "f_miss"),
        _mx_edge("fe5", "f_miss", "f_imp", "Ya"),
        _mx_edge("fe6", "f_imp", "f_norm"),
        _mx_edge("fe7", "f_miss", "f_norm", "Tidak", extra="exitX=0.5;exitY=1;entryX=0;entryY=0.5;"),
        _mx_edge("fe8", "f_norm", "f_ahc"),
        _mx_edge("fe9", "f_ahc", "f_coph"),
        _mx_edge("fe10", "f_coph", "f_sil"),
        _mx_edge("fe11", "f_sil", "f_opt"),
        _mx_edge("fe12", "f_opt", "f_tune", "Tidak", extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;"),
        _mx_edge("fe13", "f_tune", "f_ahc", dashed=True, extra="exitX=1;exitY=0.2;entryX=0.5;entryY=1;"),
        _mx_edge("fe14", "f_opt", "f_pick", "Ya", extra="exitX=1;exitY=0.75;entryX=0;entryY=0.5;"),
        _mx_edge("fe15", "f_pick", "f_prof"),
        _mx_edge("fe16", "f_prof", "f_vis"),
        _mx_edge("fe17", "f_vis", "f_end"),
    ]
    return _wrap_diagram("page2_flow_jurnal", "Page-2 Flowchart Jurnal", c, 1400, 540)


# ---------------------------------------------------------------------------
# PAGE 3: ARCHITECTURE MODIFIKASI MAHASISWA (MODULE ONLY, MULTIMODAL A/B/C)
# ---------------------------------------------------------------------------
def build_page3_arch_modifikasi() -> str:
    grp = "rounded=1;arcSize=4;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=11;spacingTop=6;strokeWidth=2;fillColor=#F8FAFC;strokeColor=#1A365D;fontColor=#0B2E59;"
    sub = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;verticalAlign=top;fontStyle=1;fontSize=11;spacingTop=4;strokeWidth=1.4;fillColor=#FFFFFF;strokeColor=#4A5568;fontColor=#1A202C;"
    mod = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fontSize=10;fillColor=#FFFFFF;strokeColor=#2B6CB0;fontColor=#111111;"
    cyl = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=11;fontSize=10;fontStyle=1;fillColor=#EDF2F7;strokeColor=#2B6CB0;fontColor=#111111;"

    c = [
        _mx_box("p3_title", "Page 3 — Architecture Modifikasi Mahasiswa: Multimodal Big Data & AI UMKM Kuliner Surabaya Timur", 40, 20, 1280, 34, "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#0B2E59;"),
        _mx_box("p3_note", "Catatan: Struktur blok mengikuti Arsitektur Mahasiswa dengan isi node khusus Module (Dataset A Tabular/Teks, Dataset B Gambar, Dataset C Dokumen).", 40, 56, 1180, 22, "text;html=1;align=left;verticalAlign=middle;fontSize=11;fontColor=#4A5568;"),

        # 1. DATA SOURCES
        _mx_box("m_ds", "DATA SOURCES (3 MODALITAS)", 40, 110, 230, 330, grp),
        _mx_box("ds_a", "Dataset A — Tabular & Teks\n(48 SWK + 385 UMKM +\n1.164 Ulasan Surabaya Timur)", 58, 150, 194, 76, mod),
        _mx_box("ds_b", "Dataset B — Gambar Indonesia\n(549 Foto UMKM/SWK +\n993 Foto Padang Cuisine / JuTISI)", 58, 242, 194, 76, mod),
        _mx_box("ds_c", "Dataset C — Dokumen\n(MIDD 708 Invoice + CORD-v2\n200 Struk Indonesia + 3 PDF)", 58, 334, 194, 76, mod),

        # 2. INGESTION & STAGING
        _mx_box("m_etl", "INGESTION & STAGING", 310, 110, 200, 330, grp),
        _mx_box("et_1", "Pengambilan Data Multi-Sumber", 328, 155, 164, 62, mod),
        _mx_box("et_2", "Penggabungan Batch Multimodal", 328, 238, 164, 62, mod),
        _mx_box("et_3", "Katalog Skema & Provenance", 328, 321, 164, 62, mod),

        # 3. DATA STORAGE
        _mx_box("m_st", "DATA STORAGE", 550, 110, 195, 420, grp),
        _mx_box("st_1", "Raw Data Zone\n(Mentah A, B, C)", 570, 150, 155, 72, cyl),
        _mx_box("st_2", "Clean Data Zone\n(Bersih Tervalidasi)", 570, 240, 155, 72, cyl),
        _mx_box("st_3", "Feature Store\n(Matriks & Embedding)", 570, 330, 155, 72, cyl),
        _mx_box("st_4", "Model Registry\n(Artefak ML & DL)", 570, 420, 155, 72, cyl),

        # 4. DATA TRANSFORMATION
        _mx_box("m_dt", "DATA TRANSFORMATION", 790, 110, 620, 420, grp),
        _mx_box("dt_pre", "Pre-processing", 808, 145, 584, 112, sub),
        _mx_box("pr_1", "Seleksi Variabel &\nImputasi Mean", 820, 178, 132, 64, mod),
        _mx_box("pr_2", "Pembersihan Teks &\nSensor PII", 964, 178, 132, 64, mod),
        _mx_box("pr_3", "Filter Visual &\nBlur Wajah", 1108, 178, 132, 64, mod),
        _mx_box("pr_4", "Parsing IOB, JSON\nStruk & PDF", 1252, 178, 132, 64, mod),

        _mx_box("dt_eda", "Exploratory Data Analysis (EDA)", 808, 272, 584, 108, sub),
        _mx_box("ed_1", "EDA Tabular & Teks\n(Dataset A)", 820, 305, 178, 60, mod),
        _mx_box("ed_2", "EDA Citra & Indikator\n(Dataset B)", 1012, 305, 178, 60, mod),
        _mx_box("ed_3", "EDA Invoice & Struk\n(Dataset C)", 1204, 305, 178, 60, mod),

        _mx_box("dt_fe", "Feature Extraction & Engineering", 808, 395, 584, 118, sub),
        _mx_box("fe_1", "Normalisasi Z-Score\n& Scaling", 820, 430, 132, 66, mod),
        _mx_box("fe_2", "Ekstraksi Fitur Teks\n& Indikator Visual", 964, 430, 132, 66, mod),
        _mx_box("fe_3", "Fusi Multimodal &\nReduksi Dimensi", 1108, 430, 132, 66, mod),
        _mx_box("fe_4", "Ekstraksi Fitur Teks\n& Metadata Dokumen", 1252, 430, 132, 66, mod),

        # 5. MODELING (MULTI-BRANCH)
        _mx_box("m_aos", "ANALYTICS & MODELING (MULTI-BRANCH)", 790, 565, 620, 210, grp),
        _mx_box("ao_1", "Branch A — Clustering\n(AHC Complete, K-Means,\nAutoencoder)", 810, 610, 184, 135, mod),
        _mx_box("ao_2", "Branch B — Multimodal\n(Klasifikasi 3 Kelas &\nRegresi Skor Kematangan)", 1008, 610, 184, 135, mod),
        _mx_box("ao_3", "Branch C — Dokumen\n(Klasifikasi 4 Layout MIDD &\nPembayaran Non-Tunai CORD)", 1206, 610, 184, 135, mod),

        # 6. EVALUATION, BDA & VISUALIZATION, APPLICATION
        _mx_box("m_me", "MODEL EVALUATION", 1460, 110, 230, 260, grp),
        _mx_box("ev_1", "Cophenetic & Silhouette", 1478, 150, 194, 46, mod),
        _mx_box("ev_2", "GroupKFold Accuracy,\nMacro-F1 & Cohen's Kappa", 1478, 206, 194, 48, mod),
        _mx_box("ev_3", "Regression RMSE & R2", 1478, 264, 194, 42, mod),
        _mx_box("ev_4", "StratifiedKFold Layout &\nReceipt Metrics", 1478, 316, 194, 44, mod),

        _mx_box("m_bda", "BIG DATA ANALYTICS & VISUALIZATION", 1460, 400, 230, 220, grp),
        _mx_box("bd_1", "Agregasi Digital Gap\n6 Kecamatan", 1478, 440, 194, 48, mod),
        _mx_box("bd_2", "Peta Sebaran Spasial &\nSpider Chart", 1478, 498, 194, 48, mod),
        _mx_box("bd_3", "Komparasi Repeat vs\nModifikasi", 1478, 556, 194, 48, mod),

        _mx_box("m_app", "INFERENCING & AI APPLICATION", 1460, 645, 230, 130, grp),
        _mx_box("ap_1", "Inference Service API &\nDashboard Rekomendasi", 1478, 690, 194, 65, mod),

        # Main Edges
        _mx_edge("me1", "m_ds", "m_etl"),
        _mx_edge("me2", "m_etl", "m_st"),
        _mx_edge("me3", "m_st", "m_dt", extra="startArrow=block;startFill=1;"),
        _mx_edge("me4", "m_dt", "m_aos", extra="startArrow=block;startFill=1;"),
        _mx_edge("me5", "m_st", "m_aos"),
        _mx_edge("me6", "m_aos", "m_me"),
        _mx_edge("me7", "m_me", "m_bda"),
        _mx_edge("me8", "m_bda", "m_app"),
    ]
    return _wrap_diagram("page3_arch_modifikasi", "Page-3 Architecture Modifikasi", c, 1740, 840)


# ---------------------------------------------------------------------------
# PAGE 4: FLOWCHART MODIFIKASI BERCABANG DATASET A, B, C (POLA FIGURE (d))
# ---------------------------------------------------------------------------
def build_page4_flow_modifikasi() -> str:
    cyl_st = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=12;fontSize=11;fontStyle=1;fillColor=#EDF2F7;strokeColor=#1A1A1A;strokeWidth=1.8;"
    proc_st = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=10.5;fontStyle=1;fillColor=#FFFFFF;strokeColor=#1A1A1A;strokeWidth=1.8;"
    b_a = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=10.5;fontStyle=1;fillColor=#EBF8FF;strokeColor=#2B6CB0;strokeWidth=1.8;"
    b_b = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=10.5;fontStyle=1;fillColor=#F0FFF4;strokeColor=#2F855A;strokeWidth=1.8;"
    b_c = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=10.5;fontStyle=1;fillColor=#FFFBEB;strokeColor=#B7791F;strokeWidth=1.8;"
    end_st = "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fontSize=11;fontStyle=1;fillColor=#E6FFFA;strokeColor=#00695C;strokeWidth=2;"

    c = [
        _mx_box("p4_title", "Page 4 — Flowchart Modifikasi Bercabang (Dataset A, Dataset B, Dataset C — Pola Figure (d))", 40, 20, 1220, 34, "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;fontColor=#0B2E59;"),

        # Top Root: Dataset -> Data Splitting / Routing
        _mx_box("r_ds", "Multi-Source\nDataset", 40, 95, 125, 68, cyl_st),
        _mx_box("r_split", "Data Splitting /\nRouting per Modalitas", 215, 100, 175, 58, proc_st),

        # Branch B (Top horizontal track — Gambar & Multimodal Fusion)
        _mx_box("cyl_b", "Dataset B\n(549 Foto UMKM +\n993 Foto Masakan ID)", 445, 90, 145, 78, cyl_st),
        _mx_box("pre_b", "Pre-processing &\nEDA B (Filter Visual,\nBlur Wajah)", 635, 96, 170, 66, b_b),
        _mx_box("fe_b", "Feature Extraction B\n(Indikator Visual +\nFusi Multimodal)", 845, 96, 170, 66, b_b),
        _mx_box("mod_b", "Modeling B\n(RF, SVM, GB, MLP\n+ Regressor)", 1055, 96, 165, 66, b_b),
        _mx_box("ev_b", "Evaluation B\n(GroupKFold F1,\nKappa, R2/RMSE)", 1260, 96, 160, 66, b_b),
        _mx_box("inf_b", "Inferences B\n(Prediksi 6 Model)", 1260, 235, 160, 62, b_b),

        # Branch A (Middle-Bottom track — Tabular & Teks SWK/UMKM Clustering)
        _mx_box("cyl_a", "Dataset A\n(48 SWK + 385 UMKM\n+ 1.164 Ulasan)", 160, 250, 155, 78, cyl_st),
        _mx_box("pre_a", "Pre-processing &\nEDA A (Imputasi,\nSensor PII, Z-Score)", 152, 375, 170, 66, b_a),
        _mx_box("cl_a", "Clustering A\n(AHC Complete,\nK-Means, Autoencoder)", 375, 375, 175, 66, b_a),
        _mx_box("mod_a", "Modeling A\n(Profil Klaster SWK\n& UMKM)", 600, 375, 165, 66, b_a),
        _mx_box("ev_a", "Evaluation A\n(Cophenetic &\nSilhouette Score)", 600, 235, 165, 62, b_a),

        # Branch C (Bottom track — Dokumen MIDD 708 Invoice + CORD-v2 200 Struk + 3 PDF)
        _mx_box("cyl_c", "Dataset C\n(MIDD 708 Invoice +\nCORD-v2 200 Struk)", 160, 505, 155, 78, cyl_st),
        _mx_box("pre_c", "Pre-processing &\nEDA C (Parsing IOB,\nJSON Struk, PDF)", 375, 511, 175, 66, b_c),
        _mx_box("fe_c", "Feature Extraction C\n(TF-IDF Invoice &\nFitur Transaksi Struk)", 600, 511, 170, 66, b_c),
        _mx_box("mod_c", "Modeling C\n(Klasifikasi Layout\n& Pembayaran Struk)", 820, 511, 170, 66, b_c),
        _mx_box("ev_c", "Evaluation C\n(StratifiedKFold\nAccuracy & F1)", 1035, 511, 160, 66, b_c),
        _mx_box("inf_c", "Inferences C\n(Ekstraksi Entitas &\nStatus Non-Tunai)", 1260, 511, 160, 66, b_c),

        # Convergence: Analysis -> Visualization
        _mx_box("ana", "Analysis\n(Spatial BDA Digital Gap\n6 Kecamatan & Komparasi\nRepeat vs Modifikasi)", 840, 225, 235, 82, end_st),
        _mx_box("vis", "Visualization\n(Dendrogram, Grouped Bar,\nPoint Map, Spider Chart,\nConfusion Matrix)", 840, 375, 235, 82, end_st),

        # Edges (Root -> 3 Dashed Branches)
        _mx_edge("pe0", "r_ds", "r_split"),
        _mx_edge("pe_sb", "r_split", "cyl_b", dashed=True),
        _mx_edge("pe_sa", "r_split", "cyl_a", dashed=True, extra="exitX=0.35;exitY=1;entryX=0.5;entryY=0;"),
        _mx_edge(
            "pe_sc",
            "r_split",
            "cyl_c",
            dashed=True,
            extra="exitX=0.1;exitY=1;entryX=0;entryY=0.5;",
            waypoints=[(232, 195), (80, 195), (80, 544)],
        ),

        # Branch B Flow
        _mx_edge("pb1", "cyl_b", "pre_b"),
        _mx_edge("pb2", "pre_b", "fe_b"),
        _mx_edge("pb3", "fe_b", "mod_b"),
        _mx_edge("pb4", "mod_b", "ev_b"),
        _mx_edge("pb5", "ev_b", "inf_b"),
        _mx_edge("pb6", "inf_b", "ana", extra="exitX=0;exitY=0.5;entryX=1;entryY=0.35;"),

        # Branch A Flow
        _mx_edge("pa1", "cyl_a", "pre_a"),
        _mx_edge("pa2", "pre_a", "cl_a"),
        _mx_edge("pa3", "cl_a", "mod_a"),
        _mx_edge("pa4", "mod_a", "ev_a"),
        _mx_edge("pa5", "ev_a", "ana", extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;"),

        # Branch C Flow
        _mx_edge("pc1", "cyl_c", "pre_c"),
        _mx_edge("pc2", "pre_c", "fe_c"),
        _mx_edge("pc3", "fe_c", "mod_c"),
        _mx_edge("pc4", "mod_c", "ev_c"),
        _mx_edge("pc5", "ev_c", "inf_c"),
        _mx_edge("pc6", "inf_c", "ana", extra="exitX=0.2;exitY=0;entryX=1;entryY=0.78;"),

        # Final Visualization
        _mx_edge("p_fin", "ana", "vis"),
    ]
    return _wrap_diagram("page4_flow_modifikasi", "Page-4 Flowchart Modifikasi", c, 1480, 640)


def _render_png_preview(title: str, subtitle: str, boxes: list[tuple], arrows: list[tuple], out_png: Path, figsize=(13.5, 7.5)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(2, 96, title, fontsize=12.5, fontweight="bold", color="#0B2E59", va="top")
    ax.text(2, 91.5, subtitle, fontsize=9, color="#4A5568", va="top")

    for x, y, w, h, txt, fc, ec, bold in boxes:
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.0, fontweight="bold" if bold else "normal", color="#111111")

    for item in arrows:
        if len(item) == 5:
            x1, y1, x2, y2, ls = item
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.4, linestyle=ls))
        else:
            pts, ls = item
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs[:-1], ys[:-1], color="#222222", lw=1.4, linestyle=ls)
            ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.4, linestyle=ls))

    plt.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_all_previews() -> None:
    # Preview Page 1
    _render_png_preview(
        "Page 1 — Architecture Jurnal Utama (Selayanti et al., 2024)",
        "Setiap node hanya menampilkan Module proses (48 SWK Surabaya, Agglomerative Hierarchical Clustering)",
        [
            (2, 48, 15, 28, "DATA SOURCE\n\nSatu Data Surabaya\n(48 SWK, 11 Var)", "#EBF8FF", "#1F4E79", True),
            (20, 20, 15, 64, "DATA STORAGE\n\n[Raw Data]\n[Processed Z-Score]\n[Output Cluster]", "#EDF2F7", "#1F4E79", True),
            (38, 52, 19, 32, "PRE-PROCESSING\n- Seleksi Variabel (4)\n- Missing Value (Mean)\n- Normalisasi Z-Score", "#FFF8E1", "#B26A00", False),
            (38, 18, 19, 28, "EDA\n- Statistika Deskriptif\n- Pelaku vs Omzet\n- Histogram & Kapasitas", "#E8F5E9", "#2E7D32", False),
            (60, 52, 18, 32, "MODELING (AHC)\n- Single Linkage\n- Average Linkage\n- Complete Linkage\n- Ward Linkage", "#F3E5F5", "#6A1B9A", False),
            (60, 18, 18, 28, "EVALUATION\n- Korelasi Cophenetic\n- Silhouette (k=2..6)", "#FCE4EC", "#AD1457", False),
            (81, 35, 17, 40, "ANALYSIS & VIS\n- Dendrogram\n- Profil 3 Cluster\n- Strategi Omzet SWK", "#E0F2F1", "#00695C", True),
        ],
        [
            (17, 62, 20, 62, "-"), (35, 68, 38, 68, "-"), (35, 32, 38, 32, "-"),
            (57, 68, 60, 68, "-"), (69, 52, 69, 46, "-"), (78, 32, 81, 45, "-"),
        ],
        DIAG_DIR / "Page1_Architecture_Jurnal.png",
    )

    # Preview Page 2
    _render_png_preview(
        "Page 2 — Flowchart Jurnal Utama (Selaras Notebook Repeat/)",
        "Alur linear + validasi optimalitas AHC Complete Linkage k=3 pada 48 SWK Surabaya",
        [
            (2, 68, 10, 14, "Start", "#E6FFFA", "#00695C", True),
            (15, 66, 15, 18, "Dataset SWK\n(48 Baris, 11 Var)", "#EDF2F7", "#1A1A1A", True),
            (33, 66, 17, 18, "EDA Deskriptif\n(Gambar 2, 3, 4)", "#FFFFFF", "#1A1A1A", False),
            (53, 66, 16, 18, "Seleksi 4 Fitur\nUtama SWK", "#FFFFFF", "#1A1A1A", False),
            (72, 66, 12, 18, "Missing?\nYa->Mean", "#FFFBEB", "#B7791F", True),
            (87, 66, 11, 18, "Z-Score\nScaling", "#FFFFFF", "#1A1A1A", False),
            (76, 38, 22, 18, "AHC (Single, Average,\nComplete, Ward Linkage)", "#F3E5F5", "#6A1B9A", True),
            (50, 38, 22, 18, "Evaluasi Cophenetic &\nSilhouette (k=2..6)", "#FCE4EC", "#AD1457", False),
            (25, 38, 20, 18, "Optimal?\nComplete k=3", "#FFFBEB", "#B7791F", True),
            (25, 10, 26, 18, "Dendrogram & Profil\nRata-Rata 3 Cluster", "#E0F2F1", "#00695C", True),
            (56, 10, 24, 18, "Visualisasi & Strategi\nOptimalisasi Omzet", "#E8F5E9", "#1B5E20", False),
            (85, 12, 11, 14, "End", "#E6FFFA", "#00695C", True),
        ],
        [
            (12, 75, 15, 75, "-"), (30, 75, 33, 75, "-"), (50, 75, 53, 75, "-"),
            (69, 75, 72, 75, "-"), (84, 75, 87, 75, "-"), (92, 66, 92, 56, "-"),
            (76, 47, 72, 47, "-"), (50, 47, 45, 47, "-"), (35, 38, 35, 28, "-"),
            (51, 19, 56, 19, "-"), (80, 19, 85, 19, "-"),
        ],
        DIAG_DIR / "Page2_Flowchart_Jurnal.png",
    )

    # Preview Page 3
    _render_png_preview(
        "Page 3 — Architecture Modifikasi Mahasiswa (Module Only, 3 Modalitas)",
        "Integrasi Dataset A (Tabular/Teks), Dataset B (Gambar Indonesia), & Dataset C (Dokumen MIDD + CORD-v2)",
        [
            (2, 22, 16, 62, "DATA SOURCES\n\nA: 48 SWK + 385 UMKM\n+ 1.164 Ulasan\n\nB: 549 Foto UMKM +\n993 Padang Cuisine\n\nC: MIDD 708 + CORD 200", "#EBF8FF", "#1F4E79", True),
            (20, 22, 13, 62, "INGESTION &\nSTAGING\n\n- Akuisisi\n- Batch Join\n- Schema", "#EDF2F7", "#1F4E79", False),
            (35, 22, 13, 62, "DATA STORAGE\n\n- Raw Zone\n- Clean Zone\n- Feature Store\n- Model Registry", "#EDF2F7", "#1F4E79", True),
            (50, 62, 27, 22, "PRE-PROCESSING\nImputasi | Sensor PII | Filter Visual &\nBlur Wajah | Parsing IOB, JSON, PDF", "#FFF8E1", "#B26A00", False),
            (50, 40, 27, 19, "EDA & FEATURE ENGINEERING\nEDA A/B/C | Z-Score | Ekstraksi Visual,\nTeks, Dokumen | Fusi & Reduksi Dimensi", "#EDE7F6", "#4527A0", False),
            (50, 14, 27, 23, "MODELING (3 CABANG)\nA: AHC, K-Means, Autoencoder\nB: RF, SVM, GB, MLP + Regressor\nC: Layout MIDD & Bayar CORD", "#F3E5F5", "#6A1B9A", True),
            (80, 56, 18, 28, "EVALUATION\n- Cophenetic/Sil\n- GroupKFold F1/Kappa\n- RMSE/R2\n- StratifiedKFold Doc", "#FCE4EC", "#AD1457", False),
            (80, 14, 18, 38, "BDA, VIS & APP\n- Digital Gap 6 Kec.\n- Peta & Spider Chart\n- Repeat vs Modifikasi\n- API & Dashboard", "#E8F5E9", "#1B5E20", True),
        ],
        [
            (18, 53, 20, 53, "-"), (33, 53, 35, 53, "-"), (48, 72, 50, 72, "-"),
            (48, 50, 50, 50, "-"), (63, 40, 63, 37, "-"), (77, 26, 80, 64, "-"), (89, 56, 89, 52, "-"),
        ],
        DIAG_DIR / "Page3_Architecture_Modifikasi.png",
    )

    # Preview Page 4
    _render_png_preview(
        "Page 4 — Flowchart Modifikasi Bercabang (Dataset A, B, C — Pola Figure (d))",
        "Tiga cabang paralel sesuai karakteristik modalitas bertemu pada tahap Analysis & Visualization",
        [
            (2, 70, 11, 16, "Multi-Source\nDataset", "#EDF2F7", "#1A1A1A", True),
            (16, 71, 13, 14, "Data Splitting\n/ Routing", "#FFFFFF", "#1A1A1A", True),
            # Branch B
            (33, 70, 12, 16, "Dataset B\n(549 UMKM +\n993 Food ID)", "#F0FFF4", "#2F855A", True),
            (48, 71, 12, 14, "Pre-proc &\nEDA B", "#F0FFF4", "#2F855A", False),
            (63, 71, 12, 14, "Feature\nExtract B", "#F0FFF4", "#2F855A", False),
            (78, 71, 10, 14, "Model B\n(ML/DL)", "#F0FFF4", "#2F855A", False),
            (90, 71, 9, 14, "Eval B", "#F0FFF4", "#2F855A", False),
            (88, 44, 11, 14, "Inferences\nB", "#F0FFF4", "#2F855A", False),
            # Branch A
            (10, 44, 13, 16, "Dataset A\n(SWK 48 +\nUMKM 385)", "#EBF8FF", "#2B6CB0", True),
            (10, 24, 13, 14, "Pre-proc &\nEDA A", "#EBF8FF", "#2B6CB0", False),
            (27, 24, 13, 14, "Clustering A\n(AHC/KM)", "#EBF8FF", "#2B6CB0", False),
            (44, 24, 13, 14, "Modeling A\n(Profil)", "#EBF8FF", "#2B6CB0", False),
            (44, 44, 13, 14, "Eval A\n(Coph/Sil)", "#EBF8FF", "#2B6CB0", False),
            # Branch C
            (10, 4, 13, 14, "Dataset C\n(MIDD+CORD)", "#FFFBEB", "#B7791F", True),
            (27, 4, 13, 14, "Pre-proc &\nEDA C", "#FFFBEB", "#B7791F", False),
            (44, 4, 13, 14, "Feature\nExtract C", "#FFFBEB", "#B7791F", False),
            (61, 4, 13, 14, "Model C\n(Layout/Pay)", "#FFFBEB", "#B7791F", False),
            (78, 4, 9, 14, "Eval C", "#FFFBEB", "#B7791F", False),
            (90, 4, 9, 14, "Infer C", "#FFFBEB", "#B7791F", False),
            # Convergence
            (64, 43, 18, 16, "Analysis\n(Spatial BDA &\nRepeat vs Modif)", "#E6FFFA", "#00695C", True),
            (64, 23, 18, 15, "Visualization\n(Peta, Spider,\nDendrogram)", "#E6FFFA", "#00695C", True),
        ],
        [
            (13, 78, 16, 78, "-"),
            (29, 78, 33, 78, "--"), (22, 71, 16, 60, "--"), ([(18, 71), (18, 65), (4.5, 65), (4.5, 11), (10, 11)], "--"),
            (45, 78, 48, 78, "-"), (60, 78, 63, 78, "-"), (75, 78, 78, 78, "-"), (88, 78, 90, 78, "-"),
            (94, 71, 94, 58, "-"), (88, 51, 82, 51, "-"),
            (16, 44, 16, 38, "-"), (23, 31, 27, 31, "-"), (40, 31, 44, 31, "-"), (50, 38, 50, 44, "-"), (57, 51, 64, 51, "-"),
            (23, 11, 27, 11, "-"), (40, 11, 44, 11, "-"), (57, 11, 61, 11, "-"), (74, 11, 78, 11, "-"), (87, 11, 90, 11, "-"),
            (94, 18, 82, 43, "-"), (73, 43, 73, 38, "-"),
        ],
        DIAG_DIR / "Page4_Flowchart_Modifikasi.png",
    )


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    mxfile = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-11-25T12:00:00.000Z" agent="Antigravity" version="24.7.0" pages="4">\n'
        + build_page1_arch_jurnal() + "\n"
        + build_page2_flow_jurnal() + "\n"
        + build_page3_arch_modifikasi() + "\n"
        + build_page4_flow_modifikasi() + "\n"
        + "</mxfile>\n"
    )
    OUT_DRAWIO.write_text(mxfile, encoding="utf-8")
    render_all_previews()
    print(f"Generated 4-page draw.io XML: {OUT_DRAWIO}")
    print("Generated 4 PNG previews in:", DIAG_DIR)


if __name__ == "__main__":
    main()
