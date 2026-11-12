"""
Antarmuka Pengguna (Client UI) Peta Digital UMKM Kuliner Surabaya Timur.

Bagian: AI-Assisted (< 40% UI Layout) & Kelompok (Integrasi Data Spasial)
Referensi Dokumentasi Resmi:
- Streamlit API Reference: https://docs.streamlit.io/develop/api-reference
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.client.api_client import BackendClient


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Peta Digital UMKM Surabaya Timur",
        layout="wide",
    )

    client = BackendClient()
    st.title("Peta Digital Kematangan UMKM Kuliner Surabaya Timur")
    st.caption(
        "Arsitektur Client-Server Terintegrasi Big Data Spasial, Machine Learning (.pkl), dan Deep Learning (.pt/.pth/.h5)"
    )

    tab_bda, tab_pred, tab_dir, tab_models = st.tabs(
        [
            "1. Analitik Spasial (BDA)",
            "2. Simulasi Prediksi ML & DL",
            "3. Direktori UMKM",
            "4. Inventaris Model & Embedding",
        ]
    )

    with tab_bda:
        st.subheader("Indeks Kesenjangan Digital Kecamatan Surabaya Timur")
        bda_rows = client.fetch_bda_summary()
        if bda_rows:
            df_bda = pd.DataFrame(bda_rows)
            st.dataframe(df_bda, use_container_width=True)
            st.bar_chart(df_bda.set_index("kecamatan")[["avg_maturity_score", "digital_gap_index"]])
        else:
            st.info("Data agregasi BDA belum tersedia. Jalankan python -m model.run_all terlebih dahulu.")

    with tab_pred:
        st.subheader("Uji Prediksi 6 Model (ML & DL: Clustering, Classification, Regression)")
        col1, col2 = st.columns(2)
        with col1:
            nama_umkm = st.text_input("Nama UMKM", value="Warung Kuliner Rungkut")
            kecamatan = st.selectbox(
                "Kecamatan",
                [
                    "Sukolilo",
                    "Gubeng",
                    "Rungkut",
                    "Mulyorejo",
                    "Tenggilis Mejoyo",
                    "Gunung Anyar",
                ],
            )
            rating = st.slider("Rating Google Maps", 1.0, 5.0, 4.5, 0.1)
            jumlah_ulasan = st.number_input("Jumlah Ulasan", min_value=0, value=40)
        with col2:
            indikator_qris = int(st.checkbox("Tersedia QRIS", value=True))
            indikator_ojol = int(st.checkbox("Terdaftar GoFood/GrabFood/ShopeeFood", value=True))
            indikator_pos = int(st.checkbox("Menggunakan Kasir POS Digital", value=False))
            indikator_edc = int(st.checkbox("Tersedia Mesin EDC", value=False))

        if st.button("Jalankan Inferensi Model"):
            payload = {
                "nama_umkm": nama_umkm,
                "kecamatan": kecamatan,
                "rating": rating,
                "jumlah_ulasan": int(jumlah_ulasan),
                "indikator_qris": indikator_qris,
                "indikator_ojol": indikator_ojol,
                "indikator_pos": indikator_pos,
                "indikator_edc": indikator_edc,
            }
            res = client.predict_umkm(payload)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown("#### Hasil Machine Learning (`.pkl`)")
                st.metric("ML Clustering (K-Means)", f"Klaster {res['ml_clustering_cluster']}")
                st.metric("ML Classification (Random Forest)", res["ml_classification_label"])
                st.metric("ML Regression (Skor Kesiapan)", f"{res['ml_regression_score']} / 100")
            with m2:
                st.markdown("#### Hasil Deep Learning (`.pt / .pth / .h5`)")
                st.metric("DL Clustering (Autoencoder)", f"Klaster {res['dl_clustering_cluster']}")
                st.metric("DL Classification (MLP)", res["dl_classification_label"])
                st.metric("DL Regression (MLP Regressor)", f"{res['dl_regression_score']} / 100")
            st.success(f"Rekomendasi Intervensi: {res['rekomendasi']}")

    with tab_dir:
        st.subheader("Data UMKM Terkurasi (Staging PySpark & Clean Parquet)")
        kec_filter = st.selectbox(
            "Filter Kecamatan",
            ["Semua", "Sukolilo", "Gubeng", "Rungkut", "Mulyorejo", "Tenggilis Mejoyo", "Gunung Anyar"],
        )
        rows = client.fetch_umkm_list(kecamatan=kec_filter)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab_models:
        st.subheader("Artefak Model & Embedding Tersimpan (`model/saved/`)")
        health = client.fetch_health()
        st.write(f"Mode Koneksi: `{health.get('status')}` | Total File Tersimpan: `{health.get('saved_models_count')}`")
        if health.get("artifacts"):
            st.dataframe(pd.DataFrame(health["artifacts"]), use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
