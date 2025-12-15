import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    layout="centered"
)

st.title("🚲 Bike Sharing Demand Prediction")
st.write(
    """
    Aplikasi ini memprediksi **kategori permintaan penyewaan sepeda**
    (*Low Demand* atau *High Demand*)  
    menggunakan **model K-Means (K=2)** dan **Logistic Regression**
    hasil pelatihan di Google Colab.
    """
)

# =========================
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_bike_k2.pkl")
    scaler = joblib.load("scaler_bike.pkl")
    logreg = joblib.load("logreg_cluster_predictor.pkl")
    features = joblib.load("clustering_features.pkl")
    return kmeans, scaler, logreg, features

kmeans, scaler, logreg, features = load_models()

# =========================
# INPUT DATA
# =========================
st.subheader("📥 Input Data")

input_method = st.radio(
    "Pilih metode input:",
    ["Input Manual", "Upload CSV"]
)

# -------------------------
# INPUT MANUAL
# -------------------------
if input_method == "Input Manual":
    input_data = {}

    for feature in features:
        input_data[feature] = st.number_input(
            f"Masukkan nilai {feature}",
            value=0.0
        )

    input_df = pd.DataFrame([input_data])

# -------------------------
# UPLOAD CSV
# -------------------------
else:
    uploaded_file = st.file_uploader(
        "Upload file CSV (harus mengandung kolom fitur yang sesuai)",
        type=["csv"]
    )

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        input_df = input_df[features]
        st.dataframe(input_df.head())
    else:
        input_df = None

# =========================
# PREDIKSI
# =========================
if input_df is not None and st.button("🔍 Prediksi Demand"):
    # Scaling
    X_scaled = scaler.transform(input_df)

    # Prediksi cluster
    cluster_pred = logreg.predict(X_scaled)

    # Mapping cluster ke label
    demand_label = [
        "🟢 Low Demand Day" if c == 0 else "🔴 High Demand Day"
        for c in cluster_pred
    ]

    input_df["Predicted Cluster"] = cluster_pred
    input_df["Demand Category"] = demand_label

    st.subheader("📊 Hasil Prediksi")
    st.dataframe(input_df)

    # =========================
    # VISUALISASI PCA
    # =========================
    st.subheader("📈 Visualisasi PCA (Referensi Cluster)")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    centroids_pca = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_pred,
        alpha=0.7,
        label="Data"
    )

    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroid"
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Prediksi Cluster dengan PCA")
    ax.legend()

    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Model: K-Means (K=2) + Logistic Regression | "
    "Dataset: Bike Sharing (day.csv)"
)