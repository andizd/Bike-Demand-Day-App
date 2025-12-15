import streamlit as st
import pandas as pd
import joblib

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    layout="centered"
)

st.title("🚲 Prediksi Permintaan Sepeda")
st.write(
    """
    Aplikasi ini memprediksi **kategori permintaan penyewaan sepeda harian**
    menggunakan:
    - **K-Means (K = 2)** untuk clustering
    - **Logistic Regression** untuk prediksi cluster
    
    Output:
    - 🟢 Low Demand Day  
    - 🔴 High Demand Day
    """
)

# =========================
# LOAD MODEL & ARTIFACT
# =========================
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_bike_k2.pkl")
    logreg = joblib.load("logreg_cluster_predictor.pkl")
    scaler = joblib.load("scaler_bike.pkl")
    features = joblib.load("clustering_features.pkl")
    return kmeans, logreg, scaler, features

kmeans, logreg, scaler, features = load_artifacts()

# =========================
# DROPDOWN MAPPING
# =========================
holiday_map = {
    "Bukan Libur": 0,
    "Libur Nasional": 1
}

season_map = {
    "Spring": 1,
    "Summer": 2,
    "Fall": 3,
    "Winter": 4
}

weathersit_map = {
    "Cerah": 1,
    "Berawan / Berkabut": 2,
    "Hujan ringan / Salju ringan": 3,
    "Cuaca ekstrem": 4
}

# =========================
# INPUT DATA MANUAL
# =========================
st.subheader("📥 Input Data")

col1, col2 = st.columns(2)

with col1:
    season_label = st.selectbox("Musim", list(season_map.keys()))
    holiday_label = st.selectbox("Hari Libur", list(holiday_map.keys()))
    weathersit_label = st.selectbox("Kondisi Cuaca", list(weathersit_map.keys()))

with col2:
    temp = st.number_input("Suhu (temp)", 0.0, 1.0, 0.3)
    atemp = st.number_input("Suhu Terasa (atemp)", 0.0, 1.0, 0.3)
    hum = st.number_input("Kelembapan (hum)", 0.0, 1.0, 0.5)
    windspeed = st.number_input("Kecepatan Angin (windspeed)", 0.0, 1.0, 0.2)

# =========================
# KONVERSI KE DATAFRAME
# =========================
input_df = pd.DataFrame([{
    "season": season_map[season_label],
    "holiday": holiday_map[holiday_label],
    "weathersit": weathersit_map[weathersit_label],
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed
}])

# Pastikan urutan fitur SAMA dengan model
input_df = input_df[features]

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Permintaan"):
    X_scaled = scaler.transform(input_df)

    # Prediksi cluster (Logistic Regression)
    cluster_pred = logreg.predict(X_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if cluster_pred == 0:
        st.success("🟢 **Low Demand Day**\n\nPermintaan penyewaan sepeda relatif rendah.")
    else:
        st.error("🔴 **High Demand Day**\n\nPermintaan penyewaan sepeda relatif tinggi.")

    # (Opsional) cek kedekatan ke centroid
    centroid_pred = kmeans.predict(X_scaled)[0]

    st.markdown("### 🔎 Ringkasan Teknis")
    st.write(f"Cluster (Logistic Regression): **{cluster_pred}**")
    st.write(f"Cluster (K-Means Centroid): **{centroid_pred}**")

    st.markdown("### 📋 Data Input (Numerik)")
    st.dataframe(input_df)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Model: K-Means (K=2) + Logistic Regression | "
    "Dataset: Bike Sharing (day.csv)"
)