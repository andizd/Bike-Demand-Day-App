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
    menggunakan **K-Means (K = 2)** dan **Logistic Regression**.
    
    Output:
    - 🟢 Low Demand Day
    - 🔴 High Demand Day
    """
)

# =========================
# LOAD MODEL & FEATURE
# =========================
@st.cache_resource
def load_artifacts():
    logreg = joblib.load("logreg_cluster_k2.pkl")
    scaler = joblib.load("scaler_k2.pkl")
    features = joblib.load("features_k2.pkl")
    return logreg, scaler, features

logreg, scaler, features = load_artifacts()

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
    temp = st.number_input("Suhu (temp)", min_value=0.0, max_value=1.0, value=0.3)
    atemp = st.number_input("Suhu Terasa (atemp)", min_value=0.0, max_value=1.0, value=0.3)
    hum = st.number_input("Kelembapan (hum)", min_value=0.0, max_value=1.0, value=0.5)
    windspeed = st.number_input("Kecepatan Angin (windspeed)", min_value=0.0, max_value=1.0, value=0.2)

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

# Pastikan urutan fitur SESUAI model
input_df = input_df[features]

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi Permintaan"):
    X_scaled = scaler.transform(input_df)
    prediction = logreg.predict(X_scaled)[0]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 0:
        st.success("🟢 **Low Demand Day**\n\nPermintaan penyewaan sepeda relatif rendah.")
    else:
        st.error("🔴 **High Demand Day**\n\nPermintaan penyewaan sepeda relatif tinggi.")

    st.markdown("### 🔎 Ringkasan Input")
    st.dataframe(input_df)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(
    "Model: K-Means (K=2) + Logistic Regression | "
    "Dataset: Bike Sharing (day.csv)"
)