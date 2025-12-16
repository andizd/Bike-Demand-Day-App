import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(
    page_title="Prediksi Permintaan Sepeda",
    page_icon="🚲",
    layout="centered"
)

st.title("🚲 Prediksi Tingkat Permintaan Sepeda")
st.write(
    "Aplikasi ini memprediksi **tingkat permintaan sepeda** "
    "berdasarkan **kondisi cuaca dan hari**, "
    "menggunakan **Machine Learning (Logistic Regression)**."
)

st.divider()

@st.cache_resource
def load_model():
    model = joblib.load("logreg_demand_model.joblib")
    scaler = joblib.load("logreg_scaler.joblib")
    features = joblib.load("logreg_features.joblib")
    return model, scaler, features

model, scaler, features = load_model()

cluster_name_map = {
    0: "Very Low Demand",
    1: "Low Demand",
    2: "Medium Demand",
    3: "High Demand"
}

cluster_description = {
    "Very Low Demand": {
        "desc": "Permintaan sepeda sangat rendah.",
        "insight": "Biasanya terjadi pada kondisi cuaca dingin atau kurang nyaman."
    },

    "Low Demand": {
        "desc": "Permintaan sepeda rendah hingga menengah.",
        "insight": "Permintaan mulai muncul namun belum optimal."
    },

    "Medium Demand": {
        "desc": "Permintaan sepeda berada pada tingkat normal.",
        "insight": "Ini adalah kondisi paling umum dalam dataset."
    },

    "High Demand": {
        "desc": "Permintaan sepeda sangat tinggi.",
        "insight": "Kondisi paling ideal untuk penggunaan sepeda."
    }
}

cluster_color_map = {
    "Very Low Demand": "🔵",
    "Low Demand": "🟢",
    "Medium Demand": "🟠",
    "High Demand": "🔴"
}

st.subheader("🔧 Masukkan Kondisi Lingkungan")

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Musim",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }[x]
    )

    holiday = st.selectbox(
        "Hari Libur",
        options=[0, 1],
        format_func=lambda x: "Libur" if x == 1 else "Bukan Libur"
    )

    weathersit = st.selectbox(
        "Kondisi Cuaca",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Cerah / Berawan",
            2: "Kabut / Mendung",
            3: "Hujan / Salju"
        }[x]
    )

with col2:
    temp = st.slider("Suhu (normalized)", 0.0, 1.0, 0.5)
    atemp = st.slider("Suhu Terasa (normalized)", 0.0, 1.0, 0.5)
    hum = st.slider("Kelembapan (normalized)", 0.0, 1.0, 0.6)
    windspeed = st.slider("Kecepatan Angin (normalized)", 0.0, 1.0, 0.3)

season_text_map = {
    1: "Spring",
    2: "Summer",
    3: "Fall",
    4: "Winter"
}

weather_text_map = {
    1: "Cerah / Berawan",
    2: "Kabut / Mendung",
    3: "Hujan / Salju"
}

season_text = season_text_map[season]
weather_text = weather_text_map[weathersit]
holiday_text = "Libur" if holiday == 1 else "Bukan Libur"

if st.button("🔍 Prediksi Permintaan"):
    input_data = pd.DataFrame([[
        season,
        holiday,
        weathersit,
        temp,
        atemp,
        hum,
        windspeed
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    demand_label = cluster_name_map[prediction]
    demand_icon = cluster_color_map[demand_label]

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    st.markdown("### 📌 Kondisi yang Anda masukkan")

    st.markdown(f"""
    - 🗓️ **Musim**: {season_text}  
    - 🎉 **Hari**: {holiday_text}  
    - 🌦️ **Cuaca**: {weather_text}  
    - 🌡️ **Suhu**: {temp:.2f}  
    - 💧 **Kelembapan**: {hum:.2f}  
    - 💨 **Kecepatan Angin**: {windspeed:.2f}  
    """)

    info = cluster_description[demand_label]

    st.markdown(f"**Deskripsi:** {info['desc']}")
    st.info(f"💡 **Insight:** {info['insight']}")


st.divider()
st.caption(
    "Model: Logistic Regression | Dataset: Bike Sharing | "
    "Metode: Clustering + Classification"
)