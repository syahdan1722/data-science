import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan fitur yang digunakan saat training
model = pickle.load(open('model_rf.pkl', 'rb'))
feature_names = pickle.load(open('model_features.pkl', 'rb'))

st.set_page_config(page_title="Attrition Prediction", layout="wide")
st.title("💼 Prediksi Karyawan Keluar (Attrition)")
st.write("Masukkan data berikut untuk memprediksi apakah seorang karyawan akan keluar atau tidak.")

def user_input_features():
    Age = st.slider('Umur', 18, 60, 30)
    DistanceFromHome = st.slider('Jarak dari Rumah ke Kantor (km)', 1, 30, 10)
    MonthlyIncome = st.slider('Pendapatan Bulanan', 1000, 20000, 5000)
    OverTime = st.selectbox('Lembur?', ('Yes', 'No'))
    JobSatisfaction = st.selectbox('Kepuasan Kerja (1=sangat rendah, 4=sangat tinggi)', [1, 2, 3, 4])
    TotalWorkingYears = st.slider('Total Tahun Bekerja', 0, 40, 10)
    YearsAtCompany = st.slider('Tahun di Perusahaan Saat Ini', 0, 40, 5)

    # Fitur input yang tersedia dari user
    data = {
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'MonthlyIncome': MonthlyIncome,
        'OverTime_Yes': 1 if OverTime == 'Yes' else 0,
        'JobSatisfaction': JobSatisfaction,
        'TotalWorkingYears': TotalWorkingYears,
        'YearsAtCompany': YearsAtCompany
    }

    return pd.DataFrame([data])

# Ambil input pengguna
input_df = user_input_features()

# Siapkan template dataframe sesuai fitur model
template = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Isi data user ke template jika cocok
for col in input_df.columns:
    if col in template.columns:
        template[col] = input_df[col].values[0]

# Tampilkan input
st.subheader("Data yang Dimasukkan:")
st.write(input_df)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(template)
    if prediction[0] == 1:
        st.error("❌ Karyawan Diprediksi Akan KELUAR")
    else:
        st.success("✅ Karyawan Diprediksi TETAP")