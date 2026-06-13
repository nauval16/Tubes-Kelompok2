import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

DATASET_PATH = "student_mental_health_burnout_clean.csv"

CLASSIFICATION_FEATURES = [
    "stress_level",
    "sleep_quality",
    "internet_quality",
    "daily_study_hours",
    "year"
]

REGRESSION_FEATURES = [
    "sleep_quality",
    "daily_study_hours",
    "daily_sleep_hours",
    "stress_level",
    "course_freq_encode"
]

CLASSIFICATION_LABEL = "burnout_level"
REGRESSION_LABEL = "cgpa"


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_model_files(folder):
    joblib_files = glob.glob(os.path.join(folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
    return joblib_files + pkl_files


def predict_burnout_label(pred):
    if pred == 0:
        return "Low Burnout / Rendah"
    elif pred == 1:
        return "Moderate Burnout / Sedang"
    elif pred == 2:
        return "High Burnout / Tinggi"
    else:
        return str(pred)


def burnout_interpretation(pred):
    if pred == 0:
        return "Risiko burnout rendah. Kondisi mahasiswa relatif stabil."
    elif pred == 1:
        return "Risiko burnout sedang. Mahasiswa perlu menjaga pola tidur, belajar, dan istirahat."
    elif pred == 2:
        return "Risiko burnout tinggi. Mahasiswa disarankan memperbaiki manajemen waktu dan mencari dukungan."
    else:
        return "Hasil tidak dikenali."


def cgpa_interpretation(value):
    if value >= 8:
        return "Prediksi CGPA tergolong sangat baik."
    elif value >= 6:
        return "Prediksi CGPA tergolong cukup baik."
    else:
        return "Prediksi CGPA tergolong rendah dan perlu ditingkatkan."


def scale_input(input_df, prediction_type):
    df = load_dataset()

    if prediction_type == "Klasifikasi":
        scaler = StandardScaler()
        scaler.fit(df[CLASSIFICATION_FEATURES])
        scaled = scaler.transform(input_df[CLASSIFICATION_FEATURES])
        return pd.DataFrame(scaled, columns=CLASSIFICATION_FEATURES)
    else:
        scaler = MinMaxScaler(feature_range=(0, 10))
        scaler.fit(df[REGRESSION_FEATURES])
        scaled = scaler.transform(input_df[REGRESSION_FEATURES])
        return pd.DataFrame(scaled, columns=REGRESSION_FEATURES)


def show_header():
    st.markdown(
        """
        <div style="background-color:#EAF4FF; padding:28px; border-radius:18px; margin-bottom:22px;">
            <h1 style="color:#1F4E79; margin-bottom:8px;">🧠 Student Mental Health & Burnout Prediction</h1>
            <p style="font-size:17px; color:#333;">
                Aplikasi machine learning berbasis web untuk memprediksi tingkat burnout dan nilai CGPA mahasiswa.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def load_selected_model(folder):
    model_files = get_model_files(folder)

    if len(model_files) == 0:
        st.error(f"Belum ada model di folder `{folder}`.")
        st.stop()

    model_names = [os.path.basename(file) for file in model_files]

    selected_model_name = st.selectbox("Pilih Model Machine Learning", model_names)
    selected_model_path = model_files[model_names.index(selected_model_name)]

    model = load_model(selected_model_path)
    return model, selected_model_name


show_header()

menu = st.sidebar.radio(
    "Menu Aplikasi",
    [
        "Home",
        "Prediksi Mahasiswa",
        "Evaluasi Model",
        "Visualisasi Data",
        "Tentang Dataset"
    ]
)

df = load_dataset()


if menu == "Home":
    st.subheader("📌 Deskripsi Studi Kasus")

    st.write(
        """
        Studi kasus ini menggunakan dataset Student Mental Health and Burnout.
        Aplikasi dibuat untuk membantu mahasiswa memperkirakan tingkat burnout dan nilai CGPA
        berdasarkan data kondisi akademik, kebiasaan belajar, kualitas tidur, tingkat stres,
        dan faktor pendukung lainnya.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Data", f"{df.shape[0]}")
    col2.metric("Jumlah Kolom", f"{df.shape[1]}")
    col3.metric("Jenis Studi Kasus", "Klasifikasi & Regresi")

    st.subheader("🎯 Tujuan Aplikasi")
    st.write(
        """
        1. Membangun aplikasi machine learning berbasis web.  
        2. Menyediakan fitur input data oleh pengguna.  
        3. Menampilkan hasil prediksi burnout level dan CGPA.  
        4. Menampilkan informasi model, evaluasi model, dan visualisasi sederhana.
        """
    )


elif menu == "Prediksi Mahasiswa":
    st.subheader("📝 Prediksi Berdasarkan Input Mahasiswa")

    prediction_choice = st.radio(
        "Pilih Jenis Prediksi",
        ["Klasifikasi Burnout Level", "Regresi CGPA"],
        horizontal=True
    )

    if prediction_choice == "Klasifikasi Burnout Level":
        mode = "Klasifikasi"
        model, selected_model_name = load_selected_model("model_klasifikasi")
        feature_names = CLASSIFICATION_FEATURES
    else:
        mode = "Regresi"
        model, selected_model_name = load_selected_model("model_regresi")
        feature_names = REGRESSION_FEATURES

    st.info(f"Model yang digunakan: {selected_model_name}")

    input_data = {}

    if mode == "Klasifikasi":
        col1, col2 = st.columns(2)

        with col1:
            input_data["stress_level"] = st.slider("Stress Level", 1, 10, 5)
            input_data["sleep_quality"] = st.slider("Sleep Quality", 1, 10, 5)
            input_data["internet_quality"] = st.slider("Internet Quality", 1, 10, 5)

        with col2:
            input_data["daily_study_hours"] = st.number_input(
                "Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5
            )
            input_data["year"] = st.selectbox("Tahun Kuliah", [1, 2, 3, 4])

    else:
        col1, col2 = st.columns(2)

        with col1:
            input_data["sleep_quality"] = st.slider("Sleep Quality", 1, 10, 5)
            input_data["daily_study_hours"] = st.number_input(
                "Daily Study Hours", min_value=0.0, max_value=24.0, value=3.0, step=0.5
            )
            input_data["daily_sleep_hours"] = st.number_input(
                "Daily Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5
            )

        with col2:
            input_data["stress_level"] = st.slider("Stress Level", 1, 10, 5)
            input_data["course_freq_encode"] = st.number_input(
                "Course Frequency Encode", min_value=0.0, value=1.0, step=0.1
            )

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    st.subheader("📄 Data Input")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🔍 Prediksi Sekarang"):
        try:
            scaled_input = scale_input(input_df, mode)
            prediction = model.predict(scaled_input)[0]

            st.subheader("🎯 Hasil Prediksi")

            if mode == "Klasifikasi":
                label = predict_burnout_label(prediction)

                if prediction == 0:
                    st.success(f"Hasil Prediksi: {label}")
                elif prediction == 1:
                    st.warning(f"Hasil Prediksi: {label}")
                else:
                    st.error(f"Hasil Prediksi: {label}")

                st.write(burnout_interpretation(prediction))

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(scaled_input)[0]
                    proba_df = pd.DataFrame({
                        "Kategori": [predict_burnout_label(cls) for cls in model.classes_],
                        "Probabilitas": proba
                    })
                    st.subheader("📊 Probabilitas Prediksi")
                    st.dataframe(proba_df, use_container_width=True)

            else:
                st.success(f"Prediksi Nilai CGPA: {prediction:.2f}")
                st.write(cgpa_interpretation(prediction))

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


elif menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Model")

    eval_choice = st.radio(
        "Pilih Evaluasi",
        ["Klasifikasi", "Regresi"],
        horizontal=True
    )

    if eval_choice == "Klasifikasi":
        model, selected_model_name = load_selected_model("model_klasifikasi")

        X = df[CLASSIFICATION_FEATURES]
        y_true = df[CLASSIFICATION_LABEL]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=CLASSIFICATION_FEATURES
        )

        y_pred = model.predict(X_scaled)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

    else:
        model, selected_model_name = load_selected_model("model_regresi")

        X = df[REGRESSION_FEATURES]
        y_true = df[REGRESSION_LABEL]

        scaler = MinMaxScaler(feature_range=(0, 10))
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=REGRESSION_FEATURES
        )

        y_pred = model.predict(X_scaled)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("R²", f"{r2:.4f}")

        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.4)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle="--")
        ax.set_xlabel("Actual CGPA")
        ax.set_ylabel("Predicted CGPA")
        ax.set_title("Actual vs Predicted CGPA")
        st.pyplot(fig)

        st.info(
            "Jika nilai R² rendah atau negatif, artinya model belum mampu menjelaskan variasi target dengan baik. "
            "Hal ini dapat dijadikan analisis dalam laporan."
        )


elif menu == "Visualisasi Data":
    st.subheader("📊 Visualisasi Sederhana Dataset")

    chart_choice = st.selectbox(
        "Pilih Visualisasi",
        [
            "Distribusi Burnout Level",
            "Distribusi CGPA",
            "Rata-rata CGPA berdasarkan Burnout Level",
            "Distribusi Stress Level"
        ]
    )

    if chart_choice == "Distribusi Burnout Level":
        counts = df["burnout_level"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Burnout Level")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Burnout Level")
        st.pyplot(fig)

    elif chart_choice == "Distribusi CGPA":
        fig, ax = plt.subplots()
        ax.hist(df["cgpa"], bins=20)
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi CGPA")
        st.pyplot(fig)

    elif chart_choice == "Rata-rata CGPA berdasarkan Burnout Level":
        avg_cgpa = df.groupby("burnout_level")["cgpa"].mean()

        fig, ax = plt.subplots()
        ax.bar(avg_cgpa.index.astype(str), avg_cgpa.values)
        ax.set_xlabel("Burnout Level")
        ax.set_ylabel("Rata-rata CGPA")
        ax.set_title("Rata-rata CGPA berdasarkan Burnout Level")
        st.pyplot(fig)

    else:
        counts = df["stress_level"].value_counts().sort_index()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Stress Level")
        st.pyplot(fig)


elif menu == "Tentang Dataset":
    st.subheader("📚 Tentang Dataset")

    st.write(
        """
        Dataset yang digunakan adalah Student Mental Health and Burnout Dataset.
        Dataset ini berisi data mahasiswa yang berkaitan dengan kondisi akademik,
        kebiasaan belajar, kualitas tidur, tingkat stres, burnout level, dan CGPA.
        """
    )

    st.write("Preview dataset:")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Informasi kolom:")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": [str(dtype) for dtype in df.dtypes]
    })
    st.dataframe(info_df, use_container_width=True)

    with st.expander("Penjelasan fitur utama"):
        st.write(
            """
            - stress_level: tingkat stres mahasiswa  
            - sleep_quality: kualitas tidur mahasiswa  
            - internet_quality: kualitas koneksi internet  
            - daily_study_hours: jumlah jam belajar per hari  
            - daily_sleep_hours: jumlah jam tidur per hari  
            - year: tahun kuliah mahasiswa  
            - burnout_level: label klasifikasi tingkat burnout  
            - cgpa: target regresi nilai akademik mahasiswa  
            """
        )