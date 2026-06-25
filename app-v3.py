import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import warnings
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

# =========================
# KONFIGURASI DATASET & FITUR
# =========================

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

COURSE_MAPPING = {
    "BBA": 0.167353,
    "BCA": 0.166487,
    "BSc": 0.165960,
    "BTech": 0.165660,
    "MBA": 0.168207,
    "MCA": 0.166333
}


# =========================
# LOAD DATA DAN MODEL
# =========================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


def get_model_files(folder):
    joblib_files = glob.glob(os.path.join(folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
    return sorted(joblib_files + pkl_files)


@st.cache_resource(show_spinner="Memuat semua model, harap tunggu...")
def preload_all_models():
    """Load semua model sekali saat app start, simpan di cache RAM."""
    models = {}
    for folder in ["model_klasifikasi", "model_regresi"]:
        files = get_model_files(folder)
        for path in files:
            name = os.path.basename(path)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Gagal load model `{name}`: {e}")
    return models


def load_selected_model(folder, key="model_selector"):
    model_files = get_model_files(folder)

    if len(model_files) == 0:
        st.error(f"Belum ada model di folder `{folder}`.")
        st.stop()

    model_names = [os.path.basename(f) for f in model_files]
    selected_model_name = st.selectbox(
        "Pilih Model Machine Learning",
        model_names,
        key=key
    )

    # Ambil dari preloaded cache, bukan load ulang dari disk
    model = ALL_MODELS.get(selected_model_name)
    if model is None:
        st.error(f"Model `{selected_model_name}` tidak ditemukan di cache.")
        st.stop()

    return model, selected_model_name


# =========================
# FUNCTION BANTUAN
# =========================

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


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def validate_columns(data, feature_names):
    return [col for col in feature_names if col not in data.columns]


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


# =========================
# PRELOAD SEMUA MODEL (sekali saat startup)
# =========================

ALL_MODELS = preload_all_models()

# =========================
# HALAMAN UTAMA
# =========================

show_header()

menu = st.sidebar.radio(
    "Menu Aplikasi",
    [
        "Home",
        "Prediksi Mahasiswa",
        "Prediksi Dataset",
        "Evaluasi Model",
        "Visualisasi Data",
        "Tentang Dataset",
        "Biodata Kelompok"
    ]
)

df = load_dataset()


# =========================
# HOME
# =========================

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
        5. Menyediakan fitur prediksi satu data mahasiswa dan prediksi banyak data melalui upload dataset.
        """
    )

    st.subheader("🤖 Status Model yang Dimuat")
    if ALL_MODELS:
        loaded_names = list(ALL_MODELS.keys())
        status_df = pd.DataFrame({
            "Nama Model": loaded_names,
            "Tipe": [type(ALL_MODELS[n]).__name__ for n in loaded_names],
            "Status": ["✅ Loaded" for _ in loaded_names]
        })
        st.dataframe(status_df, use_container_width=True)
    else:
        st.error("Tidak ada model yang berhasil di-load.")




# =========================
# PREDIKSI MAHASISWA MANUAL
# =========================

elif menu == "Prediksi Mahasiswa":
    st.subheader("📝 Prediksi Berdasarkan Input Mahasiswa")

    prediction_choice = st.radio(
        "Pilih Jenis Prediksi",
        ["Klasifikasi Burnout Level", "Regresi CGPA"],
        horizontal=True
    )

    if prediction_choice == "Klasifikasi Burnout Level":
        mode = "Klasifikasi"
        model, selected_model_name = load_selected_model(
            "model_klasifikasi",
            key="manual_klasifikasi"
        )
        feature_names = CLASSIFICATION_FEATURES
    else:
        mode = "Regresi"
        model, selected_model_name = load_selected_model(
            "model_regresi",
            key="manual_regresi"
        )
        feature_names = REGRESSION_FEATURES

    st.info(f"Model yang digunakan: **{selected_model_name}** — `{type(model).__name__}`")

    input_data = {}

    if mode == "Klasifikasi":
        col1, col2 = st.columns(2)

        with col1:
            input_data["stress_level"] = st.slider(
                "Stress Level (1 = Rendah, 10 = Tinggi)",
                min_value=1, max_value=10, value=5,
                help="Semakin tinggi nilai, semakin tinggi tingkat stres mahasiswa."
            )
            input_data["sleep_quality"] = st.slider(
                "Sleep Quality (1 = Buruk, 10 = Baik)",
                min_value=1, max_value=10, value=5,
                help="Semakin tinggi nilai, semakin baik kualitas tidur mahasiswa."
            )
            input_data["internet_quality"] = st.slider(
                "Internet Quality (1 = Buruk, 10 = Baik)",
                min_value=1, max_value=10, value=5,
                help="Semakin tinggi nilai, semakin baik kualitas koneksi internet mahasiswa."
            )

        with col2:
            input_data["daily_study_hours"] = st.number_input(
                "Daily Study Hours",
                min_value=0.0, max_value=24.0, value=3.0, step=0.5,
                help="Jumlah jam belajar mahasiswa per hari."
            )
            input_data["year"] = st.selectbox(
                "Tahun Kuliah",
                [1, 2, 3, 4],
                help="Tahun kuliah mahasiswa saat ini.",
                key="manual_year"
            )

    else:
        col1, col2 = st.columns(2)

        with col1:
            input_data["sleep_quality"] = st.slider(
                "Sleep Quality (1 = Buruk, 10 = Baik)",
                min_value=1, max_value=10, value=5,
                help="Semakin tinggi nilai, semakin baik kualitas tidur mahasiswa."
            )
            input_data["daily_study_hours"] = st.number_input(
                "Daily Study Hours",
                min_value=0.0, max_value=24.0, value=3.0, step=0.5,
                help="Jumlah jam belajar mahasiswa per hari."
            )
            input_data["daily_sleep_hours"] = st.number_input(
                "Daily Sleep Hours",
                min_value=0.0, max_value=24.0, value=7.0, step=0.5,
                help="Jumlah jam tidur mahasiswa per hari."
            )

        with col2:
            input_data["stress_level"] = st.slider(
                "Stress Level (1 = Rendah, 10 = Tinggi)",
                min_value=1, max_value=10, value=5,
                help="Semakin tinggi nilai, semakin tinggi tingkat stres mahasiswa."
            )
            selected_course = st.selectbox(
                "Program Studi",
                list(COURSE_MAPPING.keys()),
                help="Program studi akan dikonversi otomatis menggunakan Frequency Encoding.",
                key="manual_course"
            )
            input_data["course_freq_encode"] = COURSE_MAPPING[selected_course]

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


# =========================
# PREDIKSI DATASET UPLOAD
# =========================

elif menu == "Prediksi Dataset":
    st.subheader("📂 Prediksi Banyak Data dari Upload Dataset")

    prediction_choice = st.radio(
        "Pilih Jenis Prediksi",
        ["Klasifikasi Burnout Level", "Regresi CGPA"],
        horizontal=True,
        key="dataset_prediction_choice"
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            data = read_uploaded_file(uploaded_file)

            st.subheader("👀 Preview Dataset")
            st.dataframe(data.head(), use_container_width=True)
            st.write(f"Jumlah data: {data.shape[0]} baris | Jumlah kolom: {data.shape[1]} kolom")

            if prediction_choice == "Klasifikasi Burnout Level":
                mode = "Klasifikasi"
                model, selected_model_name = load_selected_model(
                    "model_klasifikasi",
                    key="dataset_klasifikasi"
                )
                feature_names = CLASSIFICATION_FEATURES
                file_name = "hasil_prediksi_burnout.csv"
            else:
                mode = "Regresi"
                model, selected_model_name = load_selected_model(
                    "model_regresi",
                    key="dataset_regresi"
                )
                feature_names = REGRESSION_FEATURES
                file_name = "hasil_prediksi_cgpa.csv"

            st.info(f"🤖 Model yang digunakan: **{selected_model_name}** — `{type(model).__name__}`")

            missing_cols = validate_columns(data, feature_names)

            if len(missing_cols) > 0:
                st.error("Kolom pada dataset belum sesuai dengan fitur yang dibutuhkan model.")
                st.write("Kolom yang belum ada:", missing_cols)
                st.write("Kolom yang harus ada:", feature_names)

            else:
                input_df = data[feature_names].copy()

                for col in feature_names:
                    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

                if input_df.isnull().sum().sum() > 0:
                    st.error("Ada nilai kosong atau data non-numerik pada kolom fitur.")
                    missing_value_info = input_df.isnull().sum()
                    st.write("Jumlah nilai bermasalah per kolom:")
                    st.write(missing_value_info[missing_value_info > 0])

                else:
                    st.success("✅ Format dataset sudah sesuai. Data siap diprediksi.")
                    st.subheader("📄 Data Fitur yang Digunakan")
                    st.dataframe(input_df.head(), use_container_width=True)

                    if st.button("🔍 Prediksi Dataset"):
                        with st.spinner("Sedang memproses prediksi..."):
                            scaled_input = scale_input(input_df, mode)
                            predictions = model.predict(scaled_input)
                            result_df = data.copy()

                            if mode == "Klasifikasi":
                                result_df["prediction"] = predictions
                                result_df["prediction_label"] = [
                                    predict_burnout_label(pred) for pred in predictions
                                ]

                                if "burnout_level" in data.columns:
                                    try:
                                        y_true = data["burnout_level"]
                                        acc = accuracy_score(y_true, predictions)
                                        precision = precision_score(y_true, predictions, average="weighted", zero_division=0)
                                        recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
                                        f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)

                                        st.subheader("📊 Evaluasi Model Klasifikasi")
                                        c1, c2, c3, c4 = st.columns(4)
                                        c1.metric("Accuracy", f"{acc:.4f}")
                                        c2.metric("Precision", f"{precision:.4f}")
                                        c3.metric("Recall", f"{recall:.4f}")
                                        c4.metric("F1 Score", f"{f1:.4f}")
                                    except:
                                        st.warning("Kolom burnout_level tidak sesuai format model sehingga evaluasi tidak dapat dihitung.")

                                if hasattr(model, "predict_proba"):
                                    probabilities = model.predict_proba(scaled_input)
                                    if probabilities.shape[1] == 3:
                                        result_df["probability_low"] = probabilities[:, 0]
                                        result_df["probability_moderate"] = probabilities[:, 1]
                                        result_df["probability_high"] = probabilities[:, 2]

                            else:
                                result_df["prediction_cgpa"] = predictions

                                if "cgpa" in data.columns:
                                    try:
                                        y_true = data["cgpa"]
                                        mae = mean_absolute_error(y_true, predictions)
                                        mse = mean_squared_error(y_true, predictions)
                                        rmse = np.sqrt(mse)
                                        r2 = r2_score(y_true, predictions)

                                        st.subheader("📈 Evaluasi Model Regresi")
                                        c1, c2, c3, c4 = st.columns(4)
                                        c1.metric("MAE", f"{mae:.4f}")
                                        c2.metric("MSE", f"{mse:.4f}")
                                        c3.metric("RMSE", f"{rmse:.4f}")
                                        c4.metric("R²", f"{r2:.4f}")
                                    except:
                                        st.warning("Kolom cgpa tidak sesuai format model sehingga evaluasi tidak dapat dihitung.")

                            st.subheader("🎯 Hasil Prediksi Dataset")
                            st.dataframe(result_df, use_container_width=True)

                            csv_result = result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Hasil Prediksi",
                                data=csv_result,
                                file_name=file_name,
                                mime="text/csv"
                            )

        except Exception as e:
            st.error(f"File gagal diproses: {e}")


# =========================
# EVALUASI MODEL
# =========================

elif menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Model")

    eval_choice = st.radio(
        "Pilih Evaluasi",
        ["Klasifikasi", "Regresi"],
        horizontal=True
    )

    if eval_choice == "Klasifikasi":
        model, selected_model_name = load_selected_model(
            "model_klasifikasi",
            key="eval_klasifikasi"
        )

        st.info(f"Model yang dievaluasi: **{selected_model_name}** — `{type(model).__name__}`")

        with st.spinner("Sedang menghitung evaluasi..."):
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
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

        st.pyplot(fig)

    else:
        model, selected_model_name = load_selected_model(
            "model_regresi",
            key="eval_regresi"
        )

        st.info(f"Model yang dievaluasi: **{selected_model_name}** — `{type(model).__name__}`")

        with st.spinner("Sedang menghitung evaluasi..."):
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
        ax.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            linestyle="--"
        )
        ax.set_xlabel("Actual CGPA")
        ax.set_ylabel("Predicted CGPA")
        ax.set_title("Actual vs Predicted CGPA")
        st.pyplot(fig)

        st.info(
            "Jika nilai R² rendah atau negatif, artinya model belum mampu menjelaskan variasi target dengan baik. "
            "Hal ini dapat dijadikan analisis dalam laporan."
        )


# =========================
# VISUALISASI DATA
# =========================

elif menu == "Visualisasi Data":
    st.subheader("📊 Visualisasi Sederhana Dataset")

    chart_choice = st.selectbox(
        "Pilih Visualisasi",
        [
            "Distribusi Burnout Level",
            "Distribusi CGPA",
            "Rata-rata CGPA berdasarkan Burnout Level",
            "Distribusi Stress Level"
        ],
        key="visualisasi_chart"
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


# =========================
# TENTANG DATASET
# =========================

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
            - daily_sleep_hours: jumlah jam tidur mahasiswa  
            - year: tahun kuliah mahasiswa  
            - course_freq_encode: hasil Frequency Encoding dari program studi mahasiswa  
            - burnout_level: label klasifikasi tingkat burnout  
            - cgpa: target regresi nilai akademik mahasiswa  
            """
        )


# =========================
# BIODATA KELOMPOK
# =========================

elif menu == "Biodata Kelompok":
    st.subheader("👥 Biodata Kelompok")

    st.markdown(
        """
        <div style="background-color:#F0F4FF; padding:20px; border-radius:14px; margin-bottom:20px;">
            <h3 style="color:#1F4E79; margin-bottom:4px;">Kelompok 2</h3>
            <p style="color:#555; margin:0;">Kelompok 2 — SIKC - Fakultas Ilmu Terapan, Telkom University</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Foto kelompok
    if os.path.exists("foto_kelompok.JPEG"):
        col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
        with col_tengah:
            st.image(
                "foto_kelompok.JPEG",
                caption="Kelompok 2 — SIKC - Fakultas Ilmu Terapan, Telkom University",
                use_container_width=True
            )
    else:
        st.info("")

    st.markdown("<br>", unsafe_allow_html=True)

    anggota = [
        {"no": "1", "nama": "Nauval Rafi A.P",   "nim": "707012400071", "icon": "👨\u200d💻"},
        {"no": "2", "nama": "Kerin Anggraeni",    "nim": "707012400066", "icon": "👩\u200d💻"},
        {"no": "3", "nama": "M. Habib Rizky D",   "nim": "707012400133", "icon": "👨\u200d💻"},
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for col, member in zip(cols, anggota):
        with col:
            st.markdown(
                f"""
                <div style="
                    background-color:#ffffff;
                    border: 1.5px solid #C7D9F5;
                    border-radius:14px;
                    padding:22px 18px;
                    text-align:center;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                ">
                    <div style="font-size:42px; margin-bottom:10px;">{member['icon']}</div>
                    <div style="
                        background-color:#1F4E79;
                        color:white;
                        border-radius:20px;
                        padding:2px 12px;
                        display:inline-block;
                        font-size:12px;
                        margin-bottom:10px;
                    ">Anggota {member['no']}</div>
                    <h4 style="color:#1F4E79; margin:8px 0 4px 0; font-size:16px;">{member['nama']}</h4>
                    <p style="color:#666; font-size:13px; margin:0;">NIM: {member['nim']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )