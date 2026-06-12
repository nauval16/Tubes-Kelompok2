import streamlit as st
import pandas as pd
import joblib
import os
import glob
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

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


def get_model_files(model_folder):
    joblib_files = glob.glob(os.path.join(model_folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(model_folder, "*.pkl"))
    return joblib_files + pkl_files


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def predict_class_label(prediction):
    if prediction == 0:
        return "Low Burnout / Rendah"
    elif prediction == 1:
        return "Moderate Burnout / Sedang"
    elif prediction == 2:
        return "High Burnout / Tinggi"
    else:
        return str(prediction)


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def validate_input_columns(df, feature_names):
    return [col for col in feature_names if col not in df.columns]


def prepare_input_data(df, feature_names):
    input_df = df[feature_names].copy()

    for col in feature_names:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    return input_df


def show_header():
    st.markdown(
        """
        <div style="background-color:#EAF4FF; padding:25px; border-radius:18px; margin-bottom:20px;">
            <h1 style="color:#1F4E79; margin-bottom:8px;">🧠 Student Mental Health & Burnout Prediction</h1>
            <p style="font-size:17px; color:#333;">
                Aplikasi ini digunakan untuk melakukan prediksi menggunakan model machine learning
                berdasarkan dataset Student Mental Health and Burnout.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_dataset_info():
    st.subheader("📌 Informasi Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "Student Mental Health")

    with col2:
        st.metric("Target Klasifikasi", "burnout_level")

    with col3:
        st.metric("Target Regresi", "cgpa")

    st.info(
        "Catatan: Pada dataset ini, performansi model bisa rendah karena hubungan fitur terhadap target cukup lemah. "
        "Hal ini tetap dapat dijadikan bahan analisis dalam laporan."
    )


def show_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    st.subheader("📈 Evaluasi Model Klasifikasi")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1-Score", f"{f1:.4f}")


def show_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    st.subheader("📈 Evaluasi Model Regresi")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")
    col4.metric("R²", f"{r2:.4f}")


show_header()
show_dataset_info()

st.sidebar.title("⚙️ Pengaturan Aplikasi")

prediction_type = st.sidebar.selectbox(
    "Pilih Jenis Prediksi",
    ["Klasifikasi", "Regresi"]
)

input_mode = st.sidebar.radio(
    "Pilih Mode Input",
    ["Manual", "CSV/Excel"]
)

if prediction_type == "Klasifikasi":
    model_folder = "model_klasifikasi"
    feature_names = CLASSIFICATION_FEATURES
    label_column = CLASSIFICATION_LABEL
else:
    model_folder = "model_regresi"
    feature_names = REGRESSION_FEATURES
    label_column = REGRESSION_LABEL


st.sidebar.subheader("🤖 Pilih Model")

model_files = get_model_files(model_folder)

if len(model_files) == 0:
    st.error(f"Belum ada file model di folder `{model_folder}/`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.sidebar.selectbox(
    "Model Machine Learning",
    model_names
)

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.sidebar.success(f"Model aktif: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Model gagal dimuat: {e}")
    st.stop()


st.subheader("🧩 Ringkasan Pengaturan")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jenis Prediksi", prediction_type)

with col2:
    st.metric("Mode Input", input_mode)

with col3:
    st.metric("Jumlah Fitur", len(feature_names))


if input_mode == "Manual":
    st.subheader("✍️ Input Data Manual")
    st.write("Masukkan nilai fitur sesuai data mahasiswa yang ingin diprediksi.")

    input_data = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(feature_names):
        if i % 3 == 0:
            with col1:
                input_data[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.4f"
                )
        elif i % 3 == 1:
            with col2:
                input_data[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.4f"
                )
        else:
            with col3:
                input_data[feature] = st.number_input(
                    label=feature,
                    value=0.0,
                    format="%.4f"
                )

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    st.subheader("📄 Data yang Dimasukkan")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Prediksi Data Manual"):
        try:
            prediction = model.predict(input_df)[0]

            st.subheader("🎯 Hasil Prediksi")

            if prediction_type == "Klasifikasi":
                prediction_label = predict_class_label(prediction)

                if prediction == 0:
                    st.success(f"Hasil prediksi: {prediction_label}")
                elif prediction == 1:
                    st.warning(f"Hasil prediksi: {prediction_label}")
                elif prediction == 2:
                    st.error(f"Hasil prediksi: {prediction_label}")
                else:
                    st.info(f"Hasil prediksi: {prediction_label}")

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]

                    if hasattr(model, "classes_"):
                        class_names = [predict_class_label(cls) for cls in model.classes_]
                    else:
                        class_names = [f"Class {i}" for i in range(len(proba))]

                    proba_df = pd.DataFrame({
                        "Class": class_names,
                        "Probability": proba
                    })

                    st.subheader("📊 Probabilitas Prediksi")
                    st.dataframe(proba_df, use_container_width=True)

            else:
                st.success(f"Hasil prediksi nilai CGPA: {prediction:.4f}")

                if prediction >= 8:
                    st.info("Interpretasi: Performa akademik sangat baik.")
                elif prediction >= 6:
                    st.info("Interpretasi: Performa akademik cukup baik.")
                else:
                    st.info("Interpretasi: Performa akademik perlu ditingkatkan.")

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")


elif input_mode == "CSV/Excel":
    st.subheader("📁 Prediksi Banyak Data dari CSV/Excel")
    st.write(
        "Upload file data dalam format CSV atau Excel. "
        "Jika file memiliki kolom label asli, evaluasi model akan dihitung otomatis."
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)

            st.subheader("👀 Preview Data")
            st.dataframe(df.head(), use_container_width=True)

            st.write(f"Jumlah data: {df.shape[0]} baris")
            st.write(f"Jumlah kolom: {df.shape[1]} kolom")

            missing_cols = validate_input_columns(df, feature_names)

            if len(missing_cols) > 0:
                st.error("Kolom pada file belum sesuai dengan fitur yang dibutuhkan model.")
                st.write("Kolom yang belum ada:")
                st.write(missing_cols)

                st.write("Kolom yang harus ada:")
                st.write(feature_names)

            else:
                input_df = prepare_input_data(df, feature_names)

                if input_df.isnull().sum().sum() > 0:
                    st.error(
                        "Ada nilai kosong atau data non-numerik pada kolom fitur. "
                        "Silakan cek kembali file CSV/Excel."
                    )

                    missing_value_info = input_df.isnull().sum()
                    missing_value_info = missing_value_info[missing_value_info > 0]

                    st.write("Jumlah nilai bermasalah per kolom:")
                    st.write(missing_value_info)

                else:
                    st.success("Format data sudah sesuai. Data siap diprediksi.")

                    st.subheader("📄 Data Fitur yang Digunakan")
                    st.dataframe(input_df.head(), use_container_width=True)

                    if st.button("Prediksi Data CSV/Excel"):
                        try:
                            predictions = model.predict(input_df)

                            result_df = df.copy()
                            result_df["prediction"] = predictions

                            st.subheader("🎯 Hasil Prediksi")

                            if prediction_type == "Klasifikasi":
                                result_df["prediction_label"] = [
                                    predict_class_label(pred) for pred in predictions
                                ]

                                if hasattr(model, "predict_proba"):
                                    probabilities = model.predict_proba(input_df)

                                    if probabilities.shape[1] == 3:
                                        result_df["probability_low"] = probabilities[:, 0]
                                        result_df["probability_moderate"] = probabilities[:, 1]
                                        result_df["probability_high"] = probabilities[:, 2]

                                if label_column in df.columns:
                                    y_true = pd.to_numeric(df[label_column], errors="coerce")

                                    valid_index = y_true.notnull()
                                    y_true_valid = y_true[valid_index]
                                    y_pred_valid = predictions[valid_index]

                                    show_classification_metrics(y_true_valid, y_pred_valid)
                                else:
                                    st.info(
                                        f"Kolom label asli `{label_column}` tidak ditemukan, "
                                        "jadi evaluasi model tidak dihitung."
                                    )

                                file_name = "hasil_prediksi_klasifikasi_burnout.csv"

                            else:
                                result_df["prediction_value"] = predictions

                                if label_column in df.columns:
                                    y_true = pd.to_numeric(df[label_column], errors="coerce")

                                    valid_index = y_true.notnull()
                                    y_true_valid = y_true[valid_index]
                                    y_pred_valid = predictions[valid_index]

                                    show_regression_metrics(y_true_valid, y_pred_valid)
                                else:
                                    st.info(
                                        f"Kolom label asli `{label_column}` tidak ditemukan, "
                                        "jadi evaluasi model tidak dihitung."
                                    )

                                file_name = "hasil_prediksi_regresi_cgpa.csv"

                            st.dataframe(result_df, use_container_width=True)

                            csv_result = result_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                label="Download Hasil Prediksi",
                                data=csv_result,
                                file_name=file_name,
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Terjadi error saat prediksi: {e}")

        except Exception as e:
            st.error(f"File gagal dibaca: {e}")