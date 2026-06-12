import streamlit as st
import pandas as pd
import joblib
import os
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

# =========================
# KONFIGURASI FITUR
# =========================

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

DATASET_PATH = "student_mental_health_burnout_clean.csv"


# =========================
# FUNCTION
# =========================

def get_model_files(folder):
    joblib_files = glob.glob(os.path.join(folder, "*.joblib"))
    pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
    return joblib_files + pkl_files


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


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
        return "Kondisi burnout mahasiswa tergolong rendah. Mahasiswa masih cukup mampu mengelola aktivitas akademik dan keseharian."
    elif pred == 1:
        return "Kondisi burnout mahasiswa tergolong sedang. Mahasiswa mulai menunjukkan risiko kelelahan akademik dan perlu menjaga pola belajar, tidur, serta istirahat."
    elif pred == 2:
        return "Kondisi burnout mahasiswa tergolong tinggi. Mahasiswa berisiko mengalami kelelahan akademik yang cukup serius dan disarankan untuk memperbaiki manajemen waktu serta mencari dukungan."
    else:
        return "Hasil prediksi tidak dikenali."


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
        scaled_data = scaler.transform(input_df[CLASSIFICATION_FEATURES])
        return pd.DataFrame(scaled_data, columns=CLASSIFICATION_FEATURES)

    else:
        scaler = MinMaxScaler(feature_range=(0, 10))
        scaler.fit(df[REGRESSION_FEATURES])
        scaled_data = scaler.transform(input_df[REGRESSION_FEATURES])
        return pd.DataFrame(scaled_data, columns=REGRESSION_FEATURES)


# =========================
# HEADER
# =========================

st.markdown(
    """
    <div style="background-color:#EAF4FF; padding:28px; border-radius:18px; margin-bottom:22px;">
        <h1 style="color:#1F4E79; margin-bottom:8px;">🧠 Student Mental Health Prediction App</h1>
        <p style="font-size:17px; color:#333;">
            Aplikasi ini membantu mahasiswa memperkirakan tingkat burnout atau nilai CGPA
            berdasarkan kondisi akademik dan kebiasaan harian yang dimasukkan secara manual.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Pengaturan")

prediction_type = st.sidebar.selectbox(
    "Pilih Jenis Prediksi",
    ["Klasifikasi Burnout Level", "Regresi CGPA"]
)

if prediction_type == "Klasifikasi Burnout Level":
    model_folder = "model_klasifikasi"
    feature_names = CLASSIFICATION_FEATURES
    mode = "Klasifikasi"
else:
    model_folder = "model_regresi"
    feature_names = REGRESSION_FEATURES
    mode = "Regresi"

model_files = get_model_files(model_folder)

if len(model_files) == 0:
    st.error(f"Belum ada model di folder `{model_folder}`.")
    st.stop()

model_names = [os.path.basename(file) for file in model_files]

selected_model_name = st.sidebar.selectbox(
    "Pilih Model",
    model_names
)

selected_model_path = model_files[model_names.index(selected_model_name)]

try:
    model = load_model(selected_model_path)
    st.sidebar.success(f"Model aktif: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"Model gagal dimuat: {e}")
    st.stop()


# =========================
# INFO APLIKASI
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jenis Prediksi", mode)

with col2:
    st.metric("Model", selected_model_name)

with col3:
    st.metric("Jumlah Fitur", len(feature_names))

st.info(
    "Catatan: Hasil prediksi merupakan estimasi berdasarkan model machine learning, "
    "bukan hasil diagnosis atau penilaian akademik resmi."
)

# =========================
# FORM INPUT USER
# =========================

st.subheader("📝 Input Data Mahasiswa")

input_data = {}

if mode == "Klasifikasi":
    col1, col2 = st.columns(2)

    with col1:
        input_data["stress_level"] = st.slider(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=5
        )

        input_data["sleep_quality"] = st.slider(
            "Sleep Quality",
            min_value=1,
            max_value=10,
            value=5
        )

        input_data["internet_quality"] = st.slider(
            "Internet Quality",
            min_value=1,
            max_value=10,
            value=5
        )

    with col2:
        input_data["daily_study_hours"] = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        input_data["year"] = st.selectbox(
            "Tahun Kuliah",
            [1, 2, 3, 4]
        )

else:
    col1, col2 = st.columns(2)

    with col1:
        input_data["sleep_quality"] = st.slider(
            "Sleep Quality",
            min_value=1,
            max_value=10,
            value=5
        )

        input_data["daily_study_hours"] = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        input_data["daily_sleep_hours"] = st.number_input(
            "Daily Sleep Hours",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

    with col2:
        input_data["stress_level"] = st.slider(
            "Stress Level",
            min_value=1,
            max_value=10,
            value=5
        )

        input_data["course_freq_encode"] = st.number_input(
            "Course Frequency Encode",
            min_value=0.0,
            value=1.0,
            step=0.1
        )


input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]

st.subheader("📄 Data yang Dimasukkan")
st.dataframe(input_df, use_container_width=True)

# =========================
# PREDIKSI
# =========================

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
            elif prediction == 2:
                st.error(f"Hasil Prediksi: {label}")
            else:
                st.info(f"Hasil Prediksi: {label}")

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
# PENJELASAN FITUR
# =========================

with st.expander("ℹ️ Penjelasan Fitur"):
    st.write("""
    **Stress Level**: tingkat stres mahasiswa.  
    **Sleep Quality**: kualitas tidur mahasiswa.  
    **Internet Quality**: kualitas koneksi internet mahasiswa.  
    **Daily Study Hours**: jumlah jam belajar per hari.  
    **Year**: tahun kuliah mahasiswa.  
    **Daily Sleep Hours**: jumlah jam tidur per hari.  
    **Course Frequency Encode**: hasil encoding dari frekuensi/kategori course pada dataset.
    """)