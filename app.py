import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model Regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Model Klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Metrik
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="ML Playground 🎯",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling CSS kustom agar UI lebih cantik
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .sub-title { text-align: center; color: gray; margin-bottom: 2rem; }
    .metric-box { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Machine Learning Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Eksperimen Regresi & Klasifikasi dengan Dataset Scikit-Learn 🧪</div>', unsafe_allow_html=True)


# ============================================================
# CACHE: Fungsi untuk memuat dataset (supaya tidak reload terus)
# ============================================================
@st.cache_data
def load_dataset(name: str):
    """Memuat dataset toy dari sklearn berdasarkan nama."""
    if name == "Diabetes 🩺":
        data = datasets.load_diabetes(as_frame=True)
    elif name == "California Housing 🏠":
        data = datasets.load_diabetes(as_frame=True)  # fallback (toy)
    elif name == "Iris 🌸":
        data = datasets.load_iris(as_frame=True)
    elif name == "Wine 🍷":
        data = datasets.load_wine(as_frame=True)
    elif name == "Breast Cancer 🎗️":
        data = datasets.load_breast_cancer(as_frame=True)
    elif name == "Digits 🔢":
        data = datasets.load_digits(as_frame=True)
    else:
        raise ValueError(f"Dataset '{name}' tidak dikenali")
    return data


# ============================================================
# SIDEBAR: Semua kontrol eksperimen
# ============================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Eksperimen")

    # 1. Pilih tipe tugas
    task_type = st.radio(
        "🎯 Pilih Tipe Tugas",
        options=["Regresi", "Klasifikasi"],
        help="Regresi memprediksi angka kontinu, Klasifikasi memprediksi kelas."
    )

    # 2. Pilih dataset berdasarkan tipe tugas
    if task_type == "Regresi":
        dataset_options = ["Diabetes 🩺"]
    else:
        dataset_options = ["Iris 🌸", "Wine 🍷", "Breast Cancer 🎗️", "Digits 🔢"]

    dataset_name = st.selectbox("📊 Pilih Dataset", dataset_options)

    # 3. Pilih model
    if task_type == "Regresi":
        model_options = [
            "Linear Regression 📏",
            "Ridge 📐",
            "Lasso 📉",
            "Decision Tree 🌳",
            "Random Forest 🌲"
        ]
    else:
        model_options = [
            "Logistic Regression 📊",
            "KNN 👥",
            "Decision Tree 🌳",
            "Random Forest 🌲",
            "SVM 🧠"
        ]

    model_name = st.selectbox("🤖 Pilih Model", model_options)

    # 4. Slider test size
    test_size = st.slider(
        "📏 Test Size",
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="Proporsi data untuk testing (0.1 – 0.5)"
    )

    # 5. Tombol train
    train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("💡 Pilih konfigurasi lalu klik **Train Model** untuk memulai.")


# ============================================================
# FUNGSI: Membangun model berdasarkan pilihan
# ============================================================
def build_model(task: str, model_name: str):
    """Mengembalikan objek model sklearn sesuai pilihan."""
    # Regresi
    if task == "Regresi":
        if "Linear Regression" in model_name:
            return LinearRegression()
        elif "Ridge" in model_name:
            return Ridge(alpha=1.0)
        elif "Lasso" in model_name:
            return Lasso(alpha=0.1)
        elif "Decision Tree" in model_name:
            return DecisionTreeRegressor(random_state=42)
        elif "Random Forest" in model_name:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    # Klasifikasi
    else:
        if "Logistic Regression" in model_name:
            return LogisticRegression(max_iter=1000)
        elif "KNN" in model_name:
            return KNeighborsClassifier(n_neighbors=5)
        elif "Decision Tree" in model_name:
            return DecisionTreeClassifier(random_state=42)
        elif "Random Forest" in model_name:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif "SVM" in model_name:
            return SVC(kernel="rbf")
    raise ValueError("Model tidak dikenali")


# ============================================================
# PROSES TRAINING SAAT TOMBOL DIKLIK
# ============================================================
if train_button:
    try:
        # --- Muat dataset ---
        with st.spinner("📥 Memuat dataset..."):
            dataset = load_dataset(dataset_name)

        # Dataframe fitur dan target
        X = dataset.data
        y = dataset.target

        # --- Split data ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task_type == "Klasifikasi" else None
        )

        # --- Scaling (penting untuk Logistic/KNN/SVM/Ridge/Lasso) ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # --- Bangun & latih model ---
        model = build_model(task_type, model_name)
        with st.spinner(f"🏋️ Melatih model {model_name}..."):
            model.fit(X_train_s, y_train)

        # --- Prediksi ---
        y_pred = model.predict(X_test_s)

        # --- Simpan ke session_state ---
        st.session_state["trained"] = True
        st.session_state["task_type"] = task_type
        st.session_state["dataset_name"] = dataset_name
        st.session_state["model_name"] = model_name
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["target_names"] = getattr(dataset, "target_names", None)

        st.success(f"✅ Model **{model_name}** berhasil dilatih pada dataset **{dataset_name}**!")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")


# ============================================================
# TAMPILAN HASIL (jika sudah ada model terlatih)
# ============================================================
if st.session_state.get("trained", False):
    task = st.session_state["task_type"]
    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]
    target_names = st.session_state["target_names"]

    st.markdown("---")
    st.subheader(
        f"📊 Hasil: **{st.session_state['model_name']}** "
        f"pada **{st.session_state['dataset_name']}** ({task})"
    )

    # Buat dua tab: Visualisasi & Metrik
    tab1, tab2 = st.tabs(["📈 Visualisasi", "📋 Evaluation Metrics"])

    # --------------------------------------------------------
    # TAB 1: Visualisasi
    # --------------------------------------------------------
    with tab1:
        if task == "Regresi":
            st.markdown("### 📈 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", color="#1f77b4")
            # Garis ideal y = x
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", lw=2, label="Ideal (y = x)")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        else:  # Klasifikasi
            st.markdown("### 🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names if target_names is not None else "auto",
                yticklabels=target_names if target_names is not None else "auto",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            plt.close(fig)

    # --------------------------------------------------------
    # TAB 2: Evaluation Metrics
    # --------------------------------------------------------
    with tab2:
        if task == "Regresi":
            st.markdown("### 📊 Metrik Regresi")
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE 📉", f"{mae:.4f}")
            col2.metric("MSE 📊", f"{mse:.4f}")
            col3.metric("RMSE 📈", f"{rmse:.4f}")
            col4.metric("R² 🎯", f"{r2:.4f}")

            st.markdown("#### 📝 Ringkasan")
            st.info(
                f"- **MAE** (Mean Absolute Error): rata-rata selisih absolut → semakin kecil semakin baik.\n"
                f"- **MSE** (Mean Squared Error): rata-rata kuadrat error → sensitif terhadap outlier.\n"
                f"- **RMSE** (Root MSE): akar dari MSE, satuannya sama dengan target.\n"
                f"- **R²** (R-squared): proporsi variansi yang dijelaskan model (maks 1.0)."
            )

        else:  # Klasifikasi
            st.markdown("### 📊 Metrik Klasifikasi")
            acc = accuracy_score(y_test, y_pred)
            # Gunakan average='weighted' agar aman untuk multiclass
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy 🎯", f"{acc:.4f}")
            col2.metric("Precision 🎪", f"{prec:.4f}")
            col3.metric("Recall 🔍", f"{rec:.4f}")
            col4.metric("F1-Score ⚖️", f"{f1:.4f}")

            st.markdown("#### 📄 Classification Report")
            report = classification_report(
                y_test, y_pred,
                target_names=target_names,
                zero_division=0
            )
            st.code(report, language="text")

else:
    # Pesan awal jika belum training
    st.info("👈 Atur konfigurasi di **sidebar** lalu klik **🚀 Train Model** untuk melihat hasil.")
    st.markdown("""
    ### 🌟 Fitur Aplikasi
    - **Regresi** 📏 : Linear, Ridge, Lasso, Decision Tree, Random Forest
    - **Klasifikasi** 🎯 : Logistic, KNN, Decision Tree, Random Forest, SVM
    - **Visualisasi**: Actual vs Predicted (regresi) & Confusion Matrix (klasifikasi)
    - **Metrik lengkap** + classification report
    - **Session state** menyimpan hasil training antar interaksi
    """)