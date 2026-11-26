import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# MongoDB
from pymongo import MongoClient
from bson import ObjectId

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model Regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Model Klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
st.markdown('<div class="sub-title">Eksperimen Regresi, Klasifikasi & Database MongoDB 🧪</div>', unsafe_allow_html=True)


# ============================================================
# CACHE: Fungsi untuk memuat dataset sklearn
# ============================================================
@st.cache_data
def load_dataset(name: str):
    """Memuat dataset toy dari sklearn berdasarkan nama."""
    if name == "Diabetes 🩺":
        data = datasets.load_diabetes(as_frame=True)
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
# MONGODB: Client cache + helper
# ============================================================
@st.cache_resource
def get_mongo_client():
    """Membuat koneksi MongoDB menggunakan URI dari st.secrets (cache_resource)."""
    try:
        uri = st.secrets["MONGODB_URI"]
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Ping untuk validasi koneksi
        client.admin.command("ping")
        return client, None
    except KeyError:
        return None, "❌ `MONGODB_URI` tidak ditemukan di `st.secrets`. Tambahkan di `.streamlit/secrets.toml`."
    except Exception as e:
        return None, f"❌ Gagal terhubung ke MongoDB: {e}"


@st.cache_data(ttl=5)
def fetch_data(_client, db_name: str, collection_name: str):
    """Mengambil semua dokumen dari collection (cache 5 detik agar cepat refresh)."""
    db = _client[db_name]
    coll = db[collection_name]
    docs = list(coll.find().sort("timestamp", -1))
    return docs


def insert_data(client, db_name, collection_name, doc: dict):
    """Insert satu dokumen ke collection."""
    client[db_name][collection_name].insert_one(doc)


def delete_data(client, db_name, collection_name, _id: str):
    """Hapus dokumen berdasarkan _id (ObjectId string)."""
    client[db_name][collection_name].delete_one({"_id": ObjectId(_id)})


def count_docs(client, db_name, collection_name) -> int:
    """Hitung jumlah dokumen (tanpa cache agar selalu akurat)."""
    return client[db_name][collection_name].count_documents({})


# ============================================================
# SIDEBAR: Semua kontrol eksperimen
# ============================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Eksperimen")

    task_type = st.radio(
        "🎯 Pilih Tipe Tugas",
        options=["Regresi", "Klasifikasi"],
        help="Regresi memprediksi angka kontinu, Klasifikasi memprediksi kelas."
    )

    if task_type == "Regresi":
        dataset_options = ["Diabetes 🩺"]
    else:
        dataset_options = ["Iris 🌸", "Wine 🍷", "Breast Cancer 🎗️", "Digits 🔢"]

    dataset_name = st.selectbox("📊 Pilih Dataset", dataset_options)

    if task_type == "Regresi":
        model_options = ["Linear Regression 📏", "Ridge 📐", "Lasso 📉",
                         "Decision Tree 🌳", "Random Forest 🌲"]
    else:
        model_options = ["Logistic Regression 📊", "KNN 👥", "Decision Tree 🌳",
                         "Random Forest 🌲", "SVM 🧠"]

    model_name = st.selectbox("🤖 Pilih Model", model_options)

    test_size = st.slider(
        "📏 Test Size",
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="Proporsi data untuk testing (0.1 – 0.5)"
    )

    train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("💡 Pilih konfigurasi lalu klik **Train Model** untuk memulai.")


# ============================================================
# FUNGSI: Membangun model
# ============================================================
def build_model(task: str, model_name: str):
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
# PROSES TRAINING (Tab 1 & 2)
# ============================================================
if train_button:
    try:
        with st.spinner("📥 Memuat dataset..."):
            dataset = load_dataset(dataset_name)

        X = dataset.data
        y = dataset.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if task_type == "Klasifikasi" else None
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_model(task_type, model_name)
        with st.spinner(f"🏋️ Melatih model {model_name}..."):
            model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

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
# TABS UTAMA: Tab 1, Tab 2, Tab 3
# ============================================================
tab1, tab2, tab3 = st.tabs(["📈 Visualisasi", "📋 Evaluation Metrics", "🗄️ Database & Retrain"])

# ------------------------------------------------------------
# TAB 1 — Visualisasi
# ------------------------------------------------------------
with tab1:
    if st.session_state.get("trained", False):
        task = st.session_state["task_type"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        target_names = st.session_state["target_names"]

        st.markdown(
            f"### 📊 Hasil: **{st.session_state['model_name']}** "
            f"pada **{st.session_state['dataset_name']}**"
        )

        if task == "Regresi":
            st.markdown("#### 📈 Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", color="#1f77b4")
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", lw=2, label="Ideal (y = x)")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown("#### 🧩 Confusion Matrix")
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
    else:
        st.info("👈 Atur konfigurasi di **sidebar** lalu klik **🚀 Train Model** untuk melihat hasil.")


# ------------------------------------------------------------
# TAB 2 — Evaluation Metrics
# ------------------------------------------------------------
with tab2:
    if st.session_state.get("trained", False):
        task = st.session_state["task_type"]
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        target_names = st.session_state["target_names"]

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

            st.info(
                f"- **MAE**: rata-rata selisih absolut → semakin kecil semakin baik.\n"
                f"- **MSE**: rata-rata kuadrat error → sensitif terhadap outlier.\n"
                f"- **RMSE**: akar dari MSE, satuannya sama dengan target.\n"
                f"- **R²**: proporsi variansi yang dijelaskan model (maks 1.0)."
            )
        else:
            st.markdown("### 📊 Metrik Klasifikasi")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy 🎯", f"{acc:.4f}")
            col2.metric("Precision 🎪", f"{prec:.4f}")
            col3.metric("Recall 🔍", f"{rec:.4f}")
            col4.metric("F1-Score ⚖️", f"{f1:.4f}")

            st.markdown("#### 📄 Classification Report")
            st.code(
                classification_report(y_test, y_pred, target_names=target_names, zero_division=0),
                language="text"
            )
    else:
        st.info("👈 Belum ada model yang dilatih. Latih dulu di sidebar.")


# ------------------------------------------------------------
# TAB 3 — Database & Retrain (MongoDB CRUD + Retrain RF)
# ------------------------------------------------------------
with tab3:
    st.markdown("## 🗄️ Database MongoDB & Retrain Model")
    st.caption("Database: `test_db` | Collection: `iris_db`")

    # --- Koneksi MongoDB ---
    client, err = get_mongo_client()
    if err or client is None:
        st.error(err or "❌ Tidak dapat terhubung ke MongoDB.")
        st.stop()

    DB_NAME = "test_db"
    COLL_NAME = "iris_db"

    # --- Statistik collection ---
    try:
        total_docs = count_docs(client, DB_NAME, COLL_NAME)
    except Exception as e:
        st.error(f"❌ Gagal menghitung dokumen: {e}")
        st.stop()

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("📦 Total Dokumen", total_docs)
    col_stat2.metric("🏷️ Database", DB_NAME)
    col_stat3.metric("📚 Collection", COLL_NAME)

    st.markdown("---")

    # ============================================
    # FORM INSERT DATA IRIS
    # ============================================
    st.markdown("### ➕ Insert Data Iris Baru")
    with st.form("form_insert_iris", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        sepal_length = c1.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = c2.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = c3.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = c4.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        species = c5.selectbox("Species", ["setosa", "versicolor", "virginica"])

        submitted = st.form_submit_button("💾 Simpan ke MongoDB", use_container_width=True, type="primary")

    if submitted:
        try:
            doc = {
                "sepal_length": float(sepal_length),
                "sepal_width": float(sepal_width),
                "petal_length": float(petal_length),
                "petal_width": float(petal_width),
                "species": species,
                "timestamp": datetime.utcnow(),
            }
            insert_data(client, DB_NAME, COLL_NAME, doc)
            st.success(f"✅ Data **{species}** berhasil disimpan!")
            st.cache_data.clear()  # bersihkan cache query
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal menyimpan data: {e}")

    st.markdown("---")

    # ============================================
    # TABEL DATA MONGO + HAPUS
    # ============================================
    st.markdown("### 📋 Data dari MongoDB")
    try:
        docs = fetch_data(client, DB_NAME, COLL_NAME)
    except Exception as e:
        st.error(f"❌ Gagal mengambil data: {e}")
        docs = []

    if docs:
        df_mongo = pd.DataFrame(docs)
        df_mongo["_id"] = df_mongo["_id"].astype(str)

        # Format tampilan: kolom _id jangan tampil sebagai ObjectId
        display_df = df_mongo.copy()
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # --- Hapus berdasarkan _id ---
        st.markdown("#### 🗑️ Hapus Data berdasarkan `_id`")
        with st.form("form_delete_iris"):
            id_list = df_mongo["_id"].tolist()
            selected_id = st.selectbox("Pilih _id yang akan dihapus", id_list)
            del_btn = st.form_submit_button("🗑️ Hapus", type="secondary", use_container_width=True)

        if del_btn:
            try:
                delete_data(client, DB_NAME, COLL_NAME, selected_id)
                st.success(f"✅ Dokumen `{selected_id}` berhasil dihapus!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menghapus: {e}")
    else:
        st.info("📭 Belum ada data di MongoDB. Silakan insert dulu di form di atas.")

    st.markdown("---")

    # ============================================
    # RETRAIN: Gabungan data sklearn + MongoDB
    # ============================================
    st.markdown("### 🔁 Retrain Model dengan Data Gabungan")
    st.caption("Data sklearn (150 sampel) + data MongoDB → Random Forest Classifier")

    if st.button("🔁 Retrain Sekarang", use_container_width=True, type="primary"):
        with st.spinner("🔄 Menggabungkan data & melatih ulang model..."):
            try:
                # 1. Data sklearn Iris
                iris = datasets.load_iris(as_frame=True)
                df_sklearn = iris.frame.copy()
                # Kolom sklearn: sepal length (cm), dst. → rename agar seragam
                df_sklearn = df_sklearn.rename(columns={
                    "sepal length (cm)": "sepal_length",
                    "sepal width (cm)": "sepal_width",
                    "petal length (cm)": "petal_length",
                    "petal width (cm)": "petal_width",
                    "target": "species_int",
                })
                # Konversi target int -> string species
                species_map_inv = {0: "setosa", 1: "versicolor", 2: "virginica"}
                df_sklearn["species"] = df_sklearn["species_int"].map(species_map_inv)
                df_sklearn = df_sklearn[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]

                # 2. Data MongoDB
                if not docs:
                    st.warning("⚠️ Tidak ada data MongoDB. Menggunakan 150 sampel sklearn saja.")
                    df_mongo_clean = pd.DataFrame(columns=df_sklearn.columns)
                else:
                    df_mongo_clean = pd.DataFrame(docs)[
                        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
                    ].copy()

                # 3. Gabung
                df_all = pd.concat([df_sklearn, df_mongo_clean], ignore_index=True)

                # 4. Encode species string → int
                species_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
                df_all = df_all[df_all["species"].isin(species_map.keys())]
                df_all["species_int"] = df_all["species"].map(species_map)

                X_all = df_all[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
                y_all = df_all["species_int"].values

                # 5. Split & train Random Forest
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
                )
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_tr, y_tr)
                y_pr = rf.predict(X_te)

                # 6. Metrik
                acc = accuracy_score(y_te, y_pr)
                st.success(
                    f"✅ Retrain selesai! Total data: **{len(df_all)}** "
                    f"(sklearn: {len(df_sklearn)}, Mongo: {len(df_mongo_clean)})"
                )

                colA, colB, colC = st.columns(3)
                colA.metric("🎯 Akurasi", f"{acc:.4f}")
                colB.metric("📦 Total Data Training", len(X_tr))
                colC.metric("🧪 Total Data Testing", len(X_te))

                st.markdown("#### 📄 Classification Report")
                st.code(
                    classification_report(
                        y_te, y_pr,
                        target_names=["setosa", "versicolor", "virginica"],
                        zero_division=0
                    ),
                    language="text"
                )

                # 7. Feature Importance
                st.markdown("#### 🌟 Feature Importance")
                importances = rf.feature_importances_
                feat_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(
                    x=importances, y=feat_names,
                    palette="viridis", ax=ax
                )
                ax.set_xlabel("Importance Score")
                ax.set_ylabel("Fitur")
                ax.set_title("Feature Importance — Random Forest")
                for i, v in enumerate(importances):
                    ax.text(v + 0.005, i, f"{v:.3f}", va="center")
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"❌ Gagal retrain: {e}")