import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# MongoDB
from pymongo import MongoClient
from bson import ObjectId
import certifi

# Scikit-Learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score
)

# ============================================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="TAS AI Research Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #8b5cf6;
        --secondary: #ec4899;
        --accent: #06b6d4;
        --warning: #f59e0b;
        --bg-card: #ffffff;
        --text-dark: #1e293b;
        --text-muted: #64748b;
        --border: #f1f5f9;
    }
    .stApp { background: linear-gradient(135deg, #fdf4ff 0%, #f0f9ff 50%, #fef2f2 100%); color: var(--text-dark); }
    [data-testid="stSidebar"] { background-color: var(--bg-card); border-right: 2px solid var(--border); box-shadow: 4px 0 15px rgba(139, 92, 246, 0.08); }
    
    /* Menu Sidebar */
    div.row-widget.stRadio > div { flex-direction: column; }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        padding: 12px 18px; margin-bottom: 8px; border-radius: 12px; background-color: transparent;
        transition: all 0.3s ease; cursor: pointer; color: var(--text-dark); font-weight: 600; border: 1px solid transparent;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover { background-color: #f5f3ff; border-color: #ddd6fe; }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%); color: white;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4); border: 1px solid #7c3aed;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label > div:first-child { display: none; }
    
    .custom-card {
        background-color: var(--bg-card); padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03); margin-bottom: 20px; border: 2px solid transparent; transition: transform 0.2s;
    }
    .custom-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08); }
    .header-title { background: linear-gradient(90deg, #8b5cf6, #ec4899, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.4rem; margin-bottom: 10px; }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%); color: white; border: none;
        border-radius: 10px; padding: 10px 24px; font-weight: 600; box-shadow: 0 4px 10px rgba(139, 92, 246, 0.3); transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); transform: translateY(-1px); }
    .stSlider [data-baseweb="slider"] > div > div > div { background: linear-gradient(90deg, #c4b5fd, #8b5cf6) !important; }
    .stSlider [role="slider"] { background-color: var(--primary) !important; border-color: var(--primary) !important; }

    /* Sub-Tab Styling (Seperti Screenshot) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #64748b; border-radius: 8px 8px 0 0;
        padding: 8px 16px; font-weight: 600; border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%) !important; color: white !important;
        box-shadow: 0 4px 10px rgba(139, 92, 246, 0.3);
    }

    /* --- STYLING KARTU METRIK KUSTOM --- */
    .metric-card {
        background-color: white; border: 2px solid var(--border); padding: 15px 20px; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02); position: relative; overflow: visible; display: flex; align-items: center; gap: 15px;
    }
    .metric-icon { font-size: 32px; background: #f8fafc; padding: 10px; border-radius: 12px; }
    .metric-content { display: flex; flex-direction: column; }
    .metric-title { font-size: 14px; color: var(--text-muted); font-weight: 600; margin-bottom: 2px; }
    .metric-value { font-weight: 800; font-size: 24px; }
    .tooltip-container { position: absolute; top: 12px; right: 12px; cursor: help; }
    .tooltip-icon { display: inline-block; width: 18px; height: 18px; background-color: #f1f5f9; color: #94a3b8; border-radius: 50%; text-align: center; font-size: 12px; font-weight: bold; line-height: 18px; }
    .tooltip-text {
        visibility: hidden; width: 220px; background-color: #1e293b; color: #fff; text-align: left; border-radius: 8px;
        padding: 10px; position: absolute; z-index: 100; top: 100%; right: 0; margin-top: 5px; opacity: 0; transition: opacity 0.3s; font-size: 12px; line-height: 1.4;
    }
    .tooltip-container:hover .tooltip-text { visibility: visible; opacity: 1; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNGSI BANTUAN
# ============================================================
def render_metric_card(icon, title, value, tooltip_text, bg_color="#8b5cf6"):
    html = f"""
    <div class="metric-card" style="border-left: 5px solid {bg_color};">
        <div class="tooltip-container">
            <span class="tooltip-icon">?</span>
            <span class="tooltip-text">{tooltip_text}</span>
        </div>
        <div class="metric-icon" style="background-color: {bg_color}20; color: {bg_color};">{icon}</div>
        <div class="metric-content">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {bg_color} !important;">{value}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

@st.cache_resource
def get_mongo_client():
    try:
        uri = st.secrets["MONGODB_URI"]
        client = MongoClient(uri, serverSelectionTimeoutMS=10000, tlsCAFile=certifi.where())
        client.admin.command("ping")
        return client, None
    except KeyError:
        return None, "❌ `MONGODB_URI` tidak ditemukan di `st.secrets`."
    except Exception as e:
        return None, f"❌ Gagal terhubung: {e}"

@st.cache_data(ttl=5)
def fetch_data(_client, db_name: str, collection_name: str):
    return list(_client[db_name][collection_name].find().sort("timestamp", -1))

def count_docs(client, db_name, collection_name) -> int:
    return client[db_name][collection_name].count_documents({})

@st.cache_data
def load_dataset(name: str):
    if name == "Diabetes 🩺": return datasets.load_diabetes(as_frame=True)
    elif name == "Iris 🌸": return datasets.load_iris(as_frame=True)
    elif name == "Wine 🍷": return datasets.load_wine(as_frame=True)
    elif name == "Breast Cancer 🎗️": return datasets.load_breast_cancer(as_frame=True)
    else: raise ValueError("Dataset tidak dikenali")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fdf4ff, #f0f9ff); padding: 18px; border-radius: 14px; border: 2px solid #e9d5ff; margin-bottom: 20px;'>
        <h3 style='margin:0; color: #8b5cf6; font-size: 1.4rem;'>🔬 TAS AI Research Lab</h3>
        <p style='margin:0; font-size: 12px; color: #64748b;'>🎓 User AI - 632025051</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:11px; font-weight:bold; color:#94a3b8;'>MODUL PENELITIAN</p>", unsafe_allow_html=True)
    menu = st.radio(
        "Navigasi", 
        ["🏠 Dashboard Beranda", "📈 Regresi Linear", "🎯 Klasifikasi", "🔵 Clustering", "🗄️ Database & Retraining", "💬 Diskusi & Komentar"], 
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 10px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)
    client, err = get_mongo_client()
    status_color = "#10b981" if client else "#ef4444"
    status_bg = "#ecfdf5" if client else "#fef2f2"
    status_text = "Aktif" if client else "Offline"
    st.markdown(f"""
    <div style='background-color: {status_bg}; padding: 15px; border-radius: 12px; border: 2px solid {status_color}40;'>
        <p style='margin:0; font-size: 14px; font-weight:bold; color: #334155;'>🗄️ MongoDB Atlas</p>
        <p style='margin:0; font-size: 12px; color: {status_color}; font-weight:600;'>🟢 {status_text}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOGIKA HALAMAN
# ============================================================

# ----------------- HALAMAN 1: DASHBOARD -----------------
if menu == "🏠 Dashboard Beranda":
    st.markdown('<div class="header-title">🔬 Dashboard Penelitian Machine Learning</div>', unsafe_allow_html=True)
    st.write("Platform eksperimen interaktif untuk Regresi, Klasifikasi, Clustering, dan integrasi Database.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown("""<div class="custom-card" style="border-top: 4px solid #8b5cf6;"><h4 style='color: #8b5cf6;'>📈 Regresi</h4><p style='font-size:14px; color:#64748b;'>Analisis prediksi nilai kontinu.</p></div>""", unsafe_allow_html=True)
    with col2: st.markdown("""<div class="custom-card" style="border-top: 4px solid #ec4899;"><h4 style='color: #ec4899;'>🎯 Klasifikasi</h4><p style='font-size:14px; color:#64748b;'>Analisis prediksi kelas diskrit.</p></div>""", unsafe_allow_html=True)
    with col3: st.markdown("""<div class="custom-card" style="border-top: 4px solid #06b6d4;"><h4 style='color: #06b6d4;'>🔵 Clustering</h4><p style='font-size:14px; color:#64748b;'>Pengelompokan data tanpa label.</p></div>""", unsafe_allow_html=True)
    with col4: st.markdown("""<div class="custom-card" style="border-top: 4px solid #f59e0b;"><h4 style='color: #f59e0b;'>🗄️ Database</h4><p style='font-size:14px; color:#64748b;'>Integrasi penyimpanan MongoDB.</p></div>""", unsafe_allow_html=True)


# ----------------- HALAMAN 2: REGRESI -----------------
elif menu == "📈 Regresi Linear":
    st.markdown('<div class="header-title">📈 Regresi Linear</div>', unsafe_allow_html=True)
    st.info("💡 Model akan dilatih ulang secara otomatis saat kamu mengubah pengaturan.")
    
    dataset_name = st.selectbox("Pilih Dataset", ["Diabetes 🩺"])
    test_size = st.slider("Proporsi Data Uji (Test Size)", 0.1, 0.5, 0.2, 0.05)
    
    # --- LIVE TRAINING ---
    data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("📊", "R² Score", f"{r2_score(y_test, y_pred):.4f}", "Koefisien Determinasi.", "#8b5cf6")
    with c2: render_metric_card("📏", "MAE", f"{mean_absolute_error(y_test, y_pred):.4f}", "Mean Absolute Error.", "#ec4899")
    with c3: render_metric_card("📉", "RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}", "Root Mean Squared Error.", "#06b6d4")
    
    # SUB-TABS SEPERTI SCREENSHOT
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Grafik & Koefisien", "🔮 Simulator Prediksi"])
    
    with tab1:
        st.write("**Data Preview:**")
        # Menampilkan dataframe persis seperti screenshot
        st.dataframe(data.frame, use_container_width=True, height=400)
        st.write("**Statistik Deskriptif:**")
        st.dataframe(data.frame.describe(), use_container_width=True)
        
    with tab2:
        st.markdown("#### Grafik Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred, alpha=0.7, color="#8b5cf6", edgecolors="#7c3aed")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "#ec4899", lw=2, linestyle="--", label="Ideal")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.legend()
        st.pyplot(fig)
        
    with tab3:
        st.markdown("#### 🔮 Simulator Prediksi")
        st.write("Masukkan nilai fitur untuk memprediksi target:")
        # Ambil nama-nama fitur
        feature_names = data.feature_names
        input_data = {}
        
        # Membuat input dinamis sesuai jumlah fitur dataset
        cols = st.columns(min(len(feature_names), 4)) 
        for i, feature in enumerate(feature_names):
            with cols[i % len(cols)]:
                input_data[feature] = st.number_input(feature, value=float(data.data[feature].mean()), step=0.1)
                
        if st.button("🔮 Prediksi Sekarang", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"Hasil Prediksi Target: **{prediction:.2f}**")


# ----------------- HALAMAN 3: KLASIFIKASI -----------------
elif menu == "🎯 Klasifikasi":
    st.markdown('<div class="header-title">🎯 Klasifikasi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1: dataset_name = st.selectbox("Pilih Dataset", ["Iris 🌸", "Wine 🍷", "Breast Cancer 🎗️"])
    with col2: model_name = st.selectbox("Pilih Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"])
        
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # --- LIVE TRAINING ---
    data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=test_size, random_state=42, stratify=data.target)
    
    if model_name == "Logistic Regression": model = LogisticRegression(max_iter=1000)
    elif model_name == "KNN": model = KNeighborsClassifier()
    elif model_name == "Decision Tree": model = DecisionTreeClassifier()
    elif model_name == "Random Forest": model = RandomForestClassifier()
    else: model = SVC(probability=True)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("🎓", "Accuracy", f"{accuracy_score(y_test, y_pred):.4f}", "Akurasi keseluruhan.", "#8b5cf6")
    with c2: render_metric_card("🎯", "Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", "Precision.", "#ec4899")
    with c3: render_metric_card("🔍", "Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", "Recall.", "#06b6d4")
    with c4: render_metric_card("⚖️", "F1-Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}", "F1-Score.", "#f59e0b")
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "🧩 Confusion Matrix", "🔮 Simulator Prediksi"])
    
    with tab1:
        st.write("**Data Preview:**")
        st.dataframe(data.frame, use_container_width=True, height=400)
        
    with tab2:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Purples", cbar=False,
                    xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
        st.pyplot(fig)
        
    with tab3:
        st.markdown("#### 🔮 Simulator Prediksi")
        feature_names = data.feature_names
        input_data = {}
        cols = st.columns(min(len(feature_names), 4)) 
        for i, feature in enumerate(feature_names):
            with cols[i % len(cols)]:
                input_data[feature] = st.number_input(feature, value=float(data.data[feature].mean()), step=0.1)
                
        if st.button("🔮 Prediksi Sekarang", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            pred_class = data.target_names[prediction]
            st.success(f"Hasil Prediksi Kelas: **{pred_class}**")


# ----------------- HALAMAN 4: CLUSTERING -----------------
elif menu == "🔵 Clustering":
    st.markdown('<div class="header-title">🔵 Clustering K-Means</div>', unsafe_allow_html=True)
    
    colA, colB = st.columns([2, 1])
    with colA: dataset_name = st.selectbox("Pilih Dataset", ["Iris 🌸", "Wine 🍷", "Breast Cancer 🎗️"])
        
    k = st.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3, step=1)
    
    data = load_dataset(dataset_name)
    X_scaled = StandardScaler().fit_transform(data.data)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("📐", "Silhouette", f"{silhouette_score(X_scaled, labels):.4f}", "Semakin dekat 1 semakin baik.", "#8b5cf6")
    with c2: render_metric_card("📊", "Davies-Bouldin", f"{davies_bouldin_score(X_scaled, labels):.4f}", "Semakin dekat 0 semakin baik.", "#06b6d4")
    with c3: render_metric_card("🔄", "Inertia", f"{kmeans.inertia_:.4f}", "Jumlah kuadrat jarak ke centroid.", "#f59e0b")
    
    tab1, tab2 = st.tabs(["📊 Dataset & Hasil Cluster", "📈 Visualisasi Grafik"])
    
    with tab1:
        df_results = data.frame.copy()
        df_results["Cluster"] = labels
        st.dataframe(df_results, use_container_width=True, height=400)
        
    with tab2:
        st.markdown("#### Visualisasi Cluster (Proyeksi PCA 2D)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="Purples", alpha=0.8, edgecolors="gray", s=80)
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c="#ec4899", marker="X", s=200, label="Centroid", edgecolors="#be185d")
        ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
        ax.legend()
        st.pyplot(fig)


# ----------------- HALAMAN 5: DATABASE -----------------
elif menu == "🗄️ Database & Retraining":
    st.markdown('<div class="header-title">🗄️ Database & Retraining</div>', unsafe_allow_html=True)
    if err or client is None:
        st.error(err or "❌ Tidak dapat terhubung ke MongoDB.")
    else:
        DB_NAME, COLL_NAME = "test_db", "iris_db"
        total_docs = count_docs(client, DB_NAME, COLL_NAME)
        st.markdown(f"### 📦 Total Data Iris di Database: **{total_docs}**")
        
        col_form, col_table = st.columns([1, 2])
        with col_form:
            st.markdown("#### ➕ Insert Data Iris")
            with st.form("form_insert", clear_on_submit=True):
                sl = st.number_input("Sepal Length", value=5.1, step=0.1)
                sw = st.number_input("Sepal Width", value=3.5, step=0.1)
                pl = st.number_input("Petal Length", value=1.4, step=0.1)
                pw = st.number_input("Petal Width", value=0.2, step=0.1)
                sp = st.selectbox("Species", ["setosa", "versicolor", "virginica"])
                if st.form_submit_button("💾 Simpan ke MongoDB", use_container_width=True):
                    client[DB_NAME][COLL_NAME].insert_one({
                        "sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw,
                        "species": sp, "timestamp": datetime.utcnow()
                    })
                    st.cache_data.clear()
                    st.rerun()
                    
        with col_table:
            st.markdown("#### 📋 Data Tersimpan")
            docs = fetch_data(client, DB_NAME, COLL_NAME)
            if docs:
                df = pd.DataFrame(docs)
                df["_id"] = df["_id"].astype(str)
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                del_id = st.selectbox("Pilih _id untuk dihapus", df["_id"].tolist())
                if st.button("🗑️ Hapus Data", type="secondary"):
                    client[DB_NAME][COLL_NAME].delete_one({"_id": ObjectId(del_id)})
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.info("Belum ada data.")
                
        st.markdown("---")
        if st.button("🔁 Retrain Sekarang", use_container_width=True, type="primary"):
            iris = datasets.load_iris(as_frame=True)
            df_sklearn = iris.frame.rename(columns={
                "sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length", "petal width (cm)": "petal_width", "target": "species_int"
            })
            df_sklearn["species"] = df_sklearn["species_int"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
            df_sklearn = df_sklearn[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
            
            if docs:
                df_mongo = pd.DataFrame(docs)[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]]
                df_all = pd.concat([df_sklearn, df_mongo], ignore_index=True)
            else:
                df_all = df_sklearn
                
            df_all["species_int"] = df_all["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
            X = df_all[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
            y = df_all["species_int"].values
            
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_tr, y_tr)
            y_pr = rf.predict(X_te)
            
            st.success(f"Retrain Selesai! Total data: {len(df_all)}")
            render_metric_card("🏆", "Akurasi Model Baru", f"{accuracy_score(y_te, y_pr):.4f}", "Akurasi model setelah retrain.", "#f59e0b")


# ----------------- HALAMAN 6: KOMENTAR -----------------
elif menu == "💬 Diskusi & Komentar":
    st.markdown('<div class="header-title">💬 Diskusi & Komentar</div>', unsafe_allow_html=True)
    if err or client is None:
        st.error(err or "❌ Tidak dapat terhubung ke MongoDB.")
    else:
        DB_NAME, COMMENTS_COLL = "test_db", "comments_db"
        
        with st.form("form_comment", clear_on_submit=True):
            st.markdown("#### ✍️ Tulis Komentar Baru")
            name = st.text_input("Nama Anda")
            comment = st.text_area("Komentar / Masukan")
            submitted = st.form_submit_button("Kirim Komentar 🚀", use_container_width=True)
            
            if submitted:
                if not name or not comment:
                    st.warning("⚠️ Nama dan Komentar tidak boleh kosong!")
                else:
                    client[DB_NAME][COMMENTS_COLL].insert_one({"name": name, "comment": comment, "timestamp": datetime.utcnow()})
                    st.cache_data.clear()
                    st.success("✅ Komentar berhasil dikirim!")
                    st.rerun()
        
        st.markdown("---")
        st.markdown("#### 🗣️ Komentar Pengunjung Lain")
        comments = list(client[DB_NAME][COMMENTS_COLL].find().sort("timestamp", -1))
        
        if comments:
            for c in comments:
                time_str = c.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"""
                <div style='background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                        <strong style='color: #8b5cf6;'>👤 {c['name']}</strong>
                        <span style='font-size: 12px; color: #94a3b8;'>🕒 {time_str}</span>
                    </div>
                    <p style='margin: 0; color: #334155;'>{c['comment']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada komentar. Jadilah yang pertama!")