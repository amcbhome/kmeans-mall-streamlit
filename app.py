import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="K-Means Clustering Lab", layout="wide")

st.title("K-Means Clustering Lab")
st.caption("Upload a CSV (e.g., Kaggle Mall Customers) or use a bundled file. Explore k-means with Elbow & Silhouette.")

# --- Data load
uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_sample = st.checkbox("Use bundled sample file (data/Mall_Customers.csv)", value=False)

@st.cache_data
def load_sample():
    try:
        return pd.read_csv("data/Mall_Customers.csv")
    except Exception as e:
        st.warning("Sample file not found. Please upload a CSV.")
        return None

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = load_sample()
else:
    st.info("Upload a CSV or tick the sample option to continue.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

# --- Basic cleaning
# Offer to drop non-numeric or encode categoricals quickly
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

with st.expander("Optional: quick encoding for categorical columns"):
    to_encode = st.multiselect("Categorical columns to one-hot encode", cat_cols, default=[])
    if to_encode:
        df = pd.get_dummies(df, columns=to_encode, drop_first=True)

st.write(f"Numeric columns detected: {', '.join(df.select_dtypes(include=[np.number]).columns)}")

# --- Feature selection
all_numeric = df.select_dtypes(include=[np.number])
if all_numeric.empty:
    st.error("No numeric features available after encoding. Please choose a different file or encode categorical columns.")
    st.stop()

default_feats = [c for c in all_numeric.columns if c.lower() in ["age","annual income (k$)","spending score (1-100)"]]
features = st.multiselect("Choose features for clustering", all_numeric.columns.tolist(),
                          default=default_feats if default_feats else all_numeric.columns.tolist()[:3])

if len(features) < 2:
    st.warning("Select at least 2 features for PCA visualisation.")
    st.stop()

X = all_numeric[features].copy()

# --- Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- k search
st.subheader("Choose number of clusters (k)")
k_min, k_max = st.slider("k range for diagnostics (Elbow/Silhouette)", 2, 12, (2, 8))
auto_k = st.checkbox("Suggest best k (by max silhouette)", value=True)

inertias, silhouettes, ks = [], [], []
for k in range(k_min, k_max+1):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    try:
        silhouettes.append(silhouette_score(X_scaled, labels))
    except Exception:
        silhouettes.append(np.nan)
    ks.append(k)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Elbow (Inertia vs k)** – look for the bend.")
    st.line_chart(pd.DataFrame({"k": ks, "inertia": inertias}).set_index("k"))
with col2:
    st.markdown("**Silhouette vs k** – higher is better (0–1).")
    st.line_chart(pd.DataFrame({"k": ks, "silhouette": silhouettes}).set_index("k"))

if auto_k and np.isfinite(silhouettes).any():
    k_best = ks[int(np.nanargmax(silhouettes))]
else:
    k_best = st.number_input("Or set k manually", min_value=2, max_value=20, value=3, step=1)

st.success(f"Selected k = {k_best}")

# --- Fit final model
model = KMeans(n_clusters=int(k_best), n_init="auto", random_state=42)
labels = model.fit_predict(X_scaled)
df_out = df.copy()
df_out["cluster"] = labels

# --- PCA for 2D visual
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
pc_df["cluster"] = labels

st.subheader("Cluster scatter (PCA 2-D)")
st.caption("PCA used only for visualisation.")
scatter_data = pc_df.copy()
st.scatter_chart(scatter_data, x="PC1", y="PC2", color="cluster")

# --- Cluster profiles
st.subheader("Cluster profiles (feature means)")
profiles = df_out.groupby("cluster")[features].mean().round(2)
st.dataframe(profiles, use_container_width=True)

# --- Download labelled data
st.subheader("Download results")
buffer = io.BytesIO()
df_out.to_csv(buffer, index=False)
st.download_button("Download labelled CSV", buffer.getvalue(), file_name="kmeans_labeled.csv", mime="text/csv")
