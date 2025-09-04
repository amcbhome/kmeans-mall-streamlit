# kmeans-mall-streamlit
# K-Means Clustering Lab (Streamlit)

Practice k-means on Kaggle's **Mall Customers** dataset (or any CSV):
- Feature selection & scaling
- Elbow and Silhouette diagnostics
- PCA 2-D visualisation
- Cluster profiles + export

## Data
Download "Mall_Customers.csv" from Kaggle and place it in `data/`  
Or upload your own CSV directly in the app.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
