# 🌸 Iris Flower Clustering & Dimensionality Reduction

An unsupervised machine learning project demonstrating **K-Means Clustering**, **Principal Component Analysis (PCA)**, the **Elbow Method**, and model persistence using `joblib` on the classic Iris dataset.

---

## 📌 Project Overview

This repository contains an end-to-end unsupervised learning pipeline designed to cluster iris flowers based on their physical measurements without using true species labels during training. 

### Key Highlights:
- **Feature Scaling**: Standardized numerical features using `StandardScaler` to ensure balanced distance calculations.
- **Optimal Cluster Selection**: Utilized the **Elbow Method** (Within-Cluster Sum of Squares - WCSS) to determine $K=3$.
- **Dimensionality Reduction**: Applied **PCA** to reduce the 4D feature space to 2D while preserving **~95.8%** of the dataset variance.
- **Cluster Evaluation**: Compared predicted clusters against ground-truth species using contingency tables (`pd.crosstab`) and the Adjusted Rand Index (ARI).
- **Model Persistence**: Serialized trained `KMeans`, `StandardScaler`, and `PCA` objects using `joblib`.

---

## 📁 Repository Structure

```text
├── Iris.csv                     # Dataset file
├── iris_clustering.py           # Main Python script
├── kmeans_iris_model.pkl        # Saved K-Means model
├── scaler_iris.pkl              # Saved StandardScaler object
├── pca_iris.pkl                 # Saved PCA transformer
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
