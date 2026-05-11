# ============================================
# USARRESTS: PCA + HIERARCHICAL + KMEANS
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# 1. LOAD DATA
# -------------------------------
USArrests = get_rdataset('USArrests').data

print("Columns:\n", USArrests.columns)
print("\nMean:\n", USArrests.mean())
print("\nVariance:\n", USArrests.var())

# -------------------------------
# 2. STANDARDIZE DATA
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(USArrests)

# -------------------------------
# 3. PCA
# -------------------------------
pca = PCA()
scores = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)

# -------------------------------
# 4. PCA BIPLOT (PC1 vs PC2)
# -------------------------------
i, j = 0, 1

plt.figure(figsize=(8, 8))
plt.scatter(scores[:, i], scores[:, j])

for k in range(pca.components_.shape[1]):
    plt.arrow(0, 0,
              pca.components_[i, k],
              pca.components_[j, k],
              color='red')
    plt.text(pca.components_[i, k],
             pca.components_[j, k],
             USArrests.columns[k])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Biplot (USArrests)")
plt.grid()
plt.show()

# -------------------------------
# 4(b). Variance Explained
# -------------------------------
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(8, 5))
plt.plot(explained_var, marker='o', label='Individual Variance')
plt.plot(cumulative_var, marker='s', label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------
# 6. K-MEANS CLUSTERING (WITH CENTROIDS)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k_clusters = kmeans.fit_predict(X_scaled)

# Get centroids (in scaled space)
centroids = kmeans.cluster_centers_

# Convert centroids to PCA space
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))

# Data points
plt.scatter(scores[:, 0], scores[:, 1],
            c=k_clusters, cmap='viridis', label='Data')

# Centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='red', s=200, marker='o', label='Centroids')

plt.title("K-Means Clustering with Centroids")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 7. HIERARCHICAL CLUSTERING
# -------------------------------

# Use complete linkage
Z = linkage(X_scaled, method='complete')

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=USArrests.index)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("States")
plt.ylabel("Distance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# -------------------------------
# 8. AGGLOMERATIVE CLUSTERING
# -------------------------------
agg = AgglomerativeClustering(n_clusters=3, linkage='complete')
a_clusters = agg.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(scores[:, 0], scores[:, 1],
            c=a_clusters, cmap='plasma')
plt.title("Agglomerative Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()

# -------------------------------
# END
# -------------------------------