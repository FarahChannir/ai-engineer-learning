import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =========================
# 1. Load data
# =========================
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =========================
# 2. Scale data
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. Elbow & Silhouette analysis
# =========================
inertia = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# # Elbow plot
# plt.figure()
# plt.plot(K_range, inertia, marker='o')
# plt.xlabel("Number of clusters (K)")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.grid(True)
# plt.show()

# # Silhouette plot
# plt.figure()
# plt.plot(K_range, silhouette_scores, marker='o')
# plt.xlabel("Number of clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score vs K")
# plt.grid(True)
# plt.show()

# =========================
# 4. Final KMeans model
# =========================
k = 4  # chosen based on elbow + silhouette

# k = K_range[silhouette_scores.index(max(silhouette_scores))]
# print(f"Best K based on silhouette=: {k}")
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels

# =========================
# 5. Cluster summary (business space)
# =========================
cluster_summary = df.groupby("Cluster")[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\nCluster Summary (Mean Values):")
print(cluster_summary)

# =========================
# 6. PCA for visualization
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Business labels derived from summary
cluster_labels = {
    0: "Average Customers",
    1: "High Income / High Spending (VIP)",
    2: "Low Income / High Spending (Deal Seekers)",
    3: "High Income / Low Spending (Potential)"
}

# =========================
# 7. Plot PCA with interpretation
# =========================
plt.figure(figsize=(8, 6))

for cluster_id in np.unique(labels):
    plt.scatter(
        X_pca[labels == cluster_id, 0],
        X_pca[labels == cluster_id, 1],
        label=cluster_labels[cluster_id],
        alpha=0.7
    )

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
for i, center in enumerate(centers_pca):
    plt.text(
        center[0],
        center[1],
        cluster_labels[i],
        fontsize=9,
        fontweight="bold"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using PCA")
plt.legend()
plt.grid(True)
plt.show()
