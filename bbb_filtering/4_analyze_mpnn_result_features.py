import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Create output directory for figures
output_dir = r"C:\Users\ayamin\OneDrive\Desktop\post_all\figures_v2"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
file_path = r"C:\Users\ayamin\OneDrive\Desktop\post_all\predictions_filtered_3d_analysis.csv"
df = pd.read_csv(file_path)

# Drop non-numeric columns
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

range_clusters = range(2, 11)

# Compute silhouette scores to determine best k
silhouette_scores = []
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(range_clusters, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different k")
plt.savefig(os.path.join(output_dir, "silhouette_scores.png"))
plt.show()

optimal_k = 3 # Optimal k from silhouette score graph
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)
df["KMeans_Cluster"] = kmeans_labels

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df["PCA1"], df["PCA2"] = pca_result[:, 0], pca_result[:, 1]

# t-SNE for Visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)
df["TSNE1"], df["TSNE2"] = tsne_result[:, 0], tsne_result[:, 1]

# Visualize Molecular Properties per Cluster
properties = ["MW", "LogP", "TPSA", "HBA", "HBD", "RotBonds", "3D_GyrationRadius", "3D_SpherocityIndex"]
for prop in properties:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df["KMeans_Cluster"], y=df[prop])
    plt.title(f"Distribution of {prop} Across Clusters")
    plt.savefig(os.path.join(output_dir, f"{prop}_per_cluster.png"))
    plt.show()

# Visualize Scaffold Distribution per Cluster
scaffold_counts_per_cluster = df.groupby("KMeans_Cluster")["Scaffold"].value_counts().unstack().fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(scaffold_counts_per_cluster, cmap="coolwarm", linewidths=0.5)
plt.title("Scaffold Distribution Across Clusters")
plt.xlabel("Scaffold")
plt.ylabel("Cluster")
plt.savefig(os.path.join(output_dir, "scaffold_distribution_per_cluster.png"))
plt.show()

# Save scaffold analysis
df.to_csv("analyzed_molecules_with_clusters_and_scaffolds.csv", index=False)
print("Analysis complete")
