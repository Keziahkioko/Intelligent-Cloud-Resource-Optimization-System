import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
# Use the absolute path for the Lightning AI Studio root directory
clean_data_path = '/teamspace/studios/this_studio/cloud_performance_features.csv'

if os.path.exists(clean_data_path):
    print("Loading dataset directly from THIS_STUDIO root...")
    df = pd.read_csv(clean_data_path)
else:
    # A quick fallback just in case the absolute path shifts slightly
    print("Trying relative parent path...")
    df = pd.read_csv('../cloud_performance_features.csv')

# Force pandas to only keep numeric columns (strips out string IDs)
df = df.select_dtypes(include=[np.number]).dropna()

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ==========================================
# 2. SWEEPS FOR OPTIMAL K (ELBOW & SILHOUETTE)
# ==========================================
print("Running K-Means sweeps to find optimal K...")
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    print(f" -> Calculating for K={k}...") # Added this so you can track progress!
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # Fast Silhouette Score using a random sample of 10,000 rows
    score = silhouette_score(X_scaled, labels, sample_size=10000, random_state=42)
    silhouette_scores.append(score)

print("Sweeps complete! Generating plots...")

# Plot Elbow and Silhouette curves side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Plot
axes[0].plot(K_range, inertia, marker='o', linestyle='--')
axes[0].set_title('Elbow Method (Inertia)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')

# Silhouette Plot
axes[1].plot(K_range, silhouette_scores, marker='s', linestyle='--', color='orange')
axes[1].set_title('Silhouette Score Method')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('kmeans_sweeps.png')
# plt.show() # Kept commented out!

# ==========================================
# 3. TRAIN FINAL MODELS & EVALUATE
# ==========================================
# Assuming optimal K is 4 based on typical sweeps (adjust based on your actual plot!)
optimal_k = 4 
print(f"\nTraining final models with K={optimal_k}...")

# Model 1: KMeans
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Model 2: DBSCAN
print("Training DBSCAN (utilizing all 32 CPU cores)...")
# Using n_jobs=-1 to explicitly parallelize DBSCAN across your 32 cores
dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Calculate metrics for KMeans
metrics = {
    "Silhouette Score": silhouette_score(X_scaled, kmeans_labels, sample_size=10000, random_state=42),
    "Davies-Bouldin Index": davies_bouldin_score(X_scaled, kmeans_labels),
    "Calinski-Harabasz Index": calinski_harabasz_score(X_scaled, kmeans_labels)
}

print("\n--- KMeans Evaluation Metrics ---")
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")

# ==========================================
# 4. 2D PCA VISUALIZATION
# ==========================================
print("\nApplying PCA for 2D visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=kmeans_labels, 
    palette='Set1', 
    alpha=0.7
)

variance_explained = pca.explained_variance_ratio_ * 100
plt.title(f'2D PCA of Cloud Performance (K={optimal_k})')
plt.xlabel(f'Principal Component 1 ({variance_explained[0]:.1f}% variance)')
plt.ylabel(f'Principal Component 2 ({variance_explained[1]:.1f}% variance)')
plt.legend(title='Cluster')
plt.savefig('pca_clusters_2d.png')
# plt.show() # Kept commented out!

print("\nPipeline complete! Visualizations saved as PNG files.")