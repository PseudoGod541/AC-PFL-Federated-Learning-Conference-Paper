import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import load_cmapss_data

def visualize_clusters():
    """
    Loads engine data and cluster assignments, performs PCA, and
    creates a scatter plot to visualize the clusters.
    """
    # --- 1. Load Data ---
    DATA_FILE = 'train_FD004.txt'
    CLUSTER_MAP_FILE = 'unit_to_cluster_map.json'

    if not os.path.exists(DATA_FILE) or not os.path.exists(CLUSTER_MAP_FILE):
        print(f"🚨 Error: Make sure '{DATA_FILE}' and '{CLUSTER_MAP_FILE}' are in your project folder.")
        print("   > You may need to run 'cluster_precompute.py' first.")
        return

    print("   > Loading data and cluster map...")
    df = load_cmapss_data(DATA_FILE)
    with open(CLUSTER_MAP_FILE, 'r') as f:
        unit_to_cluster_map = json.load(f)

    # --- 2. Recreate Feature Vectors (same as in clustering) ---
    print("   > Processing features for PCA...")
    sensor_cols = [col for col in df.columns if col.startswith('s')]
    engine_features = df.groupby('unit_id')[sensor_cols].mean()

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engine_features)

    # --- 3. Perform PCA ---
    print("   > Performing PCA to reduce to 2 dimensions...")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    # --- 4. Prepare DataFrame for Plotting ---
    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['Principal Component 1', 'Principal Component 2']
    )
    pca_df['unit_id'] = engine_features.index
    pca_df['Cluster'] = pca_df['unit_id'].astype(str).map(unit_to_cluster_map)

    # --- 5. Generate and Save Plot ---
    print("   > Generating plot...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        x="Principal Component 1", y="Principal Component 2",
        hue="Cluster",
        palette=sns.color_palette("viridis", n_colors=len(pca_df['Cluster'].unique())),
        data=pca_df,
        legend="full",
        alpha=0.9,
        s=100 # Adjust size of points
    )

    plt.title('Engine Clusters Visualized with PCA', fontsize=16, weight='bold')
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Explain the variance captured by the components
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    plt.figtext(0.5, 0.01, f'The first two principal components capture {explained_variance:.2f}% of the variance.', 
                ha='center', fontsize=10, style='italic')

    # Save the plot
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("\n✅ Plot saved to cluster_visualization.png")


if __name__ == "__main__":
    visualize_clusters()