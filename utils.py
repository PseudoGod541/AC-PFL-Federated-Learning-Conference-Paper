#utils.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import numpy as np

def load_cmapss_data(file_path: str) -> pd.DataFrame:
    """
    Load a single CMAPSS dataset file.
    Uses sep='\s+' for robust parsing of whitespace.
    """
    column_names = [
        'unit_id', 'cycle', 'setting1', 'setting2', 'setting3',
        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
        's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
        's20', 's21'
    ]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)
    df = df.dropna(axis=1, how='all')
    return df

def get_engine_clusters(df: pd.DataFrame, num_clusters: int) -> dict:
    """
    Clusters engines based on their mean sensor values after scaling.
    This is a critical step for the AC-PFL framework.
    """
    sensor_cols = [col for col in df.columns if col.startswith('s')]
    engine_features = df.groupby('unit_id')[sensor_cols].mean()
    
    if len(engine_features) < num_clusters:
        raise ValueError(f"Number of unique engines ({len(engine_features)}) is less than the number of clusters ({num_clusters}).")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(engine_features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # --- RECOMMENDED ADDITION: Validate the output of KMeans ---
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < num_clusters:
        print(f"⚠️ Warning: K-Means produced only {len(unique_clusters)} clusters, not the {num_clusters} requested.")
    # --- END ADDITION ---
    
    print("\n-- K-Means Clustering Results --")
    cluster_counts = Counter(clusters)
    # --- RECOMMENDED CHANGE: Loop over requested clusters for a complete log ---
    for cid in range(num_clusters):
        count = cluster_counts.get(cid, 0)
        print(f"  - Cluster {cid} assigned {count} engines.")
    # --- END CHANGE ---
    print("---------------------------------")
 
    return {str(int(unit_id)): int(cluster) for unit_id, cluster in zip(engine_features.index, clusters)}

def create_hard_partitions(unit_to_cluster_map: dict, num_clients: int, num_clusters: int) -> tuple:
    """
    Creates hard partitions, dedicating clients to specific clusters to ensure balance.
    """
    if num_clients < num_clusters:
        raise ValueError(f"Number of clients ({num_clients}) must be >= number of clusters ({num_clusters}) for hard partitioning.")

    cluster_to_units = defaultdict(list)
    for unit_id_str, cluster_id in unit_to_cluster_map.items():
        cluster_to_units[cluster_id].append(int(unit_id_str))

    # Determine how many clients are dedicated to each cluster
    clients_per_cluster = [0] * num_clusters
    for i in range(num_clients):
        clients_per_cluster[i % num_clusters] += 1
    
    client_partitions = [[] for _ in range(num_clients)]
    client_to_cluster_map = {}
    
    client_idx = 0
    # Iterate through clusters and assign their units to the dedicated clients
    for cluster_id in range(num_clusters):
        num_dedicated_clients = clients_per_cluster[cluster_id]
        
        # Skip if a cluster has no units or no dedicated clients
        if num_dedicated_clients == 0 or not cluster_to_units.get(cluster_id):
            continue
            
        # Split the units of this cluster among its dedicated clients
        unit_partitions = np.array_split(cluster_to_units[cluster_id], num_dedicated_clients)
        
        for part in unit_partitions:
            # Ensure the partition is not empty before assigning
            if part.size > 0 and client_idx < num_clients:
                client_partitions[client_idx] = part.tolist()
                client_to_cluster_map[str(client_idx)] = cluster_id
                client_idx += 1

    return client_partitions, client_to_cluster_map
