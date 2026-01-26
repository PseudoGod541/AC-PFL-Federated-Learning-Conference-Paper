import json
import os
from collections import Counter
from utils import load_cmapss_data, get_engine_clusters, create_hard_partitions
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Precomputation")
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters")
    args = parser.parse_args()

    DATA_PATH = 'train_FD004.txt'
    NUM_CLUSTERS = 2
    NUM_CLIENTS = 4

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"🚨 Data file not found at '{DATA_PATH}'.")

    print("Step 1: Loading data...")
    df = load_cmapss_data(DATA_PATH)
    
    print(f"Step 2: Computing {NUM_CLUSTERS} clusters for {df['unit_id'].nunique()} engines...")
    unit_to_cluster_map = get_engine_clusters(df, num_clusters=NUM_CLUSTERS)
    with open('unit_to_cluster_map.json', 'w') as f:
        json.dump(unit_to_cluster_map, f, indent=4, sort_keys=True)
    print("✅ Unit-to-Cluster map saved.")

    print("Step 3: Creating hard-balanced client partitions...")
    client_partitions, client_to_cluster_map = create_hard_partitions(
        unit_to_cluster_map, NUM_CLIENTS, NUM_CLUSTERS
    )
    with open('client_partitions.json', 'w') as f:
        json.dump(client_partitions, f, indent=4)
    print("✅ Balanced client partitions saved.")
    
    with open('client_to_cluster_map.json', 'w') as f:
        json.dump(client_to_cluster_map, f, indent=4, sort_keys=True)
    print("✅ Client-to-Cluster map saved.")

    print("\n📊 Final Configuration Summary:")
    print(f"Client-to-Cluster Assignments: {client_to_cluster_map}")
