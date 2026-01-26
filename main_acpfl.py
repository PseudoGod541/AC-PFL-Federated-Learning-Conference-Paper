#main_acpfl
import argparse
import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
from collections import Counter
import joblib

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes, Metrics, Parameters,
                         Scalar, ndarrays_to_parameters, parameters_to_ndarrays)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocess import calculate_rul, create_sequences
from utils import load_cmapss_data

# --- Config ---
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

ALL_FEATURE_COLS = ['setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
DATA_FILE = 'train_FD004.txt'
NUM_FEATURES = len(ALL_FEATURE_COLS)
NUM_CLUSTERS = 2
NUM_CLIENTS = 4
NUM_ROUNDS = 15

# --- JSON Helper ---
def load_json(path: str) -> dict:
    try:
        with open(path, "r") as f: return json.load(f)
    except FileNotFoundError:
        print(f"🚨 Error: Config file not found at {path}. Please run setup scripts first."); exit()

# --- Model, Client, and Strategy classes ---
def create_keras_model(input_shape):
    base_model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dropout(0.3), tf.keras.layers.LSTM(32), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(32, activation='relu')])
    head_model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(1)])
    inputs = tf.keras.Input(shape=input_shape)
    base_output = base_model(inputs)
    predictions = head_model(base_output)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, base_model, head_model
class CmapssClient(fl.client.NumPyClient):
    def __init__(self, model, base_model, head_model, x_train, y_train, x_val, y_val, cluster_id):
        self.model, self.base_model, self.head_model = model, base_model, head_model
        self.x_train, self.y_train, self.x_val, self.y_val = x_train, y_train, x_val, y_val
        self.cluster_id = cluster_id
    def get_parameters(self, config): return self.base_model.get_weights()
    def fit(self, parameters, config):
        self.base_model.set_weights(parameters)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        self.model.fit(self.x_train, self.y_train, epochs=20, validation_data=(self.x_val, self.y_val), callbacks=[early_stopping], batch_size=32, verbose=0)
        return self.base_model.get_weights(), len(self.x_train), {"cluster_id": self.cluster_id}
    def evaluate(self, parameters, config):
        self.base_model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"mae": mae, "cluster_id": self.cluster_id}
class ACStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_clusters: int, client_to_cluster_map: Dict[str, int], initial_parameters: Parameters, **kwargs):
        super().__init__(**kwargs)
        self.client_to_cluster_map, self.num_clusters = client_to_cluster_map, num_clusters
        self.cluster_parameters: List[Parameters] = [initial_parameters for _ in range(num_clusters)]
        print(f"ACStrategy initialized for {num_clusters} clusters.")
    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}
        print(f"\n[Round {server_round}] Received {len(results)} updates")
        for client, fit_res in results: print(f"  - Client {client.cid} (Cluster {fit_res.metrics['cluster_id']}) contributed {fit_res.num_examples} examples")
        updates = {i: [] for i in range(self.num_clusters)}
        for _, fit_res in results: updates[fit_res.metrics["cluster_id"]].append((fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters)))
        for cid, up in updates.items():
            if not up: continue
            total_ex = sum(n for n, _ in up)
            weighted_w = [[np.array(l) * n for l in w] for n, w in up]
            w_prime = [sum(l) / total_ex for l in zip(*weighted_w)]
            self.cluster_parameters[cid] = ndarrays_to_parameters(w_prime)
            print(f"  - Aggregated model for cluster {cid}")
        return None, {}
    def aggregate_evaluate(self, server_round, results, failures):
        if not results: return None, {}
        maes = {i: [] for i in range(self.num_clusters)}
        for _, res in results: maes[res.metrics["cluster_id"]].append((res.num_examples, res.metrics["mae"]))
        print(f"\n[Evaluation Round {server_round}]")
        all_maes = []
        for cid, met in maes.items():
            if not met: print(f"  - Cluster {cid}: No evaluation results."); continue
            total_ex, w_mae_sum = sum(n for n, _ in met), sum(n * m for n, m in met)
            mae = w_mae_sum / total_ex
            print(f"  - Cluster {cid} MAE: {mae:.6f}")
            all_maes.append(mae)
        avg_mae = sum(all_maes) / len(all_maes) if all_maes else 0
        print(f"  - Overall Average MAE: {avg_mae:.6f}")
        return None, {"mae": avg_mae}
    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(self.min_evaluate_clients, client_manager.num_available())
        return [(c, EvaluateIns(self.cluster_parameters[self.client_to_cluster_map.get(c.cid, 0)], {})) for c in clients]
    def configure_fit(self, server_round, parameters, client_manager):
        ss, mc = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(ss, mc)
        return [(c, FitIns(self.cluster_parameters[self.client_to_cluster_map.get(c.cid, 0)], {})) for c in clients]

def run_server():
    client_to_cluster_map = load_json("client_to_cluster_map.json")
    print(f"Server: Loaded Client-to-Cluster Map: {client_to_cluster_map}")
    _, base_model, _ = create_keras_model(input_shape=(30, NUM_FEATURES))
    initial_parameters = ndarrays_to_parameters(base_model.get_weights())
    strategy = ACStrategy(
        num_clusters=NUM_CLUSTERS, client_to_cluster_map=client_to_cluster_map,
        initial_parameters=initial_parameters, fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS
    )
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strategy)

def run_client(cid: int, seed: int):
    client_partitions = load_json("client_partitions.json")
    client_to_cluster_map = load_json("client_to_cluster_map.json")
    if cid >= len(client_partitions): raise ValueError(f"Client ID {cid} is out of bounds.")
    client_unit_ids = client_partitions[cid]
    if not client_unit_ids: return
    cluster_id = client_to_cluster_map[str(cid)]
    print(f"Client {cid} (units {client_unit_ids}) assigned to Cluster {cluster_id}")
    
    df = load_cmapss_data(DATA_FILE)
    df = calculate_rul(df)
    for col in ALL_FEATURE_COLS:
        if col not in df.columns: df[col] = 0
    df = df[['unit_id', 'cycle', 'RUL'] + ALL_FEATURE_COLS]
    
    try: scaler = joblib.load('global_scaler.pkl')
    except FileNotFoundError: print("🚨 Error: global_scaler.pkl not found."); exit()
    
    client_df = df[df['unit_id'].isin(client_unit_ids)].copy()
    client_df[ALL_FEATURE_COLS] = scaler.transform(client_df[ALL_FEATURE_COLS])
    
    X, y = create_sequences(client_df, ALL_FEATURE_COLS, 30)
    if len(X) < 2: return
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    if len(x_train) == 0 or len(x_val) == 0: return
    model, base, head = create_keras_model((X.shape[1], X.shape[2]))
    client = CmapssClient(model, base, head, x_train, y_train, x_val, y_val, cluster_id)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC-PFL Simulation")
    parser.add_argument("--mode", required=True, choices=["server", "client"])
    parser.add_argument("--cid", type=int, help="Client ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    if args.mode == "server":
        run_server()
    elif args.mode == "client" and args.cid is not None:
        run_client(args.cid, args.seed)
