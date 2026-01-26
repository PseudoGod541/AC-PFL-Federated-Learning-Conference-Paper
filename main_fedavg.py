#main_fedavg
import argparse
import os
import warnings
import json
from typing import List, Tuple
import joblib

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from flwr.common import Metrics
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
NUM_CLIENTS = 4
NUM_ROUNDS = 15

# --- Enhanced Model Definition ---
def create_keras_model(input_shape):
    """Creates a robust LSTM model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- Flower Client Definition ---
class CmapssClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val):
        self.model, self.x_train, self.y_train, self.x_val, self.y_val = model, x_train, y_train, x_val, y_val
    def get_parameters(self, config): return self.model.get_weights()
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        self.model.fit(self.x_train, self.y_train, epochs=20, validation_data=(self.x_val, self.y_val), callbacks=[early_stopping], batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}
    def evaluate(self, parameters, config):
        loss, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return loss, len(self.x_val), {"mae": mae}

# --- Server & Client Logic ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    maes = [num * m["mae"] for num, m in metrics]
    total_examples = sum(num for num, _ in metrics)
    return {"mae": sum(maes) / total_examples} if total_examples > 0 else {}

def run_server():
    """Defines and starts the FedAvg server."""
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), strategy=strategy)

def run_client(cid: int, seed: int):
    """Loads data partition, creates model, and starts the client."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    df = load_cmapss_data(DATA_FILE)
    df = calculate_rul(df)
    for col in ALL_FEATURE_COLS:
        if col not in df.columns: df[col] = 0
    df = df[['unit_id', 'cycle', 'RUL'] + ALL_FEATURE_COLS]
    
    try:
        scaler = joblib.load('global_scaler.pkl')
    except FileNotFoundError:
        print("🚨 Error: global_scaler.pkl not found. Please run precompute_scaler.py first.")
        exit()
    
    df[ALL_FEATURE_COLS] = scaler.transform(df[ALL_FEATURE_COLS])
    
    all_unit_ids = sorted(df['unit_id'].unique())
    client_partitions = np.array_split(all_unit_ids, NUM_CLIENTS)
    client_unit_ids = client_partitions[cid]
    
    client_df = df[df['unit_id'].isin(client_unit_ids)]
    X, y = create_sequences(client_df, ALL_FEATURE_COLS, 30)
    if len(X) < 2: 
        print(f"Client {cid} has insufficient data. Skipping."); return
        
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    if len(x_train) == 0 or len(x_val) == 0:
        print(f"Client {cid} has empty train/val split. Skipping."); return
    
    model = create_keras_model((X.shape[1], X.shape[2]))
    flower_client = CmapssClient(model, x_train, y_train, x_val, y_val)
    fl.client.start_client(server_address="127.0.0.1:8080", client=flower_client.to_client())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FedAvg Simulation")
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client"])
    parser.add_argument("--cid", type=int, help="Client ID (for client mode).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.mode == "server":
        run_server()
    elif args.mode == "client":
        if args.cid is None: raise ValueError("Client ID (--cid) is required for client mode.")
        run_client(args.cid, args.seed)
