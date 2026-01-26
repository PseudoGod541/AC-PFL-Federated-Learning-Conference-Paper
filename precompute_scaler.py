#precompute_scaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from utils import load_cmapss_data

ALL_FEATURE_COLS = ['setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
DATA_FILE = 'train_FD004.txt'

if __name__ == "__main__":
    print("Pre-computing and saving the global scaler...")
    df = load_cmapss_data(DATA_FILE)
    for col in ALL_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[ALL_FEATURE_COLS]
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, 'global_scaler.pkl')
    print("✅ Global scaler saved to global_scaler.pkl")