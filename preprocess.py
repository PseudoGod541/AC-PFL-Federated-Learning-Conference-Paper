#preprocess.py
import numpy as np
import pandas as pd
from typing import Tuple, List

def calculate_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Calculates the Remaining Useful Life (RUL) for each engine.
    Uses a piecewise linear degradation model and includes input validation.

    Args:
        df (pd.DataFrame): The input dataframe.
        rul_cap (int): The maximum RUL value to cap at. Defaults to 125.

    Returns:
        pd.DataFrame: DataFrame with an added 'RUL' column.
    """
    # --- Input validation ---
    if df.empty:
        raise ValueError("Input DataFrame for calculate_rul is empty.")
    required_cols = ['unit_id', 'cycle']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame is missing required columns: {required_cols}")
    
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    
    # Cap RUL at the specified value
    df['RUL'] = df['RUL'].clip(upper=rul_cap)
    return df

def create_sequences(df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the data into sequences for time-series prediction.
    Includes input validation and handles units with insufficient data.
    
    Returns:
        A tuple of (X, y) numpy arrays.
    """
    # --- Input validation ---
    if df.empty:
        raise ValueError("Input DataFrame for create_sequences is empty.")
    if not all(col in df.columns for col in feature_cols + ['RUL']):
        raise ValueError("Input DataFrame is missing required feature or RUL columns.")

    X, y = [], []
    skipped_units = 0
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        
        # Skip units with fewer cycles than the sequence length
        if len(unit_data) < sequence_length:
            skipped_units += 1
            continue
            
        for i in range(len(unit_data) - sequence_length + 1):
            X.append(unit_data[feature_cols].iloc[i:i+sequence_length].values)
            y.append(unit_data['RUL'].iloc[i+sequence_length-1])
            
    if skipped_units > 0:
        print(f"⚠️  Skipped {skipped_units} units with insufficient data for sequence creation.")
    
    # --- Validation for empty results ---
    if not X:
        raise ValueError("No sequences could be created. Check data and sequence_length parameter.")

    X_arr, y_arr = np.array(X), np.array(y)
    
    print(f"✅ Created sequences: X shape = {X_arr.shape}, y shape = {y_arr.shape}")
            
    return X_arr, y_arr