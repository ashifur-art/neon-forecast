"""
NEON-FORECAST AI
================
File: backend/data_engine.py
Phase 2 — Real Data Pipeline
Dataset: Pharma Sales Data (Kaggle - milanzdravkovic)
        salesdaily.csv  →  2,106 rows  |  6 years (2014–2019)

8 Real Drug Categories:
  M01AB → Anti-inflammatory (Acetic acid)
  M01AE → Anti-inflammatory (Propionic acid)
  N02BA → Analgesics / Aspirin family
  N02BE → Analgesics / Paracetamol family  ← highest volume drug
  N05B  → Tranquilizers / Anxiety
  N05C  → Sleeping pills / Sedatives
  R03   → Asthma / Bronchitis drugs
  R06   → Antihistamines / Allergy

Run: python backend/data_engine.py
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle

# ── Config ────────────────────────────────────────────────────────────────────
RAW_CSV       = os.path.join("data", "raw", "salesdaily.csv")
PROCESSED_DIR = os.path.join("data", "processed")
SEQ_LENGTH    = 14        # LSTM looks back 14 days to predict next day
TRAIN_SPLIT   = 0.80      # 80% train, 20% test

# Human-readable names for each drug code
DRUG_NAMES = {
    "M01AB": "Anti_Inflammatory_Acetic",
    "M01AE": "Anti_Inflammatory_Propionic",
    "N02BA": "Analgesic_Aspirin",
    "N02BE": "Analgesic_Paracetamol",
    "N05B":  "Tranquilizer",
    "N05C":  "Sedative",
    "R03":   "Asthma_Bronchitis",
    "R06":   "Antihistamine_Allergy",
}

DRUG_COLS = list(DRUG_NAMES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & Validate
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate(csv_path: str) -> pd.DataFrame:
    print("\n[1/5] Loading CSV...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n  ERROR: File not found at {csv_path}"
            f"\n  Fix: Place salesdaily.csv inside the data/raw/ folder"
        )

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"datum": "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    df = df.sort_values("date").reset_index(drop=True)

    print(f"    Rows       : {len(df):,}")
    print(f"    Date range : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"    Drug cols  : {DRUG_COLS}")
    print(f"    Missing    : {df[DRUG_COLS].isnull().sum().sum()} values")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Clean & Add Features
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Cleaning data...")

    # Fill any date gaps so the time series is continuous every day
    full_range = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )
    df = df.set_index("date").reindex(full_range).reset_index()
    df = df.rename(columns={"index": "date"})

    # Fill missing drug values
    df[DRUG_COLS] = df[DRUG_COLS].ffill().bfill()

    # Remove extreme outliers (beyond 3 standard deviations)
    for col in DRUG_COLS:
        mean = df[col].mean()
        std  = df[col].std()
        df[col] = df[col].clip(lower=0, upper=mean + 3 * std)

    # Time-based features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["is_winter"]   = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"]   = df["month"].isin([6, 7, 8]).astype(int)

    print(f"    Rows after gap-fill : {len(df):,}")
    print(f"    Remaining NaNs      : {df[DRUG_COLS].isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Normalize each drug to 0-1 range
# ─────────────────────────────────────────────────────────────────────────────
def normalize_data(df: pd.DataFrame) -> tuple:
    print("\n[3/5] Normalizing drug sales...")

    scalers = {}
    df_norm = df.copy()

    for col in DRUG_COLS:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = df[[col]].values
        df_norm[f"{col}_norm"] = scaler.fit_transform(values).flatten()
        scalers[col] = scaler
        print(f"    {col}  ({DRUG_NAMES[col]:<30})"
              f"  min={df[col].min():.1f}"
              f"  max={df[col].max():.1f}"
              f"  mean={df[col].mean():.1f}")

    return df_norm, scalers


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build Sliding Window Sequences for LSTM
# ─────────────────────────────────────────────────────────────────────────────
def build_sequences(series: np.ndarray, seq_length: int):
    """
    Example with seq_length=3:
      Input:  [10, 20, 30, 40, 50]
      X[0] = [10, 20, 30]  →  y[0] = 40
      X[1] = [20, 30, 40]  →  y[1] = 50
    """
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i: i + seq_length])
        y.append(series[i + seq_length])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Convert to PyTorch Tensors and Save
# ─────────────────────────────────────────────────────────────────────────────
def create_tensors(df_norm, scalers, seq_length, train_split, out_dir):
    print("\n[4/5] Building sequences and saving tensors...")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for col in DRUG_COLS:
        series = df_norm[f"{col}_norm"].values

        X, y   = build_sequences(series, seq_length)
        X      = X.reshape(X.shape[0], seq_length, 1)

        split      = int(len(X) * train_split)
        X_train    = X[:split]
        X_test     = X[split:]
        y_train    = y[:split]
        y_test     = y[split:]

        tensors = {
            "X_train": torch.tensor(X_train),
            "X_test":  torch.tensor(X_test),
            "y_train": torch.tensor(y_train),
            "y_test":  torch.tensor(y_test),
        }

        save_path = os.path.join(out_dir, f"{col}_tensors.pt")
        torch.save(tensors, save_path)

        summary[col] = {
            "friendly_name": DRUG_NAMES[col],
            "train": len(X_train),
            "test":  len(X_test),
        }

        print(f"    {col}  train={len(X_train):>4}  test={len(X_test):>4}"
              f"  →  saved to {save_path}")

    # Save scalers — needed in Phase 4 to convert 0-1 back to real units
    scaler_path = os.path.join(out_dir, "scalers.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)

    # Save drug name mapping — used by the API
    mapping_path = os.path.join(out_dir, "drug_names.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump(DRUG_NAMES, f)

    print(f"\n    Scalers saved   : {scaler_path}")
    print(f"    Drug map saved  : {mapping_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run everything
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(csv_path: str = RAW_CSV):
    print("=" * 60)
    print("  NEON-FORECAST AI — Data Engineering Pipeline")
    print("  Real Pharma Sales Dataset (2014-2019)")
    print("=" * 60)

    df               = load_and_validate(csv_path)
    df_clean         = clean_data(df)
    df_norm, scalers = normalize_data(df_clean)
    summary          = create_tensors(
                           df_norm, scalers,
                           SEQ_LENGTH, TRAIN_SPLIT,
                           PROCESSED_DIR
                       )

    print("\n[5/5] Pipeline Complete!")
    print("=" * 60)
    print(f"  {'Drug':<8}  {'Name':<32}  {'Train':>5}  {'Test':>5}")
    print(f"  {'-'*8}  {'-'*32}  {'-'*5}  {'-'*5}")
    for code, s in summary.items():
        print(f"  {code:<8}  {s['friendly_name']:<32}  {s['train']:>5}  {s['test']:>5}")
    print()
    print(f"  Sequence length  : {SEQ_LENGTH} days")
    print(f"  Train/Test split : {int(TRAIN_SPLIT*100)}% / {int((1-TRAIN_SPLIT)*100)}%")
    print(f"  Output folder    : {PROCESSED_DIR}/")
    print()
    print("  Ready for Phase 3 — Model Training")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()