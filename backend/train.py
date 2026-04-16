"""
NEON-FORECAST AI
================
File: backend/train.py
Phase 3 — Train LSTM model on real pharma sales data

What this does:
  1. Loads tensors from data/processed/ (created in Phase 2)
  2. Trains one LSTM model per drug (8 models total)
  3. Saves each trained model to models/saved/<DRUG>.pth
  4. Prints accuracy (MAE) for each drug

Run: python backend/train.py
Expected time: ~2-5 minutes on CPU
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

# Add project root to path so we can import model.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model import NeonForecastLSTM

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = os.path.join("models", "saved")

DRUG_COLS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]

# Training hyperparameters
EPOCHS      = 50          # number of full passes through training data
BATCH_SIZE  = 32          # samples per gradient update
LEARNING_RATE = 0.001     # Adam optimizer step size
HIDDEN_SIZE = 64          # LSTM memory units
NUM_LAYERS  = 2           # stacked LSTM layers
DROPOUT     = 0.2         # regularization

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Load scalers (needed to convert MAE back to real units)
# ─────────────────────────────────────────────────────────────────────────────
def load_scalers() -> dict:
    scaler_path = os.path.join(PROCESSED_DIR, "scalers.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Train one model for one drug
# ─────────────────────────────────────────────────────────────────────────────
def train_one_drug(drug_code: str, scalers: dict) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training: {drug_code}")
    print(f"{'='*55}")

    # ── Load tensors ──────────────────────────────────────────────────────────
    tensor_path = os.path.join(PROCESSED_DIR, f"{drug_code}_tensors.pt")
    data        = torch.load(tensor_path, weights_only=True)

    X_train = data["X_train"].to(DEVICE)   # (1673, 14, 1)
    y_train = data["y_train"].to(DEVICE)   # (1673,)
    X_test  = data["X_test"].to(DEVICE)    # (419, 14, 1)
    y_test  = data["y_test"].to(DEVICE)    # (419,)

    # Reshape y to (N, 1) for loss function
    y_train = y_train.unsqueeze(1)
    y_test  = y_test.unsqueeze(1)

    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test samples  : {X_test.shape[0]}")
    print(f"  Device        : {DEVICE}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_dataset = TensorDataset(X_train, y_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ── Model, Loss, Optimizer ────────────────────────────────────────────────
    model = NeonForecastLSTM(
        input_size  = 1,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
        output_size = 1,
    ).to(DEVICE)

    criterion = nn.MSELoss()                          # Mean Squared Error
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler — reduces LR when loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_model_state = None
    history = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'MAE (units)':>12}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*12}")

    for epoch in range(1, EPOCHS + 1):

        # ── Train mode ────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss  = criterion(val_preds, y_test).item()

            # Convert MAE back to real units using scaler
            val_preds_np = val_preds.cpu().numpy()
            y_test_np    = y_test.cpu().numpy()
            scaler       = scalers[drug_code]

            real_preds   = scaler.inverse_transform(val_preds_np)
            real_actual  = scaler.inverse_transform(y_test_np)
            mae_units    = np.mean(np.abs(real_preds - real_actual))

        train_loss_avg = np.mean(train_losses)
        scheduler.step(val_loss)
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss_avg,
            "val_loss":   val_loss,
            "mae_units":  mae_units,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = model.state_dict().copy()
            best_mae         = mae_units

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss_avg:>12.6f}  "
                  f"{val_loss:>10.6f}  {mae_units:>10.2f} u")

    # ── Save best model ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{drug_code}.pth")

    torch.save({
        "model_state_dict": best_model_state,
        "hyperparameters": {
            "input_size":   1,
            "hidden_size":  HIDDEN_SIZE,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "output_size":  1,
            "seq_length":   14,
        },
        "drug_code":  drug_code,
        "best_mae":   best_mae,
        "val_loss":   best_val_loss,
    }, save_path)

    print(f"\n  Best MAE    : {best_mae:.2f} units")
    print(f"  Model saved : {save_path}")

    return {
        "drug":     drug_code,
        "best_mae": best_mae,
        "val_loss": best_val_loss,
        "saved_to": save_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train all 8 drugs
# ─────────────────────────────────────────────────────────────────────────────
def train_all():
    print("=" * 55)
    print("  NEON-FORECAST AI — Model Training")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
    print("=" * 55)

    scalers = load_scalers()
    results = []

    for drug in DRUG_COLS:
        result = train_one_drug(drug, scalers)
        results.append(result)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  TRAINING COMPLETE — Final Results")
    print(f"{'='*55}")
    print(f"  {'Drug':<8}  {'MAE (units)':>12}  {'Status':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}")

    for r in results:
        status = "EXCELLENT" if r["best_mae"] < 3 else "GOOD" if r["best_mae"] < 7 else "OK"
        print(f"  {r['drug']:<8}  {r['best_mae']:>10.2f} u  {status:>10}")

    avg_mae = np.mean([r["best_mae"] for r in results])
    print(f"\n  Average MAE : {avg_mae:.2f} units")
    print(f"  Models in   : {MODELS_DIR}/")
    print()
    print("  Ready for Phase 4 — FastAPI Backend")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_all()