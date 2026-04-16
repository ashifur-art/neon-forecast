"""
NEON-FORECAST AI
================
File: backend/main.py
Phase 4 - FastAPI Backend Server

Endpoints:
  GET  /                        - Health check
  GET  /drugs                   - List all available drugs
  POST /predict/{drug_code}     - Predict next N days for a drug
  GET  /inventory               - Stock-out risk report for all drugs
  POST /upload                  - Upload a new CSV and re-run pipeline

Run: uvicorn backend.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

import os
import sys
import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import shutil
from contextlib import asynccontextmanager

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.model import NeonForecastLSTM

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR    = os.path.join(ROOT, "models", "saved")
RAW_DATA_DIR  = os.path.join(ROOT, "data", "raw")
SEQ_LENGTH    = 14

DRUG_CODES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]

# ── Global cache ──────────────────────────────────────────────────────────────
loaded_models  = {}
loaded_scalers = {}
drug_names     = {}


# ── Load all models at startup ────────────────────────────────────────────────
def load_all_models():
    global loaded_models, loaded_scalers, drug_names

    print("\n[STARTUP] Loading models and scalers...")

    scaler_path = os.path.join(PROCESSED_DIR, "scalers.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            loaded_scalers = pickle.load(f)
        print(f"  Scalers loaded: {list(loaded_scalers.keys())}")
    else:
        print(f"  WARNING: scalers.pkl not found at {scaler_path}")

    names_path = os.path.join(PROCESSED_DIR, "drug_names.pkl")
    if os.path.exists(names_path):
        with open(names_path, "rb") as f:
            drug_names = pickle.load(f)

    for drug in DRUG_CODES:
        model_path = os.path.join(MODELS_DIR, f"{drug}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            hp         = checkpoint["hyperparameters"]
            model = NeonForecastLSTM(
                input_size  = hp["input_size"],
                hidden_size = hp["hidden_size"],
                num_layers  = hp["num_layers"],
                dropout     = hp["dropout"],
                output_size = hp["output_size"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            loaded_models[drug] = {
                "model":    model,
                "best_mae": checkpoint.get("best_mae", 0),
            }
            print(f"  Loaded: {drug}  (MAE: {checkpoint.get('best_mae', 0):.2f} units)")
        else:
            print(f"  WARNING: Model not found for {drug}")

    print(f"[STARTUP] {len(loaded_models)}/{len(DRUG_CODES)} models ready\n")


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    load_all_models()
    yield


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NEON-FORECAST AI",
    description = "Sales & Demand Prediction API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    days_ahead:    Optional[int]   = 7
    current_stock: Optional[float] = 100.0


class PredictResponse(BaseModel):
    drug_code:           str
    drug_name:           str
    days_ahead:          int
    predictions:         list[float]
    total_predicted:     float
    current_stock:       float
    days_until_stockout: Optional[int]
    reorder_recommended: bool
    confidence:          str


class DrugInfo(BaseModel):
    code:         str
    name:         str
    model_loaded: bool


class InventoryItem(BaseModel):
    drug_code:           str
    drug_name:           str
    current_stock:       float
    avg_daily_demand:    float
    days_until_stockout: int
    risk_level:          str


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_last_sequence(drug_code: str) -> list:
    tensor_path = os.path.join(PROCESSED_DIR, f"{drug_code}_tensors.pt")
    data        = torch.load(tensor_path, map_location="cpu", weights_only=True)
    X_test      = data["X_test"].numpy()
    last_seq    = X_test[-1].tolist()   # shape: (14, 1) -> list of [val]
    return last_seq


def forecast_n_days(drug_code: str, days: int) -> list[float]:
    if drug_code not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model for {drug_code} not loaded. Run train.py first."
        )

    model    = loaded_models[drug_code]["model"]
    scaler   = loaded_scalers[drug_code]
    sequence = get_last_sequence(drug_code)

    predictions_norm = []
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            x    = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            pred = model(x).item()
            predictions_norm.append(pred)
            sequence = sequence[1:] + [[pred]]

    pred_array = np.array(predictions_norm).reshape(-1, 1)
    pred_real  = scaler.inverse_transform(pred_array).flatten()
    pred_real  = np.clip(pred_real, 0, None)

    return [round(float(v), 2) for v in pred_real]


def safe_makedirs(path: str):
    """Windows-safe makedirs — removes stale file if it blocks folder creation."""
    if os.path.isfile(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":        "ONLINE",
        "app":           "NEON-FORECAST AI",
        "version":       "1.0.0",
        "models_loaded": len(loaded_models),
        "drugs":         list(loaded_models.keys()),
    }


@app.get("/drugs", response_model=list[DrugInfo], tags=["Drugs"])
def get_drugs():
    return [
        DrugInfo(
            code         = code,
            name         = drug_names.get(code, code),
            model_loaded = code in loaded_models,
        )
        for code in DRUG_CODES
    ]


@app.post("/predict/{drug_code}", response_model=PredictResponse, tags=["Forecast"])
def predict(drug_code: str, body: PredictRequest):
    drug_code = drug_code.upper()

    if drug_code not in DRUG_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drug: {drug_code}. Valid codes: {DRUG_CODES}"
        )

    days          = max(1, min(body.days_ahead, 90))
    current_stock = body.current_stock
    predictions   = forecast_n_days(drug_code, days)
    total_demand  = sum(predictions)
    avg_daily     = total_demand / days

    days_until_stockout = None
    cumulative          = 0.0
    for i, pred in enumerate(predictions):
        cumulative += pred
        if cumulative >= current_stock:
            days_until_stockout = i + 1
            break

    reorder = (
        days_until_stockout is not None and days_until_stockout <= 7
    ) or (
        days_until_stockout is None and current_stock < avg_daily * 7
    )

    mae        = loaded_models[drug_code]["best_mae"]
    confidence = "EXCELLENT" if mae < 3 else "GOOD" if mae < 7 else "OK"

    return PredictResponse(
        drug_code            = drug_code,
        drug_name            = drug_names.get(drug_code, drug_code),
        days_ahead           = days,
        predictions          = predictions,
        total_predicted      = round(total_demand, 2),
        current_stock        = current_stock,
        days_until_stockout  = days_until_stockout,
        reorder_recommended  = reorder,
        confidence           = confidence,
    )


@app.get("/inventory", tags=["Inventory"])
def inventory_check():
    default_stocks = {
        "M01AB": 150, "M01AE": 120, "N02BA": 100,
        "N02BE": 400, "N05B":  200, "N05C":  80,
        "R03":   180, "R06":   90,
    }

    report = []
    for drug in DRUG_CODES:
        if drug not in loaded_models:
            continue

        predictions = forecast_n_days(drug, 30)
        avg_daily   = sum(predictions) / 30
        stock       = default_stocks.get(drug, 100)
        days_left   = int(stock / avg_daily) if avg_daily > 0 else 999

        if days_left <= 7:
            risk = "CRITICAL"
        elif days_left <= 14:
            risk = "WARNING"
        else:
            risk = "SAFE"

        report.append(InventoryItem(
            drug_code           = drug,
            drug_name           = drug_names.get(drug, drug),
            current_stock       = stock,
            avg_daily_demand    = round(avg_daily, 2),
            days_until_stockout = days_left,
            risk_level          = risk,
        ))

    order = {"CRITICAL": 0, "WARNING": 1, "SAFE": 2}
    report.sort(key=lambda x: order[x.risk_level])

    return {
        "total_drugs": len(report),
        "critical":    sum(1 for r in report if r.risk_level == "CRITICAL"),
        "warning":     sum(1 for r in report if r.risk_level == "WARNING"),
        "safe":        sum(1 for r in report if r.risk_level == "SAFE"),
        "items":       report,
    }


@app.post("/upload", tags=["Data"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a new salesdaily.csv.
    Saves it, re-runs the data pipeline, reloads scalers.
    NOTE: You must retrain models manually after upload for predictions to use new data.
    """

    # Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    # Validate file is not empty
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save to data/raw/
    safe_makedirs(RAW_DATA_DIR)
    save_path = os.path.join(RAW_DATA_DIR, "salesdaily.csv")

    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Run data pipeline
    try:
        # Import here to avoid circular import issues
        import importlib
        import backend.data_engine as de
        importlib.reload(de)
        de.run_pipeline(save_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}. Check that your CSV has columns: datum, M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06"
        )

    # Reload scalers so predictions use new data immediately
    try:
        global loaded_scalers
        scaler_path = os.path.join(PROCESSED_DIR, "scalers.pkl")
        with open(scaler_path, "rb") as f:
            loaded_scalers = pickle.load(f)
    except Exception as e:
        pass  # Non-fatal — old scalers still work

    return {
        "status":  "SUCCESS",
        "message": "CSV uploaded and data pipeline completed successfully.",
        "file":    save_path,
        "rows":    len(contents.decode("utf-8", errors="ignore").strip().split("\n")) - 1,
        "note":    "Re-run train.py to retrain models on the new data.",
    }