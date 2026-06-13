"""
FastAPI REST API – Shipment Delay Predictor
Endpoints:
  POST /predict          → single shipment prediction
  POST /predict/batch    → batch predictions
  GET  /health           → health check + model metadata
  GET  /features         → list required input fields
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import uvicorn

# ── Load model ───────────────────────────────────────────
MODEL_PATH    = "models/delay_model.pkl"
METADATA_PATH = "models/model_metadata.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_pipeline.py first.")

model    = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

NUMERIC_FEATURES     = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="ML-powered shipment delay risk scoring using Gradient Boosting classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────
WEATHER_OPTIONS  = ["Clear", "Cloudy", "Rainy", "Storm", "Hurricane"]
CARRIER_OPTIONS  = ["MaerskLine", "MSC", "COSCO", "EverGreen", "HapagLloyd"]
PORT_OPTIONS     = ["Shanghai", "Los Angeles", "Rotterdam", "New York",
                    "Singapore", "Hamburg", "Dubai", "Mumbai",
                    "Busan", "Seattle", "Hong Kong", "London",
                    "Guangzhou", "Sydney", "Antwerp", "Chicago"]

class ShipmentInput(BaseModel):
    origin_port:              str   = Field(..., example="Shanghai")
    destination_port:         str   = Field(..., example="Los Angeles")
    carrier:                  str   = Field(..., example="MaerskLine")
    ship_date:                str   = Field(..., example="2024-06-15")
    base_transit_days:        int   = Field(..., ge=1, le=60, example=14)
    port_congestion_score:    float = Field(..., ge=0, le=1, example=0.45)
    weather_condition:        str   = Field(..., example="Cloudy")
    customs_complexity:       int   = Field(..., ge=1, le=3, example=2)
    cargo_weight_tons:        float = Field(..., gt=0, example=120.5)
    num_stops:                int   = Field(..., ge=0, le=5, example=1)
    is_peak_season:           int   = Field(..., ge=0, le=1, example=0)
    carrier_on_time_rate:     float = Field(..., ge=0, le=1, example=0.82)
    days_since_maintenance:   int   = Field(..., ge=0, le=365, example=45)
    fuel_cost_index:          float = Field(..., gt=0, example=1.05)

    @field_validator("weather_condition")
    @classmethod
    def valid_weather(cls, v):
        if v not in WEATHER_OPTIONS:
            raise ValueError(f"weather_condition must be one of {WEATHER_OPTIONS}")
        return v

    @field_validator("carrier")
    @classmethod
    def valid_carrier(cls, v):
        if v not in CARRIER_OPTIONS:
            raise ValueError(f"carrier must be one of {CARRIER_OPTIONS}")
        return v


class PredictionResponse(BaseModel):
    shipment_id:       Optional[str]
    delay_probability: float
    risk_level:        str
    prediction:        str
    confidence:        str
    top_risk_factors:  List[str]


class BatchInput(BaseModel):
    shipments: List[ShipmentInput]


# ── Helper ────────────────────────────────────────────────
WEATHER_SEVERITY = {"Clear": 0, "Cloudy": 1, "Rainy": 2, "Storm": 3, "Hurricane": 4}

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df["ship_date"]  = pd.to_datetime(df["ship_date"])
    df["ship_month"]   = df["ship_date"].dt.month
    df["ship_quarter"] = df["ship_date"].dt.quarter
    df["ship_dayofweek"] = df["ship_date"].dt.dayofweek
    df["is_weekend_departure"] = (df["ship_dayofweek"] >= 5).astype(int)

    df["congestion_x_stops"] = df["port_congestion_score"] * df["num_stops"]
    df["weight_per_stop"]    = df["cargo_weight_tons"] / (df["num_stops"] + 1)
    df["maintenance_risk"]   = (df["days_since_maintenance"] / 180).clip(0, 1)
    df["carrier_risk_score"] = 1 - df["carrier_on_time_rate"]
    df["weather_severity"]   = df["weather_condition"].map(WEATHER_SEVERITY)
    df["high_risk_weather"]  = (df["weather_severity"] >= 3).astype(int)
    df["log_weight"]         = np.log1p(df["cargo_weight_tons"])

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.30:
        return "LOW",    "High confidence – shipment likely on time"
    elif prob < 0.55:
        return "MEDIUM", "Moderate risk – monitor closely"
    elif prob < 0.75:
        return "HIGH",   "High risk – consider contingency planning"
    else:
        return "CRITICAL","Very high risk – immediate attention required"


def top_factors(data: dict) -> List[str]:
    factors = []
    if data["weather_condition"] in ("Storm", "Hurricane"):
        factors.append(f"Severe weather: {data['weather_condition']}")
    if data["port_congestion_score"] > 0.6:
        factors.append(f"High port congestion ({data['port_congestion_score']:.0%})")
    if data["num_stops"] >= 2:
        factors.append(f"Multiple stops ({data['num_stops']})")
    if data["customs_complexity"] == 3:
        factors.append("Complex customs clearance")
    if data["is_peak_season"]:
        factors.append("Peak shipping season")
    if data["carrier_on_time_rate"] < 0.75:
        factors.append(f"Low carrier on-time rate ({data['carrier_on_time_rate']:.0%})")
    if data["days_since_maintenance"] > 120:
        factors.append(f"High maintenance gap ({data['days_since_maintenance']} days)")
    return factors[:4] if factors else ["No major risk flags detected"]


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  metadata["model_name"],
        "roc_auc": round(metadata["roc_auc"], 4),
        "avg_precision": round(metadata["avg_precision"], 4),
    }


@app.get("/features")
def features():
    return {
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "weather_options":      WEATHER_OPTIONS,
        "carrier_options":      CARRIER_OPTIONS,
        "port_options":         PORT_OPTIONS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ShipmentInput):
    try:
        data    = payload.model_dump()
        X       = engineer_features(data)
        prob    = float(model.predict_proba(X)[0, 1])
        risk, conf = risk_label(prob)
        return PredictionResponse(
            shipment_id=None,
            delay_probability=round(prob, 4),
            risk_level=risk,
            prediction="DELAYED" if prob >= 0.5 else "ON-TIME",
            confidence=conf,
            top_risk_factors=top_factors(data),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(payload: BatchInput):
    results = []
    for i, shipment in enumerate(payload.shipments):
        data  = shipment.model_dump()
        X     = engineer_features(data)
        prob  = float(model.predict_proba(X)[0, 1])
        risk, conf = risk_label(prob)
        results.append({
            "index":            i,
            "delay_probability": round(prob, 4),
            "risk_level":        risk,
            "prediction":        "DELAYED" if prob >= 0.5 else "ON-TIME",
        })
    delayed = sum(1 for r in results if r["prediction"] == "DELAYED")
    return {
        "total":    len(results),
        "delayed":  delayed,
        "on_time":  len(results) - delayed,
        "results":  results,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
