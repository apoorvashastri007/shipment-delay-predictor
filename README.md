# 🚢 Shipment Delay Prediction System

An end-to-end ML system to predict shipment delays using features like transit time, port congestion, weather, and carrier history.

---

## 📁 Project Structure

```
shipment-delay-predictor/
├── data/
│   ├── generate_data.py        # Synthetic dataset generator (8,000 shipments)
│   └── shipments.csv           # Generated dataset
├── models/
│   ├── delay_model.pkl         # Trained Gradient Boosting pipeline
│   ├── model_metadata.pkl      # Feature names, metrics
│   └── evaluation_report.png  # ROC, PR, confusion matrix, feature importance
├── api/
│   └── main.py                 # FastAPI REST API
├── dashboard/
│   └── app.py                  # Streamlit real-time dashboard
├── train_pipeline.py           # Full ML pipeline (preprocessing → training → eval)
├── Dockerfile.api               # Container for the FastAPI service
├── Dockerfile.dashboard         # Container for the Streamlit dashboard
├── docker-compose.yml           # Run both services together
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data & train model
```bash
python data/generate_data.py     # creates data/shipments.csv
python train_pipeline.py          # trains model, saves to models/
```

### 3. Start the FastAPI server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 4. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
# Opens: http://localhost:8501
```

---

## 🐳 Running with Docker

Both services can be built and run as containers, either individually or together with Docker Compose.

### Run everything with Docker Compose (recommended)
```bash
docker-compose up --build
```
- FastAPI: http://localhost:8000/docs
- Streamlit dashboard: http://localhost:8501

### Or build/run each service individually

**API:**
```bash
docker build -f Dockerfile.api -t shipment-delay-api .
docker run -p 8000:8000 shipment-delay-api
```

**Dashboard:**
```bash
docker build -f Dockerfile.dashboard -t shipment-delay-dashboard .
docker run -p 8501:8501 shipment-delay-dashboard
```

> Note: the model is trained ahead of time and shipped as part of the image (`models/delay_model.pkl`). If you change the data or pipeline, re-run `python train_pipeline.py` before rebuilding the images.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model info + AUC metrics |
| GET | `/features` | Required input fields & options |
| POST | `/predict` | Single shipment prediction |
| POST | `/predict/batch` | Batch predictions |

### Example — Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin_port": "Shanghai",
    "destination_port": "Los Angeles",
    "carrier": "MaerskLine",
    "ship_date": "2024-08-15",
    "base_transit_days": 14,
    "port_congestion_score": 0.65,
    "weather_condition": "Storm",
    "customs_complexity": 2,
    "cargo_weight_tons": 120.5,
    "num_stops": 1,
    "is_peak_season": 1,
    "carrier_on_time_rate": 0.78,
    "days_since_maintenance": 90,
    "fuel_cost_index": 1.10
  }'
```

### Response
```json
{
  "shipment_id": null,
  "delay_probability": 0.6732,
  "risk_level": "HIGH",
  "prediction": "DELAYED",
  "confidence": "High risk – consider contingency planning",
  "top_risk_factors": [
    "Severe weather: Storm",
    "High port congestion (65%)",
    "Peak shipping season"
  ]
}
```

---

## 🧠 ML Pipeline

### Features (20 total after engineering)
| Category | Features |
|----------|----------|
| Route | `origin_port`, `destination_port`, `base_transit_days`, `num_stops` |
| Carrier | `carrier`, `carrier_on_time_rate`, `carrier_risk_score` |
| Weather | `weather_condition`, `weather_severity`, `high_risk_weather` |
| Port | `port_congestion_score`, `congestion_x_stops` |
| Cargo | `cargo_weight_tons`, `log_weight`, `weight_per_stop` |
| Operations | `customs_complexity`, `days_since_maintenance`, `maintenance_risk`, `fuel_cost_index`, `is_peak_season` |
| Temporal | `ship_month`, `ship_quarter`, `ship_dayofweek`, `is_weekend_departure` |

### Models Evaluated
- Logistic Regression (baseline)
- Random Forest
- **Gradient Boosting ← Best**
- Voting Ensemble

### Preprocessing
- Numeric: Median imputation → StandardScaler
- Categorical: Mode imputation → OneHotEncoder
- Evaluation: 5-fold stratified cross-validation

---

## 📊 Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| 🟢 LOW | < 30% | No action needed |
| 🟡 MEDIUM | 30–55% | Monitor closely |
| 🔴 HIGH | 55–75% | Contingency planning |
| 🚨 CRITICAL | > 75% | Immediate attention |
