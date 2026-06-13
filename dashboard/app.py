"""
Streamlit Dashboard – Shipment Delay Risk Monitor
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import sys
from datetime import date, timedelta

# ── Path setup ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH    = os.path.join(ROOT, "models", "delay_model.pkl")
METADATA_PATH = os.path.join(ROOT, "models", "model_metadata.pkl")
DATA_PATH     = os.path.join(ROOT, "data",   "shipments.csv")

# ── Load assets ───────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, metadata = load_model()
NUMERIC_FEATURES     = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

WEATHER_SEVERITY = {"Clear": 0, "Cloudy": 1, "Rainy": 2, "Storm": 3, "Hurricane": 4}

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df["ship_date"]            = pd.to_datetime(df["ship_date"])
    df["ship_month"]           = df["ship_date"].dt.month
    df["ship_quarter"]         = df["ship_date"].dt.quarter
    df["ship_dayofweek"]       = df["ship_date"].dt.dayofweek
    df["is_weekend_departure"] = (df["ship_dayofweek"] >= 5).astype(int)
    df["congestion_x_stops"]   = df["port_congestion_score"] * df["num_stops"]
    df["weight_per_stop"]      = df["cargo_weight_tons"] / (df["num_stops"] + 1)
    df["maintenance_risk"]     = (df["days_since_maintenance"] / 180).clip(0, 1)
    df["carrier_risk_score"]   = 1 - df["carrier_on_time_rate"]
    df["weather_severity"]     = df["weather_condition"].map(WEATHER_SEVERITY)
    df["high_risk_weather"]    = (df["weather_severity"] >= 3).astype(int)
    df["log_weight"]           = np.log1p(df["cargo_weight_tons"])
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

def risk_color(level):
    return {"LOW": "#16a34a", "MEDIUM": "#d97706", "HIGH": "#dc2626", "CRITICAL": "#7c3aed"}.get(level, "#6b7280")

def risk_emoji(level):
    return {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}.get(level, "⚪")

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Shipment Delay Risk Monitor",
    page_icon="🚢",
    layout="wide",
)

st.markdown("""
<style>
  .metric-card { background:#f8fafc; border-radius:10px; padding:16px; border:1px solid #e2e8f0; }
  .risk-badge  { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:700; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/container-ship.png", width=72)
st.sidebar.title("🚢 Delay Predictor")
st.sidebar.caption(f"Model: **{metadata['model_name']}** | AUC: **{metadata['roc_auc']:.3f}**")

page = st.sidebar.radio("Navigation", ["🔍 Predict Single Shipment", "📊 Fleet Analytics", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════
# PAGE 1 – SINGLE PREDICTION
# ══════════════════════════════════════════════════════════
if page == "🔍 Predict Single Shipment":
    st.title("🔍 Shipment Delay Risk Prediction")
    st.caption("Enter shipment details below to get an instant delay risk score.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Route & Carrier")
        origin = st.selectbox("Origin Port", ["Shanghai","Rotterdam","Singapore","Dubai","Busan","Hong Kong","Guangzhou","Antwerp"])
        destination = st.selectbox("Destination Port", ["Los Angeles","New York","Hamburg","Mumbai","Seattle","London","Sydney","Chicago"])
        carrier = st.selectbox("Carrier", ["MaerskLine","MSC","COSCO","EverGreen","HapagLloyd"])
        ship_date = st.date_input("Ship Date", value=date.today() + timedelta(days=7))

    with col2:
        st.subheader("Transit & Cargo")
        base_transit = st.slider("Base Transit Days", 3, 30, 14)
        cargo_weight = st.number_input("Cargo Weight (tons)", 1.0, 5000.0, 120.0, step=10.0)
        num_stops    = st.slider("Number of Stops", 0, 5, 1)
        customs      = st.selectbox("Customs Complexity", [1, 2, 3], index=1,
                                     format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])

    with col3:
        st.subheader("Risk Factors")
        weather      = st.selectbox("Weather Condition", ["Clear","Cloudy","Rainy","Storm","Hurricane"])
        congestion   = st.slider("Port Congestion Score", 0.0, 1.0, 0.3, step=0.01)
        on_time_rate = st.slider("Carrier On-Time Rate", 0.5, 1.0, 0.82, step=0.01)
        maint_days   = st.slider("Days Since Maintenance", 0, 180, 45)
        fuel_idx     = st.slider("Fuel Cost Index", 0.7, 1.5, 1.05, step=0.01)
        peak_season  = st.checkbox("Peak Season", value=False)

    st.markdown("---")
    if st.button("🚀 Predict Delay Risk", type="primary", use_container_width=True):
        payload = {
            "origin_port": origin, "destination_port": destination,
            "carrier": carrier, "ship_date": str(ship_date),
            "base_transit_days": base_transit, "port_congestion_score": congestion,
            "weather_condition": weather, "customs_complexity": customs,
            "cargo_weight_tons": cargo_weight, "num_stops": num_stops,
            "is_peak_season": int(peak_season), "carrier_on_time_rate": on_time_rate,
            "days_since_maintenance": maint_days, "fuel_cost_index": fuel_idx,
        }
        X    = engineer_features(payload)
        prob = float(model.predict_proba(X)[0, 1])

        if prob < 0.30:   risk = "LOW"
        elif prob < 0.55: risk = "MEDIUM"
        elif prob < 0.75: risk = "HIGH"
        else:             risk = "CRITICAL"

        # Results
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Delay Probability", f"{prob:.1%}")
        r2.metric("Risk Level", f"{risk_emoji(risk)} {risk}")
        r3.metric("Prediction", "⚠️ DELAYED" if prob >= 0.5 else "✅ ON-TIME")
        r4.metric("Route", f"{origin} → {destination}")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(["Risk"], [prob], color=risk_color(risk), height=0.4)
        ax.barh(["Risk"], [1 - prob], left=[prob], color="#e2e8f0", height=0.4)
        ax.set_xlim(0, 1); ax.set_xlabel("Delay Probability")
        ax.axvline(0.5, color="#374151", linestyle="--", alpha=0.5, label="Threshold (0.5)")
        ax.set_title(f"Risk Score: {prob:.1%}  |  {risk}", fontweight="bold")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Risk factors
        st.subheader("⚠️ Key Risk Factors")
        factors = []
        if weather in ("Storm","Hurricane"): factors.append(f"🌊 Severe weather: **{weather}**")
        if congestion > 0.6:                 factors.append(f"🏗️ High port congestion: **{congestion:.0%}**")
        if num_stops >= 2:                   factors.append(f"🔄 Multiple stops: **{num_stops}**")
        if customs == 3:                     factors.append("📋 **High** customs complexity")
        if peak_season:                      factors.append("📦 **Peak season** shipping period")
        if on_time_rate < 0.75:              factors.append(f"🚢 Low carrier on-time rate: **{on_time_rate:.0%}**")
        if not factors:                      factors = ["✅ No major risk flags detected"]
        for f in factors: st.markdown(f"• {f}")


# ══════════════════════════════════════════════════════════
# PAGE 2 – FLEET ANALYTICS
# ══════════════════════════════════════════════════════════
elif page == "📊 Fleet Analytics":
    st.title("📊 Fleet Analytics Dashboard")
    df = load_data()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Shipments", f"{len(df):,}")
    k2.metric("Overall Delay Rate", f"{df['is_delayed'].mean():.1%}")
    k3.metric("Avg Transit Days", f"{df['base_transit_days'].mean():.1f}")
    k4.metric("Carriers", str(df["carrier"].nunique()))

    st.markdown("---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Fleet Analytics", fontsize=14, fontweight="bold")

    # Delay by carrier
    delay_carrier = df.groupby("carrier")["is_delayed"].mean().sort_values(ascending=False)
    axes[0,0].bar(delay_carrier.index, delay_carrier.values, color="#3b82f6")
    axes[0,0].set_title("Delay Rate by Carrier"); axes[0,0].set_ylabel("Delay Rate")
    axes[0,0].tick_params(axis="x", rotation=30)

    # Delay by weather
    delay_weather = df.groupby("weather_condition")["is_delayed"].mean().reindex(
        ["Clear","Cloudy","Rainy","Storm","Hurricane"])
    axes[0,1].bar(delay_weather.index, delay_weather.values,
                  color=["#16a34a","#84cc16","#d97706","#ef4444","#7c3aed"])
    axes[0,1].set_title("Delay Rate by Weather"); axes[0,1].set_ylabel("Delay Rate")
    axes[0,1].tick_params(axis="x", rotation=20)

    # Congestion histogram
    axes[0,2].hist(df["port_congestion_score"], bins=30, color="#60a5fa", edgecolor="white")
    axes[0,2].set_title("Port Congestion Distribution"); axes[0,2].set_xlabel("Score")

    # Delay by stops
    delay_stops = df.groupby("num_stops")["is_delayed"].mean()
    axes[1,0].bar(delay_stops.index, delay_stops.values, color="#f97316")
    axes[1,0].set_title("Delay Rate by # Stops"); axes[1,0].set_xlabel("Stops")

    # Delay rate trend (monthly)
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    df["month"] = df["ship_date"].dt.to_period("M")
    monthly = df.groupby("month")["is_delayed"].mean()
    axes[1,1].plot(range(len(monthly)), monthly.values, color="#8b5cf6", marker="o", markersize=3)
    axes[1,1].set_title("Monthly Delay Rate Trend"); axes[1,1].set_ylabel("Delay Rate")
    axes[1,1].tick_params(axis="x", rotation=45)

    # Cargo weight vs delay
    on_time = df[df["is_delayed"] == 0]["cargo_weight_tons"]
    delayed = df[df["is_delayed"] == 1]["cargo_weight_tons"]
    axes[1,2].hist(np.log1p(on_time), bins=25, alpha=0.6, label="On-Time", color="#16a34a")
    axes[1,2].hist(np.log1p(delayed), bins=25, alpha=0.6, label="Delayed",  color="#dc2626")
    axes[1,2].set_title("Log Cargo Weight by Outcome"); axes[1,2].legend()

    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════
# PAGE 3 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Report")

    eval_img = os.path.join(ROOT, "models", "evaluation_report.png")
    if os.path.exists(eval_img):
        st.image(eval_img, caption="Evaluation Report", use_container_width=True)
    else:
        st.warning("Run train_pipeline.py to generate evaluation charts.")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model",         metadata["model_name"])
    c2.metric("ROC-AUC",       f"{metadata['roc_auc']:.4f}")
    c3.metric("Avg Precision", f"{metadata['avg_precision']:.4f}")

    with st.expander("ℹ️ Model Details"):
        df_info = load_data()
        n_test = int(len(df_info) * 0.2)
        n_train = len(df_info) - n_test
        st.markdown(f"""
| Property | Value |
|---|---|
| Algorithm | {metadata['model_name']} |
| Total samples | {len(df_info):,} |
| Training samples | {n_train:,} |
| Test samples | {n_test:,} |
| Numeric features | {len(NUMERIC_FEATURES)} |
| Categorical features | {len(CATEGORICAL_FEATURES)} |
| ROC-AUC (test) | {metadata['roc_auc']:.4f} |
| Average Precision | {metadata['avg_precision']:.4f} |
        """)
